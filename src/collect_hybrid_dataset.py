##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Collect walk-to-pick dataset via FSM batched rollouts and attach per-row sample weights.

Spawn radii follow the natural FSM distribution; pick is under-represented physically
but the saved sample_weights array makes the effective training distribution match
pick_weight_target when the loader uses WeightedRandomSampler.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from src.fsm_batched_fixed import build_dataset_batched_fixed

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "walk_pick_hybrid.npz"


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three arrays from a BC-format .npz written by the FSM collector.
    Args: path: .npz file with keys unified_obs, action, phase.
    Returns: (unified_obs, action, phase) tuple of ndarrays.
    """
    blob = np.load(str(path))
    return blob["unified_obs"], blob["action"], blob["phase"]


def _phase_weights(phase: np.ndarray, pick_weight_target: float) -> np.ndarray:
    """Compute per-row sample weights so a weighted sampler matches pick_weight_target.
    Args: phase: (N,) int array with 0=walk and 1=pick; pick_weight_target: desired share of pick draws, in (0, 1).
    Returns: (N,) float32 weights normalised to mean 1.
    """
    phase = phase.astype(np.int64)
    n_pick = int((phase == 1).sum())
    n_walk = int((phase == 0).sum())
    if n_pick == 0 or n_walk == 0:
        return np.ones(phase.shape[0], dtype=np.float32)
    w_pick = pick_weight_target / max(1, n_pick)
    w_walk = (1.0 - pick_weight_target) / max(1, n_walk)
    weights = np.where(phase == 1, w_pick, w_walk).astype(np.float32)
    weights *= phase.shape[0] / weights.sum()
    return weights


def collect_hybrid_dataset(
    out_path: Path | str,
    target_transitions: int = 2_000_000,
    pick_weight_target: float = 0.5,
    spawn_r_min: float = 0.2,
    spawn_r_max: float = 5.0,
    fsm_max_steps: int = 1200,
    num_envs: int = 2048,
    device: str = "cuda:0",
    env_spacing: float = 20.0,
    seed: int = 7,
    verbose: bool = True,
) -> dict:
    """Build a BC-compatible .npz from one or more FSM rollouts with per-row sample weights.
    Args: out_path: destination .npz; target_transitions: minimum rows to collect; pick_weight_target: desired pick share under a WeightedRandomSampler; spawn_r_min: inner spawn radius; spawn_r_max: outer spawn radius; fsm_max_steps: FSM step cap per rollout; num_envs: parallel batch size; device: torch device; env_spacing: metres between env origins; seed: base RNG seed, incremented per rollout; verbose: progress prints.
    Returns: dict with transitions_total, walk_transitions, pick_transitions, pick_ratio_physical, pick_weight_target, fsm_wall_s, fsm_rollouts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="hybrid_"))

    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_phase: list[np.ndarray] = []
    rows_so_far = 0
    fsm_wall = 0.0
    rollout_idx = 0
    while rows_so_far < target_transitions:
        tmp = tmp_root / f"fsm_{rollout_idx}.npz"
        if verbose:
            print(f"[hybrid] FSM rollout {rollout_idx+1} (r=[{spawn_r_min},{spawn_r_max}], max_steps={fsm_max_steps}, rows={rows_so_far}/{target_transitions})")
        t0 = time.time()
        build_dataset_batched_fixed(
            num_envs=num_envs,
            device=device,
            out_path=tmp,
            seed=seed + rollout_idx,
            max_steps=fsm_max_steps,
            spawn_r_min=spawn_r_min,
            spawn_r_max=spawn_r_max,
            env_spacing=env_spacing,
            verbose=verbose,
        )
        fsm_wall += time.time() - t0
        obs, act, phase = _load_npz(tmp)
        all_obs.append(obs); all_act.append(act); all_phase.append(phase)
        rows_so_far += obs.shape[0]
        rollout_idx += 1

    unified_obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    action = np.concatenate(all_act, axis=0).astype(np.float32)
    phase = np.concatenate(all_phase, axis=0).astype(np.int8)
    sample_weights = _phase_weights(phase, pick_weight_target)

    np.savez(
        str(out_path),
        unified_obs=unified_obs,
        action=action,
        phase=phase,
        sample_weights=sample_weights,
    )

    n_walk = int((phase == 0).sum())
    n_pick = int((phase == 1).sum())
    total = int(phase.shape[0])
    stats = {
        "transitions_total": total,
        "walk_transitions": n_walk,
        "pick_transitions": n_pick,
        "pick_ratio_physical": n_pick / max(1, total),
        "pick_weight_target": float(pick_weight_target),
        "fsm_wall_s": fsm_wall,
        "fsm_rollouts": rollout_idx,
        "out_path": str(out_path),
    }
    if verbose:
        print(f"\n[hybrid] wrote {out_path} ({total} transitions, {out_path.stat().st_size/1e6:.1f} MB)")
        print(f"[hybrid] walk={n_walk}  pick={n_pick}  pick_ratio_physical={100*n_pick/total:.1f}%")
        print(f"[hybrid] sample_weights saved -> effective pick share under WeightedRandomSampler: {100*pick_weight_target:.0f}%")
        print(f"[hybrid] stats: {stats}")
    return stats


def main() -> None:
    """Parse CLI flags and build the hybrid walk-to-pick dataset.
    Args: none, reads out, target-transitions, pick-weight-target, r-min, r-max, fsm-max-steps, num-envs, env-spacing, device, seed from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--target-transitions", type=int, default=2_000_000,
                    help="Minimum rows to collect across FSM rollouts.")
    ap.add_argument("--pick-weight-target", type=float, default=0.5,
                    help="Effective pick share under a WeightedRandomSampler (0.5 = balanced).")
    ap.add_argument("--r-min", type=float, default=0.2, help="FSM spawn inner radius (m).")
    ap.add_argument("--r-max", type=float, default=5.0, help="FSM spawn outer radius (m).")
    ap.add_argument("--fsm-max-steps", type=int, default=1200)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--env-spacing", type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    collect_hybrid_dataset(
        out_path=Path(args.out),
        target_transitions=args.target_transitions,
        pick_weight_target=args.pick_weight_target,
        spawn_r_min=args.r_min,
        spawn_r_max=args.r_max,
        fsm_max_steps=args.fsm_max_steps,
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
