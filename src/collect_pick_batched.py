##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/itandroids/itandroids-academic              #
##########################################################################################
"""Batched pick-only dataset generation using the pick env's walk-terminal reset.
Each env resets via reset_from_walk_buffer, then the pick expert rolls up to
max_steps_per_episode sim steps; on lift, fall, or step cap it auto-resets.
Writes a .npz with unified_obs/action/phase matching src.fsm_continuous_batch.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.expert_registry import BY_NAME, resolve_ckpt
from src.fsm_continuous import _disable_terminations
from src.fsm_continuous_batch import PICK_PHASE_ID, _unified_obs_per_env
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "pick_only_batched.npz"


def _episode_outcome(base_env) -> tuple[torch.Tensor, torch.Tensor]:
    """Flag per-env lift success and fall at the current sim step.
    Args: base_env: mjlab env with robot and box entities.
    Returns: (lifted_mask (N,), fell_mask (N,)) boolean tensors on the env device.
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    up_align = -robot.data.projected_gravity_b[:, 2]
    root_z = robot.data.root_link_pos_w[:, 2]
    box_z = box.data.root_link_pos_w[:, 2]
    fell = (up_align < 0.25) | ((root_z < 0.15) & (up_align < 0.7))
    lifted = (box_z > 0.85) & (up_align > 0.8)
    return lifted, fell


def collect_pick_dataset(
    num_envs: int,
    target_episodes: int,
    out_path: Path | str,
    device: str = "cuda:0",
    max_steps_per_episode: int = 150,
    seed: int = 0,
    keep_only_successful: bool = True,
    verbose: bool = True,
) -> dict:
    """Roll parallel pick episodes until target_episodes complete, writing a BC-compatible .npz.
    Args: num_envs: parallel batch size; target_episodes: stop after this many episodes; out_path: .npz destination; device: torch device; max_steps_per_episode: per-episode step cap; seed: RNG seed; keep_only_successful: drop fallen episodes; verbose: progress prints.
    Returns: dict with transitions_written, episodes_collected, episodes_successful, episodes_fell, wall_s.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(max_steps_per_episode) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    device_t = base_env.scene["robot"].data.root_link_pos_w.device

    pick_runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    pick_runner.load(str(resolve_ckpt(BY_NAME["pick"])), load_cfg={"actor": True}, strict=True, map_location=device)
    pick_policy = pick_runner.get_inference_policy(device=device)

    base_env.reset()

    goal = sample_goal(num_envs, device=device_t, seed=seed)
    phase_ids = torch.full((num_envs,), PICK_PHASE_ID, dtype=torch.long, device=device_t)
    last_action = torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t)

    # Track lifted/fell per env to relabel the staged buffer at episode end.
    ep_obs_buf: list[torch.Tensor] = [[] for _ in range(num_envs)]
    ep_act_buf: list[torch.Tensor] = [[] for _ in range(num_envs)]
    ep_step: torch.Tensor = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    ep_fell_mask: torch.Tensor = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    ep_lifted_mask: torch.Tensor = torch.zeros(num_envs, dtype=torch.bool, device=device_t)

    kept_obs: list[np.ndarray] = []
    kept_act: list[np.ndarray] = []
    kept_phase: list[np.ndarray] = []

    episodes_collected = 0
    episodes_successful = 0
    episodes_fell = 0

    t0 = time.time()
    step_abs = 0
    while episodes_collected < target_episodes:
        unified = _unified_obs_per_env(base_env, phase_ids, goal, last_action)
        obs = env.get_observations()
        with torch.no_grad():
            action = pick_policy(obs)

        # Stage obs/action pair before stepping so it matches the pre-step state.
        unified_cpu = unified.detach().cpu()
        action_cpu = action.detach().cpu()
        for i in range(num_envs):
            ep_obs_buf[i].append(unified_cpu[i])
            ep_act_buf[i].append(action_cpu[i])

        env.step(action)
        last_action = action.detach()

        new_lifted, new_fell = _episode_outcome(base_env)
        ep_lifted_mask |= new_lifted
        ep_fell_mask |= new_fell
        ep_step += 1

        done_mask = ep_lifted_mask | ep_fell_mask | (ep_step >= max_steps_per_episode)
        if done_mask.any():
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            for i in done_ids:
                fell = bool(ep_fell_mask[i])
                lifted = bool(ep_lifted_mask[i])
                episodes_collected += 1
                if lifted:
                    episodes_successful += 1
                if fell:
                    episodes_fell += 1
                keep = (not keep_only_successful) or (lifted and not fell)
                if keep and ep_obs_buf[i]:
                    kept_obs.append(torch.stack(ep_obs_buf[i]).numpy().astype(np.float32))
                    kept_act.append(torch.stack(ep_act_buf[i]).numpy().astype(np.float32))
                    kept_phase.append(np.full(len(ep_obs_buf[i]), PICK_PHASE_ID, dtype=np.int8))
                ep_obs_buf[i] = []
                ep_act_buf[i] = []
                ep_step[i] = 0
                ep_fell_mask[i] = False
                ep_lifted_mask[i] = False

            # Reset finished envs; this re-triggers reset_from_walk_buffer.
            env_ids = torch.tensor(done_ids, dtype=torch.int32, device=device_t)
            base_env.reset(env_ids=env_ids)
            last_action[done_ids] = 0.0
            new_goal = sample_goal(len(done_ids), device=device_t, seed=seed + 1 + step_abs)
            for name in ("object_init_pos_w", "object_init_quat_w", "place_target_pos_w", "place_target_quat_w"):
                getattr(goal, name)[done_ids] = getattr(new_goal, name)

        step_abs += 1
        if verbose and step_abs % 50 == 0:
            print(f"[pick-batch] step {step_abs:4d} episodes_collected={episodes_collected}/{target_episodes}  "
                  f"successful={episodes_successful}  fell={episodes_fell}  wall={time.time()-t0:.1f}s")

    wall_s = time.time() - t0
    base_env.close()

    if kept_obs:
        unified_arr = np.concatenate(kept_obs, axis=0)
        action_arr = np.concatenate(kept_act, axis=0)
        phase_arr = np.concatenate(kept_phase, axis=0)
    else:
        unified_arr = np.zeros((0, UNIFIED_OBS_DIM), dtype=np.float32)
        action_arr = np.zeros((0, UNIFIED_ACT_DIM), dtype=np.float32)
        phase_arr = np.zeros((0,), dtype=np.int8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), unified_obs=unified_arr, action=action_arr, phase=phase_arr)

    stats = {
        "transitions_written": int(unified_arr.shape[0]),
        "episodes_collected": episodes_collected,
        "episodes_successful": episodes_successful,
        "episodes_fell": episodes_fell,
        "wall_s": wall_s,
        "episodes_per_s": episodes_collected / max(1e-9, wall_s),
        "out_path": str(out_path),
    }
    if verbose:
        print(f"\n[pick-batch] wrote {out_path}  ({unified_arr.shape[0]} transitions, {out_path.stat().st_size/1e6:.1f} MB)")
        print(f"[pick-batch] stats: {stats}")
    return stats


def main() -> None:
    """Parse CLI flags and run the batched pick-only dataset collection.
    Args: reads --num-envs, --target-episodes, --out, --device, --max-steps, --seed, --keep-all from argv.
    Returns: None.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--target-episodes", type=int, default=10_000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-all", action="store_true", help="keep failed episodes too (default drops them)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    collect_pick_dataset(
        num_envs=args.num_envs,
        target_episodes=args.target_episodes,
        out_path=Path(args.out),
        device=args.device,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        keep_only_successful=not args.keep_all,
    )


if __name__ == "__main__":
    main()
