##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Sequential wrapper around ``fsm_continuous_batch.build_dataset``: runs K
independent episodes with num_envs=1 each (one Python subprocess per episode
to prevent mjlab/warp context memory accumulating across runs), then
concatenates the per-episode .npz into one unified dataset. Optionally saves
an mp4 per episode.

Purpose: works around a batch (num_envs>=2) failure of the walk-to-pick handover
we couldn't isolate (mjlab/mjwarp physics diverges subtly between batch=1 and
batch>=2). Sequential N=1 keeps success rate around 78 to 100% per episode.
Subprocess isolation avoids the ~150 MB/episode leak that OOMs a monolithic
loop around episode 60.

Input:  staged experts at ``experts/{walk,pick}.pt``.
Output: ``outputs/walk_pick_sequential.npz`` (unified schema) + optional
        ``outputs/media/seq_ep_<ii>_seed<S>.mp4`` per episode. Per-episode
        intermediate .npz files are kept under ``/tmp/fsm_seq/`` until the
        final concatenation succeeds, so a crash mid-run can be recovered.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "walk_pick_sequential.npz"
DEFAULT_VIDEOS = ROOT / "outputs" / "media"


def _run_episode_subprocess(
    i: int, seed_i: int, tmp_npz: Path, video_path: Optional[Path],
    device: str, max_steps: int, spawn_r_min: float, spawn_r_max: float,
    use_astar: bool, fsm_root: Path,
) -> tuple[int, int, int]:
    """Launch a single ``fsm_continuous_batch`` subprocess (num_envs=1) and wait for it.
    Args: i: episode index (for log tagging); seed_i: per-episode seed; tmp_npz: output .npz path (skipped if already exists for crash-recovery); video_path: optional mp4 output; device: torch device string; max_steps: per-episode step cap; spawn_r_min/spawn_r_max: annulus radii (m); use_astar: forward the --astar flag; fsm_root: repo root for the subprocess cwd.
    Returns: (i, seed_i, returncode); returncode<0 means the npz was reused without re-running.
    """
    if tmp_npz.exists():
        return i, seed_i, -1
    cmd = [
        sys.executable, "-m", "src.fsm_continuous_batch",
        "--num-envs", "1", "--device", device,
        "--out", str(tmp_npz), "--seed", str(seed_i),
        "--max-steps", str(max_steps),
        "--r-min", str(spawn_r_min), "--r-max", str(spawn_r_max),
    ]
    if video_path is not None:
        cmd += ["--video-path", str(video_path)]
    if use_astar:
        cmd += ["--astar"]
    r = subprocess.run(
        cmd, cwd=str(fsm_root),
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not tmp_npz.exists():
        print(f"[seq] ep {i:03d} seed={seed_i}: subprocess FAILED (exit {r.returncode})")
        print(r.stderr[-1500:] if r.stderr else "(no stderr)")
    return i, seed_i, r.returncode


def build_sequential_dataset(
    K: int,
    base_seed: int = 100,
    out_path: Path | str = DEFAULT_OUT,
    device: str = "cuda:0",
    max_steps: int = 800,
    spawn_r_min: float = 1.0,
    spawn_r_max: float = 3.0,
    videos_dir: Optional[Path | str] = None,
    use_astar: bool = False,
    failed_dir: Optional[Path | str] = None,
    workers: int = 1,
) -> dict:
    """Run K single-env episodes and concatenate into one .npz, optionally saving mp4 per episode.
    Args: K: episode count; base_seed: first episode's seed; out_path: final concatenated .npz; device: torch device; max_steps: per-episode cap; spawn_r_min: inner annulus radius; spawn_r_max: outer annulus radius; videos_dir: if set, writes one mp4 per episode there; use_astar: forward --astar to each subprocess so the walk phase follows an A* path around the pick table; failed_dir: if set, per-episode .npz of fallen episodes is copied here (plus a metadata.json summarizing all episodes) so the caller can diagnose why the FSM chain fails; workers: parallel subprocess pool size (default 1 = original serial behavior; each subprocess still uses num_envs=1, so the mjwarp batch divergence that destabilised the walk-to-pick handover at num_envs>=2 is sidestepped by process isolation).
    Returns: dict with total_transitions, pick_successes, walk_kept, pick_kept, per_episode list.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out_path = Path(out_path)
    if videos_dir is not None:
        videos_dir = Path(videos_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)
    if failed_dir is not None:
        failed_dir = Path(failed_dir)
        failed_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/fsm_seq")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fsm_root = Path(__file__).resolve().parent.parent

    def paths_for(i: int) -> tuple[int, Path, Optional[Path]]:
        """Return (seed, tmp_npz, video_path) for episode index i."""
        seed_i = base_seed + i
        tmp_npz = tmp_dir / f"ep_{i:03d}_seed{seed_i}.npz"
        video_path = videos_dir / f"seq_ep_{i:03d}_seed{seed_i}.mp4" if videos_dir is not None else None
        return seed_i, tmp_npz, video_path

    # Phase 1: launch subprocesses (serial if workers=1, parallel otherwise). Each writes
    # its own tmp_npz; we only need the return codes to decide whether to process it. One
    # progress line per completed subprocess so a long run (K=2000) is not silent.
    workers = max(1, int(workers))

    def _print_progress(k_done: int, i: int, seed_i: int, rc: int) -> None:
        """Live progress print: episode just finished (rc<0 means we reused a cached npz).
        Args: k_done: how many episodes have completed so far (1-indexed); i: episode index; seed_i: episode seed; rc: subprocess returncode (<0 = reused existing tmp_npz).
        Returns: none.
        """
        status = "reused" if rc < 0 else ("done" if rc == 0 else f"FAILED rc={rc}")
        print(f"[seq] {k_done:>4d}/{K:<4d} ep {i:03d} seed={seed_i}: {status}", flush=True)

    if workers == 1:
        for i in range(K):
            seed_i, tmp_npz, video_path = paths_for(i)
            _, _, rc = _run_episode_subprocess(
                i, seed_i, tmp_npz, video_path, device, max_steps,
                spawn_r_min, spawn_r_max, use_astar, fsm_root,
            )
            _print_progress(i + 1, i, seed_i, rc)
    else:
        print(f"[seq] running with {workers} parallel subprocess workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = []
            for i in range(K):
                seed_i, tmp_npz, video_path = paths_for(i)
                futs.append(pool.submit(
                    _run_episode_subprocess,
                    i, seed_i, tmp_npz, video_path, device, max_steps,
                    spawn_r_min, spawn_r_max, use_astar, fsm_root,
                ))
            k_done = 0
            for fut in as_completed(futs):
                i_done, seed_done, rc = fut.result()  # surfaces any exception too
                k_done += 1
                _print_progress(k_done, i_done, seed_done, rc)

    # Phase 2: load each per-ep npz in index order and build per_episode summaries. Running
    # this serially (even when workers>1) keeps the output deterministic and the npz reads
    # cheap.
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_phase: list[np.ndarray] = []
    per_episode: list[dict] = []
    total_pick = 0
    dropped_fell = 0
    dropped_stuck = 0  # kept upright but never closed pick before max_steps; these frames were the majority of pre-filter data and were teaching the student to flail at the box
    for i in range(K):
        seed_i, tmp_npz, video_path = paths_for(i)
        if not tmp_npz.exists():
            continue  # subprocess failed; already logged
        blob = np.load(str(tmp_npz))
        obs_i = blob["unified_obs"]
        act_i = blob["action"]
        phase_i = blob["phase"]
        fell_i = int(blob["fell"]) if "fell" in blob.files else 0
        walk_i = int((phase_i == 0).sum())
        pick_i = int((phase_i == 1).sum())
        # Success proxy: has pick transitions AND ended before max_steps (i.e., FSM closed voluntarily).
        ep_success = 1 if (pick_i > 0 and obs_i.shape[0] < max_steps) else 0
        total_pick += ep_success

        if fell_i > 0:
            dropped_fell += 1
            failed_npz_path = None
            if failed_dir is not None:
                failed_npz_path = failed_dir / tmp_npz.name
                shutil.copy2(tmp_npz, failed_npz_path)
            per_episode.append({
                "seed": seed_i,
                "transitions": int(obs_i.shape[0]),
                "walk_kept": walk_i,
                "pick_kept": pick_i,
                "pick_success": ep_success,
                "fell": fell_i,
                "dropped": True,
                "drop_reason": "fell",
                "video": str(video_path) if video_path else None,
                "failed_npz": str(failed_npz_path) if failed_npz_path else None,
            })
            print(
                f"[seq] ep {i:03d} seed={seed_i}: DROPPED (robot fell) "
                f"transitions={obs_i.shape[0]} (walk={walk_i}, pick={pick_i})"
            )
            continue

        if ep_success == 0:
            dropped_stuck += 1
            per_episode.append({
                "seed": seed_i,
                "transitions": int(obs_i.shape[0]),
                "walk_kept": walk_i,
                "pick_kept": pick_i,
                "pick_success": 0,
                "fell": 0,
                "dropped": True,
                "drop_reason": "stuck",
                "video": str(video_path) if video_path else None,
                "failed_npz": None,
            })
            print(
                f"[seq] ep {i:03d} seed={seed_i}: DROPPED (stuck at max_steps, never lifted box) "
                f"transitions={obs_i.shape[0]} (walk={walk_i}, pick={pick_i})"
            )
            continue

        all_obs.append(obs_i); all_act.append(act_i); all_phase.append(phase_i)
        per_episode.append({
            "seed": seed_i,
            "transitions": int(obs_i.shape[0]),
            "walk_kept": walk_i,
            "pick_kept": pick_i,
            "pick_success": ep_success,
            "fell": fell_i,
            "dropped": False,
            "video": str(video_path) if video_path else None,
            "failed_npz": None,
        })
        print(
            f"[seq] ep {i:03d} seed={seed_i}: "
            f"success={ep_success}/1 transitions={obs_i.shape[0]} "
            f"(walk={walk_i}, pick={pick_i})"
        )

    obs = np.concatenate(all_obs, axis=0) if all_obs else np.zeros((0, 115), dtype=np.float32)
    act = np.concatenate(all_act, axis=0) if all_act else np.zeros((0, 29), dtype=np.float32)
    phase = np.concatenate(all_phase, axis=0) if all_phase else np.zeros((0,), dtype=np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, unified_obs=obs, action=act, phase=phase)
    walk_kept = int((phase == 0).sum())
    pick_kept = int((phase == 1).sum())
    size_mb = out_path.stat().st_size / 1e6
    kept_eps = K - dropped_fell - dropped_stuck
    print(
        f"\n[seq] wrote {out_path} ({size_mb:.1f} MB): {obs.shape[0]} transitions "
        f"(walk={walk_kept}, pick={pick_kept}) over {kept_eps}/{K} kept episodes "
        f"({dropped_fell} dropped due to fall, {dropped_stuck} dropped due to stuck-at-max-steps; "
        f"{total_pick} pick successes = {total_pick / max(K, 1) * 100:.0f}%)"
    )
    if failed_dir is not None:
        metadata_path = failed_dir / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump({
                "episodes": K,
                "base_seed": base_seed,
                "max_steps": max_steps,
                "spawn_r_min": spawn_r_min,
                "spawn_r_max": spawn_r_max,
                "use_astar": use_astar,
                "kept_episodes": kept_eps,
                "dropped_fell": dropped_fell,
                "dropped_stuck": dropped_stuck,
                "per_episode": per_episode,
            }, f, indent=2)
        print(f"[seq] wrote {metadata_path} ({dropped_fell} failed episodes archived)")

    # Concat succeeded, so per-episode intermediates are no longer needed for recovery.
    for i in range(K):
        seed_i = base_seed + i
        (tmp_dir / f"ep_{i:03d}_seed{seed_i}.npz").unlink(missing_ok=True)
    return {
        "total_transitions": int(obs.shape[0]),
        "walk_kept": walk_kept,
        "pick_kept": pick_kept,
        "pick_successes": total_pick,
        "episodes": K,
        "dropped_fell": dropped_fell,
        "dropped_stuck": dropped_stuck,
        "per_episode": per_episode,
    }


def _main() -> None:
    """Parse CLI flags and dispatch to build_sequential_dataset.
    Args: none, reads --episodes, --base-seed, --out, --videos-dir, --max-steps, --r-min, --r-max, --device, --astar from sys.argv.
    Returns: none.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--videos-dir", default=None, help="Directory to save one mp4 per episode.")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--astar", action="store_true", help="Walk phase follows an A* path around the pick table.")
    ap.add_argument("--failed-dir", default=None, help="If set, per-episode .npz of fallen episodes is copied here plus a metadata.json for offline diagnosis.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel subprocess pool size. Each subprocess still runs num_envs=1 (process isolation sidesteps the mjwarp batch divergence that destabilises the walk-to-pick handover at num_envs>=2). Cap by VRAM / num_envs=1 memory footprint, roughly 350 MiB per worker on mjlab/mjwarp.")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    build_sequential_dataset(
        K=args.episodes,
        base_seed=args.base_seed,
        out_path=args.out,
        device=args.device,
        max_steps=args.max_steps,
        spawn_r_min=args.r_min,
        spawn_r_max=args.r_max,
        videos_dir=args.videos_dir,
        use_astar=args.astar,
        failed_dir=args.failed_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    _main()
