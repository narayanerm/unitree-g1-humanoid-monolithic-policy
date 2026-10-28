##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Record a 2x2 tracking-camera grid of the walk-pick-carry-drop FSM at num_envs=4.

Input:  trained checkpoints under ``outputs/logs/`` and the batched FSM dataset builder.
Output: mp4 at ``outputs/media/wpcd_n4_grid_latest.mp4`` (or ``--out``) and printed stats.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "outputs" / "media" / "wpcd_n4_grid_latest.mp4"


def main() -> None:
    """Parse CLI flags, run the batched FSM rollout, and write the 2x2 grid mp4.
    Args: none (reads sys.argv via argparse).
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--r-min", type=float, default=0.5)
    ap.add_argument("--r-max", type=float, default=2.5)
    ap.add_argument("--env-spacing", type=float, default=20.0)
    ap.add_argument("--t2-r-min", type=float, default=1.5)
    ap.add_argument("--t2-r-max", type=float, default=3.5)
    ap.add_argument("--pane-h", type=int, default=540)
    ap.add_argument("--pane-w", type=int, default=960)
    ap.add_argument("--cam-distance", type=float, default=3.5)
    ap.add_argument("--fps", type=int, default=40,
                    help="output video fps (default: 40 matches 0.025 s policy dt for real-time playback)")
    ap.add_argument("--carry-mode", default=os.environ.get("WPCD_CARRY", "trained"),
                    choices=["walk", "trained", "pick"],
                    help="which policy drives the carry phase (default: trained)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["WPCD_CARRY"] = args.carry_mode

    from src.fsm_wpcd_batched import build_dataset_wpcd

    out_mp4 = Path(args.out).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp_npz = Path(tf.name)
    try:
        stats = build_dataset_wpcd(
            num_envs=4,
            device=args.device,
            out_path=tmp_npz,
            seed=args.seed,
            max_steps=args.max_steps,
            spawn_r_min=args.r_min,
            spawn_r_max=args.r_max,
            env_spacing=args.env_spacing,
            table2_r_min=args.t2_r_min,
            table2_r_max=args.t2_r_max,
            video_grid_path=out_mp4,
            video_grid_pane_hw=(args.pane_h, args.pane_w),
            video_grid_distance=args.cam_distance,
            video_fps=args.fps,
            verbose=True,
        )
    finally:
        if tmp_npz.exists():
            tmp_npz.unlink()

    print(f"[grid4] carry_mode={args.carry_mode}")
    print(f"[grid4] wrote {out_mp4}")
    print(f"[grid4] stats: {stats}")


if __name__ == "__main__":
    main()
