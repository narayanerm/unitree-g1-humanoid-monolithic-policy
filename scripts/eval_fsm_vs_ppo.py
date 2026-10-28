##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Two-pane mp4 comparing FSM composition and PPO fine-tuned student at matched seeds.

Input:  a PPO run folder under ``outputs/logs/g1_monolithic_ppo/`` and a list of seeds.
Output: an mp4 at ``outputs/media/fsm_vs_ppo_latest.mp4`` with per-seed FSM / PPO panes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_three_way import (  # noqa: E402
    concat_mp4s,
    extract_student_ckpt,
    rollout_fsm,
    rollout_student,
)

DEFAULT_OUT = ROOT / "outputs" / "media" / "fsm_vs_ppo_latest.mp4"


def highest_ppo_ckpt(run_dir: Path) -> Path:
    """Return the highest-iteration model_*.pt under run_dir.
    Args: run_dir: timestamped PPO run folder.
    Returns: path to the newest model_*.pt by numeric iter suffix.
    """
    cks = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if not cks:
        raise FileNotFoundError(f"no model_*.pt under {run_dir}")
    return cks[-1]


def stitch_two(left: Path, right: Path, label_l: str, label_r: str, out_mp4: Path) -> None:
    """Place two mp4s side by side with text labels using ffmpeg hstack.
    Args: left: left-pane source; right: right-pane source; label_l/r: labels drawn on each pane; out_mp4: destination mp4.
    Returns: none.
    """
    filter_complex = (
        f"[0:v]drawtext=text='{label_l}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[l];"
        f"[1:v]drawtext=text='{label_r}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[r];"
        f"[l][r]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(left),
        "-i", str(right),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg hstack=2 failed: {r.stderr.decode()[:400]}")


def main() -> None:
    """Parse CLI flags and build the FSM / PPO comparison mp4 for the requested seeds.
    Args: none; reads ``--ppo-run``, ``--ppo-ckpt``, ``--seeds``, ``--max-steps``, ``--device``, ``--out`` from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppo-run", default="2026-10-12_22-17-03",
                    help="Timestamped PPO run folder under outputs/logs/g1_monolithic_ppo/.")
    ap.add_argument("--ppo-ckpt", default=None,
                    help="Explicit model_*.pt path; defaults to the highest iter under --ppo-run.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 207, 313])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    if args.ppo_ckpt:
        ppo_ckpt = Path(args.ppo_ckpt)
    else:
        run_dir = ROOT / "outputs" / "logs" / "g1_monolithic_ppo" / args.ppo_run
        ppo_ckpt = highest_ppo_ckpt(run_dir)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="fsm_vs_ppo_"))
    print(f"[eval] temp dir : {tmp_root}")
    print(f"[eval] ppo ckpt : {ppo_ckpt}")
    print(f"[eval] seeds    : {args.seeds}  max_steps={args.max_steps}  device={args.device}")

    tmp_ppo_student = tmp_root / "ppo_student.pt"
    extract_student_ckpt(ppo_ckpt, tmp_ppo_student, device="cpu")
    print(f"[eval] extracted PPO student -> {tmp_ppo_student}")

    combined_parts: list[Path] = []
    for seed in args.seeds:
        print(f"\n[eval] seed {seed}: FSM ...")
        t0 = time.time()
        mp4_fsm = tmp_root / f"fsm_{seed}.mp4"
        rollout_fsm(seed, mp4_fsm, args.max_steps, args.device)
        print(f"[eval] seed {seed}: FSM done in {time.time()-t0:.1f}s")

        print(f"[eval] seed {seed}: PPO student ...")
        t0 = time.time()
        mp4_ppo = tmp_root / f"ppo_{seed}.mp4"
        rollout_student(tmp_ppo_student, seed, mp4_ppo, args.max_steps, args.device)
        print(f"[eval] seed {seed}: PPO done in {time.time()-t0:.1f}s")

        combined = tmp_root / f"combined_{seed}.mp4"
        stitch_two(
            mp4_fsm, mp4_ppo,
            f"FSM composition (seed {seed})",
            f"PPO fine-tuned (seed {seed})",
            combined,
        )
        combined_parts.append(combined)
        print(f"[eval] seed {seed}: stitched -> {combined}")

    concat_mp4s(combined_parts, out_path)
    print(f"\n[eval] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
