##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Three-pane mp4 comparing FSM composition, BC-only student, and PPO fine-tuned student."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "media" / "three_way_latest.mp4"


def latest_bc_ckpt() -> Path:
    """Return the newest model_best.pt under outputs/logs/g1_monolithic/*/.
    Args: none.
    Returns: path to the most recent BC checkpoint.
    """
    runs = sorted((ROOT / "outputs" / "logs" / "g1_monolithic").iterdir())
    runs = [r for r in runs if r.is_dir() and (r / "model_best.pt").exists()]
    if not runs:
        raise FileNotFoundError("no model_best.pt under outputs/logs/g1_monolithic")
    return runs[-1] / "model_best.pt"


def latest_ppo_ckpt() -> Path:
    """Return the highest-iteration model_*.pt under outputs/logs/g1_monolithic_ppo/*/.
    Args: none.
    Returns: path to the latest PPO checkpoint across the newest timestamped run.
    """
    root = ROOT / "outputs" / "logs" / "g1_monolithic_ppo"
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"no timestamped runs under {root}")
    cks = sorted(
        runs[-1].glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if not cks:
        raise FileNotFoundError(f"no model_*.pt under {runs[-1]}")
    return cks[-1]


def extract_student_ckpt(ppo_ckpt: Path, tmp_path: Path, device: str = "cpu") -> Path:
    """Pull student weights out of a PPO fine-tune ckpt and save them in BC format.
    Args: ppo_ckpt: PPO fine-tune checkpoint; tmp_path: destination for the extracted student .pt; device: torch device for the load.
    Returns: path to the written BC-format checkpoint.
    """
    from src.student_policy import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM

    blob = torch.load(str(ppo_ckpt), map_location=device, weights_only=False)
    actor_sd = blob["actor"]
    student_sd = {
        k.removeprefix("student."): v
        for k, v in actor_sd.items()
        if k.startswith("student.")
    }
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": student_sd, "obs_dim": UNIFIED_OBS_DIM, "act_dim": UNIFIED_ACT_DIM},
        str(tmp_path),
    )
    return tmp_path


def rollout_student(student_ckpt: Path, seed: int, mp4_path: Path, max_steps: int, device: str) -> None:
    """Run one monolithic rollout at the given seed and dump the tracking-camera mp4.
    Args: student_ckpt: BC-format student .pt; seed: annulus spawn and goal seed; mp4_path: destination mp4; max_steps: step cap; device: torch device.
    Returns: none.
    """
    from src.play_student import run

    run(
        ckpt_path=student_ckpt,
        num_episodes=1,
        base_seed=seed,
        device=device,
        spawn_r_min=0.5,
        spawn_r_max=2.0,
        max_steps=max_steps,
        video_path=mp4_path,
        video_hw=(720, 1280),
        video_fps=20,
        video_episodes=1,
    )


def rollout_fsm(seed: int, mp4_path: Path, max_steps: int, device: str) -> None:
    """Run one walk+pick FSM rollout at the given seed with A* on and dump its mp4.
    Args: seed: must match the student seed to align initial conditions; mp4_path: destination mp4; max_steps: step cap; device: torch device.
    Returns: none.
    """
    from src.fsm_continuous_batch import build_dataset

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp_npz = Path(tf.name)
    try:
        build_dataset(
            num_envs=1,
            device=device,
            out_path=tmp_npz,
            seed=seed,
            max_steps=max_steps,
            spawn_r_min=0.5,
            spawn_r_max=2.0,
            verbose=False,
            video_path=mp4_path,
            video_fps=20,
            video_hw=(720, 1280),
            use_astar=True,
        )
    finally:
        if tmp_npz.exists():
            tmp_npz.unlink()


def stitch_three(left: Path, mid: Path, right: Path, label_l: str, label_m: str, label_r: str, out_mp4: Path) -> None:
    """Place three mp4s side by side with text labels using ffmpeg hstack=3.
    Args: left: left-pane source; mid: middle-pane source; right: right-pane source; label_l/m/r: labels drawn on each pane; out_mp4: destination mp4.
    Returns: none.
    """
    filter_complex = (
        f"[0:v]drawtext=text='{label_l}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[l];"
        f"[1:v]drawtext=text='{label_m}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[m];"
        f"[2:v]drawtext=text='{label_r}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[r];"
        f"[l][m][r]hstack=inputs=3[v]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(left),
        "-i", str(mid),
        "-i", str(right),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg hstack=3 failed: {r.stderr.decode()[:400]}")


def concat_mp4s(parts: list[Path], out_mp4: Path) -> None:
    """Concatenate a list of mp4s in order via the ffmpeg concat demuxer.
    Args: parts: ordered list of mp4 paths; out_mp4: destination mp4.
    Returns: none.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tf:
        for p in parts:
            tf.write(f"file '{p.resolve()}'\n")
        list_path = Path(tf.name)
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_mp4)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr.decode()[:400]}")
    finally:
        list_path.unlink(missing_ok=True)


def main() -> None:
    """Parse CLI flags and build the three-way mp4 for the requested seeds.
    Args: none; reads --ppo-ckpt, --bc-ckpt, --seeds, --max-steps, --device, --out from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppo-ckpt", default=None, help="defaults to highest-iter model_*.pt under outputs/logs/g1_monolithic_ppo/*/")
    ap.add_argument("--bc-ckpt", default=None, help="defaults to newest outputs/logs/g1_monolithic/*/model_best.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 207, 313])
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    bc_ckpt = Path(args.bc_ckpt) if args.bc_ckpt else latest_bc_ckpt()
    ppo_ckpt = Path(args.ppo_ckpt) if args.ppo_ckpt else latest_ppo_ckpt()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="three_way_"))
    print(f"[eval] temp dir : {tmp_root}")
    print(f"[eval] bc ckpt  : {bc_ckpt}")
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

        print(f"[eval] seed {seed}: BC-only student ...")
        t0 = time.time()
        mp4_bc = tmp_root / f"bc_{seed}.mp4"
        rollout_student(bc_ckpt, seed, mp4_bc, args.max_steps, args.device)
        print(f"[eval] seed {seed}: BC done in {time.time()-t0:.1f}s")

        print(f"[eval] seed {seed}: PPO student ...")
        t0 = time.time()
        mp4_ppo = tmp_root / f"ppo_{seed}.mp4"
        rollout_student(tmp_ppo_student, seed, mp4_ppo, args.max_steps, args.device)
        print(f"[eval] seed {seed}: PPO done in {time.time()-t0:.1f}s")

        combined = tmp_root / f"combined_{seed}.mp4"
        stitch_three(
            mp4_fsm, mp4_bc, mp4_ppo,
            f"FSM composition (seed {seed})",
            f"BC-only (seed {seed})",
            f"PPO fine-tuned (seed {seed})",
            combined,
        )
        combined_parts.append(combined)
        print(f"[eval] seed {seed}: stitched -> {combined}")

    concat_mp4s(combined_parts, out_path)
    print(f"\n[eval] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
