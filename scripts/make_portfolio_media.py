##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Build the two README visual assets (demo gif, frame map) from a single mp4 rollout.

Input:  path to an mp4 rollout produced by the record-videos skill.
Output: docs/demo.gif (palette-optimised loop) and docs/frame_map.png (N frames concatenated).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _require_ffmpeg() -> None:
    """Abort the script with exit code 2 when the ffmpeg binary is not reachable.
    Args: none.
    Returns: none.
    """
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not on PATH. Install with `sudo apt install ffmpeg`.", file=sys.stderr)
        sys.exit(2)


def _video_duration_s(mp4: Path) -> float:
    """Query ffprobe for the duration (seconds) of an mp4 container.
    Args: mp4: path to an mp4 file readable by ffprobe.
    Returns: duration in seconds as a float.
    """
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        text=True,
    )
    return float(out.strip())


def make_gif(mp4: Path, out: Path, fps: int, width: int, max_seconds: float) -> None:
    """Encode a palette-optimised gif from mp4 via two-pass ffmpeg (palettegen, paletteuse).
    Args: mp4: source rollout; out: gif output path; fps: target framerate; width: output pixel width; max_seconds: clip trim limit.
    Returns: none.
    """
    dur = min(_video_duration_s(mp4), max_seconds)
    palette = out.with_suffix(".palette.png")
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-t", str(dur), "-i", str(mp4),
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
            str(palette),
        ])
        subprocess.check_call([
            "ffmpeg", "-y", "-t", str(dur), "-i", str(mp4), "-i", str(palette),
            "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse",
            str(out),
        ])
    finally:
        if palette.exists():
            palette.unlink()


def make_frame_map(mp4: Path, out: Path, n_frames: int, width_each: int) -> None:
    """Grab n_frames evenly spaced stills from mp4 and concatenate them horizontally into one PNG.
    Args: mp4: source rollout; out: PNG output path; n_frames: number of stills to sample; width_each: per-frame pixel width.
    Returns: none.
    """
    from PIL import Image

    dur = _video_duration_s(mp4)
    tmp_dir = out.parent / "_frames_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame_paths = []
        # Keep a small margin so ffmpeg does not land past the last decodable frame.
        margin = min(0.2, dur * 0.02)
        for i in range(n_frames):
            # Sample uniformly inside the safe interval [margin, dur - margin].
            frac = i / max(1, n_frames - 1)
            t = margin + frac * (dur - 2 * margin)
            p = tmp_dir / f"f{i:02d}.png"
            subprocess.check_call([
                "ffmpeg", "-y", "-ss", str(t), "-i", str(mp4),
                "-frames:v", "1", "-vf", f"scale={width_each}:-1:flags=lanczos",
                str(p),
            ])
            frame_paths.append(p)

        imgs = [Image.open(p) for p in frame_paths]
        h = max(im.height for im in imgs)
        total_w = sum(im.width for im in imgs)
        canvas = Image.new("RGB", (total_w, h), color=(0, 0, 0))
        x = 0
        for im in imgs:
            canvas.paste(im, (x, (h - im.height) // 2))
            x += im.width
        canvas.save(out, optimize=True)
    finally:
        for p in tmp_dir.glob("*.png"):
            p.unlink()
        tmp_dir.rmdir()


def main() -> None:
    """Parse CLI args and write docs/demo.gif + docs/frame_map.png from the input mp4.
    Args: none (reads sys.argv via argparse).
    Returns: none.
    """
    _require_ffmpeg()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, help="input rollout mp4")
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--fps", type=int, default=12, help="gif fps")
    parser.add_argument("--width", type=int, default=540, help="gif width in px")
    parser.add_argument("--max-seconds", type=float, default=8.0, help="trim gif to this duration")
    parser.add_argument("--frames", type=int, default=4, help="frame map count")
    parser.add_argument("--frame-width", type=int, default=360, help="per-frame width in frame map")
    args = parser.parse_args()

    if not args.mp4.exists():
        print(f"ERROR: {args.mp4} does not exist", file=sys.stderr)
        sys.exit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gif = args.out_dir / "demo.gif"
    frame_map = args.out_dir / "frame_map.png"

    print(f"[1/2] generating {gif}")
    make_gif(args.mp4, gif, args.fps, args.width, args.max_seconds)
    print(f"      wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")

    print(f"[2/2] generating {frame_map}")
    make_frame_map(args.mp4, frame_map, args.frames, args.frame_width)
    print(f"      wrote {frame_map}  ({frame_map.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
