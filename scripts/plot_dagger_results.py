##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Plot BC loss and DAgger per-round diagnostics from the two tfevents logs."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).resolve().parent.parent
BC_DIR = ROOT / "outputs/logs/g1_monolithic/2026-10-27_12-05-27"
DG_DIR = ROOT / "outputs/logs/g1_monolithic_dagger/2026-10-28_17-09-23"
OUT = ROOT / "outputs/media/dagger_results.png"


def load(ev_dir: Path) -> dict:
    """Read all scalar tags from a tfevents directory into (steps, values) arrays.
    Args: ev_dir: directory holding events.* file.
    Returns: dict mapping tag name to (steps numpy array, values numpy array).
    """
    ev_file = sorted(glob.glob(str(ev_dir / "events.*")))[0]
    ea = event_accumulator.EventAccumulator(ev_file, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        xs = ea.Scalars(tag)
        out[tag] = (np.array([x.step for x in xs]), np.array([x.value for x in xs]))
    return out


def main() -> None:
    """Render the 2x3 panel figure comparing BC convergence and DAgger per-round stats.
    Args: none.
    Returns: none.
    """
    bc = load(BC_DIR)
    dg = load(DG_DIR)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    # Panel (a): BC train/val loss on log scale.
    ax = axes[0, 0]
    tr_s, tr_v = bc["BC/train_loss"]
    va_s, va_v = bc["BC/val_loss"]
    ax.plot(tr_s, tr_v, color="#1f77b4", lw=1.2, label=f"train (end {tr_v[-1]:.4f})")
    ax.plot(va_s, va_v, color="#d62728", lw=1.2, label=f"val   (min {va_v.min():.4f})")
    ax.axhline(va_v.min(), color="#d62728", ls=":", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("BC epoch")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title("(a) BC convergence")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel (b): DAgger refit train loss per global epoch, round boundaries shaded.
    ax = axes[0, 1]
    tr_s, tr_v = dg["DAgger/train_loss"]
    ro_s, ro_v = dg["DAgger/round"]
    ax.plot(tr_s, tr_v, color="#2ca02c", lw=0.9, alpha=0.7)
    for i in range(1, 10):
        boundary = i * 60
        ax.axvline(boundary, color="#888888", ls=":", lw=0.5)
    ax.set_xlabel("global refit epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("(b) DAgger refit loss  (60 epochs x 10 rounds)")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f"round 0 start: {tr_v[0]:.4f}\nround 9 end:   {tr_v[-1]:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Panel (c): rollout outcomes per round (pick/carry/drop fires + falls).
    ax = axes[0, 2]
    rounds = np.arange(10)
    for tag, color, label in [
        ("Rollout/pick_fires",  "#1f77b4", "pick fires"),
        ("Rollout/carry_fires", "#2ca02c", "carry fires"),
        ("Rollout/drop_fires",  "#d62728", "drop fires"),
        ("Rollout/drops_done",  "#9467bd", "drops done"),
    ]:
        s, v = dg[tag]
        ax.plot(s, v, "o-", color=color, lw=1.4, markersize=4, label=label)
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("fires per round (1024 envs x 400 steps)")
    ax.set_title("(c) Oracle-gate transitions")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): falls per round.
    ax = axes[1, 0]
    s, v = dg["Rollout/falls"]
    ax.plot(s, v, "o-", color="#d62728", lw=1.6, markersize=5)
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("falls per round")
    ax.set_title(f"(d) Falls drop from {int(v[0])} to {int(v[-1])}")
    ax.grid(True, alpha=0.3)

    # Panel (e): per-round rollout wall time.
    ax = axes[1, 1]
    s, v = dg["Perf/rollout_s"]
    ax.bar(s, v, color="#17becf", alpha=0.8)
    ax.axhline(v.mean(), color="#444444", ls="--", lw=0.8, label=f"mean {v.mean():.1f} s")
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("rollout wall time (s)")
    ax.set_title("(e) Rollout cost per round")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (f): rollout outcomes at play-time, 4-env seed=7 bar chart.
    ax = axes[1, 2]
    labels = ["reach pick", "reach carry", "reach drop", "drop done", "falls"]
    bc_play  = [0, 0, 0, 0, 3]   # BC   model_best, seed=7, r=0.5-2.5
    dg_play  = [4, 1, 0, 0, 1]   # DAgger round_09, same config
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, bc_play, width=w, color="#1f77b4", label="BC (model_best)")
    ax.bar(x + w/2, dg_play, width=w, color="#2ca02c", label="DAgger (round 09)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("count / 4 envs (seed=7)")
    ax.set_title("(f) Play-time outcomes: DAgger beats BC")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Monolithic WPCD student: BC + DAgger (outputs_monolithic_dagger_v1)",
                 y=1.02, fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
