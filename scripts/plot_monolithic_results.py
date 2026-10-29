##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Plot BC loss curve and PPO training curves from the two tfevents logs."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).resolve().parent.parent
BC_DIR = ROOT / "outputs/logs/g1_monolithic/2026-10-27_12-05-27"
PPO_DIR = ROOT / "outputs/logs/g1_monolithic_ppo/2026-10-27_12-42-58"
OUT = ROOT / "outputs/media/monolithic_results.png"


def _load(ev_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read scalar tags from a tfevents directory into step/value arrays.
    Args: ev_dir: directory containing one or more events.* files.
    Returns: mapping tag to (steps, values) numpy array pair.
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
    """Render the 2x3 panel summary figure and write it to OUT.
    Args: none.
    Returns: none.
    """
    bc = _load(BC_DIR)
    ppo = _load(PPO_DIR)

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

    # Panel (b): PPO total return with EMA smoothing over ~1/50 of the run.
    ax = axes[0, 1]
    s, v = ppo["Train/mean_reward_total"]
    ax.plot(s, v, color="#2ca02c", lw=0.9, alpha=0.4, label="raw")
    w = max(1, len(v) // 50)
    kernel = np.ones(w) / w
    v_ema = np.convolve(v, kernel, mode="valid")
    s_ema = s[w - 1:]
    ax.plot(s_ema, v_ema, color="#2ca02c", lw=1.6, label=f"EMA (w={w})")
    ax.axhline(v[0], color="#aaaaaa", ls=":", lw=0.8, label=f"init {v[0]:+.2f}")
    ax.axhline(v[-1], color="#444444", ls="--", lw=0.8, label=f"end  {v[-1]:+.2f}")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean reward (total)")
    ax.set_title("(b) PPO return: does NOT improve on BC")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): reward components, each smoothed with the same EMA window.
    ax = axes[0, 2]
    for tag, color, label in [
        ("Train/mean_reward_env", "#1f77b4", "env (task)"),
        ("Train/mean_reward_mimic", "#ff7f0e", "mimic (BC prior)"),
        ("Train/mean_reward_walk_shape", "#9467bd", "walk shape"),
    ]:
        s, v = ppo[tag]
        v_ema = np.convolve(v, np.ones(w)/w, mode="valid")
        ax.plot(s[w-1:], v_ema, color=color, lw=1.4, label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean reward component")
    ax.set_title("(c) Reward components (EMA)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): per-iteration FSM transition fires across envs.
    ax = axes[1, 0]
    for tag, color, label in [
        ("Train/pick_fires_per_iter", "#1f77b4", "walk to pick"),
        ("Train/carry_fires_per_iter", "#2ca02c", "pick to carry"),
        ("Train/drop_fires_per_iter", "#d62728", "carry to drop"),
        ("Train/drops_done_per_iter", "#9467bd", "drop done"),
    ]:
        s, v = ppo[tag]
        v_ema = np.convolve(v, np.ones(w)/w, mode="valid")
        ax.plot(s[w-1:], v_ema, color=color, lw=1.4, label=label)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("phase transitions / iter (envs)")
    ax.set_title("(d) Phase fires: only walk to pick triggers")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (e): KL to the BC prior and PPO approx KL on log scale.
    ax = axes[1, 1]
    s, v = ppo["Loss/kl_to_prior"]
    v_ema = np.convolve(v, np.ones(w)/w, mode="valid")
    ax.plot(s[w-1:], v_ema, color="#8c564b", lw=1.4, label="KL(cur || BC prior)")
    s, v = ppo["Loss/approx_kl"]
    v_ema = np.convolve(v, np.ones(w)/w, mode="valid")
    ax.plot(s[w-1:], v_ema, color="#17becf", lw=1.4, label="approx KL (PPO)")
    ax.set_yscale("log")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("KL (log)")
    ax.set_title("(e) Policy drift (BeyondMimic anchor)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Panel (f): hand-logged play-time outcomes from seed=7, 4 envs, r=0.5-2.5.
    ax = axes[1, 2]
    labels = ["reach pick", "reach carry", "reach drop", "drop done", "falls"]
    bc_vals  = [0, 0, 0, 0, 2]
    ppo_vals = [0, 0, 0, 0, 4]
    x = np.arange(len(labels))
    w_bar = 0.38
    ax.bar(x - w_bar/2, bc_vals,  width=w_bar, color="#1f77b4", label="BC (model_999)")
    ax.bar(x + w_bar/2, ppo_vals, width=w_bar, color="#2ca02c", label="PPO (model_3999)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("count / 4 envs (seed=7)")
    ax.set_title("(f) Play-time outcomes: PPO worse than BC")
    ax.set_ylim(0, 4.5)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Monolithic WPCD student: BC + PPO fine-tune (outputs_monolithic_v2)", y=1.02, fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
