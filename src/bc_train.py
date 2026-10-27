"""Behaviour cloning driver for the unified student.

Input: outputs/walk_pick_sequential.npz (unified_obs, action, phase).
Output: outputs/logs/g1_monolithic/<timestamp>/model_*.pt plus TensorBoard scalars.

The loss is per-joint MSE on the expert action. Phase-balanced sampling keeps the
majority phase (walk) from drowning out pick and place. A small held-out split
produces the validation curve plotted by the companion notebook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.student_policy import MonolithicPolicyStudent

# One-hot slot count matching PHASE_ONE_HOT_DIM; sizes the inverse-frequency lookup.
NUM_PHASES = 4


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = ROOT / "outputs" / "logs" / "g1_monolithic"


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS for the ETA log line.
    Args: seconds: non-negative duration.
    Returns: zero-padded HH:MM:SS string.
    """
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class BCConfig:
    """Hyperparameters for the BC loop; the notebook edits fields by name."""
    lr: float = 3e-4
    batch_size: int = 4096
    epochs: int = 100
    val_frac: float = 0.05
    balance_phases: bool = True
    weight_decay: float = 1e-5
    log_every: int = 10
    save_every: int = 25
    device: str = "cuda:0"


def _make_split_indices(n: int, val_frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val index permutation; returns NumPy arrays for later slicing of GPU tensors.
    Args: n: dataset size; val_frac: fraction to hold out; seed: RNG seed.
    Returns: (train_idx, val_idx) numpy int arrays.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]


def _phase_balanced_weights_gpu(
    phase_t: torch.Tensor,
    dataset_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample sampling weights on-device for ``torch.multinomial`` phase-balanced sampling.
    Args: phase_t: (N,) long tensor of phase ids on the training device; dataset_weights: optional (N,) tensor of per-row weights that overrides inverse-frequency.
    Returns: (N,) float64 tensor of normalized sampling weights.
    """
    if dataset_weights is not None:
        return dataset_weights.double()
    counts = torch.bincount(phase_t, minlength=NUM_PHASES).double()
    counts = counts.clamp(min=1.0)
    inv_freq = 1.0 / counts
    return inv_freq[phase_t]


def train_bc(
    dataset_path: Path | str,
    cfg: BCConfig = BCConfig(),
    log_dir: Optional[Path | str] = None,
) -> Path:
    """Train the unified student by behavioural cloning and log checkpoints plus TB scalars.
    Args: dataset_path: .npz produced by the FSM dataset builder; cfg: BCConfig hyperparameters; log_dir: output directory, defaults to outputs/logs/g1_monolithic/<timestamp>/.
    Returns: Path to the final-epoch checkpoint.
    """
    from torch.utils.tensorboard import SummaryWriter

    blob = np.load(str(dataset_path))
    obs, act, phase = blob["unified_obs"], blob["action"], blob["phase"]
    sample_weights = blob["sample_weights"] if "sample_weights" in blob.files else None
    print(f"[bc] dataset: {obs.shape[0]} transitions, obs={obs.shape[1]}, act={act.shape[1]}" + (" (with sample_weights)" if sample_weights is not None else ""))

    # Drop corrupted samples with absurd action magnitudes before computing
    # the normalizer or the loss. Sanely trained experts produce actions
    # bounded by clip_actions~3 and empirically peaking around 25 in pick;
    # anything over 50 came from an unstable solver run (NaN propagation
    # through the drop expert's MLP) and would dominate the MSE.
    OUTLIER_THRESHOLD = 50.0
    keep = np.abs(act).max(axis=1) < OUTLIER_THRESHOLD
    n_drop = int((~keep).sum())
    if n_drop > 0:
        pct = 100.0 * n_drop / len(act)
        print(f"[bc] filtering {n_drop} outlier samples with |action|>{OUTLIER_THRESHOLD} ({pct:.2f}% of dataset)")
        per_phase = {int(p): int(((~keep) & (phase == p)).sum()) for p in sorted(set(phase.tolist()))}
        print(f"[bc]   outliers per phase: {per_phase}")
        obs = obs[keep]
        act = act[keep]
        phase = phase[keep]
        if sample_weights is not None:
            sample_weights = sample_weights[keep]
        print(f"[bc] dataset after filter: {obs.shape[0]} transitions")

    # Fast path: preload the whole dataset onto the GPU once, then sample
    # batches on-device via torch.multinomial. Removes the ~1.7 TB of
    # host-to-device copies the DataLoader path incurred over 1k epochs on a
    # 1.7 GB dataset. TF32 matmul is enabled for the student MLP.
    torch.set_float32_matmul_precision("high")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"[bc] moving dataset to {device} ({obs.nbytes/1e9 + act.nbytes/1e9:.2f} GB)...", flush=True)
    obs_t = torch.from_numpy(obs).float().to(device, non_blocking=True)
    act_t = torch.from_numpy(act).float().to(device, non_blocking=True)
    phase_t = torch.from_numpy(phase.astype(np.int64)).to(device, non_blocking=True)
    weights_full = None
    if sample_weights is not None:
        weights_full = torch.from_numpy(sample_weights.astype(np.float64)).to(device, non_blocking=True)

    tr_idx_np, val_idx_np = _make_split_indices(obs_t.shape[0], cfg.val_frac, seed=0)
    tr_idx = torch.from_numpy(tr_idx_np).to(device, non_blocking=True)
    val_idx = torch.from_numpy(val_idx_np).to(device, non_blocking=True)
    tr_obs = obs_t[tr_idx]
    tr_act = act_t[tr_idx]
    tr_phase = phase_t[tr_idx]
    val_obs = obs_t[val_idx]
    val_act = act_t[val_idx]
    del obs_t, act_t, phase_t

    if cfg.balance_phases:
        sampler_weights = _phase_balanced_weights_gpu(
            tr_phase, dataset_weights=(weights_full[tr_idx] if weights_full is not None else None),
        )
    else:
        sampler_weights = None

    student = MonolithicPolicyStudent(obs_dim=obs.shape[1], act_dim=act.shape[1]).to(device)
    student.normalizer.update(tr_obs)

    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    n_tr = tr_obs.shape[0]
    n_val = val_obs.shape[0]
    batches_per_epoch = max(1, n_tr // cfg.batch_size)
    print(f"[bc] train={n_tr} val={n_val} batches/epoch={batches_per_epoch} batch_size={cfg.batch_size}", flush=True)

    best_val = float("inf")
    t_start = time.time()
    ewma_epoch_s: Optional[float] = None
    for epoch in range(cfg.epochs):
        t_epoch_start = time.time()
        student.train()
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_seen = 0
        for _ in range(batches_per_epoch):
            if sampler_weights is not None:
                idx = torch.multinomial(sampler_weights, cfg.batch_size, replacement=True)
            else:
                idx = torch.randint(0, n_tr, (cfg.batch_size,), device=device)
            x = tr_obs[idx]
            y = tr_act[idx]
            pred = student(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optim.step()
            epoch_loss_sum = epoch_loss_sum + loss.detach() * cfg.batch_size
            epoch_seen += cfg.batch_size
        train_loss = (epoch_loss_sum / max(1, epoch_seen)).item()

        student.eval()
        with torch.no_grad():
            v_sum = torch.zeros((), device=device)
            v_seen = 0
            for s in range(0, n_val, cfg.batch_size):
                e = min(s + cfg.batch_size, n_val)
                v_sum = v_sum + loss_fn(student(val_obs[s:e]), val_act[s:e]) * (e - s)
                v_seen += (e - s)
            val_loss = (v_sum / max(1, v_seen)).item()

        epoch_s = time.time() - t_epoch_start
        ewma_epoch_s = epoch_s if ewma_epoch_s is None else 0.9 * ewma_epoch_s + 0.1 * epoch_s
        elapsed_s = time.time() - t_start
        eta_s = (cfg.epochs - epoch - 1) * ewma_epoch_s

        writer.add_scalar("BC/train_loss", train_loss, epoch)
        writer.add_scalar("BC/val_loss", val_loss, epoch)
        writer.add_scalar("Perf/epoch_s", epoch_s, epoch)
        writer.add_scalar("Perf/elapsed_s", elapsed_s, epoch)
        writer.add_scalar("Perf/eta_s", eta_s, epoch)
        if (epoch % cfg.log_every) == 0 or epoch == cfg.epochs - 1:
            print(
                f"  epoch {epoch:4d}/{cfg.epochs} | train {train_loss:.5f} | val {val_loss:.5f} "
                f"| epoch_s={epoch_s:.1f} elapsed={_fmt_hms(elapsed_s)} eta={_fmt_hms(eta_s)}"
            )

        if val_loss < best_val:
            best_val = val_loss
            student.save(log_dir / "model_best.pt")
        if (epoch % cfg.save_every) == 0 or epoch == cfg.epochs - 1:
            student.save(log_dir / f"model_{epoch}.pt")

    writer.close()
    final = log_dir / f"model_{cfg.epochs - 1}.pt"
    print(f"[bc] finished. best val = {best_val:.5f}. checkpoints under {log_dir}")
    return final


def main() -> None:
    """Parse CLI flags and train the student on the sequential dataset.
    Args: none: reads --dataset, --epochs, --lr, --batch-size, --device from argv.
    Returns: none.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=str(ROOT / "outputs" / "walk_pick_sequential.npz"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = BCConfig(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, device=args.device)
    train_bc(args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()
