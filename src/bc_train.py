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
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

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


def _make_split(
    obs: np.ndarray,
    act: np.ndarray,
    phase: np.ndarray,
    val_frac: float,
    seed: int = 0,
    sample_weights: np.ndarray | None = None,
) -> tuple[TensorDataset, TensorDataset, np.ndarray, np.ndarray | None]:
    """Split obs/act/phase into train and val TensorDatasets deterministically.
    Args: obs: (N, D) observations; act: (N, A) actions; phase: (N,) phase ids; val_frac: fraction to hold out; seed: RNG seed; sample_weights: optional (N,) per-row weights passed through to the train split.
    Returns: (train_ds, val_ds, train_phase_np, train_weights_np) with the per-split weight slice when provided.
    """
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    train_ds = TensorDataset(
        torch.from_numpy(obs[tr_idx]).float(),
        torch.from_numpy(act[tr_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(obs[val_idx]).float(),
        torch.from_numpy(act[val_idx]).float(),
    )
    train_weights = None if sample_weights is None else sample_weights[tr_idx]
    return train_ds, val_ds, phase[tr_idx], train_weights


def _phase_balanced_sampler(phase_np: np.ndarray, dataset_weights: np.ndarray | None = None) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler so batches mix phases, using dataset_weights when provided.
    Args: phase_np: (N_train,) int array of phase ids in [0, NUM_PHASES); dataset_weights: optional (N_train,) per-row weights produced upstream.
    Returns: WeightedRandomSampler covering N_train indices with replacement.
    """
    if dataset_weights is not None:
        weights = dataset_weights.astype(np.float64)
    else:
        counts = np.bincount(phase_np, minlength=NUM_PHASES).astype(np.float64)
        counts[counts == 0.0] = 1.0
        inv_freq = 1.0 / counts
        weights = inv_freq[phase_np]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(phase_np),
        replacement=True,
    )


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

    train_ds, val_ds, train_phase, train_weights = _make_split(obs, act, phase, cfg.val_frac, sample_weights=sample_weights)

    if cfg.balance_phases:
        sampler = _phase_balanced_sampler(train_phase, dataset_weights=train_weights)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    student = MonolithicPolicyStudent(obs_dim=obs.shape[1], act_dim=act.shape[1]).to(device)
    student.normalizer.update(torch.from_numpy(obs).float().to(device))

    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    best_val = float("inf")
    global_step = 0
    t_start = time.time()
    ewma_epoch_s: Optional[float] = None
    for epoch in range(cfg.epochs):
        t_epoch_start = time.time()
        student.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = student(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optim.step()
            running += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
            global_step += 1
        train_loss = running / max(1, seen)

        student.eval()
        with torch.no_grad():
            v_run, v_seen = 0.0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_run += float(loss_fn(student(x), y).item()) * x.shape[0]
                v_seen += x.shape[0]
            val_loss = v_run / max(1, v_seen)

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
