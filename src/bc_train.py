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
) -> tuple[TensorDataset, TensorDataset, np.ndarray]:
    """Split obs/act/phase into train and val TensorDatasets deterministically.
    Args: obs: (N, D) observations; act: (N, A) actions; phase: (N,) phase ids; val_frac: fraction to hold out; seed: RNG seed.
    Returns: (train_ds, val_ds, train_phase_np), the two datasets and the train-side phase array for the balanced sampler.
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
    return train_ds, val_ds, phase[tr_idx]


def _phase_balanced_sampler(phase_np: np.ndarray) -> WeightedRandomSampler:
    """Build an inverse-frequency WeightedRandomSampler so batches mix phases.
    Args: phase_np: (N_train,) int array of phase ids in [0, NUM_PHASES).
    Returns: WeightedRandomSampler covering N_train indices with replacement.
    """
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
    print(f"[bc] dataset: {obs.shape[0]} transitions, obs={obs.shape[1]}, act={act.shape[1]}")

    train_ds, val_ds, train_phase = _make_split(obs, act, phase, cfg.val_frac)

    if cfg.balance_phases:
        sampler = _phase_balanced_sampler(train_phase)
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
    for epoch in range(cfg.epochs):
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

        writer.add_scalar("BC/train_loss", train_loss, epoch)
        writer.add_scalar("BC/val_loss", val_loss, epoch)
        if (epoch % cfg.log_every) == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:4d} | train {train_loss:.5f} | val {val_loss:.5f}")

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
