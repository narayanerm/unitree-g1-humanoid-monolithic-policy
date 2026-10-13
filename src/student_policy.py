##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Unified goal-conditioned student policy trained by behaviour cloning.

Input: unified obs (num_envs, UNIFIED_OBS_DIM) with proprioception, goal, phase one-hot.
Output: joint position targets (num_envs, UNIFIED_ACT_DIM = 29).

The MLP (Linear, ELU x 3, Linear) mirrors the rsl-rl actor topology used by every
expert so a later PPO fine-tune can warm-start into the trained student without shape
mismatches.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Defaults mirror goal_features.py; the .npz carries the authoritative shape.
UNIFIED_OBS_DIM = 115
UNIFIED_ACT_DIM = 29


class RunningMeanStd(nn.Module):
    """Online mean/variance tracker for input normalisation; nn.Module so state_dict and device moves work."""

    def __init__(self, dim: int, eps: float = 1e-8):
        """Initialise the running-stats buffers.
        Args: dim: feature dimension; eps: variance floor to avoid divide-by-zero.
        Returns: none.
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1.0))
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update the running mean/var with a batch of observations.
        Args: x: (n, dim) tensor.
        Returns: none.
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x with the running stats.
        Args: x: (n, dim) tensor.
        Returns: (n, dim) tensor.
        """
        return (x - self.mean) / (self.var.sqrt() + self.eps)


class MonolithicPolicyStudent(nn.Module):
    """MLP goal-conditioned student policy mapping unified_obs to joint position targets."""

    def __init__(
        self,
        obs_dim: int = UNIFIED_OBS_DIM,
        act_dim: int = UNIFIED_ACT_DIM,
        hidden: tuple[int, ...] = (512, 256, 128),
    ):
        """Build the normaliser + MLP head.
        Args: obs_dim: unified obs size; act_dim: joint target size; hidden: hidden-layer widths.
        Returns: none.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.normalizer = RunningMeanStd(obs_dim)

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers += [nn.Linear(prev, act_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict joint position targets for a batch of unified observations.
        Args: obs: (n, obs_dim) tensor.
        Returns: (n, act_dim) tensor.
        """
        return self.mlp(self.normalizer(obs))

    def save(self, path: Path | str) -> Path:
        """Serialise the state dict to disk.
        Args: path: .pt destination.
        Returns: the resolved Path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "act_dim": self.act_dim}, str(path))
        return path

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MonolithicPolicyStudent":
        """Instantiate a student and load a checkpoint.
        Args: path: .pt file written by ``save``; device: device to load to.
        Returns: MonolithicPolicyStudent instance on device.
        """
        blob = torch.load(str(path), map_location=device, weights_only=False)
        obj = cls(obs_dim=blob.get("obs_dim", UNIFIED_OBS_DIM),
                  act_dim=blob.get("act_dim", UNIFIED_ACT_DIM))
        obj.load_state_dict(blob["state_dict"])
        return obj.to(device)
