##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""BeyondMimic-style PPO fine-tune of the full-chain walk-pick-carry-drop student.

Pipeline:
1. Load the BC-trained ``MonolithicPolicyStudent`` as warm-start.
2. Wrap it as a Gaussian actor with a scalar ``log_std`` (initialised small and
   frozen for the first iterations so the warm-start isn't blown up by noise).
3. Freeze a copy of the BC student as reference prior for a KL penalty and an
   action-mimic reward: the two BeyondMimic-style regularisers that keep the
   policy from drifting out of the demonstration manifold while PPO exploits
   the task reward.
4. Scene has table 1 (pickup) and table 2 (drop) per env — the same two-table
   layout the FSM builds for data gen. The policy drives all four phases
   (walk/pick/carry/drop); phase_ids are updated by physics-based gates so the
   student receives the correct one-hot in its observation.

Reward layout (per step):
- env reward: the pick task's native reward manager (grasp, lift, stability).
- walk shaping: w_walk * in_walk * (prev_dist_box - dist_box).
- carry shaping: w_carry * in_carry * (prev_dist_approach - dist_approach).
- drop shaping: w_drop * in_drop * (prev_box_z - box_z) — reward lowering the box.
- success bonus: +drop_success_bonus one-shot when box_z <= drop_z_done & box_on_t2 & upright.
- mimic reward: -w_mimic * MSE(action, bc_action).
- survival + fall penalty as before.

Episode ends on fall, drop_done (success), or timeout.
Output: ``outputs/logs/g1_monolithic_ppo/<timestamp>/model_*.pt`` + TB scalars.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fsm_continuous import RolloutConfig, _disable_terminations
from src.fsm_continuous_batch import (
    PICK_PHASE_ID,
    WALK_PHASE_ID,
    _random_spawn_annulus,
    _snap_pick_commands_batched,
    _teleport_robot_batch,
    _unified_obs_per_env,
)
from src.fsm_wpcd_batched import (
    CARRY_PHASE_ID,
    DROP_PHASE_ID,
    POST_DROP_PHASE_ID,
    TABLE2_HALF_X,
    TABLE2_HALF_Y,
    _make_two_tables_spec_fn,
    _resample_spawn_outside_t2,
    _sample_table2_positions_np,
)
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal
from src.student_policy import MonolithicPolicyStudent


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = ROOT / "outputs" / "logs" / "g1_monolithic_ppo"


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
class PPOFinetuneConfig:
    """Hyperparameters for the BeyondMimic-style PPO fine-tune."""
    num_envs: int = 2048
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    episode_max_steps: int = 1600
    spawn_r_min: float = 1.5
    spawn_r_max: float = 5.0
    env_spacing: float = 20.0

    # Table 2 placement per env (matches fsm_wpcd_batched defaults).
    table2_r_min: float = 1.5
    table2_r_max: float = 5.0
    spawn_t2_safety: float = 1.0

    learning_rate: float = 3e-4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 1e-3
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01  # adaptive lr target
    clip_actions: float = 3.0

    mimic_coef: float = 0.5           # weight on the negative squared L2 between policy and BC actions
    kl_to_prior_coef: float = 0.2     # KL(current || BC prior) added to the loss
    initial_log_std: float = -2.3     # exp(-2.3) ~= 0.1; starts near-deterministic
    freeze_log_std_iters: int = 100

    # Shaping / terminal reward.
    walk_distance_coef: float = 1.0   # in_walk * (prev_dist_box - dist_box)
    carry_shape_coef: float = 1.0     # in_carry * (prev_dist_approach - dist_approach)
    drop_shape_coef: float = 2.0      # in_drop * (prev_box_z - box_z)
    drop_success_bonus: float = 100.0 # one-shot when drop_done fires
    survival_coef: float = 0.01
    fall_penalty: float = 10.0

    # Phase gate thresholds (mirror fsm_wpcd_batched).
    walk_to_pick_dist: float = 0.55
    walk_to_pick_brake_steps: int = 25
    pick_to_carry_box_z: float = 0.85
    pick_stable_counter_min: int = 10
    pick_timeout_steps: int = 150
    carry_arrival_dist: float = 0.20
    carry_brake_steps: int = 60
    drop_z_done: float = 0.72
    pelvis_to_t2: float = 0.61

    critic_hidden: tuple[int, ...] = (512, 256, 128)

    log_interval: int = 1
    save_interval: int = 25
    device: str = "cuda:0"
    seed: int = 7


class MonolithicActor(nn.Module):
    """Gaussian actor that wraps a ``MonolithicPolicyStudent`` for stochastic rollouts."""

    def __init__(self, student: MonolithicPolicyStudent, initial_log_std: float):
        """Store the warm-start student and the learnable log-std.
        Args: student: BC-trained mean network (normaliser + MLP); initial_log_std: scalar used to init the per-action log-std vector.
        Returns: none.
        """
        super().__init__()
        self.student = student
        self.log_std = nn.Parameter(
            torch.full((student.act_dim,), float(initial_log_std))
        )

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action mean for an observation batch.
        Args: obs: (n, UNIFIED_OBS_DIM) tensor.
        Returns: (n, UNIFIED_ACT_DIM) tensor.
        """
        return self.student(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Build a diagonal Gaussian over the action space.
        Args: obs: (n, UNIFIED_OBS_DIM) tensor.
        Returns: ``Normal`` with batch shape (n, UNIFIED_ACT_DIM).
        """
        mean = self.mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action (or take the mean) and return (action, log_prob).
        Args: obs: (n, UNIFIED_OBS_DIM) tensor; deterministic: if True return the mean.
        Returns: (action: (n, A), log_prob: (n,)) tuple.
        """
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy) for stored (obs, action) pairs, used by PPO update.
        Args: obs: (n, UNIFIED_OBS_DIM); action: (n, UNIFIED_ACT_DIM).
        Returns: (log_prob: (n,), entropy: (n,)) tuple.
        """
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """MLP critic that estimates the value function on unified observations."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...]):
        """Build normalisation + MLP returning a scalar value per observation.
        Args: obs_dim: unified obs size; hidden: hidden-layer widths.
        Returns: none.
        """
        super().__init__()
        # Reuse the student's normaliser type so critic input matches the actor's.
        from src.student_policy import RunningMeanStd
        self.normalizer = RunningMeanStd(obs_dim)
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V(obs) for a batch.
        Args: obs: (n, obs_dim) tensor.
        Returns: (n,) tensor.
        """
        return self.mlp(self.normalizer(obs)).squeeze(-1)


@dataclass
class RolloutBuffer:
    """Per-step storage for a single PPO rollout; shape (T, N, ...)."""
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """Fill ``advantages`` and ``returns`` via GAE.
        Args: last_value: (N,) bootstrap value after the last stored step; gamma: discount; lam: GAE lambda.
        Returns: none.
        """
        T = self.rewards.shape[0]
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.values[t + 1]
            next_nonterm = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterm - self.values[t]
            gae = delta + gamma * lam * next_nonterm * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values

    def iterate(self, num_mini_batches: int, num_epochs: int):
        """Yield (obs, actions, log_probs, returns, advantages) minibatches, flattened to (T*N, ...).
        Args: num_mini_batches: splits per epoch; num_epochs: passes over the data.
        Returns: generator of tuples.
        """
        T, N = self.obs.shape[:2]
        total = T * N
        batch = total // num_mini_batches
        obs = self.obs.reshape(total, -1)
        actions = self.actions.reshape(total, -1)
        log_probs = self.log_probs.reshape(total)
        returns = self.returns.reshape(total)
        advantages = self.advantages.reshape(total)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(num_epochs):
            idx = torch.randperm(total, device=obs.device)
            for mb in range(num_mini_batches):
                sel = idx[mb * batch : (mb + 1) * batch]
                yield obs[sel], actions[sel], log_probs[sel], returns[sel], advantages[sel]


class _PhaseState:
    """Mutable per-env book-keeping tensors used by _advance_phases_full."""

    def __init__(self, num_envs: int, device: torch.device):
        """Allocate zero tensors for every per-env counter and latched target.
        Args: num_envs: parallel env count; device: torch device for all tensors.
        Returns: none.
        """
        self.phase_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pick_stable_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.in_pick_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.carry_brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.approach_target_xy = torch.zeros(num_envs, 2, device=device)
        self.place_target_pos_w = torch.zeros(num_envs, 3, device=device)
        self.place_target_quat_w = torch.zeros(num_envs, 4, device=device)
        self.place_target_quat_w[:, 0] = 1.0
        self.object_target_pos_w = torch.zeros(num_envs, 3, device=device)
        self.object_target_quat_w = torch.zeros(num_envs, 4, device=device)
        self.object_target_quat_w[:, 0] = 1.0
        self.drop_done_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Zero all per-env state for a subset of envs (used after reset/respawn).
        Args: env_ids: 1-D long tensor of env indices to reset.
        Returns: none.
        """
        if env_ids.numel() == 0:
            return
        self.phase_ids[env_ids] = WALK_PHASE_ID
        self.brake_counter[env_ids] = 0
        self.pick_stable_counter[env_ids] = 0
        self.in_pick_counter[env_ids] = 0
        self.carry_brake_counter[env_ids] = 0
        self.approach_target_xy[env_ids] = 0.0
        self.place_target_pos_w[env_ids] = 0.0
        self.place_target_quat_w[env_ids] = 0.0
        self.place_target_quat_w[env_ids, 0] = 1.0
        self.object_target_pos_w[env_ids] = 0.0
        self.object_target_quat_w[env_ids] = 0.0
        self.object_target_quat_w[env_ids, 0] = 1.0
        self.drop_done_ever[env_ids] = False


def _advance_phases_full(
    base_env,
    state: _PhaseState,
    table1_world_xy: torch.Tensor,
    table2_world_xy: torch.Tensor,
    cfg: PPOFinetuneConfig,
    rollout_cfg: RolloutConfig,
) -> dict:
    """Update per-env phase ids through walk-pick-carry-drop using physics gates.
    Args: base_env: mjlab env; state: _PhaseState with mutable per-env tensors; table1_world_xy / table2_world_xy: (N, 2) world positions used for distance gates and target locking; cfg: PPOFinetuneConfig thresholds; rollout_cfg: shared RolloutConfig (upright_min, pick velocity caps, etc.).
    Returns: dict with 'pick_fires', 'carry_fires', 'drop_fires', 'drop_done' boolean masks (N,) for the current step.
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]

    up_align = -robot.data.projected_gravity_b[:, 2]
    upright = up_align > rollout_cfg.upright_min
    root_pos_xy = robot.data.root_link_pos_w[:, :2]
    box_pos_w = box.data.root_link_pos_w
    box_z = box_pos_w[:, 2]
    dxy_box = box_pos_w[:, :2] - root_pos_xy
    dist_box = torch.linalg.norm(dxy_box, dim=-1)
    base_vx = robot.data.root_link_lin_vel_b[:, 0].abs()
    base_wz = robot.data.root_link_ang_vel_b[:, 2].abs()
    r_quat = robot.data.root_link_quat_w
    r_w, r_xq, r_yq, r_zq = r_quat.unbind(-1)
    r_yaw = torch.atan2(2 * (r_w * r_zq + r_xq * r_yq), 1 - 2 * (r_yq * r_yq + r_zq * r_zq))
    box_dir = torch.atan2(dxy_box[:, 1], dxy_box[:, 0])
    heading_err_box = torch.atan2(torch.sin(box_dir - r_yaw), torch.cos(box_dir - r_yaw))
    box_v = box.data.root_link_lin_vel_w.norm(dim=-1)

    in_walk = state.phase_ids == WALK_PHASE_ID
    in_pick = state.phase_ids == PICK_PHASE_ID
    in_carry = state.phase_ids == CARRY_PHASE_ID
    in_drop = state.phase_ids == DROP_PHASE_ID

    # Walk -> Pick.
    entered_brake = in_walk & (dist_box <= cfg.walk_to_pick_dist)
    brake_mask = in_walk & (entered_brake | (state.brake_counter > 0))
    state.brake_counter = torch.where(brake_mask, state.brake_counter + 1, torch.zeros_like(state.brake_counter))
    pick_fires = (
        in_walk & upright
        & (dist_box <= cfg.walk_to_pick_dist)
        & (base_vx < rollout_cfg.walk_to_pick_speed_max)
        & (base_wz < rollout_cfg.walk_to_pick_wz_max)
        & (heading_err_box.abs() < rollout_cfg.walk_to_pick_heading_err_max)
        & (state.brake_counter >= cfg.walk_to_pick_brake_steps)
    )

    # Pick -> Carry (relaxed thresholds + timeout, matches fsm_wpcd iter5+).
    pick_stable = (
        in_pick
        & (box_z > cfg.pick_to_carry_box_z)
        & upright
        & (base_vx < 0.15) & (base_wz < 0.25) & (box_v < 0.30)
    )
    state.pick_stable_counter = torch.where(
        pick_stable, state.pick_stable_counter + 1, torch.zeros_like(state.pick_stable_counter),
    )
    state.in_pick_counter = torch.where(in_pick, state.in_pick_counter + 1, torch.zeros_like(state.in_pick_counter))
    pick_timeout = in_pick & upright & (box_z > cfg.pick_to_carry_box_z) & (state.in_pick_counter >= cfg.pick_timeout_steps)
    carry_fires = in_pick & ((pick_stable & (state.pick_stable_counter >= cfg.pick_stable_counter_min)) | pick_timeout)

    # Lock approach target on carry_fire: cardinal snap toward t1 side of t2 (iter6).
    if carry_fires.any():
        dxy_t1_t2 = table1_world_xy - table2_world_xy
        abs_x = dxy_t1_t2[:, 0].abs()
        abs_y = dxy_t1_t2[:, 1].abs()
        on_x_axis = abs_x >= abs_y
        outward_x = torch.where(on_x_axis, torch.sign(dxy_t1_t2[:, 0]), torch.zeros_like(dxy_t1_t2[:, 0]))
        outward_y = torch.where(on_x_axis, torch.zeros_like(dxy_t1_t2[:, 1]), torch.sign(dxy_t1_t2[:, 1]))
        new_approach_x = table2_world_xy[:, 0] + outward_x * cfg.pelvis_to_t2
        new_approach_y = table2_world_xy[:, 1] + outward_y * cfg.pelvis_to_t2
        state.approach_target_xy[:, 0] = torch.where(carry_fires, new_approach_x, state.approach_target_xy[:, 0])
        state.approach_target_xy[:, 1] = torch.where(carry_fires, new_approach_y, state.approach_target_xy[:, 1])

    # Carry -> Drop gates.
    t2_dir_all = table2_world_xy - root_pos_xy
    t2_ang_all = torch.atan2(t2_dir_all[:, 1], t2_dir_all[:, 0])
    heading_err_t2 = torch.atan2(torch.sin(t2_ang_all - r_yaw), torch.cos(t2_ang_all - r_yaw))
    dist_approach = torch.linalg.norm(state.approach_target_xy - root_pos_xy, dim=-1)
    near_t2 = in_carry & (dist_approach <= cfg.carry_arrival_dist)
    state.carry_brake_counter = torch.where(
        near_t2, state.carry_brake_counter + 1, torch.zeros_like(state.carry_brake_counter),
    )
    drop_fires = (
        in_carry & near_t2 & upright
        & (box_z > cfg.pick_to_carry_box_z - 0.1)
        & (base_vx < 0.12) & (base_wz < 0.35)
        & (heading_err_t2.abs() < 0.10)
        & (state.carry_brake_counter >= cfg.carry_brake_steps)
    )

    # Lock place_target on drop_fire: pos at t2 center, quat snapped to robot yaw (iter7).
    if drop_fires.any():
        fire_mask = drop_fires
        state.place_target_pos_w[:, 0] = torch.where(fire_mask, table2_world_xy[:, 0], state.place_target_pos_w[:, 0])
        state.place_target_pos_w[:, 1] = torch.where(fire_mask, table2_world_xy[:, 1], state.place_target_pos_w[:, 1])
        half_r = r_yaw / 2
        zero_r = torch.zeros_like(half_r)
        yaw_quat_robot = torch.stack([torch.cos(half_r), zero_r, zero_r, torch.sin(half_r)], dim=-1)
        state.place_target_quat_w[:] = torch.where(fire_mask.unsqueeze(-1), yaw_quat_robot, state.place_target_quat_w)
        state.object_target_pos_w[:] = torch.where(fire_mask.unsqueeze(-1), box_pos_w, state.object_target_pos_w)
        cur_box_quat = box.data.root_link_quat_w
        state.object_target_quat_w[:] = torch.where(fire_mask.unsqueeze(-1), cur_box_quat, state.object_target_quat_w)

    # Drop -> Done (success).
    box_xy_rel_t2 = (box_pos_w[:, :2] - table2_world_xy).abs()
    box_on_t2 = (box_xy_rel_t2[:, 0] <= TABLE2_HALF_X) & (box_xy_rel_t2[:, 1] <= TABLE2_HALF_Y)
    drop_done = in_drop & (box_z <= cfg.drop_z_done) & upright & box_on_t2
    new_drop_done = drop_done & ~state.drop_done_ever

    # Commit phase transitions.
    state.phase_ids = torch.where(pick_fires, torch.full_like(state.phase_ids, PICK_PHASE_ID), state.phase_ids)
    state.phase_ids = torch.where(carry_fires, torch.full_like(state.phase_ids, CARRY_PHASE_ID), state.phase_ids)
    state.phase_ids = torch.where(drop_fires, torch.full_like(state.phase_ids, DROP_PHASE_ID), state.phase_ids)
    state.phase_ids = torch.where(drop_done, torch.full_like(state.phase_ids, POST_DROP_PHASE_ID), state.phase_ids)
    state.drop_done_ever = state.drop_done_ever | drop_done

    return {
        "pick_fires": pick_fires,
        "carry_fires": carry_fires,
        "drop_fires": drop_fires,
        "drop_done": drop_done,
        "new_drop_done": new_drop_done,
        "dist_box": dist_box,
        "dist_approach": dist_approach,
        "box_z": box_z,
    }


def _distance_to_box(base_env) -> torch.Tensor:
    """Planar distance from each robot's pelvis to the box (shaping term).
    Args: base_env: mjlab env.
    Returns: (N,) tensor on the env device.
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    dxy = box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    return torch.linalg.norm(dxy, dim=-1)


def _teleport_done_envs(
    base_env,
    done_ids: torch.Tensor,
    cfg: PPOFinetuneConfig,
    seed_offset: int,
    table2_world_xy: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teleport the robots of ``done_ids`` to a fresh annulus spawn, rejection-sampling positions near table 2 when provided, and return (xy, yaw) for logging.
    Args: base_env: mjlab env; done_ids: 1-D long tensor of env indices to respawn; cfg: finetune config; seed_offset: integer to shift the per-reset seed so successive resets don't collide; table2_world_xy: (num_envs, 2) world positions of each env's table 2, used to reject spawns too close.
    Returns: (spawn_xy, spawn_yaw) tensors for the respawned envs.
    """
    device = base_env.scene["robot"].data.root_link_pos_w.device
    n = int(done_ids.numel())
    box_xy_all = base_env.scene["box"].data.root_link_pos_w[:, :2]
    box_xy = box_xy_all[done_ids]
    reset_seed = cfg.seed ^ (seed_offset * 7919)
    spawn_xy, spawn_yaw = _random_spawn_annulus(
        box_xy, cfg.spawn_r_min, cfg.spawn_r_max, reset_seed, device,
    )
    if table2_world_xy is not None:
        spawn_xy = _resample_spawn_outside_t2(
            spawn_xy, box_xy, table2_world_xy[done_ids],
            r_min=cfg.spawn_r_min, r_max=cfg.spawn_r_max,
            safety=cfg.spawn_t2_safety, seed=reset_seed, device=device,
        )
    spawn_z = torch.full((n,), 0.78, device=device)
    # Compose a full-batch pose with no-op entries for untouched envs; mjlab's writer expects full-batch tensors when env_ids is given.
    robot = base_env.scene["robot"]
    pose_full = torch.cat([
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
    ], dim=-1).clone()
    pose_full[done_ids, 0] = spawn_xy[:, 0]
    pose_full[done_ids, 1] = spawn_xy[:, 1]
    pose_full[done_ids, 2] = spawn_z
    half = spawn_yaw / 2.0
    pose_full[done_ids, 3] = torch.cos(half)
    pose_full[done_ids, 4] = 0.0
    pose_full[done_ids, 5] = 0.0
    pose_full[done_ids, 6] = torch.sin(half)
    env_ids_i32 = done_ids.to(torch.int32)
    robot.write_root_link_pose_to_sim(pose_full[done_ids], env_ids=env_ids_i32)
    robot.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids_i32,
    )
    return spawn_xy, spawn_yaw


def ppo_finetune(
    bc_ckpt: Path | str,
    cfg: PPOFinetuneConfig = PPOFinetuneConfig(),
    log_dir: Optional[Path | str] = None,
    resume_ckpt: Optional[Path | str] = None,
) -> Path:
    """Fine-tune the BC student with PPO + BeyondMimic-style regularisers.
    Args: bc_ckpt: path to a ``MonolithicPolicyStudent`` warm-start; cfg: PPOFinetuneConfig; log_dir: output directory (defaults under ``outputs/logs/g1_monolithic_ppo``); resume_ckpt: optional previous PPO checkpoint to resume from.
    Returns: Path to the final checkpoint.
    """
    from dataclasses import asdict
    from torch.utils.tensorboard import SummaryWriter

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # Env setup: pick scene with table 2 added per env (walk+pick+carry+drop chain).
    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.env_spacing = cfg.env_spacing
    # Sparse Jacobian keeps efc.J small enough on GPU at high env counts; the
    # dense form at nworld=1024 already overflowed VRAM on a 31 GB RTX 5090.
    env_cfg.sim.mujoco.jacobian = "sparse"
    table2_rel_xy_np = _sample_table2_positions_np(cfg.num_envs, cfg.table2_r_min, cfg.table2_r_max, cfg.seed)
    env_cfg.scene.spec_fn = _make_two_tables_spec_fn(cfg.num_envs, cfg.env_spacing, table2_rel_xy_np)
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(cfg.episode_max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=cfg.clip_actions)
    _ = env.get_observations()

    # Per-env table 1 (box spawn location, origin + offset) and table 2 world positions.
    origins_t = base_env.scene.env_origins[:, :2]
    table1_world_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    table2_rel_torch = torch.from_numpy(table2_rel_xy_np).to(device)
    table2_world_xy = origins_t + table2_rel_torch

    # Actor warm-start + frozen BC prior for the mimic reward / KL penalty.
    student = MonolithicPolicyStudent.load(str(bc_ckpt), device=device)
    actor = MonolithicActor(student, cfg.initial_log_std).to(device)
    bc_prior = copy.deepcopy(student).to(device).eval()
    for p in bc_prior.parameters():
        p.requires_grad_(False)
    critic = Critic(obs_dim=UNIFIED_OBS_DIM, hidden=cfg.critic_hidden).to(device)

    # Keep log_std frozen for the first iterations; see cfg.freeze_log_std_iters.
    actor.log_std.requires_grad_(False)

    lr = cfg.learning_rate
    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=lr,
    )

    rollout_cfg = RolloutConfig(seed=cfg.seed, max_steps=cfg.episode_max_steps)
    rollout_cfg.walk_to_pick_dist = 0.55

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"[ppo] log dir: {log_dir}")

    start_iter = 0
    if resume_ckpt is not None:
        blob = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
        actor.load_state_dict(blob["actor"])
        critic.load_state_dict(blob["critic"])
        optim.load_state_dict(blob["optim"])
        start_iter = int(blob.get("iteration", 0)) + 1
        print(f"[ppo] resumed from {resume_ckpt} at iteration {start_iter}")

    # Initial spawn + goal per env.
    state = _PhaseState(cfg.num_envs, device)
    last_action = torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device)
    goal = sample_goal(cfg.num_envs, device=device, seed=cfg.seed)
    all_ids = torch.arange(cfg.num_envs, device=device, dtype=torch.long)
    _teleport_done_envs(base_env, all_ids, cfg, seed_offset=0, table2_world_xy=table2_world_xy)
    env.step(torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device))
    _snap_pick_commands_batched(base_env)
    # Previous-step distances/heights for shaping deltas.
    prev_dist_box = _distance_to_box(base_env)
    prev_dist_approach = torch.linalg.norm(
        state.approach_target_xy - base_env.scene["robot"].data.root_link_pos_w[:, :2], dim=-1,
    )
    prev_box_z = base_env.scene["box"].data.root_link_pos_w[:, 2]
    # Per-env episode step counter for timeout check (independent of mjlab's).
    ep_step = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

    reset_counter = 1
    t_start = time.time()
    ewma_iter_s: Optional[float] = None
    for it in range(start_iter, cfg.max_iterations):
        t_iter_start = time.time()
        if it == cfg.freeze_log_std_iters:
            actor.log_std.requires_grad_(True)
            print(f"[ppo] iter {it}: unfroze log_std (initial exp(log_std)={actor.log_std.exp().mean().item():.3f})")

        # Rollout.
        T, N = cfg.num_steps_per_env, cfg.num_envs
        obs_buf = torch.zeros(T, N, UNIFIED_OBS_DIM, device=device)
        act_buf = torch.zeros(T, N, UNIFIED_ACT_DIM, device=device)
        logp_buf = torch.zeros(T, N, device=device)
        rew_buf = torch.zeros(T, N, device=device)
        done_buf = torch.zeros(T, N, device=device)
        val_buf = torch.zeros(T, N, device=device)

        iter_env_rew = 0.0
        iter_mimic_rew = 0.0
        iter_walk_shape = 0.0
        iter_carry_shape = 0.0
        iter_drop_shape = 0.0
        iter_success_rew = 0.0
        iter_pick_fires = 0
        iter_carry_fires = 0
        iter_drop_fires = 0
        iter_drops_done = 0
        iter_falls = 0

        for t in range(T):
            unified = _unified_obs_per_env(base_env, state.phase_ids, goal, last_action)
            with torch.no_grad():
                action, log_prob = actor.act(unified)
                value = critic(unified)
                bc_action = bc_prior(unified)

            # Advance phase before stepping so the recorded phase matches the action's intent.
            fires = _advance_phases_full(base_env, state, table1_world_xy, table2_world_xy, cfg, rollout_cfg)
            iter_pick_fires += int(fires["pick_fires"].sum())
            iter_carry_fires += int(fires["carry_fires"].sum())
            iter_drop_fires += int(fires["drop_fires"].sum())
            iter_drops_done += int(fires["new_drop_done"].sum())
            if fires["pick_fires"].any():
                _snap_pick_commands_batched(
                    base_env, env_ids=fires["pick_fires"].nonzero(as_tuple=False).squeeze(-1),
                )

            _, env_rew, done, extras = env.step(action)
            last_action = action.detach()
            ep_step += 1

            # Phase-specific shape rewards (deltas from previous step).
            cur_dist_box = fires["dist_box"]
            cur_dist_approach = fires["dist_approach"]
            cur_box_z = fires["box_z"]
            in_walk_f = (state.phase_ids == WALK_PHASE_ID).float()
            in_carry_f = (state.phase_ids == CARRY_PHASE_ID).float()
            in_drop_f = (state.phase_ids == DROP_PHASE_ID).float()
            walk_shape = cfg.walk_distance_coef * in_walk_f * (prev_dist_box - cur_dist_box)
            carry_shape = cfg.carry_shape_coef * in_carry_f * (prev_dist_approach - cur_dist_approach)
            drop_shape = cfg.drop_shape_coef * in_drop_f * (prev_box_z - cur_box_z)
            prev_dist_box = cur_dist_box
            prev_dist_approach = cur_dist_approach
            prev_box_z = cur_box_z

            # One-shot terminal bonus the step drop_done fires (only first time per episode).
            success_rew = cfg.drop_success_bonus * fires["new_drop_done"].float()

            mimic_rew = -cfg.mimic_coef * ((action - bc_action) ** 2).mean(dim=-1)

            # Episode terminations: fall (from env), drop_done (success), timeout.
            fell = done.bool() & ~extras.get("time_outs", torch.zeros_like(done).bool())
            succeeded = fires["new_drop_done"]
            timeout = ep_step >= cfg.episode_max_steps
            terminated_any = fell | succeeded | timeout
            fall_pen = -cfg.fall_penalty * fell.float()
            iter_falls += int(fell.sum())
            surv = cfg.survival_coef * torch.ones_like(env_rew)

            step_rew = env_rew + walk_shape + carry_shape + drop_shape + success_rew + mimic_rew + fall_pen + surv

            obs_buf[t] = unified
            act_buf[t] = action
            logp_buf[t] = log_prob
            rew_buf[t] = step_rew
            done_buf[t] = terminated_any.float()
            val_buf[t] = value

            iter_env_rew += float(env_rew.mean().item())
            iter_mimic_rew += float(mimic_rew.mean().item())
            iter_walk_shape += float(walk_shape.mean().item())
            iter_carry_shape += float(carry_shape.mean().item())
            iter_drop_shape += float(drop_shape.mean().item())
            iter_success_rew += float(success_rew.mean().item())

            if terminated_any.any():
                done_ids = terminated_any.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                _teleport_done_envs(base_env, done_ids, cfg, seed_offset=reset_counter, table2_world_xy=table2_world_xy)
                reset_counter += 1
                state.reset_envs(done_ids)
                last_action[done_ids] = 0.0
                ep_step[done_ids] = 0
                # Resample goals only for the finished envs.
                new_goal = sample_goal(int(done_ids.numel()), device=device, seed=cfg.seed + reset_counter)
                for name in ("object_init_pos_w", "object_init_quat_w", "place_target_pos_w", "place_target_quat_w"):
                    tensor = getattr(goal, name)
                    tensor[done_ids] = getattr(new_goal, name)
                prev_dist_box = _distance_to_box(base_env)
                prev_dist_approach = torch.linalg.norm(
                    state.approach_target_xy - base_env.scene["robot"].data.root_link_pos_w[:, :2], dim=-1,
                )
                prev_box_z = base_env.scene["box"].data.root_link_pos_w[:, 2]

        # Bootstrap with the value of the final observation.
        with torch.no_grad():
            final_unified = _unified_obs_per_env(base_env, state.phase_ids, goal, last_action)
            last_val = critic(final_unified)

        buf = RolloutBuffer(obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf)
        buf.compute_gae(last_val, cfg.gamma, cfg.lam)

        # Keep actor normalizer frozen at the BC stats so cur_mean and prior_mean share the same input scale; critic updates its own.
        with torch.no_grad():
            flat_obs = obs_buf.reshape(-1, UNIFIED_OBS_DIM)
            critic.normalizer.update(flat_obs)

        # PPO update.
        approx_kl_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        kl_prior_sum = 0.0
        n_updates = 0

        for obs_mb, act_mb, old_logp, ret_mb, adv_mb in buf.iterate(cfg.num_mini_batches, cfg.num_learning_epochs):
            new_logp, entropy = actor.evaluate(obs_mb, act_mb)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = critic(obs_mb)
            value_loss = F.mse_loss(value_pred, ret_mb)

            with torch.no_grad():
                prior_mean = bc_prior(obs_mb)
            cur_mean = actor.mean(obs_mb)
            # Mean-anchor penalty to the frozen BC prior: Gaussian-mean trust region equivalent to KL(N(cur_mean, s^2) || N(prior_mean, s^2)) for any shared s^2, which drops entropy terms already handled by cfg.entropy_coef.
            kl_prior = 0.5 * ((cur_mean - prior_mean) ** 2).sum(dim=-1).mean()

            loss = (
                policy_loss
                + cfg.value_loss_coef * value_loss
                - cfg.entropy_coef * entropy.mean()
                + cfg.kl_to_prior_coef * kl_prior
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), cfg.max_grad_norm,
            )
            optim.step()

            with torch.no_grad():
                approx_kl_sum += float((old_logp - new_logp).mean().item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            entropy_sum += float(entropy.mean().item())
            kl_prior_sum += float(kl_prior.item())
            n_updates += 1

        # Adaptive LR matching rsl-rl's schedule on desired_kl.
        mean_kl = approx_kl_sum / max(1, n_updates)
        if mean_kl > 2.0 * cfg.desired_kl:
            lr = max(1e-5, lr / 1.5)
        elif mean_kl < 0.5 * cfg.desired_kl and mean_kl > 0:
            lr = min(1e-2, lr * 1.5)
        for g in optim.param_groups:
            g["lr"] = lr

        # Time accounting: EWMA of iter wall time, used to project ETA.
        iter_s = time.time() - t_iter_start
        ewma_iter_s = iter_s if ewma_iter_s is None else 0.9 * ewma_iter_s + 0.1 * iter_s
        elapsed_s = time.time() - t_start
        remaining_iters = cfg.max_iterations - it - 1
        eta_s = remaining_iters * ewma_iter_s

        # Logging.
        mean_step_rew = rew_buf.mean().item()
        writer.add_scalar("Train/mean_reward_total", mean_step_rew, it)
        writer.add_scalar("Train/mean_reward_env", iter_env_rew / T, it)
        writer.add_scalar("Train/mean_reward_mimic", iter_mimic_rew / T, it)
        writer.add_scalar("Train/mean_reward_walk_shape", iter_walk_shape / T, it)
        writer.add_scalar("Train/mean_reward_carry_shape", iter_carry_shape / T, it)
        writer.add_scalar("Train/mean_reward_drop_shape", iter_drop_shape / T, it)
        writer.add_scalar("Train/mean_reward_success", iter_success_rew / T, it)
        writer.add_scalar("Train/pick_fires_per_iter", iter_pick_fires, it)
        writer.add_scalar("Train/carry_fires_per_iter", iter_carry_fires, it)
        writer.add_scalar("Train/drop_fires_per_iter", iter_drop_fires, it)
        writer.add_scalar("Train/drops_done_per_iter", iter_drops_done, it)
        writer.add_scalar("Train/falls_per_iter", iter_falls, it)
        writer.add_scalar("Train/learning_rate", lr, it)
        writer.add_scalar("Loss/policy", policy_loss_sum / n_updates, it)
        writer.add_scalar("Loss/value", value_loss_sum / n_updates, it)
        writer.add_scalar("Loss/entropy", entropy_sum / n_updates, it)
        writer.add_scalar("Loss/kl_to_prior", kl_prior_sum / n_updates, it)
        writer.add_scalar("Loss/approx_kl", mean_kl, it)
        writer.add_scalar("Policy/mean_std", actor.log_std.exp().mean().item(), it)
        writer.add_scalar("Perf/iter_s", iter_s, it)
        writer.add_scalar("Perf/elapsed_s", elapsed_s, it)
        writer.add_scalar("Perf/eta_s", eta_s, it)

        if (it % cfg.log_interval) == 0:
            print(
                f"[ppo] iter {it:4d}/{cfg.max_iterations} | r={mean_step_rew:+.3f} "
                f"env={iter_env_rew / T:+.3f} mim={iter_mimic_rew / T:+.3f} "
                f"wlk={iter_walk_shape / T:+.3f} car={iter_carry_shape / T:+.3f} drp={iter_drop_shape / T:+.3f} "
                f"suc={iter_success_rew / T:+.3f} | P={iter_pick_fires} C={iter_carry_fires} D={iter_drop_fires} "
                f"done={iter_drops_done} falls={iter_falls} | kl={mean_kl:.4f} "
                f"std={actor.log_std.exp().mean().item():.3f} lr={lr:.1e} "
                f"| iter_s={iter_s:.1f} elapsed={_fmt_hms(elapsed_s)} eta={_fmt_hms(eta_s)}"
            )

        if (it % cfg.save_interval) == 0 or it == cfg.max_iterations - 1:
            ck = log_dir / f"model_{it}.pt"
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optim": optim.state_dict(),
                "iteration": it,
                "cfg": asdict(cfg),
            }, str(ck))

    writer.close()
    base_env.close()
    final = log_dir / f"model_{cfg.max_iterations - 1}.pt"
    print(f"[ppo] finished. checkpoints under {log_dir}")
    return final


def find_latest_ppo_checkpoint(root: Path | str = DEFAULT_LOG_ROOT) -> Optional[Path]:
    """Return the newest ``model_*.pt`` under the most recent run folder, or None.
    Args: root: directory holding timestamped run subfolders.
    Returns: Path to the latest checkpoint or None if nothing is found.
    """
    root = Path(root)
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        return None
    cks = sorted(
        runs[-1].glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    return cks[-1] if cks else None
