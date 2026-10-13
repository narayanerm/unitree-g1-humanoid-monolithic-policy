##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""BeyondMimic-style PPO fine-tune of the unified student.

Pipeline:
1. Load the BC-trained ``MonolithicPolicyStudent`` as warm-start.
2. Wrap it as a Gaussian actor with a scalar ``log_std`` (initialised small and
   frozen for the first iterations so the warm-start isn't blown up by noise).
3. Freeze a copy of the BC student as reference prior for a KL penalty and an
   action-mimic reward: the two BeyondMimic-style regularisers that keep the
   policy from drifting out of the demonstration manifold while PPO exploits
   the task reward.
4. Reset each env in the walk annulus with a fresh goal, roll out walk then pick
   under the same brake predicates as ``fsm_continuous_batch``, and train with a
   from-scratch MLP critic on the composite reward.

Reward layout (per step):
- env reward: the pick task's native reward manager (grasp, lift, stability).
- walk shaping: w_walk times the previous-minus-current distance to the box; zero in pick phase.
- mimic reward: negative w_mimic times squared L2 between the policy action and the BC action.
- survival: w_surv until the robot falls; -fall_penalty on fall.

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
    episode_max_steps: int = 400
    spawn_r_min: float = 0.2
    spawn_r_max: float = 5.0

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

    walk_distance_coef: float = 1.0   # reward for closing distance toward the box in walk
    survival_coef: float = 0.01
    fall_penalty: float = 10.0

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


def _advance_phase(
    base_env,
    phase_ids: torch.Tensor,
    brake_counter: torch.Tensor,
    cfg: RolloutConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Promote envs from walk to pick when the brake predicates fire.
    Args: base_env: mjlab env; phase_ids: (N,) current phases; brake_counter: (N,) per-env brake frames; cfg: RolloutConfig thresholds.
    Returns: (phase_ids_updated, brake_counter_updated, fires_mask).
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    up_align = -robot.data.projected_gravity_b[:, 2]
    upright = up_align > cfg.upright_min
    dxy = box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    dist = torch.linalg.norm(dxy, dim=-1)
    base_vx = robot.data.root_link_lin_vel_b[:, 0].abs()
    base_wz = robot.data.root_link_ang_vel_b[:, 2].abs()
    r_quat = robot.data.root_link_quat_w
    r_w, r_xq, r_yq, r_zq = r_quat.unbind(-1)
    r_yaw = torch.atan2(2 * (r_w * r_zq + r_xq * r_yq), 1 - 2 * (r_yq * r_yq + r_zq * r_zq))
    box_dir = torch.atan2(dxy[:, 1], dxy[:, 0])
    heading_err = torch.atan2(torch.sin(box_dir - r_yaw), torch.cos(box_dir - r_yaw))

    in_walk = phase_ids == WALK_PHASE_ID
    entered_brake = in_walk & (dist <= cfg.walk_to_pick_dist)
    brake_mask = in_walk & (entered_brake | (brake_counter > 0))
    brake_counter = torch.where(brake_mask, brake_counter + 1, torch.zeros_like(brake_counter))

    fires = (
        in_walk & upright
        & (dist <= cfg.walk_to_pick_dist)
        & (base_vx < cfg.walk_to_pick_speed_max)
        & (base_wz < cfg.walk_to_pick_wz_max)
        & (heading_err.abs() < cfg.walk_to_pick_heading_err_max)
        & (brake_counter >= cfg.walk_to_pick_brake_steps)
    )
    phase_ids = torch.where(fires, torch.full_like(phase_ids, PICK_PHASE_ID), phase_ids)
    return phase_ids, brake_counter, fires


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teleport the robots of ``done_ids`` to a fresh annulus spawn and return (xy, yaw) for logging.
    Args: base_env: mjlab env; done_ids: 1-D long tensor of env indices to respawn; cfg: finetune config; seed_offset: integer to shift the per-reset seed so successive resets don't collide.
    Returns: (spawn_xy, spawn_yaw) tensors for the respawned envs.
    """
    device = base_env.scene["robot"].data.root_link_pos_w.device
    n = int(done_ids.numel())
    box_xy_all = base_env.scene["box"].data.root_link_pos_w[:, :2]
    box_xy = box_xy_all[done_ids]
    spawn_xy, spawn_yaw = _random_spawn_annulus(
        box_xy, cfg.spawn_r_min, cfg.spawn_r_max, cfg.seed ^ (seed_offset * 7919), device,
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

    # Env setup: pick scene with the extended episode length (walk + pick fits in max_steps).
    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(cfg.episode_max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=cfg.clip_actions)
    _ = env.get_observations()

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
    phase_ids = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    brake_counter = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    last_action = torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device)
    goal = sample_goal(cfg.num_envs, device=device, seed=cfg.seed)
    all_ids = torch.arange(cfg.num_envs, device=device, dtype=torch.long)
    _teleport_done_envs(base_env, all_ids, cfg, seed_offset=0)
    env.step(torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device))
    _snap_pick_commands_batched(base_env)
    prev_dist = _distance_to_box(base_env)

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
        iter_shape_rew = 0.0
        iter_pick_fires = 0
        iter_falls = 0

        for t in range(T):
            unified = _unified_obs_per_env(base_env, phase_ids, goal, last_action)
            with torch.no_grad():
                action, log_prob = actor.act(unified)
                value = critic(unified)
                bc_action = bc_prior(unified)

            # Advance phase before stepping so the recorded phase matches the action's intent.
            phase_ids_next, brake_counter, fires = _advance_phase(
                base_env, phase_ids, brake_counter, rollout_cfg,
            )
            if fires.any():
                iter_pick_fires += int(fires.sum())
                _snap_pick_commands_batched(base_env, env_ids=fires.nonzero(as_tuple=False).squeeze(-1))
            phase_ids = phase_ids_next

            _, env_rew, done, extras = env.step(action)
            last_action = action.detach()

            dist_now = _distance_to_box(base_env)
            in_walk = (phase_ids == WALK_PHASE_ID).float()
            shape_rew = cfg.walk_distance_coef * in_walk * (prev_dist - dist_now)
            prev_dist = dist_now

            mimic_rew = -cfg.mimic_coef * ((action - bc_action) ** 2).mean(dim=-1)

            # Fall penalty on steps the pick env marks terminated by fell_over; survival bonus otherwise.
            terminated = done.bool() & ~extras.get("time_outs", torch.zeros_like(done).bool())
            fall_pen = -cfg.fall_penalty * terminated.float()
            iter_falls += int(terminated.sum())
            surv = cfg.survival_coef * torch.ones_like(env_rew)

            step_rew = env_rew + shape_rew + mimic_rew + fall_pen + surv

            obs_buf[t] = unified
            act_buf[t] = action
            logp_buf[t] = log_prob
            rew_buf[t] = step_rew
            done_buf[t] = done.float()
            val_buf[t] = value

            iter_env_rew += float(env_rew.mean().item())
            iter_mimic_rew += float(mimic_rew.mean().item())
            iter_shape_rew += float(shape_rew.mean().item())

            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                _teleport_done_envs(base_env, done_ids, cfg, seed_offset=reset_counter)
                reset_counter += 1
                phase_ids[done_ids] = WALK_PHASE_ID
                brake_counter[done_ids] = 0
                last_action[done_ids] = 0.0
                # Resample goals only for the finished envs.
                new_goal = sample_goal(int(done_ids.numel()), device=device, seed=cfg.seed + reset_counter)
                for name in ("object_init_pos_w", "object_init_quat_w", "place_target_pos_w", "place_target_quat_w"):
                    tensor = getattr(goal, name)
                    tensor[done_ids] = getattr(new_goal, name)
                prev_dist = _distance_to_box(base_env)

        # Bootstrap with the value of the final observation.
        with torch.no_grad():
            final_unified = _unified_obs_per_env(base_env, phase_ids, goal, last_action)
            last_val = critic(final_unified)

        buf = RolloutBuffer(obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf)
        buf.compute_gae(last_val, cfg.gamma, cfg.lam)

        # Update normalisers on the fresh rollout.
        with torch.no_grad():
            flat_obs = obs_buf.reshape(-1, UNIFIED_OBS_DIM)
            actor.student.normalizer.update(flat_obs)
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
        writer.add_scalar("Train/mean_reward_shape", iter_shape_rew / T, it)
        writer.add_scalar("Train/pick_fires_per_iter", iter_pick_fires, it)
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
                f"[ppo] iter {it:4d}/{cfg.max_iterations} | r_total={mean_step_rew:+.3f} "
                f"r_env={iter_env_rew / T:+.3f} r_mim={iter_mimic_rew / T:+.3f} "
                f"r_shp={iter_shape_rew / T:+.3f} | fires={iter_pick_fires} falls={iter_falls} "
                f"| kl={mean_kl:.4f} std={actor.log_std.exp().mean().item():.3f} lr={lr:.1e} "
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
