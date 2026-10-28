##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Unified student observation used across every FSM phase.

Input:  a mjlab ManagerBasedRlEnv (any of the four phase envs); the current phase id; a GoalSample.
Output: a flat (num_envs, UNIFIED_OBS_DIM) tensor in the robot base frame.

Layout (must stay consistent across phases so the student can learn):
- proprio 96 = joint_pos(29) + joint_vel(29) + base_lin_vel_b(3) + base_ang_vel_b(3) + projected_gravity_b(3) + last_action(29)
- goal    15 = object_pos_b(3) + object_quat_w(4) + place_target_pos_b(3) + place_target_quat_w(4) + object_in_hand_flag(1)
- phase    4 = one-hot(walk, pick, carry, place)
Total 115.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.expert_registry import PHASE_ONE_HOT_DIM

NUM_JOINTS = 29
PROPRIO_DIM = NUM_JOINTS * 3 + 3 + 3 + 3
GOAL_DIM = 3 + 4 + 3 + 4 + 1
UNIFIED_OBS_DIM = PROPRIO_DIM + GOAL_DIM + PHASE_ONE_HOT_DIM
UNIFIED_ACT_DIM = NUM_JOINTS


@dataclass
class GoalSample:
    """One goal draw for the whole chain, held constant while the FSM rolls out.
    Fields: object_init_pos_w (n, 3); object_init_quat_w (n, 4); place_target_pos_w (n, 3); place_target_quat_w (n, 4).
    Frame: all tensors are world-frame.
    """
    object_init_pos_w: torch.Tensor
    object_init_quat_w: torch.Tensor
    place_target_pos_w: torch.Tensor
    place_target_quat_w: torch.Tensor


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by quaternion q (wxyz).
    Args: q: (n, 4); v: (n, 3).
    Returns: (n, 3).
    """
    w, x, y, z = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of quaternion q (wxyz).
    Args: q: (n, 4); v: (n, 3).
    Returns: (n, 3).
    """
    w, x, y, z = q.unbind(-1)
    q_conj = torch.stack([w, -x, -y, -z], dim=-1)
    return _quat_apply(q_conj, v)


def _phase_one_hot(phase_id: int, num_envs: int, device) -> torch.Tensor:
    """One-hot phase vector broadcast to a batch.
    Args: phase_id: 0..3; num_envs: batch dim; device: torch device.
    Returns: (num_envs, PHASE_ONE_HOT_DIM) tensor.
    """
    oh = torch.zeros(num_envs, PHASE_ONE_HOT_DIM, device=device)
    oh[:, phase_id] = 1.0
    return oh


def _read_object_state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read world-frame object pose and the pelvis-proximity in-hand flag, zero when no box.
    Args: env: ManagerBasedRlEnv for the active phase.
    Returns: (object_pos_w, object_quat_w, in_hand_flag) shaped (n, 3), (n, 4), (n, 1).
    """
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_link_pos_w
    n, device = root_pos_w.shape[0], root_pos_w.device
    if "box" in env.scene.entities:
        box = env.scene["box"]
        pos_w = box.data.root_link_pos_w
        quat_w = box.data.root_link_quat_w
        dist_to_pelvis = torch.linalg.norm(pos_w - root_pos_w, dim=-1, keepdim=True)
        in_hand = (dist_to_pelvis < 0.40).float()
        return pos_w, quat_w, in_hand
    return (
        torch.zeros(n, 3, device=device),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).contiguous(),
        torch.zeros(n, 1, device=device),
    )


def compute_unified_obs(
    env,
    phase_id: int,
    goal: GoalSample,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """Build the phase-agnostic student observation from a mjlab env plus the current goal.
    Args: env: mjlab env for the current phase; phase_id: 0..3 slot for the one-hot; goal: per-episode goal draw; last_action: (n, NUM_JOINTS) action from the previous step.
    Returns: (num_envs, UNIFIED_OBS_DIM) float tensor.
    """
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w

    obj_pos_w, obj_quat_w, in_hand = _read_object_state(env)
    obj_pos_b = _quat_apply_inverse(root_quat_w, obj_pos_w - root_pos_w)
    tgt_pos_b = _quat_apply_inverse(root_quat_w, goal.place_target_pos_w - root_pos_w)

    proprio = torch.cat([
        robot.data.joint_pos,
        robot.data.joint_vel,
        robot.data.root_link_lin_vel_b,
        robot.data.root_link_ang_vel_b,
        robot.data.projected_gravity_b,
        last_action,
    ], dim=-1)
    goal_vec = torch.cat([
        obj_pos_b, obj_quat_w, tgt_pos_b, goal.place_target_quat_w, in_hand,
    ], dim=-1)
    phase_oh = _phase_one_hot(phase_id, root_pos_w.shape[0], root_pos_w.device)
    return torch.cat([proprio, goal_vec, phase_oh], dim=-1)


def sample_goal(num_envs: int, device: str, seed: int | None = None) -> GoalSample:
    """Draw an independent (object spawn, place target) pair for every parallel env.
    Args: num_envs: batch size; device: torch device; seed: optional RNG seed.
    Returns: GoalSample on device.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))

    obj_x = 0.55 + 0.50 * (torch.rand(num_envs, device=device, generator=g) - 0.5)
    obj_y = 0.00 + 0.30 * (torch.rand(num_envs, device=device, generator=g) - 0.5)
    obj_z = torch.full((num_envs,), 0.585, device=device)
    object_init_pos_w = torch.stack([obj_x, obj_y, obj_z], dim=-1)
    object_init_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(num_envs, 4).contiguous()

    tgt_x = 1.20 + 0.30 * (torch.rand(num_envs, device=device, generator=g) - 0.5)
    tgt_y = 0.00 + 0.40 * (torch.rand(num_envs, device=device, generator=g) - 0.5)
    tgt_z = 0.50 + 0.20 * (torch.rand(num_envs, device=device, generator=g) - 0.5)
    place_target_pos_w = torch.stack([tgt_x, tgt_y, tgt_z], dim=-1)
    place_target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(num_envs, 4).contiguous()

    return GoalSample(
        object_init_pos_w=object_init_pos_w,
        object_init_quat_w=object_init_quat_w,
        place_target_pos_w=place_target_pos_w,
        place_target_quat_w=place_target_quat_w,
    )
