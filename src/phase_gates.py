##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Shared physics gates and per-env state for the walk-pick-carry-drop chain.

Input:  mjlab base env, per-env PhaseState tensors, table 1 / table 2 world positions.
Output: updated phase ids and boolean fire masks for pick / carry / drop transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.fsm_continuous import RolloutConfig
from src.fsm_continuous_batch import (
    PICK_PHASE_ID,
    WALK_PHASE_ID,
    _random_spawn_annulus,
)
from src.fsm_wpcd_batched import (
    CARRY_PHASE_ID,
    DROP_PHASE_ID,
    POST_DROP_PHASE_ID,
    TABLE2_HALF_X,
    TABLE2_HALF_Y,
    _resample_spawn_outside_t2,
)


@dataclass
class GateConfig:
    """Physics-gate thresholds and spawn envelope shared by DAgger, play, and PPO."""
    spawn_r_min: float = 1.5
    spawn_r_max: float = 5.0
    table2_r_min: float = 1.5
    table2_r_max: float = 5.0
    spawn_t2_safety: float = 1.0
    walk_to_pick_dist: float = 0.55
    walk_to_pick_brake_steps: int = 25
    pick_to_carry_box_z: float = 0.85
    pick_stable_counter_min: int = 10
    pick_timeout_steps: int = 150
    carry_arrival_dist: float = 0.20
    carry_brake_steps: int = 60
    drop_z_done: float = 0.72
    pelvis_to_t2: float = 0.61
    seed: int = 7


class PhaseState:
    """Mutable per-env book-keeping tensors used by advance_phases_full."""

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


def advance_phases_full(
    base_env,
    state: PhaseState,
    table1_world_xy: torch.Tensor,
    table2_world_xy: torch.Tensor,
    cfg: GateConfig,
    rollout_cfg: RolloutConfig,
) -> dict:
    """Update per-env phase ids through walk-pick-carry-drop using physics gates.
    Args: base_env: mjlab env; state: PhaseState with mutable per-env tensors; table1_world_xy / table2_world_xy: (N, 2) world positions used for distance gates and target locking; cfg: GateConfig thresholds; rollout_cfg: shared RolloutConfig (upright_min, pick velocity caps).
    Returns: dict with pick_fires / carry_fires / drop_fires / drop_done boolean masks (N,) plus dist_box / dist_approach / box_z scalars.
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

    box_xy_rel_t2 = (box_pos_w[:, :2] - table2_world_xy).abs()
    box_on_t2 = (box_xy_rel_t2[:, 0] <= TABLE2_HALF_X) & (box_xy_rel_t2[:, 1] <= TABLE2_HALF_Y)
    drop_done = in_drop & (box_z <= cfg.drop_z_done) & upright & box_on_t2
    new_drop_done = drop_done & ~state.drop_done_ever

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


def distance_to_box(base_env) -> torch.Tensor:
    """Planar distance from each robot's pelvis to the box.
    Args: base_env: mjlab env.
    Returns: (N,) tensor on the env device.
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    dxy = box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    return torch.linalg.norm(dxy, dim=-1)


def teleport_done_envs(
    base_env,
    done_ids: torch.Tensor,
    cfg: GateConfig,
    seed_offset: int,
    table2_world_xy: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teleport robots of done_ids to a fresh annulus spawn, rejecting positions near table 2.
    Args: base_env: mjlab env; done_ids: 1-D long tensor of env indices to respawn; cfg: GateConfig; seed_offset: integer that shifts the per-reset seed so successive resets do not collide; table2_world_xy: (num_envs, 2) world positions of each env's table 2 for rejection sampling.
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


__all__ = [
    "GateConfig",
    "PhaseState",
    "advance_phases_full",
    "distance_to_box",
    "teleport_done_envs",
    "WALK_PHASE_ID",
    "PICK_PHASE_ID",
    "CARRY_PHASE_ID",
    "DROP_PHASE_ID",
    "POST_DROP_PHASE_ID",
]
