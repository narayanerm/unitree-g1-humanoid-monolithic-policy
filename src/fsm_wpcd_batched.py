##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/itandroids/itandroids-academic              #
##########################################################################################
"""Batched walk-pick-carry-drop FSM dataset builder scaling to 1k+ envs on one GPU.

Four phases rolled under the pick env infrastructure (shared robot + box state).
Carry (124-dim) and drop (147-dim) observations are constructed manually from
the shared state, avoiding instantiation of separate carry/drop worlds and the
teleport that would entail. Walk-pick fixes carried over: per-env A*, env_spacing,
per-env replica tables via spec_fn, _snap_pick_commands_batched at fire.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import torch

from src.astar_planner import PICK_TABLE_OBSTACLE, AstarConfig, WaypointTracker, plan_path
from src.expert_registry import BY_NAME, PHASE_ONE_HOT_DIM, resolve_ckpt
from src.fsm_batched_fixed import (
    _env_origin_offsets,
    _make_replica_tables_spec_fn,
    _plan_per_env,
    _waypoint_targets,
)
from src.fsm_continuous import (
    RolloutConfig, _compute_walk_obs, _disable_terminations,
    _load_actor_from_state_dict, _resolve_pick_indices, _twist_toward,
)
from src.fsm_continuous_batch import (
    _override_box_spawn_fsm, _random_spawn_annulus,
    _snap_pick_commands_batched, _teleport_robot_batch,
    _unified_obs_per_env,
)
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "walk_pick_carry_drop_batched.npz"

WALK_PHASE_ID = 0
PICK_PHASE_ID = 1
CARRY_PHASE_ID = 2
DROP_PHASE_ID = 3
POST_DROP_PHASE_ID = 4

# Table 2 footprint half-extents in the pick spec_fn (match _make_two_tables_spec_fn below).
TABLE2_HALF_X = 0.34
TABLE2_HALF_Y = 0.42

CARRY_OBS_DIM = 124
DROP_OBS_DIM = 147


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS for the ETA log line.
    Args: seconds: non-negative duration.
    Returns: zero-padded HH:MM:SS string.
    """
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resample_spawn_outside_t2(
    spawn_xy: torch.Tensor,
    box_xy: torch.Tensor,
    t2_xy: torch.Tensor,
    r_min: float,
    r_max: float,
    safety: float,
    seed: int,
    device: str,
    max_retries: int = 16,
) -> torch.Tensor:
    """Rejection-resample spawn_xy per-env until it clears table 2 by `safety` metres.
    Args: spawn_xy: (N, 2) current spawns; box_xy: (N, 2) table-1 reference points for the annulus; t2_xy: (N, 2) table-2 world centers to avoid; r_min/r_max: annulus around box_xy; safety: minimum distance from t2 center (m); seed: RNG seed; device: torch device; max_retries: outer loop cap.
    Returns: (N, 2) resampled spawns; envs that stay too close after max_retries keep their last draw.
    """
    sxy = spawn_xy.clone()
    g = torch.Generator(device=device).manual_seed(int(seed) ^ 0xBADF00D)
    n = sxy.shape[0]
    for _ in range(max_retries):
        d = torch.linalg.norm(sxy - t2_xy, dim=-1)
        bad = d < safety
        if not bool(bad.any()):
            break
        alpha = 2.0 * torch.pi * torch.rand(n, device=device, generator=g)
        u = torch.rand(n, device=device, generator=g)
        r = torch.sqrt(r_min * r_min + u * (r_max * r_max - r_min * r_min))
        new_xy = torch.stack([box_xy[:, 0] + r * torch.cos(alpha), box_xy[:, 1] + r * torch.sin(alpha)], dim=-1)
        sxy = torch.where(bad.unsqueeze(-1), new_xy, sxy)
    return sxy


def _sample_table2_positions_np(num_envs: int, r_min: float, r_max: float, seed: int) -> np.ndarray:
    """Sample table-2 xy positions in an annulus around table 1 center, relative to each env origin.
    Args: num_envs: parallel env count; r_min: inner radius around table 1 center (m); r_max: outer radius (m); seed: RNG seed.
    Returns: (num_envs, 2) float32 array of table-2 (x, y) relative to env origin.
    """
    from src.policy_pick import TABLE_X
    rng = np.random.default_rng(int(seed) ^ 0xDEADBEEF)
    alpha = 2.0 * np.pi * rng.random(num_envs)
    u = rng.random(num_envs)
    r = np.sqrt(r_min * r_min + u * (r_max * r_max - r_min * r_min))
    dx = r * np.cos(alpha)
    dy = r * np.sin(alpha)
    xy = np.stack([TABLE_X + dx, dy], axis=-1).astype(np.float32)
    return xy


def _make_two_tables_spec_fn(num_envs: int, env_spacing: float, table2_rel_xy: np.ndarray):
    """Build a SceneCfg.spec_fn that adds table 1 (brown) and table 2 (green) per env origin.
    Args: num_envs: parallel env count; env_spacing: SceneCfg.env_spacing in metres; table2_rel_xy: (num_envs, 2) array of table-2 xy relative to each env origin.
    Returns: Callable[[mujoco.MjSpec], None] usable as SceneCfg.spec_fn.
    """
    from src.policy_pick import TABLE_X, TABLE_HEIGHT, TABLE_TOP_HALF
    offsets = _env_origin_offsets(num_envs, env_spacing)

    def _spec_fn(spec: mujoco.MjSpec) -> None:
        leg_len_z = TABLE_HEIGHT - TABLE_TOP_HALF
        leg_half_z = leg_len_z / 2.0
        leg_center_z = leg_half_z - TABLE_HEIGHT
        legs = {
            "fl": (0.29, 0.37), "fr": (0.29, -0.37),
            "bl": (-0.29, 0.37), "br": (-0.29, -0.37),
        }
        for i, (ox, oy) in enumerate(offsets):
            prefix1 = f"t1_{i}/"
            body1 = spec.worldbody.add_body(name=f"{prefix1}table", pos=(ox + TABLE_X, oy, TABLE_HEIGHT))
            body1.add_geom(
                name=f"{prefix1}top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.34, 0.42, TABLE_TOP_HALF), rgba=(0.47, 0.40, 0.32, 1.0),
            )
            for lname, (lx, ly) in legs.items():
                body1.add_geom(
                    name=f"{prefix1}leg_{lname}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(0.025, 0.025, leg_half_z), pos=(lx, ly, leg_center_z),
                    rgba=(0.47, 0.40, 0.32, 1.0),
                )
            t2x = float(table2_rel_xy[i, 0])
            t2y = float(table2_rel_xy[i, 1])
            prefix2 = f"t2_{i}/"
            body2 = spec.worldbody.add_body(name=f"{prefix2}table", pos=(ox + t2x, oy + t2y, TABLE_HEIGHT))
            body2.add_geom(
                name=f"{prefix2}top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.34, 0.42, TABLE_TOP_HALF), rgba=(0.22, 0.55, 0.35, 1.0),
            )
            for lname, (lx, ly) in legs.items():
                body2.add_geom(
                    name=f"{prefix2}leg_{lname}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(0.025, 0.025, leg_half_z), pos=(lx, ly, leg_center_z),
                    rgba=(0.22, 0.55, 0.35, 1.0),
                )
    return _spec_fn


def _plan_carry_per_env(
    robot_rel_np: np.ndarray,
    table1_rel_np: np.ndarray,
    table2_rel_np: np.ndarray,
    approach_dist: float,
    arena_half: float,
) -> list[Optional[WaypointTracker]]:
    """Plan one A* path per env from the robot at carry start to the approach of table 2, treating both tables as obstacles.
    Args: robot_rel_np: (N, 2) robot xy at carry start relative to env origin; table1_rel_np: (N, 2) table 1 center relative to origin; table2_rel_np: (N, 2) table 2 center relative to origin; approach_dist: offset from table 2 center along the nearest edge (m); arena_half: half-size of the A* occupancy grid.
    Returns: list of N WaypointTracker (or None if planning failed) per env.
    """
    n = robot_rel_np.shape[0]
    half_x, half_y = 0.34, 0.42
    trackers: list[Optional[WaypointTracker]] = []
    for i in range(n):
        t1 = (float(table1_rel_np[i, 0]), float(table1_rel_np[i, 1]), half_x, half_y)
        t2 = (float(table2_rel_np[i, 0]), float(table2_rel_np[i, 1]), half_x, half_y)
        bx, by = t2[0], t2[1]
        candidates = [
            (bx - (t2[0] - half_x), np.array([-1.0, 0.0], dtype=np.float32)),
            ((t2[0] + half_x) - bx, np.array([+1.0, 0.0], dtype=np.float32)),
            (by - (t2[1] - half_y), np.array([0.0, -1.0], dtype=np.float32)),
            ((t2[1] + half_y) - by, np.array([0.0, +1.0], dtype=np.float32)),
        ]
        _, outward = min(candidates, key=lambda e: e[0])
        approach = np.array([bx, by], dtype=np.float32) + outward * float(approach_dist)
        acfg = AstarConfig(arena_half=arena_half, robot_r=0.30)
        path = plan_path(robot_rel_np[i], approach, [t1, t2], cfg=acfg)
        if path is not None and len(path) >= 2:
            trackers.append(WaypointTracker(path, arrival_dist=acfg.arrival_dist))
        else:
            trackers.append(None)
    return trackers


def _compute_carry_obs(
    base_env,
    twist_cmd: torch.Tensor,
    last_action: torch.Tensor,
    wrist_ids: list[int],
    left_hand_id: int,
    right_hand_id: int,
    trunk_id: int,
) -> torch.Tensor:
    """Assemble the 124-dim carry actor obs from shared pick env state and a synthesized twist.
    Args: base_env: pick env; twist_cmd: (N, 3) synthesized velocity command; last_action: (N, 29) previous action; wrist_ids: actuator ids of the six wrist actuators; left_hand_id/right_hand_id/trunk_id: body indices.
    Returns: (N, 124) tensor matching policy_carry.actor_terms layout.
    """
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    joint_default = robot.data.default_joint_pos
    joint_pos_rel = robot.data.joint_pos - joint_default
    joint_vel = robot.data.joint_vel

    body_pos = robot.data.body_link_pos_w
    left_hand_pos = body_pos[:, left_hand_id]
    right_hand_pos = body_pos[:, right_hand_id]
    trunk_pos = body_pos[:, trunk_id]
    box_pos_w = box.data.root_link_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    left_to_box = box_pos_w - left_hand_pos
    right_to_box = box_pos_w - right_hand_pos
    trunk_to_box = box_pos_w - trunk_pos

    wrist_force = robot.data.actuator_force[:, wrist_ids]

    box_rel = torch.nan_to_num(
        box_pos_w - robot_pos_w, nan=0.0, posinf=0.0, neginf=0.0,
    ).clamp(-3.0, 3.0)
    box_quat = torch.nan_to_num(
        box.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0,
    ).clamp(-1.0, 1.0)
    from src.policy_carry import BOX_HALF
    box_size = torch.full_like(box_rel, 2.0 * BOX_HALF)

    return torch.cat(
        [
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
            robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            twist_cmd,
            left_to_box, right_to_box, trunk_to_box,
            wrist_force,
            box_rel, box_quat, box_size,
        ],
        dim=-1,
    )


def _compute_drop_obs(
    base_env,
    last_action: torch.Tensor,
    wrist_ids: list[int],
    left_hand_id: int,
    right_hand_id: int,
    trunk_id: int,
    place_target_pos_w: torch.Tensor,
    place_target_quat_w: torch.Tensor,
    grasp_offset_left: torch.Tensor,
    grasp_offset_right: torch.Tensor,
    object_target_pos_w: torch.Tensor = None,
    object_target_quat_w: torch.Tensor = None,
) -> torch.Tensor:
    """Assemble the 147-dim drop actor obs from shared pick env state and snapped drop targets.
    Args: base_env: pick env; last_action: (N, 29) previous action; wrist_ids: wrist actuator ids; left_hand_id/right_hand_id/trunk_id: body indices; place_target_pos_w: (N, 3) world-frame place target; place_target_quat_w: (N, 4) target quat; grasp_offset_left/right: (N, 3) grasp vectors.
    Returns: (N, 147) tensor matching policy_drop.actor_terms layout.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    joint_default = robot.data.default_joint_pos
    joint_pos_rel = robot.data.joint_pos - joint_default
    joint_vel = robot.data.joint_vel

    body_pos = robot.data.body_link_pos_w
    left_hand_pos = body_pos[:, left_hand_id]
    right_hand_pos = body_pos[:, right_hand_id]
    trunk_pos = body_pos[:, trunk_id]
    box_pos_w = box.data.root_link_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    left_to_box = box_pos_w - left_hand_pos
    right_to_box = box_pos_w - right_hand_pos
    trunk_to_box = box_pos_w - trunk_pos

    wrist_force = robot.data.actuator_force[:, wrist_ids]

    box_rel = (box_pos_w - robot_pos_w).clamp(-3.0, 3.0)
    box_lin = box.data.root_link_lin_vel_w.clamp(-5.0, 5.0)
    box_ang = box.data.root_link_ang_vel_w.clamp(-10.0, 10.0)
    box_quat = box.data.root_link_quat_w.clamp(-1.0, 1.0)
    from src.policy_carry import BOX_HALF
    box_size = torch.full_like(box_rel, 2.0 * BOX_HALF)

    # cmd_object_target_pos: snapshot of the box world pose at drop_fire (training
    # stashes this once at reset; must stay frozen through the drop phase).
    # Fall back to current box pose if snapshot not supplied.
    if object_target_pos_w is None:
        object_target_pos_w = box_pos_w
        object_target_quat_w = box_quat
    object_target_pos_rel = quat_apply_inverse(root_quat, object_target_pos_w - robot_pos_w).clamp(-3.0, 3.0)
    object_target_quat = object_target_quat_w.clamp(-1.0, 1.0)
    place_target_rel = quat_apply_inverse(root_quat, place_target_pos_w - robot_pos_w).clamp(-3.0, 3.0)

    return torch.cat(
        [
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
            robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            left_to_box, right_to_box, trunk_to_box,
            wrist_force,
            box_rel, box_lin, box_ang, box_quat, box_size,
            grasp_offset_left, grasp_offset_right,
            object_target_pos_rel, object_target_quat,
            place_target_rel, place_target_quat_w,
        ],
        dim=-1,
    )


def _phase_one_hot(phase_ids: torch.Tensor) -> torch.Tensor:
    """Encode per-env phase ids as one-hot rows of width PHASE_ONE_HOT_DIM.
    Args: phase_ids: (N,) integer phase index per env.
    Returns: (N, PHASE_ONE_HOT_DIM) float tensor on phase_ids.device.
    """
    oh = torch.zeros(phase_ids.shape[0], PHASE_ONE_HOT_DIM, device=phase_ids.device)
    oh.scatter_(1, phase_ids.long().unsqueeze(-1), 1.0)
    return oh


def build_dataset_wpcd(
    num_envs: int = 64,
    device: str = "cuda:0",
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 7,
    max_steps: int = 1600,
    spawn_r_min: float = 1.5,
    spawn_r_max: float = 5.0,
    env_spacing: float = 20.0,
    table2_r_min: float = 1.5,
    table2_r_max: float = 5.0,
    carry_approach_dist: float = 0.55,
    carry_arrival_dist: float = 0.75,
    carry_speed_cap: float = 0.45,
    carry_brake_steps: int = 60,
    drop_box_z_done: float = 0.72,
    video_path: Optional[Path | str] = None,
    video_fps: int = 20,
    video_hw: tuple[int, int] = (720, 1280),
    video_grid_path: Optional[Path | str] = None,
    video_grid_pane_hw: tuple[int, int] = (360, 640),
    video_grid_distance: float = 3.5,
    verbose: bool = True,
) -> dict:
    """Roll the batched walk-pick-carry-drop FSM and write (unified_obs, action, phase) to disk.
    Args: num_envs: parallel env count; device: torch device; out_path: destination .npz; seed: RNG seed; max_steps: per-episode step cap; spawn_r_min: inner spawn radius; spawn_r_max: outer spawn radius; env_spacing: metres between env origins; table2_r_min/table2_r_max: annulus radii for table 2 around table 1 (m); carry_approach_dist: offset from table 2 center along the nearest edge used as the A* goal (m); carry_arrival_dist: dist to table 2 center that fires the drop (m); carry_speed_cap: forward speed cap during carry walking (m/s); carry_brake_steps: brake confirmation frames before drop fires; drop_box_z_done: box z threshold below which drop is marked complete; video_path: optional mp4 output; video_fps: mp4 framerate; video_hw: (height, width) frame size; verbose: periodic progress prints.
    Returns: dict with transitions_written, fires_pick, fires_carry, fires_drop, drops_done, falls, wall_s.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    cfg = RolloutConfig(seed=seed, max_steps=max_steps)
    cfg.walk_to_pick_dist = 0.55
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    table2_rel_xy_np = _sample_table2_positions_np(num_envs, table2_r_min, table2_r_max, seed)
    env_cfg.scene.spec_fn = _make_two_tables_spec_fn(num_envs, env_spacing, table2_rel_xy_np)
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    device_t = base_env.scene["robot"].data.root_link_pos_w.device

    pick_runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    pick_runner.load(str(resolve_ckpt(BY_NAME["pick"])), load_cfg={"actor": True}, strict=True, map_location=device)
    pick_policy = pick_runner.get_inference_policy(device=device)

    walk_policy = _load_actor_from_state_dict(
        ckpt_path=resolve_ckpt(BY_NAME["walk"]),
        obs_dim=99, action_dim=29, hidden=(512, 256, 128), device=device,
    )
    carry_policy = _load_actor_from_state_dict(
        ckpt_path=resolve_ckpt(BY_NAME["carry"]),
        obs_dim=CARRY_OBS_DIM, action_dim=29, hidden=(512, 256, 128), device=device,
    )
    drop_policy = _load_actor_from_state_dict(
        ckpt_path=resolve_ckpt(BY_NAME["place"]),
        obs_dim=DROP_OBS_DIM, action_dim=29, hidden=(512, 256, 128), device=device,
    )

    _ = env.get_observations()
    idx_cache = _resolve_pick_indices(base_env)
    wrist_ids = idx_cache["wrist_actuators"]
    left_hand_id = idx_cache["left_hand"]
    right_hand_id = idx_cache["right_hand"]
    trunk_id = idx_cache["trunk"]

    # Preload carry terminal buffer so drop handoff can sample a canonical pose.
    _carry_buf_path = ROOT / "outputs" / "carry_terminal_states.npz"
    _carry_buf = None
    if _carry_buf_path.exists():
        buf_np = np.load(str(_carry_buf_path))
        _carry_buf = {
            "joint_pos": torch.from_numpy(buf_np["joint_pos"]).to(device_t),
            "root_z": torch.from_numpy(buf_np["root_z"]).to(device_t),
            "quat_zy": torch.from_numpy(buf_np["quat_zy"]).to(device_t),
            "box_pos_by": torch.from_numpy(buf_np["box_pos_by"]).to(device_t),
            "box_quat_by": torch.from_numpy(buf_np["box_quat_by"]).to(device_t),
        }
        if verbose:
            print(f"[wpcd] loaded carry terminal buffer ({_carry_buf['joint_pos'].shape[0]} samples)")
    _override_box_spawn_fsm(base_env, box_x=0.45, jitter_x=0.02, jitter_y=0.05, seed=cfg.seed)
    box_world = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    # Reject spawns too close to table 2 (robot clearance ~0.3 m + table 2
    # half-extent ~0.42 m + margin): otherwise the robot can spawn with a leg
    # intersecting the second table and never move.
    _origins_pre = base_env.scene.env_origins[:, :2]
    _table2_world_pre = _origins_pre + torch.from_numpy(table2_rel_xy_np).to(device_t)
    spawn_xy, spawn_yaw = _random_spawn_annulus(box_world, spawn_r_min, spawn_r_max, cfg.seed, device)
    spawn_xy = _resample_spawn_outside_t2(
        spawn_xy, box_world, _table2_world_pre,
        r_min=spawn_r_min, r_max=spawn_r_max, safety=1.0, seed=cfg.seed, device=device,
    )
    spawn_z = torch.full((num_envs,), cfg.spawn_z, device=device_t)
    _teleport_robot_batch(base_env, spawn_xy, spawn_z, spawn_yaw)
    env.step(torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t))
    _snap_pick_commands_batched(base_env)

    origins_np = base_env.scene.env_origins[:, :2].detach().cpu().numpy()
    box_rel_np = (box_world - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
    spawn_rel_np = (spawn_xy - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
    # Add table 2 (origin-relative) as a per-env obstacle so walk's A* routes around
    # it instead of straight through — otherwise the robot clips table 2 on approach.
    t2_obstacles_per_env = [
        [(float(table2_rel_xy_np[i, 0]), float(table2_rel_xy_np[i, 1]), TABLE2_HALF_X, TABLE2_HALF_Y)]
        for i in range(num_envs)
    ]
    trackers = _plan_per_env(
        spawn_rel_np, box_rel_np, cfg.walk_to_pick_dist,
        arena_half=max(6.0, spawn_r_max + 1.0),
        extra_obstacles_per_env=t2_obstacles_per_env,
    )
    if verbose:
        n_ok = sum(1 for t in trackers if t is not None)
        print(f"[wpcd] A*: {n_ok}/{num_envs} paths planned")

    goal = sample_goal(num_envs, device=device_t, seed=cfg.seed)
    fsm_state = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    active = torch.ones(num_envs, dtype=torch.bool, device=device_t)
    fires_pick_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    fires_carry_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    fires_drop_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    drop_done_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    fell_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    carry_brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    pick_stable_counter = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    in_pick_counter = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    carry_pre_align_done = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    carry_trackers: list[Optional[WaypointTracker]] = [None] * num_envs
    last_action = torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t)
    box_world_np = box_world.detach().cpu().numpy()

    table2_rel_torch = torch.from_numpy(table2_rel_xy_np).to(device_t)
    origins_t = torch.from_numpy(origins_np).to(device_t)
    table2_world_xy = origins_t + table2_rel_torch
    table1_rel_np = (box_world - origins_t).detach().cpu().numpy()
    # Per-env carry destination: locked at carry_fire to the exact spot 0.61 m
    # before the table 2 center along the straight-line approach direction.
    # Matches drop training's pelvis-to-place-target distance (ROBOT_SPAWN_X_OFFSET=-0.05,
    # PLACE_TARGET_X_CENTER=0.51 ⇒ 0.56 m + ~5 cm box-forward slack).
    PELVIS_TO_T2_TRAINING = 0.61
    carry_approach_target_xy = table2_world_xy.clone()

    from src.policy_pick import TABLE_HEIGHT, TABLE_TOP_HALF, BOX_HALF
    table_top_z = TABLE_HEIGHT + TABLE_TOP_HALF
    box_rest_z = table_top_z + BOX_HALF
    place_target_pos_w = torch.zeros(num_envs, 3, device=device_t)
    place_target_pos_w[:, 0] = table2_world_xy[:, 0]
    place_target_pos_w[:, 1] = table2_world_xy[:, 1]
    place_target_pos_w[:, 2] = box_rest_z + base_env.scene.env_origins[:, 2]
    place_target_quat_w = torch.zeros(num_envs, 4, device=device_t)
    place_target_quat_w[:, 0] = 1.0
    # Snapshot of the box world pose at drop_fire (frozen for the whole drop phase).
    # cmd_object_target_pos / _quat obs read from this buffer, matching training's
    # reset-time stashing in env.extras[_CMD_KEY + "object_target_*"].
    object_target_pos_w = torch.zeros(num_envs, 3, device=device_t)
    object_target_quat_w = torch.zeros(num_envs, 4, device=device_t)
    object_target_quat_w[:, 0] = 1.0
    from src.policy_pick import GRASP_NOMINAL_Y, GRASP_NOMINAL_Z
    grasp_left = torch.tensor([0.0, GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device_t).expand(num_envs, 3).contiguous()
    grasp_right = torch.tensor([0.0, -GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device_t).expand(num_envs, 3).contiguous()

    obs_rec: list[torch.Tensor] = []
    act_rec: list[torch.Tensor] = []
    phase_rec: list[torch.Tensor] = []

    writer = None
    renderer = None
    grid_writer = None
    grid_renderers: list = []
    grid_rows = 0
    grid_cols = 0
    grid_pane_h = 0
    grid_pane_w = 0
    PHASE_LABELS = {WALK_PHASE_ID: "WALK", PICK_PHASE_ID: "PICK", CARRY_PHASE_ID: "CARRY", DROP_PHASE_ID: "DROP", POST_DROP_PHASE_ID: "STAND"}
    if video_path is not None:
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        import mediapy
        h, w = video_hw
        viewer_cfg = ViewerConfig(
            height=h, width=w,
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.5),
            distance=max(10.0, 2.0 * env_spacing),
            elevation=-55.0, azimuth=45.0,
            max_extra_envs=max(0, num_envs - 1),
            env_idx=0,
            enable_shadows=True, enable_reflections=True,
        )
        renderer = OffscreenRenderer(model=base_env.sim.mj_model, cfg=viewer_cfg, scene=base_env.scene)
        renderer.initialize()
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = mediapy.VideoWriter(str(video_path), shape=(h, w), fps=video_fps)
        writer.__enter__()
    if video_grid_path is not None:
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        import mediapy
        grid_pane_h, grid_pane_w = video_grid_pane_hw
        grid_cols = int(np.ceil(np.sqrt(num_envs)))
        grid_rows = int(np.ceil(num_envs / grid_cols))
        for env_i in range(num_envs):
            pane_cfg = ViewerConfig(
                height=grid_pane_h, width=grid_pane_w,
                origin_type=ViewerConfig.OriginType.ASSET_ROOT,
                entity_name="robot",
                distance=video_grid_distance,
                elevation=-20.0, azimuth=135.0,
                env_idx=env_i,
                max_extra_envs=0,
                enable_shadows=True, enable_reflections=True,
            )
            r_pane = OffscreenRenderer(model=base_env.sim.mj_model, cfg=pane_cfg, scene=base_env.scene)
            r_pane.initialize()
            grid_renderers.append(r_pane)
        video_grid_path = Path(video_grid_path)
        video_grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_h = grid_rows * grid_pane_h
        grid_w = grid_cols * grid_pane_w
        grid_writer = mediapy.VideoWriter(str(video_grid_path), shape=(grid_h, grid_w), fps=video_fps)
        grid_writer.__enter__()

    t0 = time.time()
    for step in range(cfg.max_steps):
        if not bool(active.any()):
            break
        robot = base_env.scene["robot"]
        box = base_env.scene["box"]

        up_align = -robot.data.projected_gravity_b[:, 2]
        upright = up_align > cfg.upright_min
        root_z = robot.data.root_link_pos_w[:, 2]
        box_z = box.data.root_link_pos_w[:, 2]
        dxy = box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
        dist = torch.linalg.norm(dxy, dim=-1)
        base_vx = robot.data.root_link_lin_vel_b[:, 0].abs()
        base_wz = robot.data.root_link_ang_vel_b[:, 2].abs()
        r_quat = robot.data.root_link_quat_w
        r_w, r_xq, r_yq, r_zq = r_quat.unbind(-1)
        r_yaw = torch.atan2(2 * (r_w * r_zq + r_xq * r_yq), 1 - 2 * (r_yq * r_yq + r_zq * r_zq))
        box_dir = torch.atan2(dxy[:, 1], dxy[:, 0])
        heading_err = torch.atan2(torch.sin(box_dir - r_yaw), torch.cos(box_dir - r_yaw))

        fell_now = active & ((up_align < 0.25) | ((root_z < 0.15) & (up_align < 0.7)))
        fell_ever |= fell_now
        active &= ~fell_now

        in_walk = active & (fsm_state == WALK_PHASE_ID)
        in_pick = active & (fsm_state == PICK_PHASE_ID)
        in_carry = active & (fsm_state == CARRY_PHASE_ID)
        in_drop = active & (fsm_state == DROP_PHASE_ID)
        in_post_drop = active & (fsm_state == POST_DROP_PHASE_ID)

        robot_rel_np = (robot.data.root_link_pos_w[:, :2] - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
        wp_world = _waypoint_targets(trackers, robot_rel_np, origins_np, box_world_np, device_t)
        walk_twist = _twist_toward(base_env, wp_world, cfg.walk_speed_cap)
        entered_brake = in_walk & (dist <= cfg.walk_to_pick_dist)
        brake_mask = in_walk & (entered_brake | (brake_counter > 0))
        if brake_mask.any():
            yaw_wz = torch.clamp(1.5 * heading_err, -0.6, 0.6)
            brake_twist = torch.stack(
                [torch.zeros_like(yaw_wz), torch.zeros_like(yaw_wz), yaw_wz], dim=-1,
            )
            walk_twist = torch.where(brake_mask.unsqueeze(-1), brake_twist, walk_twist)
        brake_counter = torch.where(brake_mask, brake_counter + 1, torch.zeros_like(brake_counter))

        walk_obs = _compute_walk_obs(base_env, walk_twist, last_action)
        action = torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t)
        with torch.no_grad():
            walk_action = walk_policy({"actor": walk_obs})
            action = torch.where(in_walk.unsqueeze(-1), walk_action, action)
            if in_pick.any():
                pick_obs = env.get_observations()
                pick_action = pick_policy(pick_obs)
                action = torch.where(in_pick.unsqueeze(-1), pick_action, action)
            if in_carry.any():
                # A* pursuit toward the approach target (0.61 m before table 2
                # along the locked carry-fire direction), routing around both
                # tables. Falls back to direct pursuit once the tracker is spent.
                robot_xy_w = robot.data.root_link_pos_w[:, :2]
                robot_rel_c_np = (robot_xy_w - origins_t).detach().cpu().numpy()
                approach_world_np = carry_approach_target_xy.detach().cpu().numpy()
                wp_world_c = _waypoint_targets(
                    carry_trackers, robot_rel_c_np, origins_np, approach_world_np, device_t,
                )
                carry_twist = _twist_toward(base_env, wp_world_c, carry_speed_cap)
                dist_tgt_now = torch.linalg.norm(
                    carry_approach_target_xy - robot.data.root_link_pos_w[:, :2], dim=-1,
                )
                # Pre-align stage (0.6 m > dist > 0.15 m from approach): if the
                # robot's yaw is off, stop and rotate here — far enough from
                # table 2 that swinging legs don't clip the table top during
                # correction. Latched so the robot doesn't oscillate between
                # rotate and pursuit once it has aligned the first time.
                t2_dir_all_c = table2_world_xy - robot.data.root_link_pos_w[:, :2]
                t2_dir_ang_c = torch.atan2(t2_dir_all_c[:, 1], t2_dir_all_c[:, 0])
                heading_err_t2_c = torch.atan2(
                    torch.sin(t2_dir_ang_c - r_yaw), torch.cos(t2_dir_ang_c - r_yaw),
                )
                in_pre_align_zone = in_carry & (dist_tgt_now <= 0.6) & (dist_tgt_now > 0.15)
                pre_align = in_pre_align_zone & (heading_err_t2_c.abs() > 0.15) & ~carry_pre_align_done
                if pre_align.any():
                    yaw_wz_pre = torch.clamp(2.0 * heading_err_t2_c, -0.4, 0.4)
                    pre_t = torch.stack(
                        [torch.zeros_like(yaw_wz_pre), torch.zeros_like(yaw_wz_pre), yaw_wz_pre],
                        dim=-1,
                    )
                    carry_twist = torch.where(pre_align.unsqueeze(-1), pre_t, carry_twist)
                # Latch once the robot aligns inside the pre-align zone.
                carry_pre_align_done = carry_pre_align_done | (in_pre_align_zone & (heading_err_t2_c.abs() <= 0.15))
                near_t2 = in_carry & (dist_tgt_now <= 0.15)
                if near_t2.any():
                    yaw_wz_c = torch.clamp(1.5 * heading_err_t2_c, -0.25, 0.25)
                    brake_t = torch.stack([torch.zeros_like(yaw_wz_c), torch.zeros_like(yaw_wz_c), yaw_wz_c], dim=-1)
                    carry_twist = torch.where(near_t2.unsqueeze(-1), brake_t, carry_twist)
                mode = os.environ.get("WPCD_CARRY", "trained")
                if mode == "trained":
                    carry_obs = _compute_carry_obs(
                        base_env, carry_twist, last_action,
                        wrist_ids, left_hand_id, right_hand_id, trunk_id,
                    )
                    carry_action = carry_policy({"actor": carry_obs})
                elif mode == "walk":
                    carry_walk_obs = _compute_walk_obs(base_env, carry_twist, last_action)
                    carry_action = walk_policy({"actor": carry_walk_obs})
                else:
                    pick_obs_c = env.get_observations()
                    carry_action = pick_policy(pick_obs_c)
                action = torch.where(in_carry.unsqueeze(-1), carry_action, action)
            if in_drop.any():
                drop_obs = _compute_drop_obs(
                    base_env, last_action,
                    wrist_ids, left_hand_id, right_hand_id, trunk_id,
                    place_target_pos_w, place_target_quat_w,
                    grasp_left, grasp_right,
                    object_target_pos_w, object_target_quat_w,
                )
                drop_action = drop_policy({"actor": drop_obs})
                action = torch.where(in_drop.unsqueeze(-1), drop_action, action)
            if in_post_drop.any():
                # Hold station after a successful drop: walk policy with zero twist
                # keeps the robot standing (action=0 collapses the pose, since
                # defaults aren't a stable PD setpoint without the policy driving).
                stand_twist = torch.zeros(num_envs, 3, device=device_t)
                stand_obs = _compute_walk_obs(base_env, stand_twist, last_action)
                stand_action = walk_policy({"actor": stand_obs})
                action = torch.where(in_post_drop.unsqueeze(-1), stand_action, action)

        # Skip recording for post-drop (stand-still steps aren't skill demos).
        # Clamp phase to <= DROP_PHASE_ID to stay within the one-hot width (4).
        record_mask = active & (fsm_state != POST_DROP_PHASE_ID)
        if record_mask.any():
            unified = _unified_obs_per_env(
                base_env, fsm_state.clamp(max=DROP_PHASE_ID), goal, last_action,
            )
            obs_rec.append(unified[record_mask].detach().cpu())
            act_rec.append(action[record_mask].detach().cpu())
            phase_rec.append(fsm_state[record_mask].detach().cpu().to(torch.int8))

        env.step(action)
        last_action = action.detach()

        if writer is not None:
            renderer.update(base_env.sim.data)
            writer.add_image(renderer.render().copy())
        if grid_writer is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont
                _pil_ok = True
                _font_size = max(18, int(grid_pane_h * 0.045))
                try:
                    _font = ImageFont.load_default(size=_font_size)
                except TypeError:
                    _font = ImageFont.load_default()
            except ImportError:
                _pil_ok = False
            panes = []
            for env_i, r_pane in enumerate(grid_renderers):
                r_pane.update(base_env.sim.data)
                img = r_pane.render().copy()
                phase_i = int(fsm_state[env_i].item())
                if bool(fell_ever[env_i]):
                    status = "FELL"
                elif not bool(active[env_i]):
                    status = "DONE" if bool(drop_done_ever[env_i]) else "STOP"
                else:
                    status = PHASE_LABELS.get(phase_i, "?")
                if _pil_ok:
                    pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil)
                    label = f"env{env_i}  {status}  t={step}"
                    bar_h = int(_font_size * 1.6)
                    bar_w = int(grid_pane_w * 0.6)
                    draw.rectangle([(0, 0), (bar_w, bar_h)], fill=(0, 0, 0))
                    draw.text((10, int(bar_h * 0.15)), label, fill=(255, 255, 255), font=_font)
                    img = np.array(pil)
                panes.append(img)
            while len(panes) < grid_rows * grid_cols:
                panes.append(np.zeros((grid_pane_h, grid_pane_w, 3), dtype=np.uint8))
            rows_imgs = [np.hstack(panes[r * grid_cols : (r + 1) * grid_cols]) for r in range(grid_rows)]
            grid_writer.add_image(np.vstack(rows_imgs))

        pick_fires = (
            in_walk & upright
            & (dist <= cfg.walk_to_pick_dist)
            & (base_vx < cfg.walk_to_pick_speed_max)
            & (base_wz < cfg.walk_to_pick_wz_max)
            & (heading_err.abs() < cfg.walk_to_pick_heading_err_max)
            & (brake_counter >= cfg.walk_to_pick_brake_steps)
        )
        if pick_fires.any():
            fire_ids = pick_fires.nonzero(as_tuple=False).squeeze(-1)
            fires_pick_ever |= pick_fires
            fsm_state = torch.where(pick_fires, torch.full_like(fsm_state, PICK_PHASE_ID), fsm_state)
            _snap_pick_commands_batched(base_env, env_ids=fire_ids)

        box_v = base_env.scene["box"].data.root_link_lin_vel_w.norm(dim=-1)
        pick_stable = (
            in_pick
            & (box_z > cfg.pick_to_carry_box_z)
            & upright
            & (base_vx < 0.15)
            & (base_wz < 0.25)
            & (box_v < 0.30)
        )
        pick_stable_counter = torch.where(pick_stable, pick_stable_counter + 1, torch.zeros_like(pick_stable_counter))
        in_pick_counter = torch.where(in_pick, in_pick_counter + 1, torch.zeros_like(in_pick_counter))
        # Timeout: once the box is lifted and the robot is upright, fire carry after
        # ~150 steps even if residual velocity never fully settles — otherwise envs
        # oscillating under the box weight sit in pick forever.
        pick_timeout = in_pick & upright & (box_z > cfg.pick_to_carry_box_z) & (in_pick_counter >= 150)
        carry_fires = in_pick & ((pick_stable & (pick_stable_counter >= 10)) | pick_timeout)
        if carry_fires.any():
            fires_carry_ever |= carry_fires
            fsm_state = torch.where(carry_fires, torch.full_like(fsm_state, CARRY_PHASE_ID), fsm_state)
            # Snap approach to the cardinal face of t2 (±x or ±y) whose outward
            # normal best aligns with the t1 direction — never diagonal. Training
            # had t1 purely on +x of t2 and the robot approached along a cardinal
            # axis; a diagonal approach would park the robot at a corner, so the
            # drop fires with box over the edge and tips off. Picking the
            # dominant axis of (t1 - t2) keeps the drop in its training manifold.
            robot_xy_cf = robot.data.root_link_pos_w[:, :2]
            dxy_t1_t2 = box_world - table2_world_xy
            abs_x = dxy_t1_t2[:, 0].abs()
            abs_y = dxy_t1_t2[:, 1].abs()
            on_x_axis = abs_x >= abs_y
            outward_x = torch.where(on_x_axis, torch.sign(dxy_t1_t2[:, 0]), torch.zeros_like(dxy_t1_t2[:, 0]))
            outward_y = torch.where(on_x_axis, torch.zeros_like(dxy_t1_t2[:, 1]), torch.sign(dxy_t1_t2[:, 1]))
            new_approach = torch.stack(
                [table2_world_xy[:, 0] + outward_x * PELVIS_TO_T2_TRAINING,
                 table2_world_xy[:, 1] + outward_y * PELVIS_TO_T2_TRAINING],
                dim=-1,
            )
            carry_approach_target_xy[:, 0] = torch.where(carry_fires, new_approach[:, 0], carry_approach_target_xy[:, 0])
            carry_approach_target_xy[:, 1] = torch.where(carry_fires, new_approach[:, 1], carry_approach_target_xy[:, 1])
            # Plan carry A* from the firing-env robot position to the approach
            # target, treating both tables as obstacles so carry doesn't walk
            # straight through table 1 (same arena / robot_r as walk).
            fire_ids_c = carry_fires.nonzero(as_tuple=False).squeeze(-1).tolist()
            robot_rel_cf_np = (robot_xy_cf - origins_t).detach().cpu().numpy()
            approach_rel_cf_np = (carry_approach_target_xy - origins_t).detach().cpu().numpy()
            for i in fire_ids_c:
                obstacles_i = [
                    (float(table1_rel_np[i, 0]), float(table1_rel_np[i, 1]), 0.34, 0.42),
                    (float(table2_rel_xy_np[i, 0]), float(table2_rel_xy_np[i, 1]), TABLE2_HALF_X, TABLE2_HALF_Y),
                ]
                acfg_c = AstarConfig(arena_half=max(6.0, spawn_r_max + 1.0), robot_r=0.30)
                path_c = plan_path(
                    robot_rel_cf_np[i].astype(np.float32),
                    approach_rel_cf_np[i].astype(np.float32),
                    obstacles_i, cfg=acfg_c,
                )
                if path_c is not None and len(path_c) >= 2:
                    carry_trackers[i] = WaypointTracker(path_c, arrival_dist=acfg_c.arrival_dist)
                else:
                    carry_trackers[i] = None
            if verbose:
                fired = int(carry_fires.sum())
                n_ok = sum(1 for i in fire_ids_c if carry_trackers[i] is not None)
                print(f"[wpcd] pick->carry fired on {fired} envs at step {step} (A* ok {n_ok}/{fired})")

        dist_t2_all = torch.linalg.norm(table2_world_xy - robot.data.root_link_pos_w[:, :2], dim=-1)
        t2_dir_all = table2_world_xy - robot.data.root_link_pos_w[:, :2]
        t2_ang_all = torch.atan2(t2_dir_all[:, 1], t2_dir_all[:, 0])
        heading_err_t2_all = torch.atan2(torch.sin(t2_ang_all - r_yaw), torch.cos(t2_ang_all - r_yaw))
        # Drop fire triggered when robot reaches the exact approach target
        # (0.61 m before table 2 center along locked direction).
        dist_approach_all = torch.linalg.norm(
            carry_approach_target_xy - robot.data.root_link_pos_w[:, :2], dim=-1,
        )
        near_t2_all = in_carry & (dist_approach_all <= 0.20)
        carry_brake_counter = torch.where(near_t2_all, carry_brake_counter + 1, torch.zeros_like(carry_brake_counter))
        # Drop fire gates match galpão FSM exactly: tighter heading_err and
        # velocity thresholds so brake has time to settle the robot into the
        # drop training's zero-velocity, yaw-aligned regime.
        drop_fires = (
            in_carry
            & near_t2_all
            & upright
            & (box_z > cfg.pick_to_carry_box_z - 0.1)
            & (base_vx < 0.12)
            & (base_wz < 0.35)
            & (heading_err_t2_all.abs() < 0.10)
            & (carry_brake_counter >= carry_brake_steps)
        )
        if drop_fires.any():
            fires_drop_ever |= drop_fires
            fsm_state = torch.where(drop_fires, torch.full_like(fsm_state, DROP_PHASE_ID), fsm_state)
            fire_mask = drop_fires
            fire_ids_t = fire_mask.nonzero(as_tuple=False).squeeze(-1)
            n_fire = fire_ids_t.numel()
            # Snap place_target to Table 2's absolute world-frame center (galpão
            # FSM style). Brake above has tightened heading_err to 0.1 rad so the
            # base-frame projection of this target is in the drop training jitter.
            place_target_pos_w[:, 0] = torch.where(fire_mask, table2_world_xy[:, 0], place_target_pos_w[:, 0])
            place_target_pos_w[:, 1] = torch.where(fire_mask, table2_world_xy[:, 1], place_target_pos_w[:, 1])
            # Snap place_target_quat_w to the robot's current yaw-only quat.
            # Training had the robot always facing +x (identity rotation) with
            # place_target_quat_w = identity, so the box (held in hands, aligned
            # with the robot) was pre-aligned to the target. In deployment the
            # robot faces whichever cardinal direction t2's chosen face dictates;
            # if we left place_target_quat_w = identity, the drop policy would
            # see the box 90°/180° off target and try to throw it sideways to
            # reorient. Matching the quat to the robot's yaw cancels that error.
            qw_r, qx_r, qy_r, qz_r = robot.data.root_link_quat_w.unbind(dim=-1)
            yaw_r_now = torch.atan2(2 * (qw_r * qz_r + qx_r * qy_r), 1 - 2 * (qy_r * qy_r + qz_r * qz_r))
            half_r = yaw_r_now / 2
            zero_r = torch.zeros_like(half_r)
            yaw_quat_robot = torch.stack([torch.cos(half_r), zero_r, zero_r, torch.sin(half_r)], dim=-1)
            place_target_quat_w[:] = torch.where(fire_mask.unsqueeze(-1), yaw_quat_robot, place_target_quat_w)
            # Snapshot current box pose into object_target buffer (training stashes
            # it at reset; this must stay frozen through the drop phase).
            cur_box_pos = base_env.scene["box"].data.root_link_pos_w
            cur_box_quat = base_env.scene["box"].data.root_link_quat_w
            object_target_pos_w[:] = torch.where(fire_mask.unsqueeze(-1), cur_box_pos, object_target_pos_w)
            object_target_quat_w[:] = torch.where(fire_mask.unsqueeze(-1), cur_box_quat, object_target_quat_w)
            # Only zero velocities + last_action: no teleport of robot or box.
            # The drop_policy must handle whatever carry-end state we hand it.
            robot.write_root_link_velocity_to_sim(
                torch.zeros(n_fire, 6, device=device_t), env_ids=fire_ids_t,
            )
            robot.write_joint_state_to_sim(
                robot.data.joint_pos[fire_ids_t],
                torch.zeros_like(robot.data.joint_vel[fire_ids_t]),
                env_ids=fire_ids_t,
            )
            base_env.scene["box"].write_root_link_velocity_to_sim(
                torch.zeros(n_fire, 6, device=device_t), env_ids=fire_ids_t,
            )
            last_action = last_action.clone()
            last_action[fire_ids_t] = 0.0
            if verbose:
                print(f"[wpcd] carry->drop fired on {int(drop_fires.sum())} envs at step {step}")

        box_xy_w = base_env.scene["box"].data.root_link_pos_w[:, :2]
        box_rel_t2 = (box_xy_w - table2_world_xy).abs()
        box_on_t2 = (box_rel_t2[:, 0] <= TABLE2_HALF_X) & (box_rel_t2[:, 1] <= TABLE2_HALF_Y)
        drop_done = in_drop & (box_z <= drop_box_z_done) & upright & box_on_t2
        if drop_done.any():
            new_done = drop_done & ~drop_done_ever
            drop_done_ever |= drop_done
            fsm_state = torch.where(drop_done, torch.full_like(fsm_state, POST_DROP_PHASE_ID), fsm_state)
            if verbose and new_done.any():
                print(f"[wpcd] drop_done -> post_drop on {int(new_done.sum())} envs at step {step}")

        if verbose and step % 50 == 0:
            dist_t2_dbg = torch.linalg.norm(table2_world_xy - robot.data.root_link_pos_w[:, :2], dim=-1)
            d_in_carry = dist_t2_dbg[in_carry] if in_carry.any() else torch.tensor([-1.0])
            elapsed_s = time.time() - t0
            pace_s = elapsed_s / max(1, step + 1)
            eta_s = max(0, cfg.max_steps - step - 1) * pace_s
            pct = 100.0 * (step + 1) / max(1, cfg.max_steps)
            print(f"[wpcd] step {step:4d}/{cfg.max_steps} ({pct:5.1f}%) "
                  f"active={int(active.sum())} "
                  f"W={int(in_walk.sum())} P={int(in_pick.sum())} C={int(in_carry.sum())} D={int(in_drop.sum())} S={int(in_post_drop.sum())} "
                  f"fires P={int(fires_pick_ever.sum())}/C={int(fires_carry_ever.sum())}/D={int(fires_drop_ever.sum())} "
                  f"done={int(drop_done_ever.sum())} falls={int(fell_ever.sum())} "
                  f"d_t2(C)=[{d_in_carry.min().item():.2f},{d_in_carry.max().item():.2f}] "
                  f"| elapsed={_fmt_hms(elapsed_s)} eta={_fmt_hms(eta_s)}")

    wall = time.time() - t0
    if writer is not None:
        writer.__exit__(None, None, None)
    if grid_writer is not None:
        grid_writer.__exit__(None, None, None)
    if renderer is not None:
        renderer.close()
    for r_pane in grid_renderers:
        r_pane.close()
    base_env.close()

    if obs_rec:
        unified_arr = torch.cat(obs_rec, dim=0).numpy().astype(np.float32)
        action_arr = torch.cat(act_rec, dim=0).numpy().astype(np.float32)
        phase_arr = torch.cat(phase_rec, dim=0).numpy().astype(np.int8)
    else:
        unified_arr = np.zeros((0, UNIFIED_OBS_DIM), dtype=np.float32)
        action_arr = np.zeros((0, UNIFIED_ACT_DIM), dtype=np.float32)
        phase_arr = np.zeros((0,), dtype=np.int8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), unified_obs=unified_arr, action=action_arr, phase=phase_arr)

    stats = {
        "transitions_written": int(unified_arr.shape[0]),
        "fires_pick": int(fires_pick_ever.sum()),
        "fires_carry": int(fires_carry_ever.sum()),
        "fires_drop": int(fires_drop_ever.sum()),
        "drops_done": int(drop_done_ever.sum()),
        "falls": int(fell_ever.sum()),
        "success_rate": float(drop_done_ever.sum()) / max(1, num_envs),
        "wall_s": wall,
        "out_path": str(out_path),
    }
    if verbose:
        print(f"[wpcd] wrote {out_path} ({unified_arr.shape[0]} transitions, {out_path.stat().st_size/1e6:.1f} MB)")
        print(f"[wpcd] stats: {stats}")
    return stats


def main() -> None:
    """Parse CLI flags and run the 4-phase batched FSM.
    Args: reads --num-envs, --out, --device, --seed, --max-steps, --r-min, --r-max, --env-spacing, --carry-steps, --carry-vx, --carry-wz, --drop-z-done from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--env-spacing", type=float, default=20.0)
    ap.add_argument("--t2-r-min", type=float, default=1.5, help="Inner radius of table 2 relative to table 1 (m).")
    ap.add_argument("--t2-r-max", type=float, default=5.0, help="Outer radius of table 2 relative to table 1 (m).")
    ap.add_argument("--carry-approach-dist", type=float, default=0.55)
    ap.add_argument("--carry-arrival-dist", type=float, default=0.40)
    ap.add_argument("--carry-speed-cap", type=float, default=0.45)
    ap.add_argument("--carry-brake-steps", type=int, default=20)
    ap.add_argument("--drop-z-done", type=float, default=0.72)
    ap.add_argument("--video-path", default=None, help="Optional mp4 path to record env 0 across all four phases.")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    build_dataset_wpcd(
        num_envs=args.num_envs,
        device=args.device,
        out_path=Path(args.out),
        seed=args.seed,
        max_steps=args.max_steps,
        spawn_r_min=args.r_min,
        spawn_r_max=args.r_max,
        env_spacing=args.env_spacing,
        table2_r_min=args.t2_r_min,
        table2_r_max=args.t2_r_max,
        carry_approach_dist=args.carry_approach_dist,
        carry_arrival_dist=args.carry_arrival_dist,
        carry_speed_cap=args.carry_speed_cap,
        carry_brake_steps=args.carry_brake_steps,
        drop_box_z_done=args.drop_z_done,
        video_path=Path(args.video_path) if args.video_path else None,
    )


if __name__ == "__main__":
    main()
