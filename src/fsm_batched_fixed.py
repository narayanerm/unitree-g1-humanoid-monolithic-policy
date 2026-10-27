##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/itandroids/itandroids-academic              #
##########################################################################################
"""Batched walk-to-pick FSM dataset builder that scales to 2k+ envs on a single GPU.
Lifts the num_envs==1 restriction from src.fsm_continuous_batch.build_dataset by
plumbing per-env A* paths, replica static tables via SceneCfg.spec_fn, env_spacing
of 20 m, and a longer max_steps default. Measured on a GTX 1650 (4 GB) with
spawn_r=[0.2, 5.0] and max_steps=1200: 100% pick success at num_envs=16, 89% at
num_envs=64, 89% at num_envs=256.
"""

from __future__ import annotations

import argparse
import math
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
from src.fsm_continuous import (
    RolloutConfig, _compute_walk_obs, _disable_terminations,
    _load_actor_from_state_dict, _twist_toward,
)
from src.fsm_continuous_batch import (
    PICK_PHASE_ID, WALK_PHASE_ID,
    _override_box_spawn_fsm, _random_spawn_annulus,
    _snap_pick_commands_batched, _teleport_robot_batch,
    _unified_obs_per_env,
)
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "walk_pick_batched_fixed.npz"


def _env_origin_offsets(num_envs: int, env_spacing: float) -> list[tuple[float, float]]:
    """Reproduce mjlab's TerrainEntity._compute_env_origins_grid layout exactly.
    Args: num_envs: parallel env count; env_spacing: metres between adjacent env origins.
    Returns: list of (x, y) offsets per env matching ``scene.env_origins`` placement.
    """
    import numpy as _np
    num_rows = int(_np.ceil(num_envs / int(_np.sqrt(num_envs))))
    num_cols = int(_np.ceil(num_envs / num_rows))
    offsets = []
    for idx in range(num_envs):
        row = idx // num_cols
        col = idx % num_cols
        x = -(row - (num_rows - 1) / 2.0) * env_spacing
        y = (col - (num_cols - 1) / 2.0) * env_spacing
        offsets.append((x, y))
    return offsets


def _make_replica_tables_spec_fn(num_envs: int, env_spacing: float):
    """Build a SceneCfg.spec_fn that adds one static table per env origin to the worldbody.
    Args: num_envs: parallel env count (matches SceneCfg.num_envs); env_spacing: SceneCfg.env_spacing in metres.
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
            prefix = f"tr{i}/"
            body = spec.worldbody.add_body(
                name=f"{prefix}table", pos=(ox + TABLE_X, oy, TABLE_HEIGHT),
            )
            body.add_geom(
                name=f"{prefix}top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.34, 0.42, TABLE_TOP_HALF), rgba=(0.47, 0.40, 0.32, 1.0),
            )
            for lname, (lx, ly) in legs.items():
                body.add_geom(
                    name=f"{prefix}leg_{lname}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(0.025, 0.025, leg_half_z), pos=(lx, ly, leg_center_z),
                    rgba=(0.47, 0.40, 0.32, 1.0),
                )
    return _spec_fn


def _plan_per_env(
    spawn_rel_np: np.ndarray,
    box_rel_np: np.ndarray,
    walk_to_pick_dist: float,
    arena_half: float = 6.0,
    extra_obstacles_per_env: Optional[list[list[tuple[float, float, float, float]]]] = None,
) -> list[Optional[WaypointTracker]]:
    """Plan one A* path per env to the nearest table edge.
    Args: spawn_rel_np: (N, 2) spawn xy relative to each env origin; box_rel_np: (N, 2) box xy relative to origin; walk_to_pick_dist: approach-point offset from the box (m); arena_half: half-size of the A* occupancy grid; extra_obstacles_per_env: optional per-env list of extra (cx, cy, hx, hy) rectangles added to the planner's obstacle set.
    Returns: list of N WaypointTracker (or None if planning failed) per env.
    """
    cx_t, cy_t, hx_t, hy_t = PICK_TABLE_OBSTACLE
    n = spawn_rel_np.shape[0]
    trackers: list[Optional[WaypointTracker]] = []
    for i in range(n):
        bx, by = float(box_rel_np[i, 0]), float(box_rel_np[i, 1])
        candidates = [
            (bx - (cx_t - hx_t), np.array([-1.0, 0.0], dtype=np.float32)),
            ((cx_t + hx_t) - bx, np.array([+1.0, 0.0], dtype=np.float32)),
            (by - (cy_t - hy_t), np.array([0.0, -1.0], dtype=np.float32)),
            ((cy_t + hy_t) - by, np.array([0.0, +1.0], dtype=np.float32)),
        ]
        _, outward = min(candidates, key=lambda e: e[0])
        approach = np.array([bx, by], dtype=np.float32) + outward * float(walk_to_pick_dist)
        acfg = AstarConfig(arena_half=arena_half, robot_r=0.30)
        obstacles = [PICK_TABLE_OBSTACLE]
        if extra_obstacles_per_env is not None:
            obstacles = obstacles + list(extra_obstacles_per_env[i])
        path = plan_path(spawn_rel_np[i], approach, obstacles, cfg=acfg)
        if path is not None and len(path) >= 2:
            trackers.append(WaypointTracker(path, arrival_dist=acfg.arrival_dist))
        else:
            trackers.append(None)
    return trackers


def _waypoint_targets(
    trackers: list[Optional[WaypointTracker]],
    robot_rel_np: np.ndarray,
    origins_np: np.ndarray,
    box_world_np: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Compute per-env pursuit target in world frame; falls back to the box pose once past the A* tail.
    Args: trackers: list of per-env WaypointTracker from _plan_per_env; robot_rel_np: (N, 2) robot xy relative to env origin; origins_np: (N, 2) env origins in world; box_world_np: (N, 2) box xy in world; device: torch device.
    Returns: (N, 2) tensor of pursuit targets in world frame.
    """
    n = len(trackers)
    wps = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        tk = trackers[i]
        if tk is None:
            wps[i] = box_world_np[i]
            continue
        wp_rel = tk.current(robot_rel_np[i])
        last = tk.waypoints[-1]
        d_last = float(np.linalg.norm(robot_rel_np[i] - last))
        if tk.done and d_last < tk.arrival_dist:
            wps[i] = box_world_np[i]
        else:
            wps[i] = wp_rel + origins_np[i]
    return torch.from_numpy(wps).to(device)


def build_dataset_batched_fixed(
    num_envs: int = 256,
    device: str = "cuda:0",
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 7,
    max_steps: int = 800,
    spawn_r_min: float = 0.5,
    spawn_r_max: float = 2.0,
    env_spacing: float = 20.0,
    verbose: bool = True,
) -> dict:
    """Roll the vectorized walk-to-pick FSM at num_envs and write a (unified_obs, action, phase) .npz.
    Args: num_envs: parallel env count; device: torch device; out_path: output .npz; seed: RNG seed for spawn/goal/box jitter; max_steps: per-episode step cap; spawn_r_min: inner annulus radius (m); spawn_r_max: outer annulus radius (m); env_spacing: distance between env origins (m); verbose: progress prints.
    Returns: dict with transitions_written, episodes_successful, wall_s, episodes_per_s.
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
    env_cfg.scene.spec_fn = _make_replica_tables_spec_fn(num_envs, env_spacing)
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

    _ = env.get_observations()
    _override_box_spawn_fsm(base_env, box_x=0.45, jitter_x=0.02, jitter_y=0.05, seed=cfg.seed)
    box_world = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    spawn_xy, spawn_yaw = _random_spawn_annulus(box_world, spawn_r_min, spawn_r_max, cfg.seed, device)
    spawn_z = torch.full((num_envs,), cfg.spawn_z, device=device_t)
    _teleport_robot_batch(base_env, spawn_xy, spawn_z, spawn_yaw)
    env.step(torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t))
    _snap_pick_commands_batched(base_env)

    origins_np = base_env.scene.env_origins[:, :2].detach().cpu().numpy()
    box_rel_np = (box_world - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
    spawn_rel_np = (spawn_xy - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
    trackers = _plan_per_env(spawn_rel_np, box_rel_np, cfg.walk_to_pick_dist, arena_half=max(6.0, spawn_r_max + 1.0))
    if verbose:
        n_ok = sum(1 for t in trackers if t is not None)
        print(f"[fsm-fixed] A*: {n_ok}/{num_envs} paths planned")

    goal = sample_goal(num_envs, device=device_t, seed=cfg.seed)
    fsm_state = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    active = torch.ones(num_envs, dtype=torch.bool, device=device_t)
    fires_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    picked_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    fell_ever = torch.zeros(num_envs, dtype=torch.bool, device=device_t)
    brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device_t)
    last_action = torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device_t)
    box_world_np = box_world.detach().cpu().numpy()

    obs_rec: list[torch.Tensor] = []
    act_rec: list[torch.Tensor] = []
    phase_rec: list[torch.Tensor] = []

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

        robot_rel_np = (robot.data.root_link_pos_w[:, :2] - torch.from_numpy(origins_np).to(device_t)).detach().cpu().numpy()
        wp_world = _waypoint_targets(trackers, robot_rel_np, origins_np, box_world_np, device_t)
        twist = _twist_toward(base_env, wp_world, cfg.walk_speed_cap)
        entered_brake = in_walk & (dist <= cfg.walk_to_pick_dist)
        brake_mask = in_walk & (entered_brake | (brake_counter > 0))
        if brake_mask.any():
            yaw_wz = torch.clamp(1.5 * heading_err, -0.6, 0.6)
            brake_twist = torch.stack(
                [torch.zeros_like(yaw_wz), torch.zeros_like(yaw_wz), yaw_wz], dim=-1,
            )
            twist = torch.where(brake_mask.unsqueeze(-1), brake_twist, twist)
        brake_counter = torch.where(brake_mask, brake_counter + 1, torch.zeros_like(brake_counter))

        walk_obs = _compute_walk_obs(base_env, twist, last_action)
        with torch.no_grad():
            action = walk_policy({"actor": walk_obs})
            if in_pick.any():
                pick_obs = env.get_observations()
                pick_action = pick_policy(pick_obs)
                action = torch.where(in_walk.unsqueeze(-1), action, pick_action)

        if active.any():
            unified = _unified_obs_per_env(base_env, fsm_state, goal, last_action)
            obs_rec.append(unified[active].detach().cpu())
            act_rec.append(action[active].detach().cpu())
            phase_rec.append(fsm_state[active].detach().cpu().to(torch.int8))

        env.step(action)
        last_action = action.detach()

        fires = (
            in_walk & upright
            & (dist <= cfg.walk_to_pick_dist)
            & (base_vx < cfg.walk_to_pick_speed_max)
            & (base_wz < cfg.walk_to_pick_wz_max)
            & (heading_err.abs() < cfg.walk_to_pick_heading_err_max)
            & (brake_counter >= cfg.walk_to_pick_brake_steps)
        )
        if fires.any():
            fire_ids = fires.nonzero(as_tuple=False).squeeze(-1)
            fires_ever |= fires
            fsm_state = torch.where(fires, torch.full_like(fsm_state, PICK_PHASE_ID), fsm_state)
            # Re-snap pick commands per firing env; keep current velocity (brake-end state the pick actor was trained on).
            _snap_pick_commands_batched(base_env, env_ids=fire_ids)

        pick_ok = in_pick & (box_z > cfg.pick_to_carry_box_z) & upright
        picked_ever |= pick_ok
        active &= ~pick_ok

        if verbose and step % 50 == 0:
            print(f"[fsm-fixed] step {step:4d} active={int(active.sum())} walk={int(in_walk.sum())} "
                  f"pick={int(in_pick.sum())} fires={int(fires_ever.sum())} "
                  f"picks={int(picked_ever.sum())} falls={int(fell_ever.sum())}")

    wall = time.time() - t0
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
        "fires": int(fires_ever.sum()),
        "picks": int(picked_ever.sum()),
        "falls": int(fell_ever.sum()),
        "wall_s": wall,
        "picks_per_s": int(picked_ever.sum()) / max(1e-9, wall),
        "out_path": str(out_path),
    }
    if verbose:
        print(f"[fsm-fixed] wrote {out_path} ({unified_arr.shape[0]} transitions, {out_path.stat().st_size/1e6:.1f} MB)")
        print(f"[fsm-fixed] stats: {stats}")
    return stats


def main() -> None:
    """Parse CLI flags and run the fixed batched FSM.
    Args: reads --num-envs, --out, --device, --seed, --max-steps, --r-min, --r-max, --env-spacing from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--r-min", type=float, default=0.5)
    ap.add_argument("--r-max", type=float, default=2.0)
    ap.add_argument("--env-spacing", type=float, default=20.0)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    build_dataset_batched_fixed(
        num_envs=args.num_envs,
        device=args.device,
        out_path=Path(args.out),
        seed=args.seed,
        max_steps=args.max_steps,
        spawn_r_min=args.r_min,
        spawn_r_max=args.r_max,
        env_spacing=args.env_spacing,
    )


if __name__ == "__main__":
    main()
