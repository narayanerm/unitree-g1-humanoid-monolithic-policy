##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Batched adaptation of ``fsm_continuous.py`` for BC dataset generation.

Direct fork of ``fsm_continuous.rollout``, same setup (pick env + table2,
seed=7 defaults, thresholds, snap helpers), but with ``num_envs=N`` and per-env
FSM state so many episodes accumulate in parallel. Only walk-to-pick is kept;
carry/drop/walk_return are dropped since we only distill that segment. Records
``(unified_obs, action, phase)`` per active env per sim step and writes a .npz
matching the schema consumed by ``src.bc_train`` (unified student).

Everything single-env in the fork parent is either kept verbatim (transition
predicates, brake logic, ``_snap_pick_commands_to_robot``'s batched form) or
replaced with a per-env vectorised variant (teleport, velocity zero, one-hot
phase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.astar_planner import PICK_TABLE_OBSTACLE, AstarConfig, WaypointTracker, plan_path
from src.expert_registry import BY_NAME, PHASE_ONE_HOT_DIM, resolve_ckpt
from src.fsm_continuous import (
    RolloutConfig,
    _compute_walk_obs,
    _disable_terminations,
    _load_actor_from_state_dict,
    _resolve_pick_indices,
    _twist_toward,
)
from src.goal_features import (
    UNIFIED_ACT_DIM,
    UNIFIED_OBS_DIM,
    GoalSample,
    _quat_apply_inverse,
    _read_object_state,
    sample_goal,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "walk_pick_continuous.npz"

WALK_PHASE_ID = 0
PICK_PHASE_ID = 1


@dataclass
class BatchStats:
    """Rollout counters returned by ``build_dataset`` for post-hoc reporting."""
    total_transitions: int
    per_phase_kept: dict[str, int]
    walk_to_pick_fires: int
    pick_successes: int
    fell_envs: int


def _teleport_robot_batch(base_env, xy: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor) -> None:
    """Teleport each env's robot free joint to the requested pose and zero its twist.
    Args: base_env: mjlab env; xy: (N, 2) planar positions; z: (N,) heights; yaw: (N,) headings.
    Returns: none.
    """
    robot = base_env.scene["robot"]
    device = robot.data.root_link_pos_w.device
    n = base_env.num_envs
    pose = torch.zeros(n, 7, device=device)
    pose[:, 0] = xy[:, 0]
    pose[:, 1] = xy[:, 1]
    pose[:, 2] = z
    half = yaw / 2.0
    pose[:, 3] = torch.cos(half)
    pose[:, 6] = torch.sin(half)
    env_ids = torch.arange(n, dtype=torch.int32, device=device)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def _random_spawn_annulus(
    box_xy: torch.Tensor, r_min: float, r_max: float, seed: int, device: str,
    front_sector_half_width: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-env spawn xy in the full 360 deg annulus around each box with random yaw.
    Args: box_xy: (N, 2) world box positions; r_min: inner radius (m); r_max: outer radius (m); seed: RNG seed; device: torch device; front_sector_half_width: ignored, kept for backward-compat.
    Returns: (spawn_xy (N, 2), spawn_yaw (N,) uniform in [-pi, pi)).
    """
    del front_sector_half_width  # kept in signature so older callers don't break
    n = box_xy.shape[0]
    g = torch.Generator(device=device).manual_seed(int(seed))
    alpha = 2.0 * torch.pi * torch.rand(n, device=device, generator=g)
    u = torch.rand(n, device=device, generator=g)
    r = torch.sqrt(r_min * r_min + u * (r_max * r_max - r_min * r_min))
    dx = r * torch.cos(alpha)
    dy = r * torch.sin(alpha)
    spawn_xy = torch.stack([box_xy[:, 0] + dx, box_xy[:, 1] + dy], dim=-1)
    yaw = 2.0 * torch.pi * torch.rand(n, device=device, generator=g) - torch.pi
    return spawn_xy, yaw


def _twist_toward_box_batch(base_env, walk_speed_cap: float) -> torch.Tensor:
    """Build the walk-expert twist command that steers each env toward its own box.
    Args: base_env: mjlab env; walk_speed_cap: max forward speed (m/s) forwarded to ``_twist_toward``.
    Returns: (N, 3) twist tensor consumable by the walk expert.
    """
    box_xy = base_env.scene["box"].data.root_link_pos_w[:, :2]
    return _twist_toward(base_env, box_xy, walk_speed_cap)


def _twist_along_astar(base_env, tracker: WaypointTracker, walk_speed_cap: float) -> torch.Tensor:
    """Build the walk-expert twist along the A* path, switching to reactive pursuit at the end.
    Args: base_env: mjlab env with num_envs==1; tracker: per-episode ``WaypointTracker`` over
    the A* path; walk_speed_cap: max forward speed (m/s) forwarded to ``_twist_toward``.
    Returns: (1, 3) twist tensor consumable by the walk expert.
    Once the tracker is on its last waypoint and the robot is inside the arrival radius of it,
    steering falls back to the box directly so the last ~10 cm close the walk->pick handover
    envelope (the tracker's pure-pursuit toward the final waypoint is asymptotic and would
    otherwise orbit at a fixed radius).
    """
    robot = base_env.scene["robot"]
    device = robot.data.root_link_pos_w.device
    robot_xy_np = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
    _ = tracker.current(robot_xy_np)
    last_wp = tracker.waypoints[-1]
    d_last = float(np.linalg.norm(robot_xy_np - last_wp))
    if tracker.done and d_last < tracker.arrival_dist:
        return _twist_toward_box_batch(base_env, walk_speed_cap)
    wp = tracker.current(robot_xy_np)
    wp_t = torch.tensor([wp[0], wp[1]], device=device, dtype=torch.float32).unsqueeze(0)
    return _twist_toward(base_env, wp_t, walk_speed_cap)


def _snap_pick_commands_batched(base_env, env_ids: Optional[torch.Tensor] = None) -> None:
    """Anchor pick-expert command extras per env, fixing the yaw[0] broadcast of the single-env path.
    Args: base_env: mjlab env; env_ids: optional (K,) subset to update (avoids re-anchoring envs already past handover).
    Returns: none.
    """
    from src.policy_pick import (
        _CMD_KEY, LIFT_X_CENTER, CHEST_Z,
        GRASP_NOMINAL_Y, GRASP_NOMINAL_Z,
    )
    robot = base_env.scene["robot"]
    box = base_env.scene["box"]
    device = robot.data.root_link_pos_w.device
    n = base_env.num_envs
    if env_ids is None:
        env_ids = torch.arange(n, device=device)
    if env_ids.numel() == 0:
        return
    idx = env_ids.long()
    k = idx.numel()

    robot_pos = robot.data.root_link_pos_w[idx]
    box_pos = box.data.root_link_pos_w[idx]
    ident_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(k, 4).contiguous()

    base_env.extras[_CMD_KEY + "object_target_pos"][idx] = box_pos
    base_env.extras[_CMD_KEY + "object_target_quat"][idx] = ident_quat

    w, xq, yq, zq = robot.data.root_link_quat_w[idx].unbind(-1)
    yaw = torch.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
    cy = torch.cos(yaw); sy = torch.sin(yaw)
    tox, toy, toz = LIFT_X_CENTER, 0.0, CHEST_Z - 0.65
    off = torch.stack([cy * tox - sy * toy, sy * tox + cy * toy, torch.full((k,), toz, device=device)], dim=-1)
    base_env.extras[_CMD_KEY + "lift_target_pos"][idx] = robot_pos + off
    base_env.extras[_CMD_KEY + "lift_target_quat"][idx] = ident_quat

    gl = torch.tensor([0.0, GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device).expand(k, 3).contiguous()
    gr = torch.tensor([0.0, -GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device).expand(k, 3).contiguous()
    base_env.extras[_CMD_KEY + "grasp_offset_left"][idx] = gl
    base_env.extras[_CMD_KEY + "grasp_offset_right"][idx] = gr


def _phase_one_hot(phase_ids: torch.Tensor) -> torch.Tensor:
    """Encode per-env FSM phase ids as one-hot rows of width ``PHASE_ONE_HOT_DIM``.
    Args: phase_ids: (N,) integer phase index per env.
    Returns: (N, PHASE_ONE_HOT_DIM) float tensor on ``phase_ids.device``.
    """
    oh = torch.zeros(phase_ids.shape[0], PHASE_ONE_HOT_DIM, device=phase_ids.device)
    oh.scatter_(1, phase_ids.long().unsqueeze(-1), 1.0)
    return oh


def _unified_obs_per_env(base_env, phase_ids: torch.Tensor, goal: GoalSample, last_action: torch.Tensor) -> torch.Tensor:
    """Assemble the student-facing unified observation tensor with per-env phase ids.
    Args: base_env: mjlab env; phase_ids: (N,) FSM phase per env; goal: sampled goal for this batch; last_action: (N, A) previous student action.
    Returns: (N, UNIFIED_OBS_DIM) tensor laid out as proprio + goal + phase one-hot.
    """
    robot = base_env.scene["robot"]
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w
    obj_pos_w, obj_quat_w, in_hand = _read_object_state(base_env)
    obj_pos_b = _quat_apply_inverse(root_quat_w, obj_pos_w - root_pos_w)
    tgt_pos_b = _quat_apply_inverse(root_quat_w, goal.place_target_pos_w - root_pos_w)
    proprio = torch.cat([
        robot.data.joint_pos, robot.data.joint_vel,
        robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b,
        robot.data.projected_gravity_b, last_action,
    ], dim=-1)
    goal_vec = torch.cat([obj_pos_b, obj_quat_w, tgt_pos_b, goal.place_target_quat_w, in_hand], dim=-1)
    return torch.cat([proprio, goal_vec, _phase_one_hot(phase_ids)], dim=-1)


def build_dataset(
    num_envs: int = 128,
    device: str = "cuda:0",
    out_path: Path | str = DEFAULT_OUT,
    seed: int = 7,
    max_steps: int = 800,
    spawn_r_min: float = 1.0,
    spawn_r_max: float = 3.0,
    cfg: Optional[RolloutConfig] = None,
    verbose: bool = True,
    video_path: Optional[Path | str] = None,
    video_fps: int = 20,
    video_hw: tuple[int, int] = (720, 1280),
    use_astar: bool = False,
) -> BatchStats:
    """Roll a batched walk-to-pick FSM and record (unified_obs, action, phase) transitions to disk.
    Args: num_envs: parallel envs; device: torch device; out_path: .npz destination; seed: RNG seed; max_steps: sim step budget; spawn_r_min/spawn_r_max: annulus radii (m) around each table; cfg: optional override for ``RolloutConfig``; verbose: periodic progress prints; video_path: optional mp4 output; video_fps: mp4 framerate; video_hw: (height, width) frame size; use_astar: if True, walk phase follows an 8-connected A* path around the pick table (requires num_envs==1); if False, walk phase uses reactive pure-pursuit straight at the box.
    Returns: ``BatchStats`` with transition counts, handover fires, pick successes, and fall count.
    """
    if use_astar and num_envs != 1:
        raise ValueError(f"use_astar requires num_envs==1, got {num_envs}")
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.entity import EntityCfg
    from src.policy_pick import register as register_pick, TASK_ID as PICK_TASK_ID

    if cfg is None:
        cfg = RolloutConfig(seed=seed, max_steps=max_steps)
        # Widen brake envelope to 0.55 m so the swing foot clears the table front edge (box sits ~9 cm behind it) while still landing inside pick's grip reach.
        cfg.walk_to_pick_dist = 0.55

    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = num_envs
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(PICK_TASK_ID) or MjlabOnPolicyRunner
    pick_runner = runner_cls(env, asdict(agent_cfg), device=device)
    pick_runner.load(str(resolve_ckpt(BY_NAME["pick"])), load_cfg={"actor": True}, strict=True, map_location=device)
    pick_policy = pick_runner.get_inference_policy(device=device)

    walk_policy = _load_actor_from_state_dict(
        ckpt_path=resolve_ckpt(BY_NAME["walk"]),
        obs_dim=99, action_dim=29, hidden=(512, 256, 128), device=device,
    )

    _ = env.get_observations()
    pick_idx_cache = _resolve_pick_indices(base_env)

    box_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    spawn_xy, spawn_yaw = _random_spawn_annulus(box_xy, spawn_r_min, spawn_r_max, cfg.seed, device)
    spawn_z = torch.full((num_envs,), cfg.spawn_z, device=device)
    _teleport_robot_batch(base_env, spawn_xy, spawn_z, spawn_yaw)
    # Flush the teleport with one zero-action step; robot.data.* only reflects queued pose writes after a sim step.
    env.step(torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device))

    astar_tracker: Optional[WaypointTracker] = None
    if use_astar:
        # num_envs==1 was already asserted above.
        spawn_np = spawn_xy[0].detach().cpu().numpy()
        box_np = box_xy[0].detach().cpu().numpy()
        # Aim at the point on a circle of radius ``approach_r`` around the box that is closest
        # to the spawn, so A* enters from the spawn side of the table rather than always
        # detouring to the same snapped free cell behind it. approach_r stays under
        # cfg.walk_to_pick_dist so the last waypoint sits inside the walk->pick handover
        # envelope, letting the FSM fire naturally once the tracker exhausts the path.
        approach_r = min(0.45, cfg.walk_to_pick_dist - 0.05)
        dxy = spawn_np - box_np
        d = float(np.linalg.norm(dxy))
        if d < 1e-3:
            approach_np = box_np + np.array([approach_r, 0.0], dtype=np.float32)
        else:
            approach_np = box_np + (dxy / d) * approach_r
        # robot_r=0.15 uses the pelvis footprint (not the full arm-extended reach) so the
        # planner can approach the tabletop close enough to hit the walk->pick handover
        # envelope. Table half-extent (0.30) already provides margin.
        astar_cfg = AstarConfig(arena_half=max(6.0, spawn_r_max + 1.0), robot_r=0.15)
        path = plan_path(spawn_np, approach_np, [PICK_TABLE_OBSTACLE], cfg=astar_cfg)
        if path is None or len(path) < 2:
            if verbose:
                print(f"[cont-batch] A* returned no path from {spawn_np} to {approach_np}; falling back to reactive")
        else:
            astar_tracker = WaypointTracker(path, arrival_dist=astar_cfg.arrival_dist)
            if verbose:
                print(f"[cont-batch] A* path: {len(path)} waypoints, spawn={spawn_np}, approach={approach_np}, box={box_np}")

    goal = sample_goal(num_envs, device=device, seed=cfg.seed)

    fsm_state = torch.zeros(num_envs, dtype=torch.long, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    fires_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
    picked_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
    brake_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    last_action = torch.zeros(num_envs, UNIFIED_ACT_DIM, device=device)
    obs_rec: list[torch.Tensor] = []
    act_rec: list[torch.Tensor] = []
    phase_rec: list[torch.Tensor] = []

    _snap_pick_commands_batched(base_env)

    # Optional cinematic tracking camera matching the single-env fsm_continuous.py path.
    writer = None
    renderer = None
    if video_path is not None:
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        import mediapy
        h, w = video_hw
        viewer_cfg = ViewerConfig(
            height=h, width=w,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=4.0, elevation=-15.0, azimuth=135.0,
            max_extra_envs=0, env_idx=0,
            enable_shadows=True, enable_reflections=True,
        )
        renderer = OffscreenRenderer(model=base_env.sim.mj_model, cfg=viewer_cfg, scene=base_env.scene)
        renderer.initialize()
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = mediapy.VideoWriter(str(video_path), shape=(h, w), fps=video_fps)
        writer.__enter__()

    for step in range(cfg.max_steps):
        if not bool(active.any()):
            break

        robot = base_env.scene["robot"]
        box = base_env.scene["box"]
        up_align = -robot.data.projected_gravity_b[:, 2]
        upright = up_align > cfg.upright_min
        root_z = robot.data.root_link_pos_w[:, 2]
        box_z = box.data.root_link_pos_w[:, 2]
        dist = torch.linalg.norm(
            box.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1,
        )
        base_vx = robot.data.root_link_lin_vel_b[:, 0].abs()

        fell_now = active & ((up_align < 0.25) | ((root_z < 0.15) & (up_align < 0.7)))
        if fell_now.any():
            fell_ever |= fell_now
            active &= ~fell_now

        in_walk = active & (fsm_state == WALK_PHASE_ID)
        in_pick = active & (fsm_state == PICK_PHASE_ID)

        if astar_tracker is not None:
            twist = _twist_along_astar(base_env, astar_tracker, cfg.walk_speed_cap)
        else:
            twist = _twist_toward_box_batch(base_env, cfg.walk_speed_cap)
        brake_mask = in_walk & (dist <= cfg.walk_to_pick_dist)
        if brake_mask.any():
            twist = torch.where(brake_mask.unsqueeze(-1), torch.zeros_like(twist), twist)
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

        if writer is not None:
            renderer.update(base_env.sim.data)
            writer.add_image(renderer.render().copy())

        fires = (
            in_walk & upright
            & (dist <= cfg.walk_to_pick_dist)
            & (base_vx < cfg.walk_to_pick_speed_max)
            & (brake_counter >= cfg.walk_to_pick_brake_steps)
        )
        if fires.any():
            fires_ever |= fires
            fsm_state = torch.where(fires, torch.full_like(fsm_state, PICK_PHASE_ID), fsm_state)
            _snap_pick_commands_batched(base_env, env_ids=fires.nonzero(as_tuple=False).squeeze(-1))

        pick_ok = in_pick & (box_z > cfg.pick_to_carry_box_z) & upright
        if pick_ok.any():
            picked_ever |= pick_ok
            active &= ~pick_ok

        if verbose and (step % 50 == 0):
            print(
                f"[cont-batch] step {step:4d} active={int(active.sum())} walk={int(in_walk.sum())} "
                f"pick={int(in_pick.sum())} fires={int(fires_ever.sum())} "
                f"picks={int(picked_ever.sum())} falls={int(fell_ever.sum())}"
            )

    if writer is not None:
        writer.__exit__(None, None, None)
    if renderer is not None:
        renderer.close()
    base_env.close()

    if obs_rec:
        unified_arr = torch.cat(obs_rec, dim=0).numpy().astype(np.float32)
        action_arr = torch.cat(act_rec, dim=0).numpy().astype(np.float32)
        phase_arr = torch.cat(phase_rec, dim=0).numpy().astype(np.int8)
    else:
        unified_arr = np.zeros((0, UNIFIED_OBS_DIM), dtype=np.float32)
        action_arr = np.zeros((0, UNIFIED_ACT_DIM), dtype=np.float32)
        phase_arr = np.zeros((0,), dtype=np.int8)

    total = int(unified_arr.shape[0])
    walk_kept = int((phase_arr == WALK_PHASE_ID).sum())
    pick_kept = int((phase_arr == PICK_PHASE_ID).sum())

    fell_count = int(fell_ever.sum())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        unified_obs=unified_arr,
        action=action_arr,
        phase=phase_arr,
        fell=np.int32(fell_count),
    )
    print(
        f"[cont-batch] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
        f"{total} transitions: walk={walk_kept}, pick={pick_kept}, fell={fell_count})"
    )
    return BatchStats(
        total_transitions=total,
        per_phase_kept={"walk": walk_kept, "pick": pick_kept},
        walk_to_pick_fires=int(fires_ever.sum()),
        pick_successes=int(picked_ever.sum()),
        fell_envs=int(fell_ever.sum()),
    )


def _main() -> None:
    """Parse CLI flags and invoke ``build_dataset`` to write one batched walk-to-pick .npz.
    Args: none, reads --num-envs, --device, --out, --seed, --max-steps, --r-min, --r-max, --video-path, --astar from argv.
    Returns: none.
    """
    import argparse, os
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--video-path", default=None, help="Optional mp4 path to record the rollout.")
    ap.add_argument("--astar", action="store_true", help="Walk phase follows an A* path around the pick table (num_envs must be 1).")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    stats = build_dataset(
        num_envs=args.num_envs, device=args.device, out_path=args.out,
        seed=args.seed, max_steps=args.max_steps,
        spawn_r_min=args.r_min, spawn_r_max=args.r_max,
        video_path=args.video_path,
        use_astar=args.astar,
    )
    print(f"[cont-batch] stats: {stats}")


if __name__ == "__main__":
    _main()
