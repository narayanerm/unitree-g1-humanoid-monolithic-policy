##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""G1 carry task: walk while carrying a box, warm-started from the walk policy.

Input: play flag (bool) for g1_carry_env_cfg.
Output: ManagerBasedRlEnvCfg (env), RslRlOnPolicyRunnerCfg (PPO), and register() -> TASK_ID.
"""

from __future__ import annotations

import math
import re

import mujoco
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME, get_spec
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr, joint_pos_rel, joint_vel_rel, last_action, time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import base_ang_vel, base_lin_vel, projected_gravity, UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# Shared with walk: sign-consistency penalty for commanded yaw (kills startup rotation bias).
from src.policy_walk import wz_sign_penalty


# Local helpers duplicated from policy_pick to keep carry independent; mirror changes in policy_pick.py.
PICK_BUFFER_PATH = "outputs/pick_terminal_states.npz"

BOX_HALF     = 0.105
BOX_MASS_MIN = 0.30
BOX_MASS_MAX = 3.00

G1_FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def _get_box_spec() -> mujoco.MjSpec:
    """Build the MjSpec for the carry box entity.
    Args: none.
    Returns: mujoco.MjSpec with a single free-jointed box body.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    freejoint = body.add_freejoint()
    freejoint.name = "box_free"
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(BOX_HALF, BOX_HALF, BOX_HALF),
        mass=BOX_MASS_MIN,
        rgba=(0.85, 0.22, 0.18, 1.0),
        condim=6,
        friction=(4.0, 0.25, 0.08),
        solref=(0.004, 1.0),
        solimp=(0.95, 0.995, 0.001, 0.5, 2.0),
    )
    return spec


def _entity_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return world-frame position of an entity body or its root link.
    Args: env: RL env; asset_cfg: scene entity selector (optionally body-scoped).
    Returns: (num_envs, 3) tensor of world positions.
    """
    entity: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_names is not None or isinstance(asset_cfg.body_ids, list):
        return entity.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)
    return entity.data.root_link_pos_w


def body_vector(env, source_cfg, target_cfg):
    """Compute world-frame vector from source body to target body.
    Args: env: RL env; source_cfg: origin entity; target_cfg: destination entity.
    Returns: (num_envs, 3) world-frame delta tensor.
    """
    return _entity_pos(env, target_cfg) - _entity_pos(env, source_cfg)


def box_position_obs(env, box_cfg):
    """Robot-to-box position clipped to a safe observation range.
    Args: env: RL env; box_cfg: box scene entity.
    Returns: (num_envs, 3) tensor clamped to [-3, 3].
    """
    box_pos = env.scene[box_cfg.name].data.root_link_pos_w
    robot_pos = env.scene["robot"].data.root_link_pos_w
    rel = box_pos - robot_pos
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)


def box_lin_vel_obs(env, box_cfg):
    """Box world-frame linear velocity clipped for observation stability.
    Args: env: RL env; box_cfg: box scene entity.
    Returns: (num_envs, 3) tensor clamped to [-5, 5].
    """
    v = env.scene[box_cfg.name].data.root_link_lin_vel_w
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)


def box_ang_vel_obs(env, box_cfg):
    """Box world-frame angular velocity clipped for observation stability.
    Args: env: RL env; box_cfg: box scene entity.
    Returns: (num_envs, 3) tensor clamped to [-10, 10].
    """
    w = env.scene[box_cfg.name].data.root_link_ang_vel_w
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)


def box_orientation_obs(env, box_cfg):
    """Box world-frame quaternion clipped to unit range.
    Args: env: RL env; box_cfg: box scene entity.
    Returns: (num_envs, 4) wxyz quaternion tensor.
    """
    q = env.scene[box_cfg.name].data.root_link_quat_w
    return torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def box_size_obs(env):
    """Constant per-env box half-extents broadcast for the observation vector.
    Args: env: RL env.
    Returns: (num_envs, 3) tensor of full box side lengths.
    """
    root_pos = env.scene["robot"].data.root_link_pos_w
    size = torch.tensor([2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF], device=root_pos.device)
    return size.expand(root_pos.shape[0], -1)


def wrist_actuator_force_obs(env, asset_cfg: SceneEntityCfg):
    """Extract wrist actuator forces for the requested actuator subset.
    Args: env: RL env; asset_cfg: robot entity selecting wrist actuators.
    Returns: (num_envs, num_wrist_actuators) force tensor.
    """
    robot: Entity = env.scene[asset_cfg.name]
    forces = robot.data.actuator_force
    if asset_cfg.actuator_ids is not None and not isinstance(asset_cfg.actuator_ids, slice):
        idx = asset_cfg.actuator_ids
        if isinstance(idx, (list, tuple)):
            forces = forces[:, list(idx)]
        else:
            forces = forces[:, idx]
    return forces


def non_hand_box_contact_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg, box_cfg,
    threshold: float = 0.22,
):
    """Penalize the box coming too close to trunk, hips, or shanks.
    Args: env: RL env; trunk_cfg/left_shank_cfg/right_shank_cfg/left_hip_cfg/right_hip_cfg: body selectors; box_cfg: box selector; threshold: proximity distance in meters.
    Returns: (num_envs,) non-negative penalty summed across body parts.
    """
    box_pos = _entity_pos(env, box_cfg)
    penalty = torch.zeros(box_pos.shape[0], device=box_pos.device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg):
        body_pos = _entity_pos(env, cfg)
        dist = torch.linalg.norm(body_pos - box_pos, dim=-1)
        penalty += torch.clamp(threshold - dist, min=0.0) / threshold
    return penalty


_PICK_BUFFER_CACHE: dict[str, torch.Tensor] | None = None


def _load_pick_buffer(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load and cache the pick terminal-state buffer (robot + held box) as device tensors.
    Args: path: buffer file path; device: target torch device.
    Returns: dict of tensors keyed by buffer field, cached across calls.
    """
    global _PICK_BUFFER_CACHE
    if _PICK_BUFFER_CACHE is None or _PICK_BUFFER_CACHE.get("_path") != path:
        import numpy as np
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"pick buffer not found at {p.resolve()}. "
                f"Run: python -m src.collect_buffer --policy-module src.policy_pick --checkpoint <pick_ckpt> --out {p}"
            )
        arr = np.load(p)
        required = ("root_z", "quat_zy", "lin_vel_by", "ang_vel_by", "joint_pos", "joint_vel",
                    "box_pos_by", "box_quat_by", "box_lin_vel_by", "box_ang_vel_by")
        missing = [k for k in required if k not in arr.files]
        if missing:
            raise ValueError(
                f"pick buffer at {p} missing fields {missing}. "
                f"Re-generate with policy_pick's SNAPSHOT_ENTITIES = ['robot', 'box']."
            )
        _PICK_BUFFER_CACHE = {k: torch.from_numpy(arr[k]).float().to(device) for k in arr.files}
        _PICK_BUFFER_CACHE["_path"] = path  # type: ignore[assignment]
    return _PICK_BUFFER_CACHE


def reset_from_pick_buffer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    buffer_path: str,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
) -> None:
    """Warm-start robot + held box by sampling from the pick terminal-state buffer.
    Args: env: RL env; env_ids: envs to reset (None = all); buffer_path: pick buffer file; xy_range/yaw_range: jitter ranges applied jointly to robot+box; asset_cfg: robot entity selector; box_cfg: box entity selector.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import (
        quat_apply, quat_from_euler_xyz, quat_mul, sample_uniform,
    )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    buf = _load_pick_buffer(buffer_path, env.device)
    total = buf["root_z"].shape[0]
    idx = torch.randint(0, total, (n,), device=env.device)

    root_z         = buf["root_z"][idx]
    quat_zy        = buf["quat_zy"][idx]
    lin_vel_by     = buf["lin_vel_by"][idx]
    ang_vel_by     = buf["ang_vel_by"][idx]
    joint_pos      = buf["joint_pos"][idx]
    joint_vel      = buf["joint_vel"][idx]
    box_pos_by     = buf["box_pos_by"][idx]
    box_quat_by    = buf["box_quat_by"][idx]
    box_lin_vel_by = buf["box_lin_vel_by"][idx]
    box_ang_vel_by = buf["box_ang_vel_by"][idx]

    xy_jitter  = sample_uniform(xy_range[0],  xy_range[1],  (n, 2), device=env.device)
    yaw_jitter = sample_uniform(yaw_range[0], yaw_range[1], (n,),    device=env.device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)

    robot: Entity = env.scene[asset_cfg.name]
    default_root_state = robot.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    root_xy = default_root_state[:, 0:2] + xy_jitter + origins[:, 0:2]
    root_z_w = root_z + origins[:, 2:3]
    root_pos_w = torch.cat([root_xy, root_z_w], dim=-1)
    root_quat_w = quat_mul(q_yaw, quat_zy)

    robot.write_root_link_pose_to_sim(
        torch.cat([root_pos_w, root_quat_w], dim=-1), env_ids=env_ids,
    )
    robot.write_root_link_velocity_to_sim(
        torch.cat([quat_apply(q_yaw, lin_vel_by), quat_apply(q_yaw, ang_vel_by)], dim=-1),
        env_ids=env_ids,
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    robot.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids,
    )

    # Box snapshots live in robot body-yaw frame; re-anchor with sampled q_yaw so robot+box stay physically consistent.
    box_pos_w = root_pos_w + quat_apply(q_yaw, box_pos_by)
    box_quat_w = quat_mul(q_yaw, box_quat_by)
    box: Entity = env.scene[box_cfg.name]
    box.write_root_link_pose_to_sim(
        torch.cat([box_pos_w, box_quat_w], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.cat([quat_apply(q_yaw, box_lin_vel_by), quat_apply(q_yaw, box_ang_vel_by)], dim=-1),
        env_ids=env_ids,
    )


TASK_ID = "Mjlab-G1-Carry"

# Contract for src.collect_buffer (carry produces carry_terminal_states.npz for drop).
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True

_spec_model = get_spec().compile()
G1_JOINTS = tuple(
    mujoco.mj_id2name(_spec_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(_spec_model.njnt)
    if _spec_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
)
G1_HOME_ROOT_POS = KNEES_BENT_KEYFRAME.pos
G1_HOME_ROOT_ROT = KNEES_BENT_KEYFRAME.rot
G1_HOME_JOINT_POS = {
    j: next((v for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items() if re.fullmatch(pat, j)), 0.0)
    for j in G1_JOINTS
}

# Carry geometry: box target height, spawn offsets, drop threshold.
CHEST_Z            = 0.93
BOX_HOLD_X_OFFSET  = 0.32
BOX_HOLD_Z_ABOVE_ROOT = 0.17
BOX_HOLD_Z         = CHEST_Z
BOX_DROP_Z         = CHEST_Z - 0.40

# Legacy hugging-arm pose: main carry warm-starts from pick_buffer; kept because policy_carry_soft (ball-lattice box) still needs a procedural reset (pick buffer's rigid box doesn't match the lattice scene).
HOLDING_ARM_POSE = {
    "left_shoulder_pitch_joint":  -0.7,
    "left_shoulder_roll_joint":    0.20,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.6,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -0.7,
    "right_shoulder_roll_joint":  -0.20,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.6,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}
HOLDING_ARM_JOINTS = tuple(HOLDING_ARM_POSE.keys())


def reset_arms_to_holding(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Snap the 14 arm joints to HOLDING_ARM_POSE (legacy, used only by policy_carry_soft).
    Args: env: RL env; env_ids: envs to reset (None = all); asset_cfg: robot entity selecting the 14 arm joints.
    Returns: none.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids_t = torch.tensor(joint_ids, device=env.device)
    else:
        joint_ids_t = joint_ids
    pose = torch.tensor(
        [HOLDING_ARM_POSE[j] for j in asset_cfg.joint_names],
        device=env.device,
    ).unsqueeze(0).expand(n, -1).clone()
    vel = torch.zeros_like(pose)
    asset.write_joint_state_to_sim(pose, vel, joint_ids=joint_ids_t, env_ids=env_ids)


# Mirrored joint pairs (left, right, sign): +1 = same sign in symmetric pose, -1 = opposite (roll axes).
_SYMMETRIC_PAIRS = (
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", +1),
    ("left_shoulder_roll_joint",  "right_shoulder_roll_joint",  -1),
    ("left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",   +1),
    ("left_elbow_joint",          "right_elbow_joint",          +1),
    ("left_wrist_roll_joint",     "right_wrist_roll_joint",     -1),
    ("left_wrist_pitch_joint",    "right_wrist_pitch_joint",    +1),
    ("left_wrist_yaw_joint",      "right_wrist_yaw_joint",      +1),
    ("left_hip_pitch_joint",      "right_hip_pitch_joint",      +1),
    ("left_hip_roll_joint",       "right_hip_roll_joint",       -1),
    ("left_hip_yaw_joint",        "right_hip_yaw_joint",        +1),
    ("left_knee_joint",           "right_knee_joint",           +1),
    ("left_ankle_pitch_joint",    "right_ankle_pitch_joint",    +1),
    ("left_ankle_roll_joint",     "right_ankle_roll_joint",     -1),
)
_LEFT_SYM_JOINTS  = tuple(p[0] for p in _SYMMETRIC_PAIRS)
_RIGHT_SYM_JOINTS = tuple(p[1] for p in _SYMMETRIC_PAIRS)
_SYM_SIGNS        = tuple(float(p[2]) for p in _SYMMETRIC_PAIRS)

def hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg, target_z: float = CHEST_Z) -> torch.Tensor:
    """Reward symmetric hand grip on the box side faces AND box at target chest height.
    Args: env: RL env; left_hand_cfg/right_hand_cfg: hand body selectors; box_cfg: box entity; target_z: desired box z (absorbs former box_at_chest term).
    Returns: (num_envs,) reward in (0, 1].
    """
    left  = _entity_pos(env, left_hand_cfg)
    right = _entity_pos(env, right_hand_cfg)
    box   = _entity_pos(env, box_cfg)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    qw, qx, qy, qz = box_quat.unbind(dim=-1)
    box_yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = torch.cos(box_yaw), torch.sin(box_yaw)
    # Box +y outward normal in world xy is (-sin_yaw, cos_yaw).
    left_offset_xy  = torch.stack([-0.13 * sy,  0.13 * cy], dim=-1)
    right_offset_xy = -left_offset_xy
    desired_left  = box.clone()
    desired_left[:, :2]  += left_offset_xy
    desired_left[:, 2]   += 0.02
    desired_right = box.clone()
    desired_right[:, :2] += right_offset_xy
    desired_right[:, 2]  += 0.02
    error = (
        torch.linalg.norm(left  - desired_left,  dim=-1)
        + torch.linalg.norm(right - desired_right, dim=-1)
        + torch.abs(box[:, 2] - target_z)
    )
    return torch.exp(-8.0 * error)


def box_orientation_reward(env, box_cfg) -> torch.Tensor:
    """Fused orientation reward: box +z aligned with world +z AND box +x aligned with robot heading.
    Args: env: RL env; box_cfg: box entity.
    Returns: (num_envs,) reward in [0, 2] (sum of two clamped cosines).
    """
    entity: Entity = env.scene[box_cfg.name]
    quat = entity.data.root_link_quat_w
    qw, qx, qy, qz = quat.unbind(dim=-1)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    front_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    front_y = 2.0 * (qx * qy + qw * qz)
    robot_yaw = env.scene["robot"].data.heading_w
    cos_diff = front_x * torch.cos(robot_yaw) + front_y * torch.sin(robot_yaw)
    return torch.clamp(up_z, 0.0, 1.0) + torch.clamp(cos_diff, 0.0, 1.0)


def upright_torso_reward(env, trunk_cfg) -> torch.Tensor:
    """Reward torso local +z aligned with world +z.
    Args: env: RL env; trunk_cfg: torso body selector.
    Returns: (num_envs,) reward clamped to [0, 1].
    """
    entity: Entity = env.scene[trunk_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(up_z, 0.0, 1.0)


def symmetric_pose_penalty(
    env,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    feet_moving_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize left/right asymmetry on mirrored joints while the robot is moving.
    Args: env: RL env; left_cfg/right_cfg: mirrored joint selectors; feet_moving_threshold: base xy speed gate in m/s.
    Returns: (num_envs,) squared-error penalty (0 when standing still).
    """
    robot = env.scene[left_cfg.name]
    if not hasattr(env, "_sym_signs_tensor"):
        env._sym_signs_tensor = torch.tensor(
            _SYM_SIGNS, device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype,
        )
    qL = robot.data.joint_pos[:, left_cfg.joint_ids]
    qR = robot.data.joint_pos[:, right_cfg.joint_ids]
    diff = qL - env._sym_signs_tensor * qR
    err = (diff * diff).sum(dim=-1)
    v = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    return err * (v > feet_moving_threshold).float()


def single_stance_reward(env, sensor_name: str, command_name: str | None = None,
                         command_threshold: float = 0.05) -> torch.Tensor:
    """Reward exactly one foot in ground contact, gated by twist magnitude.
    Args: env: RL env; sensor_name: contact-sensor name; command_name: twist command key (None keeps the reward always active); command_threshold: minimum |v_xy| + |wz| to activate.
    Returns: (num_envs,) 0/1 reward zeroed when the command is near zero (so the policy is not paid to step in place while idle).
    """
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    reward = (in_contact.sum(dim=1) == 1).float()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            active = ((linear_norm + angular_norm) > command_threshold).float()
            reward = reward * active
    return reward


def action_rate_l2(env) -> torch.Tensor:
    """Sum of squared action deltas between consecutive steps.
    Args: env: RL env.
    Returns: (num_envs,) non-negative smoothness penalty.
    """
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def carry_failed(
    env,
    box_cfg,
    trunk_cfg,
    limit_angle: float = math.radians(65.0),
    box_min_z: float = BOX_DROP_Z,
) -> torch.Tensor:
    """Termination flag when the torso tips over or the box drops.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: torso body; limit_angle: max torso tilt in radians; box_min_z: drop height threshold.
    Returns: (num_envs,) boolean termination mask.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    proj_grav = entity.data.projected_gravity_b
    up_alignment = -proj_grav[:, 2]
    tilted = up_alignment < math.cos(limit_angle)
    box_z = _entity_pos(env, box_cfg)[:, 2]
    box_dropped = box_z < box_min_z
    return tilted | box_dropped


def build_g1_carry_mdp():
    """Assemble the manager-cfg dict (scene, obs, rewards, events) for the carry task.
    Args: none.
    Returns: dict of manager configuration blocks consumed by g1_carry_env_cfg.
    """
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=G1_HOME_ROOT_POS,
        rot=G1_HOME_ROOT_ROT,
        joint_pos=G1_HOME_JOINT_POS,
    )
    box_cfg = EntityCfg(
        spec_fn=_get_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(BOX_HOLD_X_OFFSET, 0.0, BOX_HOLD_Z),
            joint_pos={},
        ),
    )

    g1_asset      = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    left_sym      = SceneEntityCfg("robot", joint_names=_LEFT_SYM_JOINTS,  preserve_order=True)
    right_sym     = SceneEntityCfg("robot", joint_names=_RIGHT_SYM_JOINTS, preserve_order=True)
    wrist_acts    = SceneEntityCfg("robot", actuator_names=(r".*wrist.*",), preserve_order=True)
    trunk       = SceneEntityCfg("robot", body_names=("torso_link",))
    left_hand   = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand  = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_shank  = SceneEntityCfg("robot", body_names=("left_knee_link",))
    right_shank = SceneEntityCfg("robot", body_names=("right_knee_link",))
    left_hip    = SceneEntityCfg("robot", body_names=("left_hip_pitch_link",))
    right_hip   = SceneEntityCfg("robot", body_names=("right_hip_pitch_link",))
    feet_sites  = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
    box         = SceneEntityCfg("box")

    # Sensors: identical to walk (foot-ground contact, self-collision, foot height scan).
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot")
                    for s in ("left_foot", "right_foot")),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )

    # Observations: walk-style base state + proprioception + twist command + box.
    actor_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_lin_vel,      noise=Unoise(n_min=-0.2, n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=base_ang_vel,      noise=Unoise(n_min=-0.1, n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.5, n_max=0.5)),
        "actions":   ObservationTermCfg(func=last_action),
        "command":       ObservationTermCfg(func=vel_mdp.generated_commands, params={"command_name": "twist"}),
        "left_hand_to_box":  ObservationTermCfg(func=body_vector, params={"source_cfg": left_hand, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "right_hand_to_box": ObservationTermCfg(func=body_vector, params={"source_cfg": right_hand, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "trunk_to_box":      ObservationTermCfg(func=body_vector, params={"source_cfg": trunk, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "wrist_torque":      ObservationTermCfg(func=wrist_actuator_force_obs, params={"asset_cfg": wrist_acts}, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "box_pos":         ObservationTermCfg(func=box_position_obs,    params={"box_cfg": box}, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "box_orientation": ObservationTermCfg(func=box_orientation_obs, params={"box_cfg": box}),
        "box_size":        ObservationTermCfg(func=box_size_obs),
    }
    observations = {
        "actor":  ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    # Actions: same 29-dim joint targets as walk (warm-start transfers 1:1).
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=G1_JOINTS,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        )
    }

    # Twist command: matches policy_walk.py so warm-start transfers 1:1 and no distribution shift on velocity.
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.15,
            rel_forward_envs=0.05,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.6, 0.6),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    rewards = {
        "track_linear_velocity":  RewardTermCfg(func=vel_mdp.track_linear_velocity,  weight=2.5,
                                                params={"command_name": "twist", "std": math.sqrt(0.25)}),
        "track_angular_velocity": RewardTermCfg(func=vel_mdp.track_angular_velocity, weight=10.0,
                                                params={"command_name": "twist", "std": math.sqrt(0.10)}),
        "wz_sign_penalty": RewardTermCfg(func=wz_sign_penalty, weight=2.0,
                                         params={"command_name": "twist"}),
        "hold_box":       RewardTermCfg(func=hold_box_reward,       weight=4.5,
                                        params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                                                "box_cfg": box, "target_z": CHEST_Z}),
        "box_orientation": RewardTermCfg(func=box_orientation_reward, weight=1.5, params={"box_cfg": box}),
        "upright":        RewardTermCfg(func=upright_torso_reward,  weight=1.0, params={"trunk_cfg": trunk}),
        # Small weight so PPO doesn't learn "don't move" as a zero-penalty strategy.
        "symmetric_pose": RewardTermCfg(
            func=symmetric_pose_penalty, weight=-0.3,
            params={"left_cfg": left_sym, "right_cfg": right_sym, "feet_moving_threshold": 0.1},
        ),
        "single_stance":  RewardTermCfg(func=single_stance_reward,  weight=0.8,
                                        params={"sensor_name": "feet_ground_contact",
                                                "command_name": "twist", "command_threshold": 0.05}),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2,        weight=-0.05),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": g1_asset}),
        "body_ang_vel": RewardTermCfg(
            func=vel_mdp.body_angular_velocity_penalty,
            weight=-0.05, params={"asset_cfg": trunk},
        ),
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        "non_hand_box": RewardTermCfg(
            func=non_hand_box_contact_penalty, weight=-2.0,
            params={"trunk_cfg": trunk, "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
                    "left_hip_cfg": left_hip, "right_hip_cfg": right_hip, "box_cfg": box},
        ),
    }

    # Events (reset order matters): default scene, then place robot+box together, then arms to hold pose, then DR.
    events = {
        "reset_scene": EventTermCfg(
            func=reset_scene_to_default,
            mode="reset",
            params={},
        ),
        "reset_from_pick_buffer": EventTermCfg(
            func=reset_from_pick_buffer,
            mode="reset",
            params={
                "buffer_path": PICK_BUFFER_PATH,
                "xy_range":  (-0.05, 0.05),
                "yaw_range": (-math.pi, math.pi),
                "asset_cfg": g1_asset,
                "box_cfg":   box,
            },
        ),
        "push_robot": EventTermCfg(
            func=vel_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                "roll":  (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
            }},
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
        "box_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
                "operation": "abs",
                "ranges": (BOX_MASS_MIN, BOX_MASS_MAX),
            },
        ),
    }

    terminations = {
        "time_out":  TerminationTermCfg(func=time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=carry_failed,
            params={"box_cfg": box, "trunk_cfg": trunk,
                    "limit_angle": math.radians(65.0), "box_min_z": BOX_DROP_Z},
        ),
    }

    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=vel_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0,           "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.3, 0.3)},
                    {"step":  2_000_000,  "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.45, 0.45)},
                    {"step":  6_000_000,  "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.6, 0.6)},
                ],
            },
        ),
    }

    return {
        "robot_cfg":  robot_cfg,
        "box_cfg":    box_cfg,
        "sensors":    (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg),
        "observations": observations,
        "actions":      actions,
        "commands":     commands,
        "rewards":      rewards,
        "events":       events,
        "terminations": terminations,
        "curriculum":   curriculum,
    }


def g1_carry_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the ManagerBasedRlEnvCfg for the carry task.
    Args: play: when True disables observation noise and push events.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_carry_mdp()
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": parts["robot_cfg"], "box": parts["box_cfg"]},
            sensors=parts["sensors"],
            num_envs=1,
            extent=2.0,
        ),
        observations=parts["observations"],
        actions=parts["actions"],
        commands=parts["commands"],
        events=parts["events"],
        rewards=parts["rewards"],
        terminations=parts["terminations"],
        curriculum=parts["curriculum"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.5,
            elevation=-10.0,
            azimuth=125.0,
        ),
        sim=SimulationCfg(
            nconmax=120,
            njmax=800,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=5,
        episode_length_s=8.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_carry_ppo_runner_cfg():
    """Build the RSL-RL PPO runner configuration for the carry task.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg with actor/critic/algorithm settings.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="g1_carry", save_interval=50, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the carry task with mjlab's task registry (idempotent).
    Args: none.
    Returns: task id string.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_carry_env_cfg(),
        play_env_cfg=g1_carry_env_cfg(play=True),
        rl_cfg=g1_carry_ppo_runner_cfg(),
    )
    return TASK_ID
