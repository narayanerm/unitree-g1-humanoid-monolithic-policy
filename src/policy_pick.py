##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""G1 pick task definition shared by training and analysis notebooks.

The G1 humanoid stands in front of a table, reaches for a box with both hands,
grasps it, and lifts it to a target height. Exposes TASK_ID plus the mjlab env
and PPO runner factories consumed by the training entrypoint.
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
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
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
from mjlab.tasks.velocity.mdp import base_ang_vel, base_lin_vel, projected_gravity
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


TASK_ID = "Mjlab-G1-Pick"

# Contract for src.collect_buffer (pick produces pick_terminal_states.npz for carry).
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True

WALK_BUFFER_PATH = "outputs/walk_terminal_states.npz"

spec_model = get_spec().compile()
G1_JOINTS = tuple(
    mujoco.mj_id2name(spec_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(spec_model.njnt)
    if spec_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
)
G1_HOME_ROOT_POS = KNEES_BENT_KEYFRAME.pos
G1_HOME_ROOT_ROT = KNEES_BENT_KEYFRAME.rot
G1_HOME_JOINT_POS = {
    j: next((v for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items() if re.fullmatch(pat, j)), 0.0)
    for j in G1_JOINTS
}
G1_FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


# Terminal arm pose for the chest-hold phase; used inside CARRY_READY_POSE so pick's terminal snapshot matches the posture carry expects at reset (shoulders less pitched-down, elbows flexed, rolls opened outward for chest-hug).
HOLDING_ARM_POSE = {
    "left_shoulder_pitch_joint":  -0.70,
    "left_shoulder_roll_joint":    0.20,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.60,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -0.70,
    "right_shoulder_roll_joint":  -0.20,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.60,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}


# Full-body squat-and-grip pose used as the curriculum starting state (body_bias=1.0). Robot spawns deeply squatting IN FRONT OF the table (SQUAT_GRIP_ROOT_POS pushes pelvis to x~0.20 world) with waist pitched forward and both wrists positioned at the box side-face anchors (x~0.51, y~+/-0.13, z~0.605) so the policy can learn the grip/lift dynamics from an ideal start before pick_curriculum ramps body_bias to 0.0 (walk-buffer sample at world origin). Values are geometric approximations: verify in viewer and tune the leg-flex / waist-pitch / shoulder-pitch / elbow quartet if wrists don't sit exactly at the box on reset.
SQUAT_GRIP_POSE = {
    # Legs: moderate squat (not deep) so knees stay clear of the table front edge (x=0.26) while pelvis is at x=0. hip+knee+ankle sum to zero so feet land flat.
    "left_hip_pitch_joint":   -0.80,
    "left_hip_roll_joint":     0.12,
    "left_hip_yaw_joint":      0.0,
    "left_knee_joint":         1.40,
    "left_ankle_pitch_joint": -0.60,
    "left_ankle_roll_joint":   0.0,
    "right_hip_pitch_joint":  -0.80,
    "right_hip_roll_joint":   -0.12,
    "right_hip_yaw_joint":     0.0,
    "right_knee_joint":        1.40,
    "right_ankle_pitch_joint":-0.60,
    "right_ankle_roll_joint":  0.0,
    # Waist: heavy forward pitch (~52 deg) brings the shoulders forward so arms can reach the box side faces without moving the pelvis into the table footprint.
    "waist_yaw_joint":         0.0,
    "waist_roll_joint":        0.0,
    "waist_pitch_joint":       0.90,
    # Arms: strong shoulder_pitch forward-down, shoulder_roll opened outward to y~+/-0.13, elbow moderately bent (~63 deg) so the pose is visibly bent. Empirical FK sweep places the wrists at (0.502, +/-0.107, 0.627): 3 cm from the target box side-face anchor at (0.51, +/-0.13, 0.605).
    "left_shoulder_pitch_joint":  -1.60,
    "left_shoulder_roll_joint":    0.25,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            1.10,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -1.60,
    "right_shoulder_roll_joint":  -0.25,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           1.10,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}
assert set(SQUAT_GRIP_POSE.keys()) == set(G1_JOINTS), (
    f"SQUAT_GRIP_POSE must cover all 29 G1 joints; missing "
    f"{set(G1_JOINTS) - set(SQUAT_GRIP_POSE.keys())}, extra "
    f"{set(SQUAT_GRIP_POSE.keys()) - set(G1_JOINTS)}."
)
# Pelvis at world (0.0, 0, 0.65): x=0 keeps the pelvis behind the table front edge (0.26) so the moderately-flexed knees (at x~0.24) stay just clear of the table; z=0.65 puts the feet flat on the ground with the current hip/knee/ankle flexion. reset_from_walk_buffer lerps root_xy between the walk default (0,0) and this squat position by body_bias.
SQUAT_GRIP_ROOT_POS = (0.0, 0.0, 0.65)
SQUAT_GRIP_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity: waist_pitch carries the trunk lean via joints


# Full-body carry-ready pose: KNEES_BENT_KEYFRAME legs (mjlab canonical G1 home used by walk/carry) + waist upright + HOLDING_ARM_POSE arms wrapped around the box. Used as the target for carry_ready_pose_reward so a walk -> pick -> carry state-machine handoff sees consistent postures at every transition.
CARRY_READY_POSE = {
    # Legs (KNEES_BENT_KEYFRAME): hips pitched back slightly, knees bent, ankles compensating.
    "left_hip_pitch_joint":   -0.312,
    "left_hip_roll_joint":     0.0,
    "left_hip_yaw_joint":      0.0,
    "left_knee_joint":         0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint":   0.0,
    "right_hip_pitch_joint":  -0.312,
    "right_hip_roll_joint":    0.0,
    "right_hip_yaw_joint":     0.0,
    "right_knee_joint":        0.669,
    "right_ankle_pitch_joint":-0.363,
    "right_ankle_roll_joint":  0.0,
    # Waist: upright trunk (no yaw / roll / pitch).
    "waist_yaw_joint":         0.0,
    "waist_roll_joint":        0.0,
    "waist_pitch_joint":       0.0,
    # Arms: reuse HOLDING_ARM_POSE (chest-hug: wrists sit against the box side faces).
    **HOLDING_ARM_POSE,
}
assert set(CARRY_READY_POSE.keys()) == set(G1_JOINTS), (
    f"CARRY_READY_POSE must cover all 29 G1 joints; missing "
    f"{set(G1_JOINTS) - set(CARRY_READY_POSE.keys())}, extra "
    f"{set(CARRY_READY_POSE.keys()) - set(G1_JOINTS)}."
)
CARRY_READY_TARGET = tuple(CARRY_READY_POSE[j] for j in G1_JOINTS)  # aligned with g1_asset.joint_ids


TABLE_X          = 0.60
TABLE_HEIGHT     = 0.44
TABLE_TOP_HALF   = 0.04
TABLE_TOP_Z      = TABLE_HEIGHT + TABLE_TOP_HALF   # top surface where the box sits
BOX_HALF         = 0.105
BOX_START_X      = TABLE_X - 0.09
BOX_START_Z      = TABLE_TOP_Z + BOX_HALF
BOX_MASS_MIN     = 0.30
BOX_MASS_MAX     = 3.00
LIFT_TARGET_Z    = TABLE_TOP_Z + 0.34   # for notebook prints and analysis

# Centre of the lift-target z range (matches carry's chest-hold height).
CHEST_Z          = 0.93


# 20-D per-reset command: grasp offsets, object spawn pose, lift target (yaw-only quats).
GRASP_NOMINAL_Y   = 0.13   # left-hand nominal y in box frame (right mirrors)
GRASP_NOMINAL_Z   = 0.02
GRASP_JITTER      = 0.03

# Box always spawns resting on the table with tight +/-5 cm xy and +/-20 deg yaw jitter; the pick curriculum only modulates the robot's starting pose (SQUAT_GRIP -> walk buffer), never the box spawn.
OBJECT_XY_JITTER_X = 0.05
OBJECT_XY_JITTER_Y = 0.05
OBJECT_YAW_RANGE   = math.radians(20.0)

LIFT_X_CENTER   = 0.30
LIFT_X_HALF     = 0.10
LIFT_Y_HALF     = 0.20
LIFT_Z_HALF     = 0.15
LIFT_YAW_RANGE  = 0.30

_CMD_KEY = "pick_cmd_"    # env.extras namespace prefix

CMD_DIMS = {
    "grasp_offset_left":   3,
    "grasp_offset_right":  3,
    "object_target_pos":   3,
    "object_target_quat":  4,
    "lift_target_pos":     3,
    "lift_target_quat":    4,
}


# Arms and wrists only; leg symmetry excluded to allow weight-shifting toward the box.
_SYMMETRIC_PAIRS = (
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", +1),
    ("left_shoulder_roll_joint",  "right_shoulder_roll_joint",  -1),
    ("left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",   +1),
    ("left_elbow_joint",          "right_elbow_joint",          +1),
    ("left_wrist_roll_joint",     "right_wrist_roll_joint",     -1),
    ("left_wrist_pitch_joint",    "right_wrist_pitch_joint",    +1),
    ("left_wrist_yaw_joint",      "right_wrist_yaw_joint",      +1),
)
_LEFT_SYM_JOINTS  = tuple(p[0] for p in _SYMMETRIC_PAIRS)
_RIGHT_SYM_JOINTS = tuple(p[1] for p in _SYMMETRIC_PAIRS)
_SYM_SIGNS        = tuple(float(p[2]) for p in _SYMMETRIC_PAIRS)


def _get_table_spec() -> mujoco.MjSpec:
    """Build the MjSpec for the table (top + four legs).
    Args: none.
    Returns: MjSpec containing the table body.
    """
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table", pos=(TABLE_X, 0.0, TABLE_HEIGHT))
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.34, 0.42, TABLE_TOP_HALF),
        rgba=(0.47, 0.40, 0.32, 1.0),
    )
    leg_len_z    = TABLE_HEIGHT - TABLE_TOP_HALF
    leg_half_z   = leg_len_z / 2.0
    leg_center_z = leg_half_z - TABLE_HEIGHT
    for name, pos_xy in {
        "table_leg_fl": ( 0.29,  0.37),
        "table_leg_fr": ( 0.29, -0.37),
        "table_leg_bl": (-0.29,  0.37),
        "table_leg_br": (-0.29, -0.37),
    }.items():
        body.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.025, 0.025, leg_half_z),
            pos=(*pos_xy, leg_center_z),
            rgba=(0.47, 0.40, 0.32, 1.0),
        )
    return spec


def _get_box_spec() -> mujoco.MjSpec:
    """Build the MjSpec for the graspable box with a free joint.
    Args: none.
    Returns: MjSpec containing the box body.
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
    """Return the world-frame position of the entity or one of its bodies.
    Args: env: the RL env; asset_cfg: scene entity config, optionally with body_ids.
    Returns: (num_envs, 3) tensor of positions.
    """
    entity: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_names is not None or isinstance(asset_cfg.body_ids, list):
        return entity.data.body_link_pos_w[:, asset_cfg.body_ids].squeeze(1)
    return entity.data.root_link_pos_w


def body_vector(env, source_cfg, target_cfg):
    """Compute the world-frame vector from source body to target body.
    Args: env: the RL env; source_cfg: origin entity; target_cfg: endpoint entity.
    Returns: (num_envs, 3) tensor of displacement.
    """
    return _entity_pos(env, target_cfg) - _entity_pos(env, source_cfg)


# Box signals sanitized so MuJoCo divergences never trip rsl_rl.check_nan.
def box_position_obs(env, box_cfg):
    """Box position relative to the robot base, sanitized and clamped.
    Args: env: the RL env; box_cfg: scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    box_pos = env.scene[box_cfg.name].data.root_link_pos_w
    robot_pos = env.scene["robot"].data.root_link_pos_w
    rel = box_pos - robot_pos
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0)


def box_lin_vel_obs(env, box_cfg):
    """Box linear velocity in world frame, sanitized and clamped.
    Args: env: the RL env; box_cfg: scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    v = env.scene[box_cfg.name].data.root_link_lin_vel_w
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)


def box_ang_vel_obs(env, box_cfg):
    """Box angular velocity in world frame, sanitized and clamped.
    Args: env: the RL env; box_cfg: scene entity config for the box.
    Returns: (num_envs, 3) tensor.
    """
    w = env.scene[box_cfg.name].data.root_link_ang_vel_w
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)


def box_orientation_obs(env, box_cfg):
    """Box orientation quaternion in world frame, sanitized and clamped.
    Args: env: the RL env; box_cfg: scene entity config for the box.
    Returns: (num_envs, 4) tensor.
    """
    q = env.scene[box_cfg.name].data.root_link_quat_w
    return torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def box_size_obs(env):
    """Broadcast the fixed box size to every env for observation input.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor of box dimensions.
    """
    root_pos = env.scene["robot"].data.root_link_pos_w
    size = torch.tensor([2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF], device=root_pos.device)
    return size.expand(root_pos.shape[0], -1)


def wrist_actuator_force_obs(env, asset_cfg: SceneEntityCfg):
    """Actuator force at wrist joints; spikes when the hand is loaded during grasp.
    Args: env: the RL env; asset_cfg: entity config selecting wrist actuators.
    Returns: (num_envs, n_actuators) tensor of forces.
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


# Pick command buffers live on env.extras and are allocated on first reset/obs.
def _ensure_cmd_buffers(env: ManagerBasedRlEnv) -> None:
    """Lazily allocate per-env command tensors on env.extras.
    Args: env: the RL env.
    Returns: none.
    """
    for name, dim in CMD_DIMS.items():
        key = _CMD_KEY + name
        buf = env.extras.get(key)
        if buf is None or buf.shape[0] != env.num_envs:
            env.extras[key] = torch.zeros(env.num_envs, dim, device=env.device)
    # Identity quat default for yaw-only slots.
    for qname in ("object_target_quat", "lift_target_quat"):
        env.extras[_CMD_KEY + qname][:, 0] = 1.0


def reset_pick_command_and_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Sample the 20-D command and write the box spawn (always resting on the table with fixed jitter ranges).
    Args: env: the RL env; env_ids: envs to reset (all if None).
    Returns: none.
    """
    from mjlab.utils.lab_api.math import sample_uniform, quat_from_euler_xyz

    _ensure_cmd_buffers(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    origins = env.scene.env_origins[env_ids]
    zero = torch.zeros(n, device=device)

    # Box always resting on the table with fixed +/-5 cm xy / +/-20 deg yaw jitter; no curriculum on the box itself so the task the policy solves stays identical across training.
    ox = sample_uniform(-OBJECT_XY_JITTER_X, OBJECT_XY_JITTER_X, (n,), device=device)
    oy = sample_uniform(-OBJECT_XY_JITTER_Y, OBJECT_XY_JITTER_Y, (n,), device=device)
    o_yaw = sample_uniform(-OBJECT_YAW_RANGE, OBJECT_YAW_RANGE, (n,), device=device)
    obj_pos_w = torch.stack([
        origins[:, 0] + BOX_START_X + ox,
        origins[:, 1] + oy,
        origins[:, 2] + BOX_START_Z,
    ], dim=-1)
    obj_quat_w = quat_from_euler_xyz(zero, zero, o_yaw)

    box: Entity = env.scene["box"]
    box.write_root_link_pose_to_sim(
        torch.cat([obj_pos_w, obj_quat_w], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )

    # object_target_* echoes the spawn pose exactly.
    env.extras[_CMD_KEY + "object_target_pos"][env_ids]  = obj_pos_w
    env.extras[_CMD_KEY + "object_target_quat"][env_ids] = obj_quat_w

    # Grasp offsets in box frame; right mirrors left across sagittal plane.
    gdx = sample_uniform(-GRASP_JITTER, GRASP_JITTER, (n,), device=device)
    gdy = sample_uniform(-GRASP_JITTER, GRASP_JITTER, (n,), device=device)
    gdz = sample_uniform(-GRASP_JITTER, GRASP_JITTER, (n,), device=device)
    grasp_left = torch.stack([
        gdx,
        torch.full((n,), GRASP_NOMINAL_Y, device=device) + gdy,
        torch.full((n,), GRASP_NOMINAL_Z, device=device) + gdz,
    ], dim=-1)
    mirror = torch.tensor([1.0, -1.0, 1.0], device=device)
    grasp_right = grasp_left * mirror
    env.extras[_CMD_KEY + "grasp_offset_left"][env_ids]  = grasp_left
    env.extras[_CMD_KEY + "grasp_offset_right"][env_ids] = grasp_right

    # Lift target pose.
    lx = sample_uniform(-LIFT_X_HALF, LIFT_X_HALF, (n,), device=device)
    ly = sample_uniform(-LIFT_Y_HALF, LIFT_Y_HALF, (n,), device=device)
    lz = sample_uniform(-LIFT_Z_HALF, LIFT_Z_HALF, (n,), device=device)
    lift_pos_w = torch.stack([
        origins[:, 0] + LIFT_X_CENTER + lx,
        origins[:, 1] + ly,
        origins[:, 2] + CHEST_Z + lz,
    ], dim=-1)
    l_yaw = sample_uniform(-LIFT_YAW_RANGE, LIFT_YAW_RANGE, (n,), device=device)
    lift_quat_w = quat_from_euler_xyz(zero, zero, l_yaw)
    env.extras[_CMD_KEY + "lift_target_pos"][env_ids]  = lift_pos_w
    env.extras[_CMD_KEY + "lift_target_quat"][env_ids] = lift_quat_w

    # Reseed the per-episode max-box-z tracker used by box_fell_back termination to the current spawn z; without this, max persists across episodes and causes spurious early terminations.
    max_key = "pick_max_box_z"
    if max_key not in env.extras or env.extras[max_key].shape[0] != env.num_envs:
        env.extras[max_key] = torch.zeros(env.num_envs, device=env.device)
    env.extras[max_key][env_ids] = obj_pos_w[:, 2]


def _pos_in_base_frame(env: ManagerBasedRlEnv, world_pos: torch.Tensor) -> torch.Tensor:
    """Transform a world-frame position tensor to the robot base frame.
    Args: env: the RL env; world_pos: (num_envs, 3) tensor of world positions.
    Returns: (num_envs, 3) tensor in base frame.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    robot: Entity = env.scene["robot"]
    delta = world_pos - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, delta)


def cmd_grasp_offset_left_obs(env):
    """Observation for left-hand grasp offset in box frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "grasp_offset_left"]


def cmd_grasp_offset_right_obs(env):
    """Observation for right-hand grasp offset in box frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "grasp_offset_right"]


def cmd_object_target_pos_obs(env):
    """Object-target position expressed in robot base frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "object_target_pos"])


def cmd_object_target_quat_obs(env):
    """Object-target orientation quaternion in world frame.
    Args: env: the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "object_target_quat"]


def cmd_lift_target_pos_obs(env):
    """Lift-target position expressed in robot base frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "lift_target_pos"])


def cmd_lift_target_quat_obs(env):
    """Lift-target orientation quaternion in world frame.
    Args: env: the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "lift_target_quat"]


# Fixed side-face grasp target that forces wrist-pinching geometry: left wrist targets +y face, right wrist targets -y face; deterministic (no per-episode jitter) so the policy converges to the pinch pose.
def _desired_grasp_pos_w(env, box_cfg, side: str) -> torch.Tensor:
    """Return world-frame position of the wrist-pinch target on the box side face.
    Args: env: the RL env; box_cfg: box entity config; side: "left" or "right".
    Returns: (num_envs, 3) tensor of pinch points on the box side.
    """
    box_w    = _entity_pos(env, box_cfg)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    qw, qx, qy, qz = box_quat.unbind(dim=-1)
    box_yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = torch.cos(box_yaw), torch.sin(box_yaw)
    sign = 1.0 if side == "left" else -1.0
    dx = -sign * GRASP_NOMINAL_Y * sy
    dy =  sign * GRASP_NOMINAL_Y * cy
    dz = torch.full_like(box_w[:, 2], GRASP_NOMINAL_Z)
    return box_w + torch.stack([dx, dy, dz], dim=-1)


def reach_reward(env, hand_cfg, box_cfg, side: str, std: float = 0.20):
    """Gaussian shaping on hand distance to the commanded grasp point.
    Args: env: the RL env; hand_cfg: hand entity config; box_cfg: box entity config; side: "left" or "right"; std: kernel width.
    Returns: (num_envs,) tensor of reward values in (0, 1].
    """
    hand_w = _entity_pos(env, hand_cfg)
    d = torch.linalg.norm(_desired_grasp_pos_w(env, box_cfg, side) - hand_w, dim=-1)
    return torch.exp(-d * d / (2.0 * std * std))


def both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, threshold: float = 0.08):
    """+1 when both hands are within threshold of their commanded side grasps.
    Args: env: the RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity; threshold: hit distance.
    Returns: (num_envs,) tensor of 0/1 values.
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    return ((dl < threshold) & (dr < threshold)).float()


def palm_inward_reward(env, hand_cfg, palm_axis_local, target_axis_body):
    """Reward palm-normal alignment with a target axis in the base frame.
    Args: env: the RL env; hand_cfg: hand entity; palm_axis_local: palm normal in hand frame; target_axis_body: target axis in base frame.
    Returns: (num_envs,) tensor in [0, 1], max when aligned.
    """
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
    robot = env.scene["robot"]
    asset = env.scene[hand_cfg.name]
    hand_quat = asset.data.body_link_quat_w[:, hand_cfg.body_ids[0], :]
    root_quat = robot.data.root_link_quat_w
    palm_local = torch.tensor(palm_axis_local, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    palm_world = quat_apply(hand_quat, palm_local)
    palm_body  = quat_apply_inverse(root_quat, palm_world)
    tgt = torch.tensor(target_axis_body, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    cos_a = (palm_body * tgt).sum(dim=-1)
    return 0.5 * (1.0 + cos_a)


def feet_on_ground_reward(env, sensor_name: str):
    """Graded ground-contact reward: 0.5 for one foot, 1.0 for both.
    Args: env: the RL env; sensor_name: key of the feet-contact sensor in the scene.
    Returns: (num_envs,) tensor in [0, 1].
    """
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    return 0.5 * in_contact.sum(dim=1).clamp(0.0, 2.0)


def stability_gate(env, trunk_cfg, sensor_name):
    """Binary gate that returns 1 when torso is upright (<=25 deg tilt) AND both feet are in contact, used to zero task credit while falling so the policy solves balance first.
    Args: env: RL env; trunk_cfg: trunk entity; sensor_name: feet contact sensor key.
    Returns: (num_envs,) 0/1 tensor.
    """
    entity = env.scene[trunk_cfg.name]
    up_align = -entity.data.projected_gravity_b[:, 2]   # 1.0 upright, 0 sideways
    upright = up_align > 0.9
    sensor = env.scene[sensor_name]
    both_feet = (sensor.data.found > 0).sum(dim=1) >= 2
    return (upright & both_feet).float()


def stay_still_penalty(
    env,
    left_hand_cfg: SceneEntityCfg | None = None,
    right_hand_cfg: SceneEntityCfg | None = None,
    box_cfg: SceneEntityCfg | None = None,
    gate_dist: float = 0.20,
):
    """Penalise base linear velocity, gated on both hands being near the grasp.
    Args: env: the RL env; left_hand_cfg: left hand entity or None; right_hand_cfg: right hand entity or None; box_cfg: box entity or None; gate_dist: activation threshold in metres.
    Returns: (num_envs,) tensor of |v|^2 masked by the gate.
    """
    root_vel = env.scene["robot"].data.root_link_lin_vel_w
    v_sq = torch.sum(root_vel * root_vel, dim=-1)
    if left_hand_cfg is None or right_hand_cfg is None or box_cfg is None:
        return v_sq  # ungated fallback for backward compatibility

    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    gate = ((dl < gate_dist) & (dr < gate_dist)).float()
    return v_sq * gate


def base_ang_vel_penalty(env):
    """Squared L2 penalty on the base angular velocity.
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    root_ang_vel = env.scene["robot"].data.root_link_ang_vel_w
    return torch.sum(root_ang_vel * root_ang_vel, dim=-1)


def action_rate_l2(env):
    """Squared L2 penalty on action rate between consecutive steps.
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def action_2nd_order_l2(env):
    """Squared L2 penalty on action second-order difference (jerk); complements action_rate per the ETH ANYmal smoothness bundle.
    Args: env: the RL env.
    Returns: (num_envs,) tensor of |a_t − 2·a_{t−1} + a_{t−2}|² summed over joints.
    """
    a = env.action_manager.action
    a1 = env.action_manager.prev_action
    if not hasattr(env, "_prev_prev_action") or env._prev_prev_action.shape != a.shape:
        env._prev_prev_action = a1.clone()
    a2 = env._prev_prev_action
    diff = a - 2.0 * a1 + a2
    env._prev_prev_action = a1.clone()
    return torch.sum(diff * diff, dim=-1)


def dof_vel_l2(env):
    """Squared L2 penalty on joint velocities; discourages fast joints (ETH standard).
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    v = env.scene["robot"].data.joint_vel
    return torch.sum(v * v, dim=-1)


def dof_acc_l2(env):
    """Squared L2 penalty on joint accelerations; most effective smoothing regulariser in the ETH bundle.
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    a = env.scene["robot"].data.joint_acc
    return torch.sum(a * a, dim=-1)


def dof_torque_l2(env):
    """Squared L2 penalty on joint torques as an energy proxy (ETH standard).
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    tau = env.scene["robot"].data.qfrc_actuator
    return torch.sum(tau * tau, dim=-1)


def grip_force_reward(env, sensor_name: str = "hand_box_contact",
                      target_force: float = 5.0):
    """Reward positive normal force between robot and box, saturating at target_force.
    Args: env: RL env; sensor_name: ContactSensor name; target_force: N at which reward saturates.
    Returns: (num_envs,) tensor in [0, 1]. Ground-truth signal that grip is firm.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        force_sum = force_mag.sum(dim=-1)
    else:
        force_sum = sensor.data.found.sum(dim=-1).float()
    return torch.clamp(force_sum / target_force, 0.0, 1.0)


def hand_approach_velocity_penalty(env, left_hand_cfg, right_hand_cfg, box_cfg,
                                    gate_dist: float = 0.20):
    """Penalise hand linear velocity when either hand is within gate_dist of the box, forcing a gentle approach; uses body_link_lin_vel_w to avoid the reset-spike bug.
    Args: env: RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity; gate_dist: activation distance in metres.
    Returns: (num_envs,) tensor summing both hand speeds when the gate fires.
    """
    lh_pos = _entity_pos(env, left_hand_cfg)
    rh_pos = _entity_pos(env, right_hand_cfg)
    box_pos = _entity_pos(env, box_cfg)
    dl = torch.linalg.norm(lh_pos - box_pos, dim=-1)
    dr = torch.linalg.norm(rh_pos - box_pos, dim=-1)
    close_gate = (torch.minimum(dl, dr) < gate_dist).float()

    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    if not hasattr(env, "_hand_body_idx"):
        lh_name = left_hand_cfg.body_names[0] if isinstance(left_hand_cfg.body_names, (list, tuple)) else left_hand_cfg.body_names
        rh_name = right_hand_cfg.body_names[0] if isinstance(right_hand_cfg.body_names, (list, tuple)) else right_hand_cfg.body_names
        env._hand_body_idx = (body_names.index(lh_name), body_names.index(rh_name))
    lv = robot.data.body_link_lin_vel_w[:, env._hand_body_idx[0]]
    rv = robot.data.body_link_lin_vel_w[:, env._hand_body_idx[1]]
    speed = torch.linalg.norm(lv, dim=-1) + torch.linalg.norm(rv, dim=-1)
    return speed * close_gate


def palm_inward_bilateral_reward(env, left_hand_cfg, right_hand_cfg,
                                  left_palm_axis_local=(0.0, -1.0, 0.0),
                                  right_palm_axis_local=(0.0,  1.0, 0.0),
                                  left_target_axis_body=(0.0, -1.0, 0.0),
                                  right_target_axis_body=(0.0,  1.0, 0.0),
                                  w_left: float = 0.5, w_right: float = 0.5):
    """Fused palm-inward alignment for both hands. Weighted sum of per-hand rewards.
    Args: env: RL env; left/right_hand_cfg: hand entities; left/right_palm_axis_local: palm normal in hand frame; left/right_target_axis_body: target axis in base frame; w_left/w_right: per-side weights folded in.
    Returns: (num_envs,) tensor.
    """
    lr = palm_inward_reward(env, left_hand_cfg,  left_palm_axis_local,  left_target_axis_body)
    rr = palm_inward_reward(env, right_hand_cfg, right_palm_axis_local, right_target_axis_body)
    return w_left * lr + w_right * rr


def body_collision_bundle_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    left_hand_cfg, right_hand_cfg, box_cfg,
    non_hand_box_threshold: float = 0.22,
    table_margin_xy: float = 0.02,
    table_z_slack: float = 0.02,
    w_non_hand_box: float = -2.0,
    w_body_table: float = -2.0,
):
    """Fused body-vs-box proximity + body-vs-table volumetric penalty sharing one body-position read (box uses the 5 non-hand bodies, table uses all 7).
    Args: env: RL env; trunk_cfg: trunk; left_shank_cfg/right_shank_cfg: shank entities; left_hip_cfg/right_hip_cfg: hip entities; left_hand_cfg/right_hand_cfg: hand entities; box_cfg: box entity; non_hand_box_threshold: proximity threshold; table_margin_xy: footprint shrink; table_z_slack: vertical tolerance; w_non_hand_box/w_body_table: per-signal weights (negative).
    Returns: (num_envs,) tensor of the weighted sum.
    """
    device = env.scene["robot"].data.root_link_pos_w.device
    n_envs = env.scene["robot"].data.root_link_pos_w.shape[0]
    box_pos = _entity_pos(env, box_cfg)

    non_hand_positions = [_entity_pos(env, cfg) for cfg in
                          (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg)]
    hand_positions = [_entity_pos(env, cfg) for cfg in (left_hand_cfg, right_hand_cfg)]

    non_hand_box_pen = torch.zeros(n_envs, device=device)
    for pos in non_hand_positions:
        dist = torch.linalg.norm(pos - box_pos, dim=-1)
        non_hand_box_pen += torch.clamp(non_hand_box_threshold - dist, min=0.0) / non_hand_box_threshold

    t_xy = torch.tensor([TABLE_X, 0.0], device=device)
    t_half_xy = torch.tensor([0.34 - table_margin_xy, 0.42 - table_margin_xy], device=device)
    top_z = TABLE_TOP_Z + TABLE_TOP_HALF - table_z_slack
    body_table_pen = torch.zeros(n_envs, device=device)
    for pos in non_hand_positions + hand_positions:
        xy_into = t_half_xy - torch.abs(pos[:, :2] - t_xy)
        z_below = top_z - pos[:, 2]
        depth = torch.min(
            torch.stack([xy_into[:, 0], xy_into[:, 1], z_below], dim=-1),
            dim=-1,
        ).values
        body_table_pen += torch.clamp(depth, min=0.0)

    return w_non_hand_box * non_hand_box_pen + w_body_table * body_table_pen


def stability_bundle_reward(env, trunk_cfg, sensor_name, pelvis_cfg=None,
                             w_upright: float = 10.0,
                             w_upright_pelvis: float = 5.0,
                             w_feet: float = 5.0):
    """Fused torso-upright + optional pelvis-upright + feet-on-ground stability.
    Args: env: RL env; trunk_cfg: trunk entity; sensor_name: feet contact sensor; pelvis_cfg: optional pelvis entity (if given, adds a second upright term); w_upright/w_upright_pelvis/w_feet: per-signal weights folded in.
    Returns: (num_envs,) tensor.
    """
    total = (w_upright * upright_torso_reward(env, trunk_cfg)
           + w_feet    * feet_on_ground_reward(env, sensor_name))
    if pelvis_cfg is not None:
        total = total + w_upright_pelvis * upright_torso_reward(env, pelvis_cfg)
    return total


def box_at_chest_reward(env, box_cfg, target_z: float = CHEST_Z):
    """Gaussian reward on box z-height around chest height. Copied from policy_carry.
    Args: env: RL env; box_cfg: box entity; target_z: desired box height in metres.
    Returns: (num_envs,) reward in (0, 1].
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    diff = box_z - target_z
    return torch.exp(-(diff ** 2) / 0.05)


def box_upright_reward(env, box_cfg):
    """Reward alignment of box local +z with world +z. Copied from policy_carry.
    Args: env: RL env; box_cfg: box entity.
    Returns: (num_envs,) reward clamped to [0, 1].
    """
    entity = env.scene[box_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(up_z, 0.0, 1.0)


def box_front_aligned_reward(env, box_cfg):
    """Reward alignment of box local +x with robot heading. Copied from policy_carry.
    Args: env: RL env; box_cfg: box entity.
    Returns: (num_envs,) reward clamped to [0, 1].
    """
    entity = env.scene[box_cfg.name]
    quat = entity.data.root_link_quat_w
    qw, qx, qy, qz = quat.unbind(dim=-1)
    front_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    front_y = 2.0 * (qx * qy + qw * qz)
    robot_yaw = env.scene["robot"].data.heading_w
    rx = torch.cos(robot_yaw)
    ry = torch.sin(robot_yaw)
    cos_diff = front_x * rx + front_y * ry
    return torch.clamp(cos_diff, 0.0, 1.0)


def box_carry_pose_bundle_reward(env, box_cfg,
                                  w_upright: float = 1.5,
                                  w_front: float = 1.5,
                                  w_chest: float = 1.5,
                                  lift_threshold: float = 0.05):
    """Fused carry-style box pose (upright + front aligned + at chest height), gated on the box being lifted off the table so the "throw-yourself-backwards to collect the free 3.14/step" local optimum stays closed.
    Args: env: RL env; box_cfg: box entity; w_upright/w_front/w_chest: per-signal weights folded in; lift_threshold: box_z clearance above tabletop required for the reward to fire.
    Returns: (num_envs,) tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    lifted = (box_z > TABLE_TOP_Z + lift_threshold).float()
    return lifted * (w_upright * box_upright_reward(env, box_cfg)
                   + w_front   * box_front_aligned_reward(env, box_cfg)
                   + w_chest   * box_at_chest_reward(env, box_cfg))


def hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg):
    """Sharp Gaussian on the sum of hand errors to commanded grasp anchors.
    Args: env: the RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity.
    Returns: (num_envs,) tensor in (0, 1].
    """
    left  = _entity_pos(env, left_hand_cfg)
    right = _entity_pos(env, right_hand_cfg)
    error = (
        torch.linalg.norm(left  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
        + torch.linalg.norm(right - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    )
    return torch.exp(-8.0 * error)


def hand_placement_bundle_reward(
    env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name,
    w_reach: float = 2.5, w_hit: float = 3.0, w_hold: float = 3.0,
    std_reach: float = 0.20, hit_threshold: float = 0.08,
):
    """Fused reach_left + reach_right + hit_bonus + hold_box sharing one (dl, dr) hand-to-anchor read and one stability_gate multiply (equivalent to running the four separate terms).
    Args: env: RL env; left_hand_cfg/right_hand_cfg: hand entities; box_cfg: box entity; trunk_cfg: trunk for gate; sensor_name: feet sensor for gate; w_reach: per-side reach weight; w_hit: hit bonus weight; w_hold: hold weight; std_reach: reach Gaussian bandwidth; hit_threshold: binary hit distance.
    Returns: (num_envs,) tensor of the gated weighted sum.
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    gate = stability_gate(env, trunk_cfg, sensor_name)
    two_std2 = 2.0 * std_reach * std_reach
    reach = torch.exp(-dl * dl / two_std2) + torch.exp(-dr * dr / two_std2)
    hit = ((dl < hit_threshold) & (dr < hit_threshold)).float()
    hold = torch.exp(-8.0 * (dl + dr))
    return (w_reach * reach + w_hit * hit + w_hold * hold) * gate


def carry_ready_pose_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg, robot_cfg,
    target_values: tuple[float, ...] = CARRY_READY_TARGET,
    sigma_legs: float = 0.20,
    sigma_waist: float = 0.20,
    sigma_arms: float = 0.25,
    lift_threshold: float = 0.15,
    hold_dist: float = 0.15,
):
    """Shape the terminal full-body pose toward CARRY_READY_POSE (KNEES_BENT legs + upright waist + HOLDING chest-hug arms) via a three-group Gaussian averaged so one very-wrong group can't erase the whole reward, double-gated on (box lifted) AND (both hands near anchors) so reach/lift stay undisturbed.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities for the hold gate; robot_cfg: SceneEntityCfg selecting all 29 joints in G1_JOINTS order (typically g1_asset); target_values: per-joint targets aligned with robot_cfg.joint_ids (CARRY_READY_TARGET); sigma_legs/waist/arms: per-group Gaussian bandwidths (rad); lift_threshold: min box clearance above tabletop; hold_dist: max hand-to-anchor distance for the gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    gate = ((box_z > TABLE_TOP_Z + lift_threshold) & (dl < hold_dist) & (dr < hold_dist)).float()

    robot = env.scene["robot"]
    joints = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (N, 29) in G1_JOINTS order
    if not hasattr(env, "_carry_ready_target_cache"):
        env._carry_ready_target_cache = torch.tensor(
            target_values, device=env.device, dtype=joints.dtype,
        )
    err = joints - env._carry_ready_target_cache  # (N, 29)

    # Slice layout matches G1_JOINTS: 0-11 legs, 12-14 waist, 15-28 arms.
    legs_r  = torch.exp(-(err[:, 0:12]  ** 2).mean(dim=-1) / (2.0 * sigma_legs  * sigma_legs))
    waist_r = torch.exp(-(err[:, 12:15] ** 2).mean(dim=-1) / (2.0 * sigma_waist * sigma_waist))
    arms_r  = torch.exp(-(err[:, 15:29] ** 2).mean(dim=-1) / (2.0 * sigma_arms  * sigma_arms))
    return gate * (legs_r + waist_r + arms_r) / 3.0


def box_at_lift_target_pos_reward(
    env, box_cfg,
    sigma_xy: float = 0.30,
    sigma_z_below: float = 0.30,
    sigma_z_above: float = 0.08,
    plateau_xy: float = 0.06,
    plateau_z: float = 0.05,
):
    """Additive box-to-lift-target tracking with a broad-below/narrow-above z asymmetry so a squat-then-lift gets strong pulling gradient but overshoot becomes a sharp cliff; additive form keeps signal alive when only one axis is far. Reward saturates to 1.0 inside a (plateau_xy, plateau_z) success zone so the policy stops micro-correcting once close: kills post-grasp wiggle without touching shaping outside the zone.
    Args: env: RL env; box_cfg: box entity; sigma_xy: symmetric xy Gaussian bandwidth; sigma_z_below: z bandwidth when the box is below target; sigma_z_above: z bandwidth when above; plateau_xy: half-width of the flat success zone in xy; plateau_z: half-height of the flat success zone in z.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box = _entity_pos(env, box_cfg)
    tgt = env.extras[_CMD_KEY + "lift_target_pos"]
    xy_dist = torch.linalg.norm(box[:, :2] - tgt[:, :2], dim=-1)
    z_diff  = box[:, 2] - tgt[:, 2]
    sigma_z = torch.where(z_diff < 0,
                          torch.full_like(z_diff, sigma_z_below),
                          torch.full_like(z_diff, sigma_z_above))
    xy_r = torch.exp(-(xy_dist ** 2) / (2.0 * sigma_xy ** 2))
    z_r  = torch.exp(-(z_diff  ** 2) / (2.0 * sigma_z  ** 2))
    shaped = 0.5 * (xy_r + z_r)
    in_zone = (xy_dist < plateau_xy) & (z_diff.abs() < plateau_z)
    return torch.where(in_zone, torch.ones_like(shaped), shaped)


def lift_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    hold_dist: float = 0.10,
):
    """Linear reward proportional to box height above the tabletop, gated on both hands being close to the commanded grasp anchors (not the box centre: anchors sit on the side face 13 cm from centre, so a centre-based gate never fires).
    Args: env: RL env; box_cfg: box entity; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; hold_dist: max hand-to-anchor distance for the gate in metres.
    Returns: (num_envs,) tensor in [0, 1] scaled by the holding gate.
    """
    box = _entity_pos(env, box_cfg)
    height_norm = torch.clamp((box[:, 2] - TABLE_TOP_Z) / (CHEST_Z - TABLE_TOP_Z), 0.0, 1.0)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    holding = ((dl < hold_dist) & (dr < hold_dist)).float()
    return height_norm * holding


def box_at_lift_target_quat_reward(env, box_cfg):
    """SO(3) similarity |q_box . q_target|^2, invariant to quaternion double cover.
    Args: env: the RL env; box_cfg: box entity.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    tgt_quat = env.extras[_CMD_KEY + "lift_target_quat"]
    q_dot = (box_quat * tgt_quat).sum(dim=-1)
    return q_dot * q_dot


def upright_torso_reward(env, trunk_cfg):
    """Quat-based upright score, saturating at 1 when the torso z-axis is up.
    Args: env: the RL env; trunk_cfg: trunk entity config.
    Returns: (num_envs,) tensor in [0, 1].
    """
    entity: Entity = env.scene[trunk_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(up_z, 0.0, 1.0)


def fell_over_penalty(env, trunk_cfg,
                      tilt_threshold: float = 0.5,
                      early_steps: int = 30):
    """Per-step penalty when the trunk has tilted past tilt_threshold within the first early_steps steps, so each early falling step costs far more than the "throw-yourself-backwards" cheat can earn before termination.
    Args: env: RL env; trunk_cfg: trunk entity; tilt_threshold: up_z value below which the trunk counts as fallen (1 = upright, 0 = horizontal); early_steps: window in policy steps.
    Returns: (num_envs,) 0/1 tensor.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    quat = entity.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    fallen = (up_z < tilt_threshold).float()
    early = (env.episode_length_buf < early_steps).float()
    return fallen * early


def symmetric_pose_penalty(env, left_cfg, right_cfg):
    """Sum of squared mirrored-joint deviations across arm/wrist pairs.
    Args: env: the RL env; left_cfg: left joint set entity; right_cfg: right joint set entity.
    Returns: (num_envs,) tensor of the squared-error sum.
    """
    robot = env.scene[left_cfg.name]
    if not hasattr(env, "_sym_signs_tensor"):
        env._sym_signs_tensor = torch.tensor(
            _SYM_SIGNS, device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype,
        )
    qL = robot.data.joint_pos[:, left_cfg.joint_ids]
    qR = robot.data.joint_pos[:, right_cfg.joint_ids]
    diff = qL - env._sym_signs_tensor * qR
    return (diff * diff).sum(dim=-1)


# Task rewards gated by stability_gate so a falling robot earns zero task credit.
def gated_reach_reward(env, hand_cfg, box_cfg, trunk_cfg, sensor_name, side: str, std: float = 0.20):
    """Stability-gated wrapper over reach_reward for one hand.
    Args: env: RL env; hand_cfg: hand entity; box_cfg: box entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate; side: "left" or "right"; std: Gaussian bandwidth.
    Returns: (num_envs,) tensor equal to reach_reward × stability_gate.
    """
    return reach_reward(env, hand_cfg, box_cfg, side, std) * stability_gate(env, trunk_cfg, sensor_name)


def gated_both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name, threshold: float = 0.08):
    """Stability-gated wrapper over both_hands_hit_bonus.
    Args: env: RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate; threshold: hit distance in metres.
    Returns: (num_envs,) tensor equal to both_hands_hit_bonus × stability_gate.
    """
    return both_hands_hit_bonus(env, left_hand_cfg, right_hand_cfg, box_cfg, threshold) * stability_gate(env, trunk_cfg, sensor_name)


def gated_hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name):
    """Stability-gated wrapper over hold_box_reward.
    Args: env: RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate.
    Returns: (num_envs,) tensor equal to hold_box_reward × stability_gate.
    """
    return hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def gated_box_at_lift_target_pos_reward(
    env, box_cfg, trunk_cfg, sensor_name,
    sigma_xy: float = 0.30,
    sigma_z_below: float = 0.30,
    sigma_z_above: float = 0.08,
):
    """Stability-gated wrapper over box_at_lift_target_pos_reward.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate; sigma_xy: symmetric xy bandwidth; sigma_z_below/sigma_z_above: asymmetric z bandwidths.
    Returns: (num_envs,) tensor equal to box_at_lift_target_pos_reward × stability_gate.
    """
    return box_at_lift_target_pos_reward(
        env, box_cfg, sigma_xy=sigma_xy,
        sigma_z_below=sigma_z_below, sigma_z_above=sigma_z_above,
    ) * stability_gate(env, trunk_cfg, sensor_name)


def gated_box_at_lift_target_quat_reward(env, box_cfg, trunk_cfg, sensor_name):
    """Stability-gated wrapper over box_at_lift_target_quat_reward.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate.
    Returns: (num_envs,) tensor equal to box_at_lift_target_quat_reward × stability_gate.
    """
    return box_at_lift_target_quat_reward(env, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def gated_lift_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    trunk_cfg, sensor_name, hold_dist: float = 0.10,
):
    """Stability-gated wrapper over lift_progress_reward.
    Args: env: RL env; box_cfg: box entity; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; trunk_cfg: trunk entity for the gate; sensor_name: feet contact sensor for the gate; hold_dist: max hand-to-anchor distance for the holding gate.
    Returns: (num_envs,) tensor equal to lift_progress_reward × stability_gate.
    """
    return lift_progress_reward(
        env, box_cfg, left_hand_cfg, right_hand_cfg, hold_dist,
    ) * stability_gate(env, trunk_cfg, sensor_name)


def non_hand_box_contact_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg, box_cfg,
    threshold: float = 0.22,
):
    """Proximity ramp discouraging non-hand body parts from touching the box.
    Args: env: the RL env; trunk_cfg: torso entity; left_shank_cfg: left knee entity; right_shank_cfg: right knee entity; left_hip_cfg: left hip entity; right_hip_cfg: right hip entity; box_cfg: box entity; threshold: activation distance.
    Returns: (num_envs,) tensor of accumulated penalty.
    """
    box_pos = _entity_pos(env, box_cfg)
    penalty = torch.zeros(box_pos.shape[0], device=box_pos.device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg):
        body_pos = _entity_pos(env, cfg)
        dist = torch.linalg.norm(body_pos - box_pos, dim=-1)
        penalty += torch.clamp(threshold - dist, min=0.0) / threshold
    return penalty


def body_table_collision_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    left_hand_cfg, right_hand_cfg,
    margin_xy: float = 0.02,
    z_slack: float = 0.02,
):
    """Volumetric penalty for entering the tabletop shadow extended to the floor; hands are included to block the "hand under the table" cheat that pushes the box off from below.
    Args: env: the RL env; trunk_cfg: torso entity; left_shank_cfg: left knee; right_shank_cfg: right knee; left_hip_cfg: left hip; right_hip_cfg: right hip; left_hand_cfg: left hand; right_hand_cfg: right hand; margin_xy: footprint shrink; z_slack: vertical tolerance above the top.
    Returns: (num_envs,) tensor of accumulated penetration depth.
    """
    device = env.scene["robot"].data.root_link_pos_w.device
    n_envs = env.scene["robot"].data.root_link_pos_w.shape[0]
    t_xy = torch.tensor([TABLE_X, 0.0], device=device)
    t_half_xy = torch.tensor([0.34 - margin_xy, 0.42 - margin_xy], device=device)
    top_z = TABLE_TOP_Z + TABLE_TOP_HALF - z_slack
    penalty = torch.zeros(n_envs, device=device)
    for cfg in (trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
                left_hand_cfg, right_hand_cfg):
        pos = _entity_pos(env, cfg)
        xy_into = t_half_xy - torch.abs(pos[:, :2] - t_xy)   # positive inside footprint
        z_below = top_z - pos[:, 2]                          # positive below top surface
        depth = torch.min(
            torch.stack([xy_into[:, 0], xy_into[:, 1], z_below], dim=-1),
            dim=-1,
        ).values
        penalty += torch.clamp(depth, min=0.0)
    return penalty


_BOX_FLOOR_LIMIT_Z = TABLE_TOP_Z - 0.12   # box fell off the table


def fell_over(env, box_cfg, trunk_cfg, limit_angle: float = math.radians(65.0)):
    """Terminate if the torso tilts past limit_angle or the box falls to the floor.
    Args: env: the RL env; box_cfg: box entity; trunk_cfg: trunk entity; limit_angle: max tilt in radians.
    Returns: (num_envs,) bool tensor of termination flags.
    """
    entity: Entity = env.scene[trunk_cfg.name]
    proj_grav = entity.data.projected_gravity_b
    up_alignment = -proj_grav[:, 2]
    tilted = up_alignment < math.cos(limit_angle)
    box_pos = _entity_pos(env, box_cfg)
    box_fell = box_pos[:, 2] < _BOX_FLOOR_LIMIT_Z
    return tilted | box_fell


def box_touched_body(
    env, box_cfg, trunk_cfg,
    left_hip_cfg, right_hip_cfg,
    left_shank_cfg, right_shank_cfg,
    threshold: float = 0.15,
):
    """Terminate when the box collapses into trunk, hips or shanks; kills the "let box fall into the lap" attractor (a hand-held box in front of the torso cannot get this close).
    Args: env: RL env; box_cfg: box entity; trunk_cfg/left_hip_cfg/right_hip_cfg/left_shank_cfg/right_shank_cfg: body entities; threshold: box-centre-to-body-origin distance below which the box counts as touching.
    Returns: (num_envs,) bool tensor.
    """
    box_pos = _entity_pos(env, box_cfg)
    hit = torch.zeros(box_pos.shape[0], dtype=torch.bool, device=box_pos.device)
    for cfg in (trunk_cfg, left_hip_cfg, right_hip_cfg, left_shank_cfg, right_shank_cfg):
        dist = torch.linalg.norm(_entity_pos(env, cfg) - box_pos, dim=-1)
        hit |= dist < threshold
    return hit


def body_touched_table(env, sensor_name: str = "body_table_contact",
                        force_threshold: float = 5.0):
    """Terminate on ground-truth contact between torso/hip/thigh/shin subtree geoms and the table body; catches slump, thigh-on-edge, and collapse-forward without any geometric threshold.
    Args: env: RL env; sensor_name: ContactSensor name; force_threshold: newtons above which contact counts (used only when the sensor exposes force).
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


def box_fell_back(env, box_cfg,
                  high_thresh: float = 0.10,
                  settle_thresh: float = 0.03,
                  min_step: int = 30):
    """Terminate when the box, after ever rising high_thresh above the resting-on-table position, drops back within settle_thresh of it; grace-gated on episode_length_buf >= min_step so early transient contacts at reset can't kill the episode before the policy has time to grip.
    Args: env: RL env; box_cfg: box entity; high_thresh: clearance above resting height that counts as "lifted"; settle_thresh: proximity to resting height that counts as "back on the table"; min_step: grace period in policy steps before this termination can fire.
    Returns: (num_envs,) bool tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    key = "pick_max_box_z"
    if key not in env.extras or env.extras[key].shape[0] != env.num_envs:
        env.extras[key] = torch.zeros(env.num_envs, device=env.device)
    env.extras[key] = torch.maximum(env.extras[key], box_z)
    max_z = env.extras[key]
    rested_z = TABLE_TOP_Z + BOX_HALF
    was_ever_high = max_z > rested_z + high_thresh
    now_settled   = box_z < rested_z + settle_thresh
    past_grace    = env.episode_length_buf >= min_step
    return was_ever_high & now_settled & past_grace


def body_touched_box(env, sensor_name: str = "body_box_contact",
                      force_threshold: float = 5.0):
    """Terminate on ground-truth contact between non-hand body geoms and the box; catches leg-on-box and torso-on-box that box_touched_body misses (a shank on top of the box sits at delta≈0.155 m, just outside the 0.15 m geometric threshold).
    Args: env: RL env; sensor_name: ContactSensor name; force_threshold: newtons above which contact counts.
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


def foot_displaced(env, left_foot_cfg, right_foot_cfg,
                   threshold: float = 0.06,
                   grace_steps: int = 15):
    """Terminate when either foot moves more than threshold (xy) from its post-reset xy, snapshotted at episode_length_buf == grace_steps so the reset transient settles first; teaches "do not walk during pick" by paying with termination for any drag or side-step.
    Args: env: RL env; left_foot_cfg: left foot entity (usually left_ankle_roll_link); right_foot_cfg: right foot entity; threshold: max allowed xy displacement in metres; grace_steps: steps to wait before snapshotting the reference.
    Returns: (num_envs,) bool tensor.
    """
    lf = _entity_pos(env, left_foot_cfg)[:, :2]
    rf = _entity_pos(env, right_foot_cfg)[:, :2]
    key_l = "pick_foot_init_xy_l"
    key_r = "pick_foot_init_xy_r"
    if key_l not in env.extras or env.extras[key_l].shape != lf.shape:
        env.extras[key_l] = lf.clone()
        env.extras[key_r] = rf.clone()
    at_grace = env.episode_length_buf == grace_steps
    if at_grace.any():
        mask = at_grace.unsqueeze(-1)
        env.extras[key_l] = torch.where(mask, lf, env.extras[key_l])
        env.extras[key_r] = torch.where(mask, rf, env.extras[key_r])
    dl = torch.linalg.norm(lf - env.extras[key_l], dim=-1)
    dr = torch.linalg.norm(rf - env.extras[key_r], dim=-1)
    past_grace = env.episode_length_buf > grace_steps
    return past_grace & ((dl > threshold) | (dr > threshold))


_WALK_BUFFER_CACHE: dict[str, torch.Tensor] | None = None


def _load_walk_buffer(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load the .npz walk-terminal buffer once and cache tensors on device.
    Args: path: buffer file; device: torch device.
    Returns: dict of buffer name to tensor (plus a "_path" sentinel).
    """
    global _WALK_BUFFER_CACHE
    if _WALK_BUFFER_CACHE is None or _WALK_BUFFER_CACHE.get("_path") != path:
        import numpy as np
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"walk buffer not found at {p.resolve()}. "
                f"Run: python -m src.collect_buffer --policy-module src.policy_walk --checkpoint <walk_ckpt> --out {p}"
            )
        arr = np.load(p)
        cache = {k: torch.from_numpy(arr[k]).float().to(device) for k in arr.files}
        cache["_path"] = path  # type: ignore[assignment]
        _WALK_BUFFER_CACHE = cache
    return _WALK_BUFFER_CACHE


def reset_from_home_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset the robot to KNEES_BENT_KEYFRAME with small xy/yaw jitter as a canonical stable start, avoiding the mid-walk snapshots in the walk buffer that catapult early training into 9-step falls.
    Args: env: RL env; env_ids: envs to reset (None = all); xy_range: additive xy jitter bounds; yaw_range: additive yaw jitter bounds in rad; asset_cfg: robot entity config.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    asset: Entity = env.scene[asset_cfg.name]

    home_root_pos = torch.tensor(G1_HOME_ROOT_POS, device=device, dtype=torch.float32)
    home_root_rot = torch.tensor(G1_HOME_ROOT_ROT, device=device, dtype=torch.float32)
    home_joint_pos = torch.tensor(
        [G1_HOME_JOINT_POS[j] for j in asset_cfg.joint_names],
        device=device, dtype=torch.float32,
    )

    xy_jitter = sample_uniform(xy_range[0], xy_range[1], (n, 2), device=device)
    yaw_jitter = sample_uniform(yaw_range[0], yaw_range[1], (n,), device=device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)
    root_quat = quat_mul(q_yaw, home_root_rot.unsqueeze(0).expand(n, -1))

    origins = env.scene.env_origins[env_ids]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    root_xy = default_root_state[:, 0:2] + xy_jitter + origins[:, 0:2]
    root_z  = torch.full((n, 1), float(home_root_pos[2]), device=device) + origins[:, 2:3]
    positions = torch.cat([root_xy, root_z], dim=-1)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, root_quat], dim=-1), env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )

    joint_pos = home_joint_pos.unsqueeze(0).expand(n, -1).clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)


def reset_from_walk_buffer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    buffer_path: str,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    body_bias: float = 0.0,
) -> None:
    """Reset the full body by linearly interpolating between SQUAT_GRIP_POSE (body_bias=1.0, robot spawns already gripping the box) and a sampled walk-terminal state (body_bias=0.0, realistic post-walk posture). Joint positions, root z and root quat all lerp together so pick_curriculum can move the starting state along a single monotonic axis without the reward-magnitude / task-swap discontinuities the old hand_bias / arm_bias two-axis curriculum introduced.
    Args: env: the RL env; env_ids: envs to reset (all if None); buffer_path: path to the .npz walk buffer; xy_range: additive xy jitter bounds; yaw_range: additive yaw jitter bounds in rad; asset_cfg: robot entity config covering all 29 G1 joints in G1_JOINTS order; body_bias: curriculum weight in [0, 1], 1.0 = pure SQUAT_GRIP, 0.0 = pure walk sample.
    Returns: none.
    """
    from mjlab.utils.lab_api.math import (
        quat_from_euler_xyz, quat_mul, sample_uniform,
    )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    bb = float(max(0.0, min(1.0, body_bias)))
    device = env.device

    # Walk-buffer end of the interpolation.
    buf = _load_walk_buffer(buffer_path, device)
    total = buf["root_z"].shape[0]
    idx = torch.randint(0, total, (n,), device=device)
    walk_z     = buf["root_z"][idx]                    # (n,) or (n, 1)
    walk_quat  = buf["quat_zy"][idx]                   # (n, 4)
    walk_joint = buf["joint_pos"][idx]                 # (n, 29) in G1_JOINTS order

    # SQUAT_GRIP end of the interpolation.
    squat_joint = torch.tensor(
        [SQUAT_GRIP_POSE[j] for j in asset_cfg.joint_names],
        device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()
    squat_z = torch.full(walk_z.shape, SQUAT_GRIP_ROOT_POS[2], device=device)
    squat_xy_offset = torch.tensor(
        SQUAT_GRIP_ROOT_POS[:2], device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()
    squat_quat = torch.tensor(
        SQUAT_GRIP_ROOT_ROT, device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()

    # Full-body lerp; renormalise quaternion since lerp of unit quats drifts off the sphere.
    joint_pos = bb * squat_joint + (1.0 - bb) * walk_joint
    root_z    = bb * squat_z + (1.0 - bb) * walk_z
    root_quat_base = bb * squat_quat + (1.0 - bb) * walk_quat
    root_quat_base = root_quat_base / torch.linalg.norm(
        root_quat_base, dim=-1, keepdim=True,
    ).clamp(min=1e-6)

    # Zero velocities at reset: inherit posture without inertia (residual base/joint velocities used to catapult the robot into unrecoverable tilts).
    joint_vel = torch.zeros_like(joint_pos)
    lin_vel = torch.zeros(n, 3, device=device)
    ang_vel = torch.zeros(n, 3, device=device)

    # Scale jitter by (1 - body_bias) so squat mode spawns the robot deterministically in front of the box (facing +x, at SQUAT_GRIP_ROOT_POS xy) and walk mode keeps the full DR range.
    jitter_scale = 1.0 - bb
    xy_jitter = sample_uniform(xy_range[0] * jitter_scale, xy_range[1] * jitter_scale, (n, 2), device=device)
    yaw_jitter = sample_uniform(yaw_range[0] * jitter_scale, yaw_range[1] * jitter_scale, (n,), device=device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)
    root_quat = quat_mul(q_yaw, root_quat_base)

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]

    # Blend root_xy between the walk-default (from default_root_state) and the SQUAT offset (which is an absolute world offset from the env origin, not additive to walk default). Using (1-bb)*default + bb*squat keeps both endpoints correct: at bb=0 the pelvis lands at walk default, at bb=1 at exactly SQUAT_GRIP_ROOT_POS[:2].
    root_xy = (1.0 - bb) * default_root_state[:, 0:2] + bb * squat_xy_offset + xy_jitter + origins[:, 0:2]
    root_z_col = root_z if root_z.dim() == 2 else root_z.unsqueeze(-1)
    root_z_w = root_z_col + origins[:, 2:3]
    positions = torch.cat([root_xy, root_z_w], dim=-1)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, root_quat], dim=-1), env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(
        torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids,
    )

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=device)
    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids,
    )


def _smoothstep(t: float) -> float:
    """Hermite smoothstep giving a monotonic C1 S-curve with zero derivative at t=0 and t=1, avoiding the gradient spike linear interpolation leaves at ramp boundaries.
    Args: t: parameter clamped to [0, 1].
    Returns: smoothed value in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Optional curriculum freeze: when set, pick_curriculum reports this step forever so the viewer path can run at the same body_bias the last training checkpoint converged to; None = training mode.
_PICK_CURRICULUM_LOCKED_STEP: int | None = None


def lock_curriculum_at_iter(iter_num: int | None, num_steps_per_env: int = 24) -> None:
    """Freeze pick_curriculum at the step corresponding to a given PPO iteration so a resumed viewer run shows the trained body_bias instead of restarting from the SQUAT_GRIP warmup phase.
    Args: iter_num: PPO iteration count parsed from a checkpoint filename; pass None to unlock; num_steps_per_env: must match the PPO cfg (default 24, matches g1_pick_ppo_runner_cfg).
    Returns: none.
    """
    global _PICK_CURRICULUM_LOCKED_STEP
    if iter_num is None:
        _PICK_CURRICULUM_LOCKED_STEP = None
        return
    _PICK_CURRICULUM_LOCKED_STEP = int(iter_num) * int(num_steps_per_env)


def pick_curriculum(env, env_ids):
    """Single-axis curriculum ramping body_bias from 1.0 (robot spawns in SQUAT_GRIP_POSE, hands already at the box on the table) to 0.0 (robot spawns from a walk-terminal sample). No box curriculum: the box always spawns on the table so the task the policy solves stays identical across training, which keeps the reward curve monotonic (no reward-ceiling shifts) unlike the previous two-axis hand_bias / arm_bias / scale ramp.
    Args: env: the RL env; env_ids: envs being updated (unused, curriculum is global).
    Returns: dict with pick_body_bias for logging.
    """
    # Per-env counter (fresh at every runner) instead of env.common_step_counter, which persists across runner.load() and made the curriculum skip past the ramp on resume.
    if _PICK_CURRICULUM_LOCKED_STEP is not None:
        step = _PICK_CURRICULUM_LOCKED_STEP
    else:
        if not hasattr(env, "_pick_curriculum_step"):
            env._pick_curriculum_step = 0
        env._pick_curriculum_step += 1
        step = env._pick_curriculum_step
    # Two-phase smooth ramp: hold body_bias=1.0 for the first 24k steps so the policy learns the pick/lift dynamics from an ideal start, then smoothstep 1.0 -> 0.0 across the next 72k steps as the robot gradually starts from realistic walk poses.
    if step < 24_000:
        body_bias = 1.0
    elif step < 96_000:
        body_bias = 1.0 - _smoothstep((step - 24_000) / (96_000 - 24_000))
    else:
        body_bias = 0.0
    try:
        term_cfg = env.event_manager.get_term_cfg("reset_base")
        term_cfg.params["body_bias"] = body_bias
    except Exception:
        pass
    return {"pick_body_bias": body_bias}


def build_g1_pick_mdp():
    """Assemble the MDP dict (entities, sensors, obs, actions, rewards, events, terminations, curriculum).
    Args: none.
    Returns: dict of MDP components consumed by g1_pick_env_cfg.
    """
    robot_cfg = get_g1_robot_cfg()
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=G1_HOME_ROOT_POS,
        rot=G1_HOME_ROOT_ROT,
        joint_pos=G1_HOME_JOINT_POS,
    )
    table_cfg = EntityCfg(spec_fn=_get_table_spec)
    box_cfg_e = EntityCfg(
        spec_fn=_get_box_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(BOX_START_X, 0.0, BOX_START_Z),
            joint_pos={},
        ),
    )

    g1_asset    = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    left_sym    = SceneEntityCfg("robot", joint_names=_LEFT_SYM_JOINTS,  preserve_order=True)
    right_sym   = SceneEntityCfg("robot", joint_names=_RIGHT_SYM_JOINTS, preserve_order=True)
    wrist_acts  = SceneEntityCfg("robot", actuator_names=(r".*wrist.*",), preserve_order=True)
    trunk       = SceneEntityCfg("robot", body_names=("torso_link",))
    pelvis      = SceneEntityCfg("robot", body_names=("pelvis",))
    left_hand   = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand  = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_shank  = SceneEntityCfg("robot", body_names=("left_knee_link",))
    right_shank = SceneEntityCfg("robot", body_names=("right_knee_link",))
    left_hip    = SceneEntityCfg("robot", body_names=("left_hip_pitch_link",))
    right_hip   = SceneEntityCfg("robot", body_names=("right_hip_pitch_link",))
    left_foot   = SceneEntityCfg("robot", body_names=("left_ankle_roll_link",))
    right_foot  = SceneEntityCfg("robot", body_names=("right_ankle_roll_link",))
    box         = SceneEntityCfg("box")

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
    hand_box_contact_cfg = ContactSensorCfg(
        name="hand_box_contact",
        # Narrowed to wrist subtrees so grip_force_reward is not paid for leg / torso contact.
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left|right)_wrist_yaw_link$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="box", entity="box"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=4,
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
    body_table_contact_cfg = ContactSensorCfg(
        name="body_table_contact",
        # Torso + real-limb bodies (arms/feet excluded, hip_roll/hip_yaw pivots also excluded: they are small internal pivot links that sit right at the table_top elevation in SQUAT_GRIP and produce false-positive terminations at reset; hip_pitch is the "real hip" limb that would actually collapse onto the table).
        primary=ContactMatch(
            mode="body",
            pattern=r"^(torso_link"
                     r"|left_hip_pitch_link|left_knee_link|left_ankle_pitch_link"
                     r"|right_hip_pitch_link|right_knee_link|right_ankle_pitch_link"
                     r")$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="table", entity="table"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    body_box_contact_cfg = ContactSensorCfg(
        name="body_box_contact",
        # Same real-limb list as body_table_contact (excludes hip_roll/hip_yaw pivots).
        primary=ContactMatch(
            mode="body",
            pattern=r"^(torso_link"
                     r"|left_hip_pitch_link|left_knee_link|left_ankle_pitch_link"
                     r"|right_hip_pitch_link|right_knee_link|right_ankle_pitch_link"
                     r")$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="box", entity="box"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
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

    # Observations: base proprioception + task-relative vectors + 20-D command.
    actor_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_lin_vel,      noise=Unoise(n_min=-0.2,   n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=base_ang_vel,      noise=Unoise(n_min=-0.1,   n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.5,  n_max=0.5)),
        "actions":   ObservationTermCfg(func=last_action),
        # Task-relative vectors in world frame; the policy learns to un-rotate.
        "left_hand_to_box":  ObservationTermCfg(func=body_vector, params={"source_cfg": left_hand,  "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "right_hand_to_box": ObservationTermCfg(func=body_vector, params={"source_cfg": right_hand, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "trunk_to_box":      ObservationTermCfg(func=body_vector, params={"source_cfg": trunk,      "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        # Grasp detection signal.
        "wrist_torque":      ObservationTermCfg(func=wrist_actuator_force_obs, params={"asset_cfg": wrist_acts}, noise=Unoise(n_min=-0.2, n_max=0.2)),
        # Full sanitized box state in world frame.
        "box_pos":           ObservationTermCfg(func=box_position_obs,    params={"box_cfg": box}, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "box_lin_vel":       ObservationTermCfg(func=box_lin_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_ang_vel":       ObservationTermCfg(func=box_ang_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_orientation":   ObservationTermCfg(func=box_orientation_obs, params={"box_cfg": box}),
        "box_size":          ObservationTermCfg(func=box_size_obs),
        # 20-D command: grasp offsets in box frame plus base-frame target poses.
        "cmd_grasp_offset_left":  ObservationTermCfg(func=cmd_grasp_offset_left_obs),
        "cmd_grasp_offset_right": ObservationTermCfg(func=cmd_grasp_offset_right_obs),
        "cmd_object_target_pos":  ObservationTermCfg(func=cmd_object_target_pos_obs),
        "cmd_object_target_quat": ObservationTermCfg(func=cmd_object_target_quat_obs),
        "cmd_lift_target_pos":    ObservationTermCfg(func=cmd_lift_target_pos_obs),
        "cmd_lift_target_quat":   ObservationTermCfg(func=cmd_lift_target_quat_obs),
    }
    observations = {
        "actor":  ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=G1_JOINTS,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        )
    }

    # Task rewards gated on upright torso (<=25 deg) AND both feet in contact; palm-inward stays ungated.
    rewards = {
        # Fused reach_left + reach_right + hit_bonus + hold_box into hand_placement_bundle (all shared the same stability_gate and (dl, dr) distances; computed once, weights folded in).
        "hand_placement": RewardTermCfg(
            func=hand_placement_bundle_reward, weight=1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "trunk_cfg": trunk,
                    "sensor_name": "feet_ground_contact",
                    "w_reach": 2.5, "w_hit": 3.0, "w_hold": 3.0,
                    "std_reach": 0.20, "hit_threshold": 0.08},
        ),
        # Fused: palm_inward_left + palm_inward_right to palm_inward_bilateral (w=0.5 each folded in).
        "palm_inward": RewardTermCfg(
            func=palm_inward_bilateral_reward, weight=1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "w_left": 0.5, "w_right": 0.5},
        ),
        # Real contact-force reward replacing the removed gentle_grasp penalty (which taught weak grip). Saturates at 20 N: 5 N let a 2-3 N barely-touching grip claim 60-80% of the reward with no incentive to squeeze.
        "grip_force": RewardTermCfg(func=grip_force_reward, weight=5.0,
                                    params={"sensor_name": "hand_box_contact",
                                            "target_force": 20.0}),
        # Lift target with asymmetric z-kernel (broad below to pull up from squat, narrow above to punish overshoot); weight bumped 2.0 -> 4.0.
        "box_at_lift_target_pos":  RewardTermCfg(func=gated_box_at_lift_target_pos_reward,  weight=4.0,
                                                 params={"box_cfg": box, "trunk_cfg": trunk,
                                                         "sensor_name": "feet_ground_contact",
                                                         "sigma_xy": 0.30,
                                                         "sigma_z_below": 0.30,
                                                         "sigma_z_above": 0.08}),
        # Fused: box_upright + box_front_aligned + box_at_chest to box_carry_pose_bundle (w=1.5 each folded in).
        "box_carry_pose": RewardTermCfg(
            func=box_carry_pose_bundle_reward, weight=1.0,
            params={"box_cfg": box, "w_upright": 1.5, "w_front": 1.5, "w_chest": 1.5},
        ),
        # Shapes the terminal full-body pose toward CARRY_READY_POSE so pick's terminal snapshot matches the walk/carry canonical posture at every state-machine transition (three-group Gaussian; double-gated on lifted AND hands-near-anchors so reach/lift stay undisturbed).
        "carry_ready_pose": RewardTermCfg(
            func=carry_ready_pose_reward, weight=1.5,
            params={"box_cfg": box, "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "robot_cfg": g1_asset,
                    "target_values": CARRY_READY_TARGET,
                    "sigma_legs": 0.20, "sigma_waist": 0.20, "sigma_arms": 0.25,
                    "lift_threshold": 0.15, "hold_dist": 0.15},
        ),
        # Dense linear progress on box height above the table (gated on holding): every cm of lift pays, breaking the squat local optimum the Gaussian above tolerates.
        "lift_progress": RewardTermCfg(func=gated_lift_progress_reward, weight=3.0,
                                       params={"box_cfg": box,
                                               "left_hand_cfg": left_hand,
                                               "right_hand_cfg": right_hand,
                                               "trunk_cfg": trunk,
                                               "sensor_name": "feet_ground_contact",
                                               "hold_dist": 0.10}),
        # Fused upright + feet_on_ground stability bundle (weights bumped 1.0 -> 5.0 to fix critic collapse: posture was too small vs task rewards to shape a non-trivial value function).
        "stability": RewardTermCfg(
            func=stability_bundle_reward, weight=1.0,
            params={"trunk_cfg": trunk, "pelvis_cfg": pelvis,
                    "sensor_name": "feet_ground_contact",
                    "w_upright": 10.0, "w_upright_pelvis": 5.0, "w_feet": 5.0},
        ),
        # Weight bumped -0.5 -> -2.0 to stop free arm jitter during the hold phase (at -0.5 the penalty was trivial vs +30 stability + +5 grip_force).
        "stay_still":     RewardTermCfg(func=stay_still_penalty,    weight=-2.0,
                                        params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                                                "box_cfg": box, "gate_dist": 0.20}),
        "symmetric_pose": RewardTermCfg(func=symmetric_pose_penalty, weight=-0.3,
                                        params={"left_cfg": left_sym, "right_cfg": right_sym}),
        # ETH ANYmal-style smoothness bundle: action_rate and dof_vel bumped 2× (action_rate -3e-2 -> -6e-2, dof_vel -5e-3 -> -1e-2) to further suppress the aggressive/whippy motion + post-grasp wiggle observed at eval: the previous scaling still let the mean policy learn oscillatory targets.
        "action_rate_l2": RewardTermCfg(func=action_rate_l2,           weight=-6.0e-2),
        # Second-order (jerk) term: |a_t − 2a_{t−1} + a_{t−2}|².
        "action_2nd_order": RewardTermCfg(func=action_2nd_order_l2,    weight=-1.0e-2),
        # Joint velocity: penalise fast joints in general.
        "dof_vel_l2":       RewardTermCfg(func=dof_vel_l2,             weight=-1.0e-2),
        # Weight relaxed -3.0 -> -1.0: hands legitimately move fast near the box during a squeeze, and the stronger penalty was fighting the pick task (last run showed accumulated -1.59 growing throughout training).
        "hand_approach_vel": RewardTermCfg(
            func=hand_approach_velocity_penalty, weight=-1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "gate_dist": 0.25},
        ),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": g1_asset}),
        "base_ang_vel":   RewardTermCfg(func=base_ang_vel_penalty,     weight=-0.05),
        # Contact penalties.
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        # Fused: non_hand_box + body_table to body_collision_bundle (w=-2.0 each folded in).
        "body_collision": RewardTermCfg(
            func=body_collision_bundle_penalty, weight=1.0,
            params={"trunk_cfg": trunk, "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
                    "left_hip_cfg": left_hip, "right_hip_cfg": right_hip,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand, "box_cfg": box,
                    "w_non_hand_box": -2.0, "w_body_table": -2.0},
        ),
        # Explicit fall penalty at -50/step over the first 30 steps: termination alone lost to the "throw-yourself-backwards to collect box_carry_pose free lunch" attractor (ep=9, reward<10 before this term).
        "fell_over_penalty": RewardTermCfg(
            func=fell_over_penalty, weight=-50.0,
            params={"trunk_cfg": trunk, "tilt_threshold": 0.5, "early_steps": 30},
        ),
    }

    # Events: full-body reset with body_bias curriculum, command/box reset, domain randomization.
    events = {
        # Interpolates the full body between SQUAT_GRIP_POSE (body_bias=1.0) and a walk-terminal sample; body_bias is overwritten every step by pick_curriculum.
        "reset_base": EventTermCfg(
            func=reset_from_walk_buffer,
            mode="reset",
            params={
                "buffer_path": WALK_BUFFER_PATH,
                "xy_range":  (-0.12, 0.12),
                "yaw_range": (-0.30, 0.30),
                "asset_cfg": g1_asset,
                "body_bias": 1.0,
            },
        ),
        # Samples the 20-D command and writes the box spawn (always on the table).
        "reset_pick_cmd": EventTermCfg(
            func=reset_pick_command_and_object,
            mode="reset",
            params={},
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
            func=fell_over, params={"box_cfg": box, "trunk_cfg": trunk},
        ),
        # Kill the "let box collapse into the lap" attractor.
        "box_touched_body": TerminationTermCfg(
            func=box_touched_body,
            params={
                "box_cfg": box, "trunk_cfg": trunk,
                "left_hip_cfg": left_hip, "right_hip_cfg": right_hip,
                "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
            },
        ),
        # Kill the "slump / lean onto the tabletop" attractor via ground-truth physics contact between torso / hip / thigh / shin subtree geoms and the table body.
        "body_touched_table": TerminationTermCfg(
            func=body_touched_table,
            params={"sensor_name": body_table_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
        # "Grabbed but didn't squeeze": box ever 10 cm above the resting-on-table position and now within 3 cm of it; fires whether the box slipped from hands or was lifted and dropped (min_step=30 grace covers early transient contacts at reset).
        "box_fell_back": TerminationTermCfg(
            func=box_fell_back,
            params={"box_cfg": box,
                    "high_thresh": 0.10,
                    "settle_thresh": 0.03,
                    "min_step": 30},
        ),
        # Kill the "rest legs on the box" attractor via ground-truth physics contact between torso / hip / thigh / shin geoms and the box body.
        "body_touched_box": TerminationTermCfg(
            func=body_touched_box,
            params={"sensor_name": body_box_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
        # Feet must stay planted (xy) after settling from the reset transient: any drag / side-step past `threshold` metres kills the episode. Threshold at 6 cm allows small in-place wobble but forbids stepping.
        "foot_displaced": TerminationTermCfg(
            func=foot_displaced,
            params={"left_foot_cfg": left_foot, "right_foot_cfg": right_foot,
                    "threshold": 0.06, "grace_steps": 15},
        ),
    }

    # Curriculum widens base-reset and command ranges together.
    curriculum = {
        "reset_stages": CurriculumTermCfg(func=pick_curriculum, params={}),
    }

    return {
        "robot_cfg": robot_cfg,
        "table_cfg": table_cfg,
        "box_cfg":   box_cfg_e,
        "sensors":   (feet_ground_cfg, hand_box_contact_cfg, self_collision_cfg, body_table_contact_cfg, body_box_contact_cfg, foot_height_scan_cfg),
        "observations": observations,
        "actions":      actions,
        "rewards":      rewards,
        "events":       events,
        "terminations": terminations,
        "curriculum":   curriculum,
    }


def g1_pick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Assemble the ManagerBasedRlEnvCfg from the MDP pieces.
    Args: play: if True, disable noise/pushes and use a long episode for playback.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_pick_mdp()
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": parts["robot_cfg"], "table": parts["table_cfg"], "box": parts["box_cfg"]},
            sensors=parts["sensors"],
            num_envs=1,
            extent=2.0,
        ),
        observations=parts["observations"],
        actions=parts["actions"],
        events=parts["events"],
        rewards=parts["rewards"],
        terminations=parts["terminations"],
        curriculum=parts["curriculum"],
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=1.7,
            elevation=-12.0,
            azimuth=125.0,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=800,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=30),
        ),
        decimation=5,
        episode_length_s=3.25,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_pick_ppo_runner_cfg():
    """Build the RSL-RL PPO runner config used to train the pick policy.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.45, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        clip_actions=3.0,
        experiment_name="g1_pick", save_interval=50, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the pick task with mjlab; safe to call multiple times.
    Args: none.
    Returns: the registered task id.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_pick_env_cfg(),
        play_env_cfg=g1_pick_env_cfg(play=True),
        rl_cfg=g1_pick_ppo_runner_cfg(),
    )
    return TASK_ID
