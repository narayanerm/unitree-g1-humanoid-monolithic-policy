##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""G1 drop-only task: robot starts holding the box and places it flat and smooth on the tabletop.

Input:  a carry-buffer terminal state or procedural held-box reset.
Output: a policy that carries the box down to a commanded place target, lets go, and returns to a walk-ready posture.
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


TASK_ID = "Mjlab-G1-Drop"

# Contract for src.collect_buffer (drop is a leaf; kept for eval use).
SNAPSHOT_ENTITIES = ["robot", "box"]
SNAPSHOT_YAW_ALIGN = True

CARRY_BUFFER_PATH = "outputs/carry_terminal_states.npz"

# Legacy procedural reset (superseded by reset_from_carry_buffer); shoulder rolls tightened from 0.20 to 0.10 so the hands press inward on the box side faces at tick 0.
HOLDING_ARM_POSE = {
    "left_shoulder_pitch_joint":  -0.7,
    "left_shoulder_roll_joint":    0.10,
    "left_shoulder_yaw_joint":     0.0,
    "left_elbow_joint":            0.6,
    "left_wrist_roll_joint":       0.0,
    "left_wrist_pitch_joint":      0.0,
    "left_wrist_yaw_joint":        0.0,
    "right_shoulder_pitch_joint": -0.7,
    "right_shoulder_roll_joint":  -0.10,
    "right_shoulder_yaw_joint":    0.0,
    "right_elbow_joint":           0.6,
    "right_wrist_roll_joint":      0.0,
    "right_wrist_pitch_joint":     0.0,
    "right_wrist_yaw_joint":       0.0,
}
HOLDING_ARM_JOINTS = tuple(HOLDING_ARM_POSE.keys())

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

# Walk-ready arm target ordered to match HOLDING_ARM_JOINTS so arm_pose_walk_ready_reward can index either arm_asset.joint_ids or HOLDING_ARM_JOINTS interchangeably; shapes the drop's terminal snapshot for a clean hand-off back to the walk policy.
WALK_READY_ARM_TARGET = tuple(G1_HOME_JOINT_POS[j] for j in HOLDING_ARM_JOINTS)


# Scene and target geometry.
TABLE_X          = 0.60
TABLE_HEIGHT     = 0.44
TABLE_TOP_HALF   = 0.04
TABLE_TOP_Z      = TABLE_HEIGHT + TABLE_TOP_HALF
BOX_HALF         = 0.105
BOX_START_X      = TABLE_X - 0.09
BOX_REST_Z       = TABLE_TOP_Z + BOX_HALF   # box centre when resting flat on the table
BOX_MASS_MIN     = 0.30
BOX_MASS_MAX     = 3.00

# Held-carry chest height: used by descent_progress to normalise altitude.
CHEST_Z          = 0.93


# Grasp anchors (identical to pick: hands stay on the box side faces).
GRASP_NOMINAL_Y  = 0.13
GRASP_NOMINAL_Z  = 0.02
GRASP_JITTER     = 0.03

# Place target: anchored on the tabletop where the box comes to rest. Jitter mirrors ../unitree-g1-humanoid-pick/src/policy_pick.py OBJECT_XY_JITTER (+/-5 cm x/y, +/-20 deg yaw): with the reverse curriculum starting from the SQUAT_GRIP pose the robot's hands are fixed at ~(0.502, +/-0.107, 0.627), so a wider box jitter breaks the "hands on box" assumption at bb=1.0.
PLACE_TARGET_X_CENTER = BOX_START_X
PLACE_TARGET_X_HALF   = 0.05
PLACE_TARGET_Y_HALF   = 0.05
PLACE_TARGET_Z_CENTER = BOX_REST_Z
PLACE_TARGET_Z_HALF   = 0.02
PLACE_YAW_RANGE       = math.radians(20.0)

# Held-carry spawn (mirrors carry_normal): robot root offset chosen so the held box (at BOX_HOLD_X_OFFSET forward) lands over PLACE_TARGET_X_CENTER; yaw range kept tight so the robot spawns facing the table.
BOX_HOLD_X_OFFSET     = 0.32
BOX_HOLD_Z_ABOVE_ROOT = 0.17
# Pelvis pulled back to x=-0.05 so thigh capsules sit ~31 cm behind the table front edge; held box lands at x~0.27 and only needs ~24 cm of forward alignment to reach PLACE_TARGET_X_CENTER=0.51: short approach, not a walk.
ROBOT_SPAWN_X_OFFSET  = -0.05
ROBOT_SPAWN_XY_JITTER = 0.03
ROBOT_SPAWN_YAW_RANGE = (-0.20, 0.20)
BOX_SPAWN_XY_JITTER   = 0.015

# Standing spawn: root Z chosen so the KNEES_BENT_KEYFRAME leg pose puts feet on the ground; leg joints are left at reset_scene defaults.
STANDING_ROOT_Z = 0.76


_CMD_KEY = "drop_cmd_"

CMD_DIMS = {
    "grasp_offset_left":   3,
    "grasp_offset_right":  3,
    "object_target_pos":   3,   # echoes the box spawn pose; kept for obs-shape parity with pick
    "object_target_quat":  4,
    "place_target_pos":    3,
    "place_target_quat":   4,
}


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
    for qname in ("object_target_quat", "place_target_quat"):
        env.extras[_CMD_KEY + qname][:, 0] = 1.0


_CARRY_BUFFER_CACHE: dict[str, torch.Tensor] | None = None


def _load_carry_buffer(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load and cache the carry terminal-state buffer (robot + held box) as device tensors.
    Args: path: buffer file path; device: target torch device.
    Returns: dict of tensors keyed by buffer field, cached across calls.
    """
    global _CARRY_BUFFER_CACHE
    if _CARRY_BUFFER_CACHE is None or _CARRY_BUFFER_CACHE.get("_path") != path:
        import numpy as np
        from pathlib import Path
        buffer_path = Path(path)
        if not buffer_path.exists():
            raise FileNotFoundError(
                f"carry buffer not found at {buffer_path.resolve()}. "
                f"Run: python -m src.collect_buffer --policy-module src.policy_carry --checkpoint <carry_ckpt> --out {buffer_path}"
            )
        arr = np.load(buffer_path)
        required = ("root_z", "quat_zy", "joint_pos", "joint_vel",
                    "box_pos_by", "box_quat_by", "box_lin_vel_by", "box_ang_vel_by")
        missing = [k for k in required if k not in arr.files]
        if missing:
            raise ValueError(
                f"carry buffer at {buffer_path} missing fields {missing}. "
                f"Re-generate with policy_carry's SNAPSHOT_ENTITIES = ['robot', 'box']."
            )
        _CARRY_BUFFER_CACHE = {k: torch.from_numpy(arr[k]).float().to(device) for k in arr.files}
        _CARRY_BUFFER_CACHE["_path"] = path  # type: ignore[assignment]
    return _CARRY_BUFFER_CACHE


# PLACED endpoint of the reverse curriculum: robot deeply squatting in front of the table with waist pitched forward and both wrists positioned at the box side-face anchors on the tabletop: the drop analogue of pick's SQUAT_GRIP_POSE. Copied verbatim from ../unitree-g1-humanoid-pick/src/policy_pick.py (the box side-face anchors at x~0.51, y~+/-0.13 are shared between pick and drop, so the same pose lands the wrists in the same spot on the placed box); the geometric sweep that produced these joint targets is documented there.
PLACED_JOINT_POS = {
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
    # Arms: strong shoulder_pitch forward-down, shoulder_roll opened outward to y~+/-0.13, elbow moderately bent (~63 deg) so the pose is visibly bent. Empirical FK sweep places the wrists at (0.502, +/-0.107, 0.627), a few cm from the box side-face anchors.
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
assert set(PLACED_JOINT_POS.keys()) == set(G1_JOINTS), (
    f"PLACED_JOINT_POS must cover all 29 G1 joints; missing "
    f"{set(G1_JOINTS) - set(PLACED_JOINT_POS.keys())}, extra "
    f"{set(PLACED_JOINT_POS.keys()) - set(G1_JOINTS)}."
)
# Pelvis at world (0.0, 0, 0.65): x=0 keeps pelvis behind the table front edge (0.26) so the moderately-flexed knees (at x~0.24) stay clear; z=0.65 puts feet flat with the current hip/knee/ankle flexion. Copied from pick's SQUAT_GRIP_ROOT_POS since the table geometry is shared.
PLACED_ROOT_POS = (0.0, 0.0, 0.65)
PLACED_ROOT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity: waist_pitch carries the trunk lean via joints

_CMD_PLACED_BOX_POS  = _CMD_KEY + "placed_box_pos"
_CMD_PLACED_BOX_QUAT = _CMD_KEY + "placed_box_quat"


def _sample_placed_box_pose(env, n, origins, scale, device):
    """Sample world-frame box pose on the tabletop with the same jitter reset_drop_command_and_object uses for the place target.
    Args: env: RL env; n: number of envs; origins: (n, 3) env origins; scale: curriculum range multiplier (pinned to 1.0 with the reverse curriculum); device: target device.
    Returns: (box_pos_w, box_quat_w) tensors shaped (n, 3) and (n, 4).
    """
    from mjlab.utils.lab_api.math import sample_uniform, quat_from_euler_xyz
    zero = torch.zeros(n, device=device)
    lx = sample_uniform(-PLACE_TARGET_X_HALF * scale, PLACE_TARGET_X_HALF * scale, (n,), device=device) + PLACE_TARGET_X_CENTER
    ly = sample_uniform(-PLACE_TARGET_Y_HALF * scale, PLACE_TARGET_Y_HALF * scale, (n,), device=device)
    box_pos_w = torch.stack([origins[:, 0] + lx, origins[:, 1] + ly, origins[:, 2] + BOX_REST_Z], dim=-1)
    p_yaw = sample_uniform(-PLACE_YAW_RANGE * scale, PLACE_YAW_RANGE * scale, (n,), device=device)
    box_quat_w = quat_from_euler_xyz(zero, zero, p_yaw)
    return box_pos_w, box_quat_w


def reset_from_carry_buffer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    buffer_path: str,
    xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    place_bias: float = 0.0,
    target_scale: float = 1.0,
) -> None:
    """Lerp robot and box between the PLACED endpoint (bb=1) and a carry-buffer sample (bb=0), stashing the sampled placed_box pose in env.extras so reset_drop_command_and_object locks the place-target onto the same pose.
    Args: env: RL env; env_ids: envs to reset (None = all); buffer_path: carry-terminal-states .npz path; xy_range / yaw_range: additive jitter scaled by (1-bb); asset_cfg: robot entity covering all 29 G1 joints in G1_JOINTS order; box_cfg: box entity; place_bias: curriculum weight in [0, 1]; target_scale: placed-box jitter multiplier, pinned to 1.0.
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

    bb = float(max(0.0, min(1.0, place_bias)))
    device = env.device
    origins = env.scene.env_origins[env_ids]

    # Carry-buffer endpoint (bb=0).
    buf = _load_carry_buffer(buffer_path, device)
    total = buf["root_z"].shape[0]
    idx = torch.randint(0, total, (n,), device=device)
    carry_root_z      = buf["root_z"][idx]                 # (n,) or (n, 1)
    carry_quat_zy     = buf["quat_zy"][idx]                # (n, 4)
    carry_joint_pos   = buf["joint_pos"][idx]              # (n, 29) in G1_JOINTS order
    carry_box_pos_by  = buf["box_pos_by"][idx]             # (n, 3), body-yaw frame
    carry_box_quat_by = buf["box_quat_by"][idx]            # (n, 4), body-yaw frame

    # PLACED endpoint (bb=1).
    placed_joint_pos = torch.tensor(
        [PLACED_JOINT_POS[j] for j in asset_cfg.joint_names],
        device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()
    placed_root_z_val = torch.full(carry_root_z.shape, PLACED_ROOT_POS[2], device=device)
    placed_root_xy_offset = torch.tensor(
        PLACED_ROOT_POS[:2], device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()
    placed_quat = torch.tensor(
        PLACED_ROOT_ROT, device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1).clone()
    placed_box_pos_w, placed_box_quat_w = _sample_placed_box_pose(env, n, origins, target_scale, device)

    # Stash the placed box pose so reset_drop_command_and_object binds place_target to it (at bb=1 the box endpoint already sits exactly on the target).
    if _CMD_PLACED_BOX_POS not in env.extras or env.extras[_CMD_PLACED_BOX_POS].shape[0] != env.num_envs:
        env.extras[_CMD_PLACED_BOX_POS]  = torch.zeros(env.num_envs, 3, device=device)
        env.extras[_CMD_PLACED_BOX_QUAT] = torch.zeros(env.num_envs, 4, device=device)
        env.extras[_CMD_PLACED_BOX_QUAT][:, 0] = 1.0
    env.extras[_CMD_PLACED_BOX_POS][env_ids]  = placed_box_pos_w
    env.extras[_CMD_PLACED_BOX_QUAT][env_ids] = placed_box_quat_w

    # Lerp robot state.
    joint_pos = bb * placed_joint_pos + (1.0 - bb) * carry_joint_pos
    root_z    = bb * placed_root_z_val + (1.0 - bb) * carry_root_z
    root_quat_base = bb * placed_quat + (1.0 - bb) * carry_quat_zy
    # Lerp of unit quats drifts off the sphere; renormalise so both endpoints stay valid rotations.
    root_quat_base = root_quat_base / torch.linalg.norm(
        root_quat_base, dim=-1, keepdim=True,
    ).clamp(min=1e-6)

    # Scale additive jitter by (1 - bb) so the placed endpoint spawns deterministically at PLACED_ROOT_POS and the carry endpoint keeps its DR range (matches pick's reset_from_walk_buffer).
    jitter_scale = 1.0 - bb
    xy_jitter  = sample_uniform(xy_range[0] * jitter_scale, xy_range[1] * jitter_scale, (n, 2), device=device)
    yaw_jitter = sample_uniform(yaw_range[0] * jitter_scale, yaw_range[1] * jitter_scale, (n,),   device=device)
    zero = torch.zeros_like(yaw_jitter)
    q_yaw = quat_from_euler_xyz(zero, zero, yaw_jitter)
    root_quat_w = quat_mul(q_yaw, root_quat_base)

    robot: Entity = env.scene[asset_cfg.name]
    default_root_state = robot.data.default_root_state[env_ids].clone()

    # Blend root_xy between the walk-default (from default_root_state, plus the +X spawn offset used at bb=0 to line the held box up over the tabletop) and the PLACED absolute offset from the env origin: (1-bb)*(default+offset) + bb*placed keeps both endpoints correct.
    carry_root_xy_offset = torch.tensor(
        [ROBOT_SPAWN_X_OFFSET, 0.0], device=device,
    ).unsqueeze(0).expand(n, -1)
    root_xy = ((1.0 - bb) * (default_root_state[:, 0:2] + carry_root_xy_offset)
               + bb * placed_root_xy_offset
               + xy_jitter
               + origins[:, 0:2])
    root_z_col = root_z if root_z.dim() == 2 else root_z.unsqueeze(-1)
    root_z_w = root_z_col + origins[:, 2:3]
    root_pos_w = torch.cat([root_xy, root_z_w], dim=-1)

    # Lerp box state.
    carry_box_pos_w  = root_pos_w + quat_apply(q_yaw, carry_box_pos_by)
    carry_box_quat_w = quat_mul(q_yaw, carry_box_quat_by)
    box_pos_w = bb * placed_box_pos_w + (1.0 - bb) * carry_box_pos_w
    box_quat_w = bb * placed_box_quat_w + (1.0 - bb) * carry_box_quat_w
    box_quat_w = box_quat_w / torch.linalg.norm(
        box_quat_w, dim=-1, keepdim=True,
    ).clamp(min=1e-6)

    # Physics writes (velocities zeroed on both entities).
    robot.write_root_link_pose_to_sim(
        torch.cat([root_pos_w, root_quat_w], dim=-1), env_ids=env_ids,
    )
    robot.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=device)
    robot.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos),
        env_ids=env_ids, joint_ids=joint_ids,
    )
    box: Entity = env.scene[box_cfg.name]
    box.write_root_link_pose_to_sim(
        torch.cat([box_pos_w, box_quat_w], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )


def reset_robot_and_box_held(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    box_cfg: SceneEntityCfg,
) -> None:
    """Place standing robot and held box in one consistent write (mirrors carry_normal).
    Args: env: RL env; env_ids: envs to reset (None = all); box_cfg: box entity.
    Returns: none.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    device = env.device
    yaw   = torch.empty(n, device=device).uniform_(*ROBOT_SPAWN_YAW_RANGE)
    x_jit = torch.empty(n, device=device).uniform_(-ROBOT_SPAWN_XY_JITTER, ROBOT_SPAWN_XY_JITTER)
    y_jit = torch.empty(n, device=device).uniform_(-ROBOT_SPAWN_XY_JITTER, ROBOT_SPAWN_XY_JITTER)

    origins = env.scene.env_origins[env_ids]
    robot_x = origins[:, 0] + ROBOT_SPAWN_X_OFFSET + x_jit
    robot_y = origins[:, 1] + y_jit
    robot_z = STANDING_ROOT_Z + origins[:, 2]

    half = yaw * 0.5
    robot_quat = torch.stack([
        torch.cos(half),
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        torch.sin(half),
    ], dim=-1)
    robot_pos = torch.stack([robot_x, robot_y, robot_z], dim=-1)
    robot: Entity = env.scene["robot"]
    robot.write_root_link_pose_to_sim(
        torch.cat([robot_pos, robot_quat], dim=-1), env_ids=env_ids,
    )
    robot.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )

    cx = torch.cos(yaw)
    sy = torch.sin(yaw)
    box_jx = torch.empty(n, device=device).uniform_(-BOX_SPAWN_XY_JITTER, BOX_SPAWN_XY_JITTER)
    box_jy = torch.empty(n, device=device).uniform_(-BOX_SPAWN_XY_JITTER, BOX_SPAWN_XY_JITTER)
    box_x = robot_x + BOX_HOLD_X_OFFSET * cx + box_jx
    box_y = robot_y + BOX_HOLD_X_OFFSET * sy + box_jy
    box_z = STANDING_ROOT_Z + BOX_HOLD_Z_ABOVE_ROOT + origins[:, 2]

    box_quat = torch.zeros(n, 4, device=device)
    box_quat[:, 0] = 1.0
    box_pos = torch.stack([box_x, box_y, box_z], dim=-1)
    box: Entity = env.scene[box_cfg.name]
    box.write_root_link_pose_to_sim(
        torch.cat([box_pos, box_quat], dim=-1), env_ids=env_ids,
    )
    box.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids,
    )


def reset_arms_to_holding(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Snap the 14 arm joints to HOLDING_ARM_POSE so the hands press on the box side faces from tick 0.
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


def reset_drop_command_and_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale: float = 1.0,
) -> None:
    """Snapshot the initial box z and sample the per-episode place target and grasp offsets.
    Args: env: RL env; env_ids: envs to reset (all if None); scale: curriculum range multiplier in [0, 1].
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

    box: Entity = env.scene["box"]
    box_pos_w  = box.data.root_link_pos_w[env_ids]
    box_quat_w = box.data.root_link_quat_w[env_ids]

    # object_target echoes the held-box spawn pose; kept for obs-space parity with the pick task.
    env.extras[_CMD_KEY + "object_target_pos"][env_ids]  = box_pos_w
    env.extras[_CMD_KEY + "object_target_quat"][env_ids] = box_quat_w

    # Snapshot per-env initial box z so hand_lost_grip can detect early drops.
    if "drop_initial_box_z" not in env.extras or env.extras["drop_initial_box_z"].shape[0] != env.num_envs:
        env.extras["drop_initial_box_z"] = torch.zeros(env.num_envs, device=device)
    env.extras["drop_initial_box_z"][env_ids] = box_pos_w[:, 2]

    jitter = GRASP_JITTER * scale
    gdx = sample_uniform(-jitter, jitter, (n,), device=device)
    gdy = sample_uniform(-jitter, jitter, (n,), device=device)
    gdz = sample_uniform(-jitter, jitter, (n,), device=device)
    grasp_left = torch.stack([
        gdx,
        torch.full((n,), GRASP_NOMINAL_Y, device=device) + gdy,
        torch.full((n,), GRASP_NOMINAL_Z, device=device) + gdz,
    ], dim=-1)
    mirror = torch.tensor([1.0, -1.0, 1.0], device=device)
    grasp_right = grasp_left * mirror
    env.extras[_CMD_KEY + "grasp_offset_left"][env_ids]  = grasp_left
    env.extras[_CMD_KEY + "grasp_offset_right"][env_ids] = grasp_right

    # Place target is bound to the placed_box pose stashed by reset_from_carry_buffer so bb=1 envs start with box exactly on the target (hand_release / box_flat_at_target fire from step 0). At bb=0 the sampled placed pose still serves as the "put it here" target the held box must reach: same sample either way, so the lerp of the box between held and placed slides toward the exact spot the reward is checking.
    placed_pos  = env.extras.get(_CMD_PLACED_BOX_POS)
    placed_quat = env.extras.get(_CMD_PLACED_BOX_QUAT)
    if placed_pos is None or placed_quat is None:
        # Fallback: sample directly (e.g. reset_from_carry_buffer skipped for some reason).
        lx = sample_uniform(-PLACE_TARGET_X_HALF * scale, PLACE_TARGET_X_HALF * scale, (n,), device=device) + PLACE_TARGET_X_CENTER
        ly = sample_uniform(-PLACE_TARGET_Y_HALF * scale, PLACE_TARGET_Y_HALF * scale, (n,), device=device)
        place_pos_w = torch.stack([
            origins[:, 0] + lx,
            origins[:, 1] + ly,
            origins[:, 2] + BOX_REST_Z,
        ], dim=-1)
        p_yaw = sample_uniform(-PLACE_YAW_RANGE * scale, PLACE_YAW_RANGE * scale, (n,), device=device)
        place_quat_w = quat_from_euler_xyz(zero, zero, p_yaw)
    else:
        place_pos_w  = placed_pos[env_ids]
        place_quat_w = placed_quat[env_ids]

    env.extras[_CMD_KEY + "place_target_pos"][env_ids]  = place_pos_w
    env.extras[_CMD_KEY + "place_target_quat"][env_ids] = place_quat_w


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
    """Object-target position (box spawn pose) in the robot base frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "object_target_pos"])


def cmd_object_target_quat_obs(env):
    """Object-target orientation quaternion (box spawn pose) in world frame.
    Args: env: the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "object_target_quat"]


def cmd_place_target_pos_obs(env):
    """Place-target position expressed in the robot base frame.
    Args: env: the RL env.
    Returns: (num_envs, 3) tensor.
    """
    _ensure_cmd_buffers(env)
    return _pos_in_base_frame(env, env.extras[_CMD_KEY + "place_target_pos"])


def cmd_place_target_quat_obs(env):
    """Place-target orientation quaternion in world frame.
    Args: env: the RL env.
    Returns: (num_envs, 4) tensor.
    """
    _ensure_cmd_buffers(env)
    return env.extras[_CMD_KEY + "place_target_quat"]


def _desired_grasp_pos_w(env, box_cfg, side: str) -> torch.Tensor:
    """Return world-frame position of the commanded grasp point for one hand.
    Args: env: the RL env; box_cfg: box entity config; side: "left" or "right".
    Returns: (num_envs, 3) tensor of grasp points on the box side.
    """
    _ensure_cmd_buffers(env)
    box_w    = _entity_pos(env, box_cfg)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    qw, qx, qy, qz = box_quat.unbind(dim=-1)
    box_yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
    cy, sy = torch.cos(box_yaw), torch.sin(box_yaw)
    off = env.extras[_CMD_KEY + f"grasp_offset_{side}"]
    ox, oy, oz = off.unbind(dim=-1)
    wx = cy * ox - sy * oy
    wy = sy * ox + cy * oy
    return box_w + torch.stack([wx, wy, oz], dim=-1)


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
    """Binary gate: 1 if torso upright (<=25 deg tilt) AND both feet in contact.
    Args: env: the RL env; trunk_cfg: trunk entity; sensor_name: feet-contact sensor.
    Returns: (num_envs,) tensor of 0/1 gate values.
    """
    entity = env.scene[trunk_cfg.name]
    up_align = -entity.data.projected_gravity_b[:, 2]
    upright = up_align > 0.9
    sensor = env.scene[sensor_name]
    both_feet = (sensor.data.found > 0).sum(dim=1) >= 2
    return (upright & both_feet).float()


def hand_touching_box_gate(env, sensor_name: str = "hand_box_contact",
                            force_threshold: float = 1.0):
    """Binary gate: 1 if any wrist-vs-box contact force exceeds `force_threshold`.
    Args: env: RL env; sensor_name: narrowed hand-vs-box ContactSensor; force_threshold: N above which the hand counts as still holding.
    Returns: (num_envs,) tensor of 0/1 gate values.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1).float()
    return (sensor.data.found > 0).any(dim=-1).float()


def hand_placement_gate(env, box_cfg,
                         sensor_name: str = "hand_box_contact",
                         force_threshold: float = 1.0,
                         air_margin: float = 0.03):
    """Context-aware placement gate: reward careful descent while airborne, reward release once the box has landed.
    Args: env: RL env; box_cfg: box entity; sensor_name: hand-vs-box ContactSensor; force_threshold: N above which the hand counts as still holding; air_margin: how close the box centre must be to BOX_REST_Z to count as landed.
    Returns: (num_envs,) tensor of 0/1 gate values.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    box_on_table = box_z < BOX_REST_Z + air_margin
    hand_touching = hand_touching_box_gate(env, sensor_name, force_threshold) > 0.5
    return torch.where(box_on_table, ~hand_touching, hand_touching).float()


def box_impact_velocity_penalty(env, box_cfg, gate_z: float = 0.10,
                                 w_lin: float = 1.0, w_ang: float = 0.0):
    """Penalise box linear and angular speed squared as it approaches the tabletop, ramped over the last gate_z metres.
    Args: env: RL env; box_cfg: box entity; gate_z: height above BOX_REST_Z where the ramp begins; w_lin: weight on box linear speed squared; w_ang: weight on box angular speed squared.
    Returns: (num_envs,) tensor.
    """
    # Low speed near the tabletop requires the hands to accompany the box, shaping a controlled descent without any release gate; w_ang catches the flick-and-let-go pattern on release.
    box = env.scene[box_cfg.name]
    box_z = box.data.root_link_pos_w[:, 2]
    box_v = box.data.root_link_lin_vel_w
    box_w = box.data.root_link_ang_vel_w
    height_above_rest = box_z - BOX_REST_Z
    gate = torch.clamp(1.0 - height_above_rest / gate_z, min=0.0, max=1.0)
    lin_sq = (box_v ** 2).sum(dim=-1)
    ang_sq = (box_w ** 2).sum(dim=-1)
    return gate * (w_lin * lin_sq + w_ang * ang_sq)


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
        return v_sq

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
    """Squared L2 penalty on action second-order difference (jerk analogue).
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
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
    """Squared L2 on joint velocities.
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    v = env.scene["robot"].data.joint_vel
    return torch.sum(v * v, dim=-1)


def dof_acc_l2(env):
    """Squared L2 on joint accelerations.
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    a = env.scene["robot"].data.joint_acc
    return torch.sum(a * a, dim=-1)


def dof_torque_l2(env):
    """Squared L2 on joint torques (energy proxy).
    Args: env: the RL env.
    Returns: (num_envs,) tensor.
    """
    tau = env.scene["robot"].data.qfrc_actuator
    return torch.sum(tau * tau, dim=-1)


def grip_force_reward(env, sensor_name: str = "hand_box_contact",
                      target_force: float = 5.0,
                      box_cfg: SceneEntityCfg | None = None,
                      air_margin: float = 0.03,
                      air_ramp: float = 0.05):
    """Reward positive normal force between robot and box, saturating at target_force and smoothly decayed as the box approaches the tabletop.
    Args: env: RL env; sensor_name: ContactSensor name; target_force: N at which reward saturates; box_cfg: box entity (enables the airborne ramp); air_margin: clearance above BOX_REST_Z at which grip_force fully zeros; air_ramp: vertical span over which grip_force ramps from full to zero.
    Returns: (num_envs,) tensor in [0, 1].
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        force_sum = force_mag.sum(dim=-1)
    else:
        force_sum = sensor.data.found.sum(dim=-1).float()
    base = torch.clamp(force_sum / target_force, 0.0, 1.0)
    if box_cfg is None:
        return base
    box_z = _entity_pos(env, box_cfg)[:, 2]
    # Smooth ramp instead of a hard step at BOX_REST_Z + air_margin: the previous cliff created a "safe hover" zone (~0.70 m) paying full 8/step, wiped out in one tick at landing.
    airborne = torch.clamp(
        (box_z - (BOX_REST_Z + air_margin)) / air_ramp, 0.0, 1.0
    )
    return base * airborne


def hand_approach_velocity_penalty(env, left_hand_cfg, right_hand_cfg, box_cfg,
                                    gate_dist: float = 0.20):
    """Penalise hand linear velocity when close to the box (forces gentle descent).
    Args: env: the RL env; left_hand_cfg: left hand entity; right_hand_cfg: right hand entity; box_cfg: box entity; gate_dist: hand-close activation threshold in metres.
    Returns: (num_envs,) tensor of speed sum masked by the gate.
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
    """Weighted sum of per-hand palm-inward alignment rewards.
    Args: env: RL env; left/right_hand_cfg: hand entities; per-side palm/target axes; w_left/w_right: per-side weights folded in.
    Returns: (num_envs,) tensor.
    """
    lr = palm_inward_reward(env, left_hand_cfg,  left_palm_axis_local,  left_target_axis_body)
    rr = palm_inward_reward(env, right_hand_cfg, right_palm_axis_local, right_target_axis_body)
    return w_left * lr + w_right * rr


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
                                  w_chest: float = 1.5):
    """Fused carry-style box pose: upright + front aligned + at chest height.
    Args: env: RL env; box_cfg: box entity; w_upright/w_front/w_chest: per-signal weights folded in.
    Returns: (num_envs,) tensor.
    """
    return (w_upright * box_upright_reward(env, box_cfg)
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


def box_at_place_target_pos_reward(
    env, box_cfg,
    sigma_xy: float = 0.30,
    sigma_z_above: float = 0.30,
    sigma_z_below: float = 0.05,
):
    """Additive tracking reward with broad-above and narrow-below asymmetry in z (pull down from chest height, punish dipping below the table).
    Args: env: RL env; box_cfg: box entity; sigma_xy: xy gaussian width; sigma_z_above: z width when box above target; sigma_z_below: z width when box below target.
    Returns: (num_envs,) tensor in (0, 1].
    """
    _ensure_cmd_buffers(env)
    box = _entity_pos(env, box_cfg)
    tgt = env.extras[_CMD_KEY + "place_target_pos"]
    xy_dist = torch.linalg.norm(box[:, :2] - tgt[:, :2], dim=-1)
    z_diff  = box[:, 2] - tgt[:, 2]
    sigma_z = torch.where(z_diff > 0,
                          torch.full_like(z_diff, sigma_z_above),
                          torch.full_like(z_diff, sigma_z_below))
    xy_r = torch.exp(-(xy_dist ** 2) / (2.0 * sigma_xy ** 2))
    z_r  = torch.exp(-(z_diff  ** 2) / (2.0 * sigma_z  ** 2))
    return 0.5 * (xy_r + z_r)


def descent_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    hold_dist: float = 0.10,
    xy_gate_dist: float = 0.15,
):
    """Linear reward for descent progress toward the target, gated on hands-near-anchors and box-over-target xy.
    Args: env: RL env; box_cfg: box entity; left_hand_cfg: left hand; right_hand_cfg: right hand; hold_dist: max hand-to-anchor distance for the holding gate; xy_gate_dist: max box-to-target xy distance for the over-target gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    # Gate uses distance to the grasp anchor (side face ~13 cm from centre), not centre: a centre-distance gate with hold_dist=0.10 never fires (2026-07-25 bug). xy_gate (2026-08-11) blocks the "lower beside body" farm.
    _ensure_cmd_buffers(env)
    box = _entity_pos(env, box_cfg)
    tgt = env.extras[_CMD_KEY + "place_target_pos"]
    z_span = max(CHEST_Z - tgt[0, 2].item(), 1e-3)   # scalar normaliser
    descent = torch.clamp((CHEST_Z - box[:, 2]) / z_span, 0.0, 1.0)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    holding = ((dl < hold_dist) & (dr < hold_dist)).float()
    xy_dist = torch.linalg.norm(box[:, :2] - tgt[:, :2], dim=-1)
    over_target = (xy_dist < xy_gate_dist).float()
    return descent * holding * over_target


def release_shoulder_open_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    trigger_gap: float = 0.07,
    spread_target: float = 0.6,
    hand_close_dist: float = 0.15,
    xy_gate_dist: float = 0.07,
):
    """Reward shoulder abduction once the box is about to land, gated on still-holding and box-over-target-xy to block the "release early" cheat.
    Args: env: RL env; box_cfg: box entity; left_hand_cfg / right_hand_cfg: hand entities; trigger_gap: vertical clearance below which the incentive kicks in; spread_target: shoulder-roll magnitude (rad) at which the reward saturates; hand_close_dist: max hand-to-anchor distance for the holding gate; xy_gate_dist: max box-to-place-target xy distance for the xy_close gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box_pos = _entity_pos(env, box_cfg)
    clearance = (box_pos[:, 2] - BOX_HALF) - TABLE_TOP_Z
    near_table = (clearance <= trigger_gap).float()

    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    holding = ((dl < hand_close_dist) & (dr < hand_close_dist)).float()

    place_target = env.extras[_CMD_KEY + "place_target_pos"]
    xy_dist = torch.linalg.norm(box_pos[:, :2] - place_target[:, :2], dim=-1)
    xy_close = (xy_dist <= xy_gate_dist).float()

    robot = env.scene["robot"]
    if not hasattr(env, "_shoulder_roll_idx"):
        names = robot.joint_names
        env._shoulder_roll_idx = (
            names.index("left_shoulder_roll_joint"),
            names.index("right_shoulder_roll_joint"),
        )
    q = robot.data.joint_pos
    lsr = q[:, env._shoulder_roll_idx[0]].abs()
    rsr = q[:, env._shoulder_roll_idx[1]].abs()
    spread = torch.clamp((lsr + rsr) / (2.0 * spread_target), 0.0, 1.0)
    return near_table * holding * xy_close * spread


def hand_placement_bundle_reward(env, box_cfg, left_hand_cfg, right_hand_cfg,
                                  trunk_cfg, sensor_name,
                                  w_hold_sharp: float = 15.0,
                                  w_hold_broad: float = 8.0,
                                  broad_near_dist: float = 0.05,
                                  broad_kernel_std: float = 0.05,
                                  air_margin: float = 0.03,
                                  air_ramp: float = 0.05):
    """Fused hold_box (gated sharp) + hands_on_anchor (ungated broad), airborne-gated to zero as the box lands.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hands; trunk_cfg: trunk; sensor_name: feet sensor; w_hold_sharp / w_hold_broad: per-signal weights; broad_near_dist / broad_kernel_std: broad-kernel geometry; air_margin: clearance at which airborne ramp reaches zero; air_ramp: vertical span over which the ramp saturates.
    Returns: (num_envs,) tensor.
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    hold_sharp = torch.exp(-8.0 * (dl + dr)) * stability_gate(env, trunk_cfg, sensor_name)
    excess_l = torch.clamp(dl - broad_near_dist, min=0.0)
    excess_r = torch.clamp(dr - broad_near_dist, min=0.0)
    hold_broad = torch.exp(-(excess_l ** 2 + excess_r ** 2) / (2.0 * broad_kernel_std ** 2))
    box_z = _entity_pos(env, box_cfg)[:, 2]
    airborne = torch.clamp((box_z - (BOX_REST_Z + air_margin)) / air_ramp, 0.0, 1.0)
    return (w_hold_sharp * hold_sharp + w_hold_broad * hold_broad) * airborne


def hand_release_reward(env, box_cfg, left_hand_cfg, right_hand_cfg,
                        release_dist: float = 0.15,
                        air_margin: float = 0.03):
    """Positive reward once the box has landed and the hands have moved away from the grasp anchors: pulls the arms off the placed box.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; release_dist: hand-to-anchor distance at which the reward saturates; air_margin: clearance above BOX_REST_Z below which the box counts as landed.
    Returns: (num_envs,) tensor in [0, 1].
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    landed = torch.clamp(1.0 - (box_z - BOX_REST_Z) / air_margin, 0.0, 1.0)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    away = torch.clamp(torch.minimum(dl, dr) / release_dist, 0.0, 1.0)
    return landed * away


def arm_pose_walk_ready_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg, arm_cfg,
    target_values: tuple[float, ...] = WALK_READY_ARM_TARGET,
    sigma: float = 0.30,
    air_margin: float = 0.03,
    release_dist: float = 0.15,
):
    """Shape the terminal arm pose toward G1_HOME_JOINT_POS so the drop's snapshot feeds cleanly into the walk policy; double-gated on landed and released.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities for the release gate; arm_cfg: SceneEntityCfg selecting the 14 arm joints in HOLDING_ARM_JOINTS order; target_values: per-joint targets aligned with arm_cfg.joint_ids; sigma: Gaussian bandwidth (rad) on the RMS joint error; air_margin: clearance above BOX_REST_Z below which the box counts as landed; release_dist: min hand-to-anchor distance for the release gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    landed = (box_z < BOX_REST_Z + air_margin)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    released = (dl > release_dist) & (dr > release_dist)
    gate = (landed & released).float()

    robot = env.scene["robot"]
    joints = robot.data.joint_pos[:, arm_cfg.joint_ids]
    if not hasattr(env, "_walk_ready_arm_target_cache"):
        env._walk_ready_arm_target_cache = torch.tensor(
            target_values, device=env.device, dtype=joints.dtype,
        )
    err = joints - env._walk_ready_arm_target_cache
    mean_sq_err = (err * err).mean(dim=-1)
    return gate * torch.exp(-mean_sq_err / (2.0 * sigma * sigma))


def box_landed_flat_bonus(env, box_cfg,
                           air_margin: float = 0.03,
                           up_threshold: float = 0.95):
    """Small step-wise reward while the box is landed AND flat, independent of hand state.
    Args: env: RL env; box_cfg: box entity; air_margin: clearance below which the box counts as landed; up_threshold: box up_z above which it counts as flat.
    Returns: (num_envs,) tensor in {0, 1}.
    """
    # No hand-contact or release gate: let the policy prove a landing and see a positive gradient without also releasing simultaneously (the 2026-08-10 run had fully-gated rewards that never fired).
    box_z = _entity_pos(env, box_cfg)[:, 2]
    landed = (box_z < BOX_REST_Z + air_margin).float()
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    x, y = box_quat[:, 1], box_quat[:, 2]
    up_z = torch.clamp(1.0 - 2.0 * (x * x + y * y), 0.0, 1.0)
    flat = (up_z > up_threshold).float()
    return landed * flat


def hands_on_anchor_reward(env, box_cfg, left_hand_cfg, right_hand_cfg,
                            near_dist: float = 0.05,
                            kernel_std: float = 0.05):
    """Ungated Gaussian on hand-to-anchor distance, providing a dense first-frame signal that survives shape-mismatched warm-starts.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; near_dist: target hand-to-anchor distance for full credit; kernel_std: Gaussian scale for the smooth decay.
    Returns: (num_envs,) tensor in (0, 1].
    """
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    excess_l = torch.clamp(dl - near_dist, min=0.0)
    excess_r = torch.clamp(dr - near_dist, min=0.0)
    return torch.exp(-(excess_l ** 2 + excess_r ** 2) / (2.0 * kernel_std ** 2))


def hand_lost_grip_penalty(env, box_cfg):
    """Per-step penalty when the box falls far from its spawn z and is not on the table within the first 60 policy steps.
    Args: env: the RL env; box_cfg: box entity.
    Returns: (num_envs,) 0/1 tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    init_z = env.extras.get("drop_initial_box_z")
    if init_z is None:
        return torch.zeros(env.num_envs, device=env.device)
    fell_far = (box_z < init_z - 0.15).float()
    not_on_table = (box_z < TABLE_TOP_Z + 0.05).float()
    early = (env.episode_length_buf < 60).float()
    return fell_far * not_on_table * early


def hand_lost_grip(env, box_cfg):
    """Termination flag mirroring ``hand_lost_grip_penalty``.
    Args: env: the RL env; box_cfg: box entity.
    Returns: (num_envs,) bool tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    init_z = env.extras.get("drop_initial_box_z")
    if init_z is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fell_far = box_z < init_z - 0.15
    not_on_table = box_z < TABLE_TOP_Z + 0.05
    early = env.episode_length_buf < 60
    return fell_far & not_on_table & early


def body_touched_table(env, sensor_name: str = "body_table_contact",
                        force_threshold: float = 5.0):
    """Termination on physics contact between monitored body geoms and the table: catches slumping, thigh-on-edge, and collapse-onto-box.
    Args: env: RL env; sensor_name: ContactSensor name; force_threshold: N above which contact counts (used only when the sensor exposes force).
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


def hand_lost_anchor_airborne(env, box_cfg, left_hand_cfg, right_hand_cfg,
                               anchor_dist_threshold: float = 0.20,
                               air_margin: float = 0.03):
    """Termination when either hand strays far from its grasp anchor while the box is still airborne: anchor-distance twin of box_released_airborne that catches grazing slip contacts.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; anchor_dist_threshold: hand-to-anchor distance beyond which the grip counts as lost; air_margin: how close the box centre must be to BOX_REST_Z for release to be legal.
    Returns: (num_envs,) bool tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    airborne = box_z > BOX_REST_Z + air_margin
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    slipped = torch.minimum(dl, dr) > anchor_dist_threshold
    return slipped & airborne


def box_released_airborne(env, box_cfg,
                           hand_box_sensor_name: str = "hand_box_contact",
                           hand_force_threshold: float = 1.0,
                           air_margin: float = 0.03):
    """Termination when the box loses hand contact while still airborne: forces "place with the hands until it lands" instead of releasing mid-transit.
    Args: env: RL env; box_cfg: box entity; hand_box_sensor_name: narrowed hand-vs-box ContactSensor; hand_force_threshold: N above which the hand counts as still holding; air_margin: how close the box centre must be to its rest height for release to be legal.
    Returns: (num_envs,) bool tensor.
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    box_on_table = box_z < BOX_REST_Z + air_margin
    sensor = env.scene[hand_box_sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        hand_touching = (force_mag > hand_force_threshold).any(dim=-1)
    else:
        hand_touching = (sensor.data.found > 0).any(dim=-1)
    return (~hand_touching) & (~box_on_table)


def body_touched_box(env, sensor_name: str = "body_box_contact",
                      force_threshold: float = 5.0):
    """Termination on physics contact between any non-hand body geom and the box: catches leg-on-box that the geometric box_touched_body test misses at shank origin ~0.155 m.
    Args: env: RL env; sensor_name: ContactSensor name; force_threshold: N above which contact counts.
    Returns: (num_envs,) bool tensor.
    """
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
        force_mag = torch.linalg.norm(sensor.data.force, dim=-1)
        return (force_mag > force_threshold).any(dim=-1)
    return (sensor.data.found > 0).any(dim=-1)


def stability_bundle_reward(env, trunk_cfg, pelvis_cfg, sensor_name,
                             w_upright: float = 10.0,
                             w_upright_pelvis: float = 5.0,
                             w_feet: float = 5.0):
    """Fused torso-upright + pelvis-upright + feet-on-ground stability.
    Args: env: RL env; trunk_cfg: trunk entity; pelvis_cfg: pelvis entity; sensor_name: feet contact sensor; w_upright/w_upright_pelvis/w_feet: per-signal weights folded in.
    Returns: (num_envs,) tensor.
    """
    return (w_upright        * upright_torso_reward(env, trunk_cfg)
          + w_upright_pelvis * upright_torso_reward(env, pelvis_cfg)
          + w_feet           * feet_on_ground_reward(env, sensor_name))


def box_unsupported_bundle_penalty(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    vel_threshold: float = 0.30,
    hand_far_dist: float = 0.15,
    z_floor: float = 0.10,
    w_free_fall: float = -25.0,
    w_hand_lost_grip: float = -50.0,
    w_box_jammed: float = -10.0,
):
    """Fused box-unsupported penalty: free_fall + hand_lost_grip + box_jammed sharing dl, dr, box_z.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; vel_threshold: free-fall vz threshold; hand_far_dist: hand-anchor distance beyond which hand doesn't count as supporting; z_floor: box_jammed depth clamp; w_free_fall/w_hand_lost_grip/w_box_jammed: per-signal weights folded in.
    Returns: (num_envs,) tensor.
    """
    box_pos = _entity_pos(env, box_cfg)
    box_z = box_pos[:, 2]
    box_v = env.scene[box_cfg.name].data.root_link_lin_vel_w
    vz = torch.nan_to_num(box_v[:, 2], nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)

    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    unsupported = ((torch.minimum(dl, dr)) > hand_far_dist).float()

    falling = (vz < -vel_threshold).float()
    ff_signal = falling * unsupported * (-vz)

    init_z = env.extras.get("drop_initial_box_z")
    if init_z is None:
        hlg_signal = torch.zeros_like(box_z)
    else:
        fell_far = (box_z < init_z - 0.15).float()
        not_on_table = (box_z < TABLE_TOP_Z + 0.05).float()
        early = (env.episode_length_buf < 60).float()
        hlg_signal = fell_far * not_on_table * early

    depth = torch.clamp(TABLE_TOP_Z - box_z, min=0.0, max=z_floor)
    bj_signal = depth * unsupported

    return (w_free_fall      * ff_signal
          + w_hand_lost_grip * hlg_signal
          + w_box_jammed     * bj_signal)


def box_jammed_penalty(env, box_cfg, left_hand_cfg, right_hand_cfg,
                       hand_far_dist: float = 0.15,
                       z_floor: float = 0.10):
    """Per-step penalty when the box is jammed below the tabletop and the hands are not supporting it, scaled by depth below the top.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; hand_far_dist: hand-to-anchor distance beyond which the hand does not count as supporting; z_floor: clamp on depth to bound the per-step magnitude.
    Returns: (num_envs,) tensor >= 0 (weighted negatively at the reward config).
    """
    box_z = _entity_pos(env, box_cfg)[:, 2]
    depth = torch.clamp(TABLE_TOP_Z - box_z, min=0.0, max=z_floor)
    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    unsupported = ((torch.minimum(dl, dr)) > hand_far_dist).float()
    return depth * unsupported


def walk_to_target_bundle_reward(env, sigma: float = 1.0,
                                  w_target: float = 2.0, w_progress: float = 1.0,
                                  box_cfg: SceneEntityCfg | None = None,
                                  air_margin: float = 0.03,
                                  air_ramp: float = 0.05):
    """Fused Gaussian-on-distance + velocity-along-direction to the place target, sharing the horizontal delta vector.
    Args: env: RL env; sigma: Gaussian bandwidth; w_target / w_progress: per-signal weights; box_cfg: box entity (enables airborne gate; omit for the pre-2026-08-11 always-on behaviour); air_margin: clearance at which the gate reaches zero; air_ramp: vertical span over which the gate ramps back to one.
    Returns: (num_envs,) tensor.
    """
    # Airborne ramp mirrors hand_placement: zeroing the approach signal on landing blocks the "one extra step until it bumps the table" farm.
    _ensure_cmd_buffers(env)
    target = env.extras[_CMD_KEY + "place_target_pos"]
    robot: Entity = env.scene["robot"]
    root_pos = robot.data.root_link_pos_w
    delta = target[:, :2] - root_pos[:, :2]
    dist_full = torch.linalg.norm(delta, dim=-1)
    target_r = torch.exp(-(dist_full ** 2) / (2.0 * sigma * sigma))
    dist_clamped = dist_full.clamp(min=1e-4).unsqueeze(-1)
    direction = delta / dist_clamped
    lin_vel_w = robot.data.root_link_lin_vel_w[:, :2]
    v_align = torch.sum(direction * lin_vel_w, dim=-1)
    progress_r = torch.clamp(v_align, -1.5, 1.5)
    bundle = w_target * target_r + w_progress * progress_r
    if box_cfg is None:
        return bundle
    box_z = _entity_pos(env, box_cfg)[:, 2]
    airborne = torch.clamp((box_z - (BOX_REST_Z + air_margin)) / air_ramp, 0.0, 1.0)
    return bundle * airborne


def walk_to_place_target_reward(env, sigma: float = 1.0):
    """Gaussian shaping reward on horizontal robot-to-place-target distance.
    Args: env: RL env; sigma: Gaussian bandwidth in meters.
    Returns: (num_envs,) reward in (0, 1].
    """
    _ensure_cmd_buffers(env)
    target = env.extras[_CMD_KEY + "place_target_pos"]
    root_pos = env.scene["robot"].data.root_link_pos_w
    dist = torch.linalg.norm(target[:, :2] - root_pos[:, :2], dim=-1)
    return torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))


def walk_to_place_progress_reward(env):
    """Base linear velocity component along the direction to the place target, clamped.
    Args: env: RL env.
    Returns: (num_envs,) reward clamped to [-1.5, 1.5].
    """
    _ensure_cmd_buffers(env)
    target = env.extras[_CMD_KEY + "place_target_pos"]
    robot: Entity = env.scene["robot"]
    root_pos = robot.data.root_link_pos_w
    delta = target[:, :2] - root_pos[:, :2]
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp(min=1e-4)
    direction = delta / dist
    lin_vel_w = robot.data.root_link_lin_vel_w[:, :2]
    v_align = torch.sum(direction * lin_vel_w, dim=-1)
    return torch.clamp(v_align, -1.5, 1.5)


def free_fall_penalty(env, box_cfg, left_hand_cfg, right_hand_cfg,
                      vel_threshold: float = 0.30,
                      hand_far_dist: float = 0.15):
    """Penalise the box being in free fall while the hands are away from it, scaled by downward speed magnitude.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; vel_threshold: downward speed threshold (m/s); hand_far_dist: max hand-to-anchor distance to still count as supported.
    Returns: (num_envs,) tensor of |vz| when the gate is active, else 0.
    """
    box_v = env.scene[box_cfg.name].data.root_link_lin_vel_w
    vz = torch.nan_to_num(box_v[:, 2], nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)
    falling = (vz < -vel_threshold).float()

    dl = torch.linalg.norm(_entity_pos(env, left_hand_cfg)  - _desired_grasp_pos_w(env, box_cfg, "left"),  dim=-1)
    dr = torch.linalg.norm(_entity_pos(env, right_hand_cfg) - _desired_grasp_pos_w(env, box_cfg, "right"), dim=-1)
    unsupported = ((torch.minimum(dl, dr)) > hand_far_dist).float()

    return falling * unsupported * (-vz)


def box_at_place_target_quat_reward(env, box_cfg):
    """SO(3) similarity |q_box . q_target|^2, invariant to quaternion double cover.
    Args: env: the RL env; box_cfg: box entity.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    tgt_quat = env.extras[_CMD_KEY + "place_target_quat"]
    q_dot = (box_quat * tgt_quat).sum(dim=-1)
    return q_dot * q_dot


def box_flat_at_target_reward(env, box_cfg, near_target_dist: float = 0.20):
    """Reward the box being flat (local z aligned with world +z) near the place target, using up_z**4 for a near-binary flat/not-flat shape.
    Args: env: RL env; box_cfg: box entity; near_target_dist: activation ramp.
    Returns: (num_envs,) tensor in [0, 1].
    """
    _ensure_cmd_buffers(env)
    box = _entity_pos(env, box_cfg)
    tgt = env.extras[_CMD_KEY + "place_target_pos"]
    dist = torch.linalg.norm(box - tgt, dim=-1)
    near_gate = torch.clamp(1.0 - dist / near_target_dist, 0.0, 1.0)

    box_quat = env.scene[box_cfg.name].data.root_link_quat_w
    x, y = box_quat[:, 1], box_quat[:, 2]
    up_z = torch.clamp(1.0 - 2.0 * (x * x + y * y), 0.0, 1.0)
    return near_gate * (up_z ** 4)


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


def torso_pitch_penalty(env, trunk_cfg):
    """Squared penalty on trunk forward/back tilt from the +x component of projected gravity in the base frame.
    Args: env: the RL env; trunk_cfg: trunk entity.
    Returns: (num_envs,) tensor greater than or equal to 0.
    """
    # upright_torso_reward is symmetric in pitch (up_z stays ~0.94 at 20 deg), so a robot balancing a 3 kg box out front can lean the trunk back for free. This grows as |sin(pitch)|^2, independent of roll.
    entity: Entity = env.scene[trunk_cfg.name]
    gx = entity.data.projected_gravity_b[:, 0]
    return gx * gx


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


def gated_hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg, trunk_cfg, sensor_name):
    """Multiply hold_box_reward by the stability_gate so the sharp grasp signal only pays when the robot is upright and grounded.
    Args: env: RL env; left/right_hand_cfg: hand entities; box_cfg: box entity; trunk_cfg: trunk entity; sensor_name: feet sensor for the stability gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    return hold_box_reward(env, left_hand_cfg, right_hand_cfg, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def box_at_place_target_pos_fused_reward(
    env, box_cfg, trunk_cfg, sensor_name,
    hand_box_sensor_name: str = "hand_box_contact",
    hand_force_threshold: float = 1.0,
    air_margin: float = 0.03,
    sigma_xy: float = 0.30,
    sigma_z_above: float = 0.30,
    sigma_z_below: float = 0.05,
    w_ungated: float = 2.0,
    w_gated: float = 4.0,
):
    """Fused ungated + double-gated tracking on box-to-place-target distance.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: trunk; sensor_name: feet sensor; hand_box_sensor_name: hand-vs-box sensor; hand_force_threshold: N above which the hand counts as holding; air_margin: clearance flipping airborne/landed; sigma_xy / sigma_z_above / sigma_z_below: Gaussian widths; w_ungated: always-on pull weight; w_gated: extra weight while both gates open.
    Returns: (num_envs,) tensor.
    """
    # Base = box_at_place_target_pos_reward; total = base * (w_ungated + w_gated * stability_gate * hand_placement_gate). Fused so the Gaussian evaluates once per step instead of twice.
    base = box_at_place_target_pos_reward(
        env, box_cfg,
        sigma_xy=sigma_xy, sigma_z_above=sigma_z_above, sigma_z_below=sigma_z_below,
    )
    gate = (stability_gate(env, trunk_cfg, sensor_name)
            * hand_placement_gate(env, box_cfg, hand_box_sensor_name, hand_force_threshold, air_margin))
    return base * (w_ungated + w_gated * gate)


def gated_box_at_place_target_quat_reward(env, box_cfg, trunk_cfg, sensor_name):
    """Multiply the SO(3) place-target reward by stability_gate so orientation credit only pays when the robot is upright and grounded.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: trunk entity; sensor_name: feet sensor for the stability gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    return box_at_place_target_quat_reward(env, box_cfg) * stability_gate(env, trunk_cfg, sensor_name)


def gated_descent_progress_reward(
    env, box_cfg, left_hand_cfg, right_hand_cfg,
    trunk_cfg, sensor_name, hold_dist: float = 0.10,
    xy_gate_dist: float = 0.15,
):
    """Multiply descent_progress_reward by stability_gate so descent credit only accrues when the robot is upright and grounded.
    Args: env: RL env; box_cfg: box entity; left/right_hand_cfg: hand entities; trunk_cfg: trunk entity; sensor_name: feet sensor; hold_dist: max hand-to-anchor distance for the descent gate; xy_gate_dist: max box-to-target xy distance for the "over target" gate.
    Returns: (num_envs,) tensor in [0, 1].
    """
    return descent_progress_reward(
        env, box_cfg, left_hand_cfg, right_hand_cfg, hold_dist, xy_gate_dist,
    ) * stability_gate(env, trunk_cfg, sensor_name)


def gated_box_flat_at_target_reward(
    env, box_cfg, trunk_cfg, sensor_name,
    hand_box_sensor_name: str = "hand_box_contact",
    hand_force_threshold: float = 1.0,
    air_margin: float = 0.03,
    near_target_dist: float = 0.20,
):
    """Multiply box_flat_at_target_reward by stability_gate and hand_placement_gate so the "flat and smooth" bonus only fires during a stable, hand-supported placement.
    Args: env: RL env; box_cfg: box entity; trunk_cfg: trunk entity; sensor_name: feet sensor; hand_box_sensor_name: hand-vs-box ContactSensor; hand_force_threshold: N above which the hand counts as holding; air_margin: clearance below which the box counts as landed; near_target_dist: activation radius of the underlying flat reward.
    Returns: (num_envs,) tensor in [0, 1].
    """
    return (box_flat_at_target_reward(env, box_cfg, near_target_dist)
            * stability_gate(env, trunk_cfg, sensor_name)
            * hand_placement_gate(env, box_cfg, hand_box_sensor_name, hand_force_threshold, air_margin))


def body_collision_bundle_penalty(
    env, trunk_cfg, left_shank_cfg, right_shank_cfg, left_hip_cfg, right_hip_cfg,
    left_hand_cfg, right_hand_cfg, box_cfg,
    non_hand_box_threshold: float = 0.22,
    table_margin_xy: float = 0.02,
    table_z_slack: float = 0.02,
    w_non_hand_box: float = -2.0,
    w_body_table: float = -2.0,
):
    """Fused body-vs-box proximity + body-vs-table volumetric penalty sharing the 7 body positions.
    Args: env: RL env; body cfgs; box_cfg: box entity; non_hand_box_threshold: proximity threshold; table_margin_xy: footprint shrink; table_z_slack: vertical tolerance; w_non_hand_box/w_body_table: per-signal weights (negative).
    Returns: (num_envs,) tensor.
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
    """Volumetric penalty for entering the tabletop shadow extended to the floor.
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
        xy_into = t_half_xy - torch.abs(pos[:, :2] - t_xy)
        z_below = top_z - pos[:, 2]
        depth = torch.min(
            torch.stack([xy_into[:, 0], xy_into[:, 1], z_below], dim=-1),
            dim=-1,
        ).values
        penalty += torch.clamp(depth, min=0.0)
    return penalty


_BOX_FLOOR_LIMIT_Z = TABLE_TOP_Z - 0.12


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


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: monotonic C1 S-curve with zero derivative at t=0 and t=1.
    Args: t: parameter (clamped to [0, 1]).
    Returns: smoothed value in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# When set, drop_curriculum ignores its per-env counter and reports this step forever; used by _view.py to freeze the viewer at the trained curriculum state via lock_curriculum_at_iter (None = training mode).
_DROP_CURRICULUM_LOCKED_STEP: int | None = None


def lock_curriculum_at_iter(iter_num: int | None, num_steps_per_env: int = 24) -> None:
    """Freeze drop_curriculum at the step corresponding to a given PPO iteration.
    Args: iter_num: PPO iteration count parsed from a checkpoint filename; pass None to unlock; num_steps_per_env: must match the PPO cfg (default 24).
    Returns: none.
    """
    global _DROP_CURRICULUM_LOCKED_STEP
    if iter_num is None:
        _DROP_CURRICULUM_LOCKED_STEP = None
        return
    _DROP_CURRICULUM_LOCKED_STEP = int(iter_num) * int(num_steps_per_env)


def place_bias_curriculum(env, env_ids):
    """Reverse curriculum: ramp place_bias from 1.0 (all envs placed) to 0.0 (all envs from carry-terminal buffer).
    Args: env: the RL env; env_ids: envs being updated (unused, curriculum is global).
    Returns: dict with current place_bias for logging.
    """
    # At bb=1.0 the policy harvests hand_release + arm_pose_walk_ready + box_landed_flat + box_flat_at_target from step 0 (terminal "release + stand"); as bb decays, carry-buffer envs mix in. Env-owned counter (fresh per runner) matches pick_curriculum so resume always replays the ramp from scratch.
    if _DROP_CURRICULUM_LOCKED_STEP is not None:
        step = _DROP_CURRICULUM_LOCKED_STEP    # viewer / play mode: freeze at trained state
    else:
        if not hasattr(env, "_drop_curriculum_step"):
            env._drop_curriculum_step = 0
        env._drop_curriculum_step += 1
        step = env._drop_curriculum_step
    # Schedule copied from pick_curriculum (num_steps_per_env=24 -> ~1000 iters hold + ~3000 iters ramp, fits inside the current 4000-iter budget).
    if step < 24_000:                      # iter 0-1000: pure placed spawn
        place_bias = 1.0
    elif step < 96_000:                    # iter 1000-4000: smoothstep 1.0 -> 0.0
        place_bias = 1.0 - _smoothstep((step - 24_000) / (96_000 - 24_000))
    else:                                  # iter 4000+: pure carry-buffer spawn
        place_bias = 0.0
    try:
        term_cfg = env.event_manager.get_term_cfg("reset_base")
        term_cfg.params["place_bias"] = place_bias
    except Exception:
        pass
    return {"place_bias": place_bias}


def build_g1_drop_mdp():
    """Assemble the MDP dict for the drop task.
    Args: none.
    Returns: dict of MDP components consumed by g1_drop_env_cfg.
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
            pos=(BOX_START_X, 0.0, BOX_REST_Z),
            joint_pos={},
        ),
    )

    g1_asset      = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    arm_asset     = SceneEntityCfg("robot", joint_names=HOLDING_ARM_JOINTS, preserve_order=True)
    left_sym      = SceneEntityCfg("robot", joint_names=_LEFT_SYM_JOINTS,  preserve_order=True)
    right_sym     = SceneEntityCfg("robot", joint_names=_RIGHT_SYM_JOINTS, preserve_order=True)
    wrist_acts    = SceneEntityCfg("robot", actuator_names=(r".*wrist.*",), preserve_order=True)
    trunk         = SceneEntityCfg("robot", body_names=("torso_link",))
    pelvis        = SceneEntityCfg("robot", body_names=("pelvis",))
    left_hand     = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand    = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_shank    = SceneEntityCfg("robot", body_names=("left_knee_link",))
    right_shank   = SceneEntityCfg("robot", body_names=("right_knee_link",))
    left_hip      = SceneEntityCfg("robot", body_names=("left_hip_pitch_link",))
    right_hip     = SceneEntityCfg("robot", body_names=("right_hip_pitch_link",))
    box           = SceneEntityCfg("box")

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
        # Narrowed from the whole-robot pelvis subtree to just the wrists so grip_force_reward is not paid for leg or torso contact.
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
        # Torso + real-limb bodies (arms/feet excluded, hip_roll/hip_yaw pivots also excluded: they are small internal pivot links that sit right at the table_top elevation in PLACED / SQUAT_GRIP and produce false-positive terminations at reset; hip_pitch is the "real hip" limb that would actually collapse onto the table). Copied from ../unitree-g1-humanoid-pick/src/policy_pick.py after the previous subtree pattern triggered 27 terminations per episode at bb=1.0 because the arms/wrist subtree grazed the tabletop.
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
        # Same real-limb list as body_table_contact (arms/feet excluded, hip_roll/hip_yaw pivots also excluded: small internal pivot links at box elevation in PLACED / SQUAT_GRIP produce false-positive terminations at reset). Mirrors ../unitree-g1-humanoid-pick/src/policy_pick.py body_box_contact_cfg.
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

    actor_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_lin_vel,      noise=Unoise(n_min=-0.2,   n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=base_ang_vel,      noise=Unoise(n_min=-0.1,   n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.5,  n_max=0.5)),
        "actions":   ObservationTermCfg(func=last_action),
        "left_hand_to_box":  ObservationTermCfg(func=body_vector, params={"source_cfg": left_hand,  "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "right_hand_to_box": ObservationTermCfg(func=body_vector, params={"source_cfg": right_hand, "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "trunk_to_box":      ObservationTermCfg(func=body_vector, params={"source_cfg": trunk,      "target_cfg": box}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "wrist_torque":      ObservationTermCfg(func=wrist_actuator_force_obs, params={"asset_cfg": wrist_acts}, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "box_pos":           ObservationTermCfg(func=box_position_obs,    params={"box_cfg": box}, noise=Unoise(n_min=-0.005, n_max=0.005)),
        "box_lin_vel":       ObservationTermCfg(func=box_lin_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_ang_vel":       ObservationTermCfg(func=box_ang_vel_obs,     params={"box_cfg": box}, noise=Unoise(n_min=-0.05,  n_max=0.05)),
        "box_orientation":   ObservationTermCfg(func=box_orientation_obs, params={"box_cfg": box}),
        "box_size":          ObservationTermCfg(func=box_size_obs),
        "cmd_grasp_offset_left":  ObservationTermCfg(func=cmd_grasp_offset_left_obs),
        "cmd_grasp_offset_right": ObservationTermCfg(func=cmd_grasp_offset_right_obs),
        # Kept for obs-space parity with the pick task (same 21-term layout).
        "cmd_object_target_pos":  ObservationTermCfg(func=cmd_object_target_pos_obs),
        "cmd_object_target_quat": ObservationTermCfg(func=cmd_object_target_quat_obs),
        "cmd_place_target_pos":   ObservationTermCfg(func=cmd_place_target_pos_obs),
        "cmd_place_target_quat":  ObservationTermCfg(func=cmd_place_target_quat_obs),
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

    rewards = {
        # Fused hold_box (gated sharp w=15) + hands_on_anchor (ungated broad w=8); airborne-gated so the policy is not paid ~23 pts/step for staying stuck to the placed box (the "never release" attractor from earlier runs). Weight 15 was needed after run 4 showed hold_box=0.07/ep: the policy could earn 3.9+3.9 from upright+feet by standing still without holding.
        "hand_placement": RewardTermCfg(
            func=hand_placement_bundle_reward, weight=1.0,
            params={"box_cfg": box,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "trunk_cfg": trunk, "sensor_name": "feet_ground_contact",
                    "w_hold_sharp": 15.0, "w_hold_broad": 8.0,
                    "broad_near_dist": 0.05, "broad_kernel_std": 0.05,
                    "air_margin": 0.03, "air_ramp": 0.05},
        ),
        # Real contact-force reward; replaces the removed gentle_grasp penalty that taught the policy to grip weakly to avoid box angular oscillations.
        "grip_force": RewardTermCfg(func=grip_force_reward, weight=8.0,
                                    params={"sensor_name": "hand_box_contact",
                                            "target_force": 5.0,
                                            "box_cfg": box,
                                            "air_margin": 0.03}),
        # Fused palm_inward_left + palm_inward_right into palm_inward_bilateral (w=0.5 each folded in).
        "palm_inward": RewardTermCfg(
            func=palm_inward_bilateral_reward, weight=1.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "w_left": 0.5, "w_right": 0.5},
        ),
        # Fused ungated + double-gated pull: w_ungated=2 gives dense placement gradient (unlocks descent before gates open), w_gated=4 rewards stable hand-supported descent, Gaussian computed once per step.
        "box_at_place_target_pos": RewardTermCfg(
            func=box_at_place_target_pos_fused_reward, weight=1.0,
            # sigma_xy tightened 0.30 -> 0.12 and w_ungated 2 -> 4 (2026-08-11): the broad kernel + weak weight left a shallow xy gradient, so lowering straight down beside the body paid ~0.6 of the placement reward.
            params={"box_cfg": box, "trunk_cfg": trunk,
                    "sensor_name": "feet_ground_contact",
                    "hand_box_sensor_name": "hand_box_contact",
                    "hand_force_threshold": 1.0, "air_margin": 0.03,
                    "sigma_xy": 0.12, "sigma_z_above": 0.30, "sigma_z_below": 0.05,
                    "w_ungated": 4.0, "w_gated": 4.0},
        ),
        # Fused box_upright + box_front_aligned + box_at_chest into box_carry_pose_bundle (w=1.5 each folded in).
        "box_carry_pose": RewardTermCfg(
            func=box_carry_pose_bundle_reward, weight=1.0,
            # w_chest=0: the box_at_chest attractor inherited from policy_carry fought the descent; keep upright + front-aligned, drop the chest pull so lowering the box has no reward cost.
            params={"box_cfg": box, "w_upright": 1.5, "w_front": 1.5, "w_chest": 0.0},
        ),
        # Dense per-cm descent signal, gated on holding AND box xy over target (without the xy gate the policy farms descent by lowering beside the body).
        "descent_progress": RewardTermCfg(func=gated_descent_progress_reward, weight=3.0,
                                          params={"box_cfg": box,
                                                  "left_hand_cfg": left_hand,
                                                  "right_hand_cfg": right_hand,
                                                  "trunk_cfg": trunk,
                                                  "sensor_name": "feet_ground_contact",
                                                  "hold_dist": 0.10,
                                                  "xy_gate_dist": 0.15}),
        # Explicit "flat" bonus near the target: up_z^4 punishes any residual tilt.
        "box_flat_at_target": RewardTermCfg(func=gated_box_flat_at_target_reward, weight=5.0,
                                            params={"box_cfg": box, "trunk_cfg": trunk,
                                                    "sensor_name": "feet_ground_contact",
                                                    "near_target_dist": 0.20}),
        # Release incentive that fires only while the box is within 7 cm of the tabletop AND both hands are still on the grasp anchors; weight was dropped from 6 to 1.5 to remove the dominant "release-early" attractor observed in the 500-iter run.
        "release_shoulder_open": RewardTermCfg(func=release_shoulder_open_reward, weight=1.5,
                                               params={"box_cfg": box,
                                                       "left_hand_cfg": left_hand,
                                                       "right_hand_cfg": right_hand,
                                                       "trigger_gap": 0.07,
                                                       "spread_target": 0.6,
                                                       "hand_close_dist": 0.15,
                                                       "xy_gate_dist": 0.07}),
        # Post-landing release incentive: pays only once the box has landed AND the closer hand has moved >= release_dist from its grasp anchor; complements hand_placement's airborne gate with a positive gradient that pulls the arms away.
        "hand_release": RewardTermCfg(
            func=hand_release_reward, weight=15.0,
            params={"box_cfg": box,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "release_dist": 0.15, "air_margin": 0.03},
        ),
        # Shapes the terminal arm posture toward G1_HOME_JOINT_POS so the drop's terminal snapshot feeds cleanly into the walk policy; double-gated on (box landed) AND (both hands released) so it stays silent during descent.
        "arm_pose_walk_ready": RewardTermCfg(
            func=arm_pose_walk_ready_reward, weight=1.5,
            params={"box_cfg": box, "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "arm_cfg": arm_asset,
                    "target_values": WALK_READY_ARM_TARGET,
                    "sigma": 0.30, "air_margin": 0.03, "release_dist": 0.15},
        ),
        # Step-wise landed+flat bonus with no hand gate: lets the policy cross the airborne-to-landed threshold and see a positive gradient before also releasing, breaking the hover-just-above local optimum.
        "box_landed_flat": RewardTermCfg(
            func=box_landed_flat_bonus, weight=3.0,
            params={"box_cfg": box, "air_margin": 0.03, "up_threshold": 0.95},
        ),
        # Fused free_fall + hand_lost_grip + box_jammed into box_unsupported_bundle; all three read (dl, dr, box_z, box_vz), computed once and shared.
        "box_unsupported": RewardTermCfg(
            func=box_unsupported_bundle_penalty, weight=1.0,
            params={"box_cfg": box,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "vel_threshold": 0.30, "hand_far_dist": 0.15, "z_floor": 0.10,
                    "w_free_fall": -25.0, "w_hand_lost_grip": -50.0, "w_box_jammed": -10.0},
        ),
        # Fused upright + upright_pelvis + feet_on_ground into stability_bundle.
        "stability": RewardTermCfg(
            func=stability_bundle_reward, weight=1.0,
            params={"trunk_cfg": trunk, "pelvis_cfg": pelvis,
                    "sensor_name": "feet_ground_contact",
                    "w_upright": 10.0, "w_upright_pelvis": 5.0, "w_feet": 5.0},
        ),
        # upright_torso is pitch-symmetric so the policy leaned the trunk back to balance a heavy held box for free; +x projected gravity penalises tilt directly. Weight bumped -0.5 -> -5.0 (2026-08-11): -0.5 gave -0.058/step (drowned by the +23/step hand_placement attractor), -5.0 gives -0.58/step, comparable to other shaping terms.
        "torso_pitch": RewardTermCfg(
            func=torso_pitch_penalty, weight=-5.0,
            params={"trunk_cfg": trunk},
        ),
        # Fused walk_to_place_target + walk_to_place_progress (w=2.0 + w=1.0 folded in) sharing the horizontal delta. Airborne-gated (2026-08-11): keeping the approach signal on after landing taught the policy to take one extra step until it grazed the tabletop.
        "walk_to_target": RewardTermCfg(
            func=walk_to_target_bundle_reward, weight=1.0,
            params={"sigma": 1.0, "w_target": 2.0, "w_progress": 1.0,
                    "box_cfg": box, "air_margin": 0.03, "air_ramp": 0.05},
        ),
        # stay_still removed: hands-near-anchors gate is always open while holding the box, so its -0.5*|v|^2 penalised the lean-forward motion drop needs.
        "symmetric_pose": RewardTermCfg(func=symmetric_pose_penalty, weight=-0.3,
                                        params={"left_cfg": left_sym, "right_cfg": right_sym}),
        # Smoothness bundle (pick_normal ETH-style weights).
        "action_rate_l2":   RewardTermCfg(func=action_rate_l2,     weight=-1.0e-2),
        "action_2nd_order": RewardTermCfg(func=action_2nd_order_l2, weight=-1.0e-2),
        "dof_vel_l2":       RewardTermCfg(func=dof_vel_l2,          weight=-1.0e-3),
        # Punish box kinetic energy across the descent (gate_z 0.05 -> 0.35 gives a dense gradient from chest height; w_ang catches the flick-of-the-wrist on early release). Previous 5cm gate + w=-1.5 was inert against the +15 hand_release cliff.
        "box_impact_vel": RewardTermCfg(
            func=box_impact_velocity_penalty, weight=-6.0,
            params={"box_cfg": box, "gate_z": 0.35, "w_lin": 1.0, "w_ang": 0.1},
        ),
        "hand_approach_vel": RewardTermCfg(
            func=hand_approach_velocity_penalty, weight=-3.0,
            params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "box_cfg": box, "gate_dist": 0.25},
        ),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": g1_asset}),
        "base_ang_vel":   RewardTermCfg(func=base_ang_vel_penalty,     weight=-0.05),
        "self_collisions": RewardTermCfg(
            func=vel_mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        # Fused non_hand_box + body_table into body_collision_bundle. w_non_hand_box bumped -2.0 -> -6.0 on 2026-08-11: the shallow ramp (-2.0 across a 22 cm activation) topped out at ~-1.1/step even with the box resting on the torso, so the policy had no gradient pushing the body away from the box before the termination fired.
        "body_collision": RewardTermCfg(
            func=body_collision_bundle_penalty, weight=1.0,
            params={"trunk_cfg": trunk, "left_shank_cfg": left_shank, "right_shank_cfg": right_shank,
                    "left_hip_cfg": left_hip, "right_hip_cfg": right_hip,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand, "box_cfg": box,
                    "w_non_hand_box": -6.0, "w_body_table": -2.0},
        ),
    }

    # Reset order matters: default_scene wipes to defaults -> reset_base splits per env (placed vs carry-buffer) -> reset_drop_cmd samples targets and, for placed envs, snaps the target onto the placed box so hand_release fires at step 0.
    events = {
        "reset_scene": EventTermCfg(
            func=reset_scene_to_default,
            mode="reset",
            params={},
        ),
        "reset_base": EventTermCfg(
            func=reset_from_carry_buffer,
            mode="reset",
            params={
                "buffer_path": CARRY_BUFFER_PATH,
                "xy_range":  (-ROBOT_SPAWN_XY_JITTER, ROBOT_SPAWN_XY_JITTER),
                # Was (-pi, pi): the buffer stores the box in body-yaw frame, so sampling full 360 degrees rotated robot+box together while the table stayed fixed in world +X: half the episodes spawned the robot facing away from the table.
                "yaw_range": ROBOT_SPAWN_YAW_RANGE,
                "asset_cfg": g1_asset,
                "box_cfg":   box,
                # Overwritten every step by place_bias_curriculum; 1.0 = pure placed spawn, 0.0 = pure carry-buffer spawn.
                "place_bias": 1.0,
                # Kept at 1.0: the drop_curriculum scale ramp was removed in favour of the reverse curriculum on place_bias, so placed-box jitter always matches reset_drop_command_and_object's full jitter range.
                "target_scale": 1.0,
            },
        ),
        "reset_drop_cmd": EventTermCfg(
            func=reset_drop_command_and_object,
            mode="reset",
            # scale=1.0: place-target jitter is always at full range; the reverse curriculum now runs on where the robot starts (place_bias), not on how narrow the target box is.
            params={"scale": 1.0},
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
        # Kill the "let box fall in the first frames, then farm upright" attractor: hard-terminate so V(s')=0 and the future stream of upright rewards vanishes.
        "hand_lost_grip": TerminationTermCfg(
            func=hand_lost_grip, params={"box_cfg": box},
        ),
        # Kill the "slump onto the tabletop after placing the box" attractor via physics contact between torso/hip/knee geoms and the table: catches lying on the box, thigh on the edge, etc.
        "body_touched_table": TerminationTermCfg(
            func=body_touched_table,
            params={"sensor_name": body_table_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
        # Kill the "let box collapse into the lap / rest legs on the box" attractor via physics contact between torso/hip/knee/ankle geoms and the box. Threshold restored 1.0 -> 5.0 on 2026-08-11 (matching pick): with the reverse curriculum starting from the SQUAT_GRIP pose the torso is intentionally close to the box and contacts up to ~3 N are legitimate at reset; the "rest box on body" exploit is now blocked by teaching the released terminal shape directly at bb=1.0 instead of by a hair-trigger threshold.
        "body_touched_box": TerminationTermCfg(
            func=body_touched_box,
            params={"sensor_name": body_box_contact_cfg.name,
                    "force_threshold": 5.0},
        ),
        # Kill the "release mid-air, let it fall onto the target" attractor: end the episode if wrist-vs-box contact drops out while the box centre is still > 3 cm above its rest height.
        "box_released_airborne": TerminationTermCfg(
            func=box_released_airborne,
            params={"box_cfg": box,
                    "hand_box_sensor_name": hand_box_contact_cfg.name,
                    "hand_force_threshold": 1.0,
                    "air_margin": 0.03},
        ),
        # Anchor-distance twin of box_released_airborne that catches slow slips keeping a grazing hand-vs-box contact (so the force sensor never trips) but letting the box escape the pinch.
        "hand_lost_anchor_airborne": TerminationTermCfg(
            func=hand_lost_anchor_airborne,
            params={"box_cfg": box,
                    "left_hand_cfg": left_hand, "right_hand_cfg": right_hand,
                    "anchor_dist_threshold": 0.20, "air_margin": 0.03},
        ),
    }

    curriculum = {
        "reset_stages": CurriculumTermCfg(func=place_bias_curriculum, params={}),
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


def g1_drop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Assemble the ManagerBasedRlEnvCfg for the drop task.
    Args: play: if True, disable noise/pushes and use a long episode for playback.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_drop_mdp()
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
        # decimation 5: policy at 40 Hz, physics at 200 Hz. Reverted 10 -> 5 to match pick and cut per-iter wall clock by ~22%; raise action_rate_l2 / action_2nd_order first before bumping decimation back up.
        decimation=5,
        episode_length_s=3.25,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
    return cfg


def g1_drop_ppo_runner_cfg():
    """Build the RSL-RL PPO runner config for the drop policy.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.30, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        clip_actions=3.0,
        experiment_name="g1_drop", save_interval=100, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the drop task with mjlab; safe to call multiple times.
    Args: none.
    Returns: the registered task id.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_drop_env_cfg(),
        play_env_cfg=g1_drop_env_cfg(play=True),
        rl_cfg=g1_drop_ppo_runner_cfg(),
    )
    return TASK_ID
