##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""G1 flat-velocity locomotion task (walk): env + PPO cfg, registered under TASK_ID.

Input:  play flag (bool) for g1_flat_env_cfg; no external args elsewhere.
Output: ManagerBasedRlEnvCfg (env), RslRlOnPolicyRunnerCfg (PPO), and register() -> TASK_ID.
"""

from __future__ import annotations

import math
import re

import mujoco
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME, get_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1-Custom"

# Contract for src.collect_buffer.
SNAPSHOT_ENTITIES = ["robot"]
SNAPSHOT_YAW_ALIGN = True

# Actuated joints in canonical MuJoCo order.
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

# Foot collision geoms targeted by foot-friction domain randomisation.
G1_FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def single_stance_reward(env, sensor_name, command_name=None, command_threshold=0.05):
    """Reward +1 when exactly one foot contacts ground, gated by command magnitude.
    Args: env: RL env; sensor_name: foot contact sensor; command_name: twist command key; command_threshold: activation floor.
    Returns: per-env reward tensor.
    """
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    num_contact = in_contact.sum(dim=1)
    reward = (num_contact == 1).float()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            active = ((linear_norm + angular_norm) > command_threshold).float()
            reward = reward * active
    return reward


def swing_height_error(env, sensor_name, height_sensor_name, target_height, sigma,
                       command_name="twist", command_threshold=0.05):
    """Quadratic penalty on foot height error while in swing (fuses foot_clearance + foot_swing_height).
    Args: env: RL env; sensor_name: foot contact sensor; height_sensor_name: foot height scan; target_height/sigma: Gaussian center and width in meters; command_name/command_threshold: twist activity gate.
    Returns: (num_envs,) non-negative penalty tensor summed across feet.
    """
    contact = env.scene[sensor_name]
    height_sensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = (contact.data.found == 0).float()
    err = (foot_heights - target_height) / sigma
    cost = torch.sum(err * err * in_air, dim=1)
    cmd = env.command_manager.get_command(command_name)
    if cmd is not None:
        active = ((torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold).float()
        cost = cost * active
    return cost


def wz_sign_penalty(env, command_name="twist"):
    """Penalise yaw-rate whose sign disagrees with commanded wz to remove rotation bias.
    Args: env: RL env; command_name: twist command key.
    Returns: per-env penalty tensor.
    """
    cmd = env.command_manager.get_term(command_name).vel_command_b[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    active = (torch.abs(cmd) > 0.05).float()
    penalty = -torch.relu(-torch.sign(cmd) * actual) * active
    return penalty


def build_g1_mdp():
    """Assemble MDP pieces for the G1 walking task.
    Args: none.
    Returns: dict with robot_cfg, sensors, observations, actions, commands, events, rewards, terminations, curriculum.
    """
    robot_cfg = get_g1_robot_cfg()

    g1_asset = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    torso = SceneEntityCfg("robot", body_names=("torso_link",))
    feet_sites = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)

    # Foot-ground contact, self-collision, and foot height scan sensors.
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

    # Actor and critic share terms; only actor sees corruption noise.
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-1.0, n_max=1.0)),
        "actions":   ObservationTermCfg(func=mdp.last_action),
        "command":   ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    }
    observations = {
        "actor":  ObservationGroupCfg(terms=actor_terms,     concatenate_terms=True, enable_corruption=True),
        "critic": ObservationGroupCfg(terms={**actor_terms}, concatenate_terms=True, enable_corruption=False),
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

    # Reset perturbations, interval pushes, startup domain randomisation.
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-math.pi, math.pi)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": g1_asset,
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll":  (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw":   (-0.78, 0.78),
                },
            },
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
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
    }

    # Velocity tracking, upright, posture, gait shaping, and penalty terms.
    rewards = {
        "track_linear_velocity":  RewardTermCfg(func=mdp.track_linear_velocity,  weight=2.5, params={"command_name": "twist", "std": math.sqrt(0.25)}),
        "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=6.0, params={"command_name": "twist", "std": math.sqrt(0.10)}),
        "wz_sign_penalty": RewardTermCfg(func=wz_sign_penalty, weight=2.0, params={"command_name": "twist"}),
        "upright":                RewardTermCfg(func=mdp.upright,                weight=1.0, params={"std": math.sqrt(0.2), "asset_cfg": torso}),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=0.8,
            params={
                "asset_cfg": g1_asset,
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*":    0.45,
                    r".*hip_roll.*":     0.15,
                    r".*hip_yaw.*":      0.15,
                    r".*knee.*":         0.50,
                    r".*ankle_pitch.*":  0.25,
                    r".*ankle_roll.*":   0.10,
                    r".*waist_yaw.*":    0.20,
                    r".*waist_roll.*":   0.08,
                    r".*waist_pitch.*":  0.10,
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*":  0.15,
                    r".*shoulder_yaw.*":   0.10,
                    r".*elbow.*":          0.15,
                    r".*wrist.*":          0.30,
                },
                "std_running": {
                    r".*hip_pitch.*":    0.50,
                    r".*hip_roll.*":     0.20,
                    r".*hip_yaw.*":      0.20,
                    r".*knee.*":         0.60,
                    r".*ankle_pitch.*":  0.35,
                    r".*ankle_roll.*":   0.15,
                    r".*waist_yaw.*":    0.30,
                    r".*waist_roll.*":   0.08,
                    r".*waist_pitch.*":  0.20,
                    r".*shoulder_pitch.*": 0.50,
                    r".*shoulder_roll.*":  0.20,
                    r".*shoulder_yaw.*":   0.15,
                    r".*elbow.*":          0.35,
                    r".*wrist.*":          0.30,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.2,
            },
        ),
        "body_ang_vel":   RewardTermCfg(func=mdp.body_angular_velocity_penalty, weight=-0.05, params={"asset_cfg": torso}),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits,              weight=-1.0,  params={"asset_cfg": g1_asset}),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2,                weight=-0.1),
        "single_stance": RewardTermCfg(
            func=single_stance_reward,
            weight=1.5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "swing_height_error": RewardTermCfg(
            func=swing_height_error,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "target_height": 0.10,
                "sigma": 0.08,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "feet_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": feet_sites,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
    }

    terminations = {
        "time_out":  TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(65.0)}),
    }

    # Velocity command ranges widen as env-step count grows.
    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
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

    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    return {
        "robot_cfg": robot_cfg,
        "sensors": (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg),
        "observations": observations,
        "actions": actions,
        "commands": commands,
        "events": events,
        "rewards": rewards,
        "terminations": terminations,
        "curriculum": curriculum,
    }


def g1_flat_env_cfg(play=False):
    """Assemble the full ManagerBasedRlEnvCfg for training or evaluation.
    Args: play: when True disables noise/push and widens command ranges by 1.2x.
    Returns: ManagerBasedRlEnvCfg instance.
    """
    parts = build_g1_mdp()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": parts["robot_cfg"]},
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
        metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.0,
            elevation=-8.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=70,
            njmax=600,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=5,
        episode_length_s=20.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.commands["twist"].ranges.lin_vel_x = (-1.2, 1.2)
        cfg.commands["twist"].ranges.lin_vel_y = (-1.2, 1.2)
        cfg.commands["twist"].ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


def g1_ppo_runner_cfg():
    """Build the PPO runner configuration for G1 walking.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg instance.
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
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="g1_walk", save_interval=50, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the walking task with mjlab's task registry (idempotent).
    Args: none.
    Returns: TASK_ID string.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_flat_env_cfg(),
        play_env_cfg=g1_flat_env_cfg(play=True),
        rl_cfg=g1_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
    return TASK_ID
