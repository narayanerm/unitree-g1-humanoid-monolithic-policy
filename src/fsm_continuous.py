##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Helpers shared by ``src.fsm_continuous_batch`` for the walk->pick FSM path.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "outputs" / "media" / "fsm_continuous_latest.mp4"


@dataclass
class RolloutConfig:
    """Hyperparameters for the continuous rollout."""
    max_steps: int = 2000
    device: str = "cuda:0"
    video_path: Path = DEFAULT_VIDEO
    video_fps: int = 20
    frame_stride: int = 1
    render_hw: tuple[int, int] = (1080, 1920)
    camera_name: str = "tracking"
    spawn_xy: tuple[float, float] = (-1.5, 0.0)  # 2 m behind the table so walk phase is visible
    spawn_z: float = 0.72
    seed: int = 7
    walk_to_pick_dist: float = 0.40       # positional envelope: distance to box for pick's trained state distribution
    walk_to_pick_speed_max: float = 0.10  # velocity envelope: pick cannot absorb full walk momentum, brake first
    walk_to_pick_brake_steps: int = 25    # consecutive frames with twist=0 inside the brake region before the handover fires; pick's reset distribution has joint_vel=lin_vel=ang_vel=0, so firing on the first (dist, vx) match leaves 1.3 rad/s pelvis spin + 7 rad/s joint motion that pushes the pick out of its training manifold (87% of falls happened within 1.6 s of handover before this).
    pick_to_carry_box_z: float = 0.85     # box lifted well above tabletop
    carry_arrival_dist: float = 0.22      # radial gate; combined with carry_x_max=-1.90 fires reliably for the various carry end-poses (y-drift up to 0.16 m)
    drop_success_box_z_max: float = 0.68  # box resting on table 2 top (0.585 + 0.1 margin)
    drop_success_horiz: float = 0.40      # box within Table 2 top footprint (0.34 x 0.42 half-extent)
    place_descent_steps: int = 12          # bypasses trained drop expert (unstable at carry->drop handover); fast lerp of box from hands to Table 2 top over ~0.24 s
    place_settle_steps: int = 40           # frames box sits on table + robot balances before walk_return
    place_forward_shift: float = 0.05      # unused (kept for tuning knob compatibility)
    return_walk_target_xy: tuple[float, float] = (-0.70, 0.0)  # drop ends near world_x=-1.7; -0.7 gives 1 m walk in front of camera
    return_walk_arrival_dist: float = 0.30
    rise_steps: int = 80                   # walk expert gets zero twist for this many steps so knees straighten from deep drop squat (~0.15 m pelvis) back to walking height (~0.65 m)
    rise_min_pelvis_z: float = 0.55        # start walking once pelvis is back above this height
    drop_settle_min_steps: int = 60        # drop expert keeps running at least this long after box_placed; hands over only when robot held upright+tall for drop_settle_confirm frames (fixes "walk while still squatting" fall)
    drop_settle_confirm: int = 20
    drop_settle_timeout_steps: int = 300   # hard cap so a stuck drop cannot hang the rollout
    walk_return_warmup_steps: int = 100    # ramp the return twist from zero to full over this window
    walk_return_backoff_steps: int = 40    # pure backward-walk frames right after drop so robot clears Table 2 before turning; otherwise the 180 deg turn swings arms into the table
    walk_return_backoff_speed: float = 0.35  # magnitude (m/s) of the backward twist during the backoff window
    upright_min: float = 0.80
    walk_speed_cap: float = 0.7            # base-frame vx cap for locomotion experts
    carry_speed_cap: float = 0.45          # slower so the box does not slosh out
    carry_pursuit_lookahead: float = 0.6   # metres ahead of robot x for pure-pursuit waypoint on y=0; smaller = tighter lateral correction (more oscillation), larger = smoother (more drift); 0.6 m ~= 1.5 stride
    carry_y_abs_max_tight: float = 0.20    # relaxed 0.08 -> 0.20 (iter3, 2026-08-17): tight gate hung the FSM because carry converges asymptotically to ~15 cm lateral offset under load; drop now snaps place_target to robot_y so any offset within TABLE2_HALF_Y=0.42 still lands the box on the tabletop
    table2_xy: tuple[float, float] = (-2.5, 0.0)  # behind spawn so carry is a 180 deg turn + straight walk (mirrors iteration 23 no-table-2 run that worked)
    carry_target_xy: tuple[float, float] = (-2.03, 0.0)  # -2.00 balances policy stability (further makes it fall ~50% of runs) with box landing on Table 2 top; box sits BOX_HOLD_X_OFFSET=0.32 ahead so lands between x=-2.17 and -2.47
    carry_x_max: float = -1.90             # hand-off gate: robot past x=-1.90 (asymptotic end given target -2.03 with 0.15 arrival radius); guarantees held box sits ~x<=-2.22, at least 6 cm past Table 2's front edge -2.16
    carry_y_abs_max: float = 0.20          # hand-off gate: loosened 0.10 -> 0.20 to absorb carry drift at this seed; carry_brake stage below shapes the handoff state, not this cap
    carry_brake_min_steps: int = 20        # intermediate state mirroring walk's brake-before-pick: carry keeps running (so box stays gripped) with twist=zero; exit when |base_vx| < carry_brake_speed_max AND |base_wz| < carry_brake_ang_speed_max or max-step fires; brings (pose, velocity) inside drop's training envelope (drop was trained from resets with all velocities zeroed)
    carry_brake_max_steps: int = 400       # bumped 200 -> 300 -> 400 to give two-stage brake time to reposition and yaw-align
    carry_brake_speed_max: float = 0.12    # base_lin_vel_b norm below this to declare settled
    carry_brake_ang_speed_max: float = 0.35
    carry_brake_heading_err_max: float = 0.10   # ~5.7 deg residual yaw to Table 2 before handoff to drop
    carry_brake_lateral_err_max: float = 0.25   # relaxed 0.05 -> 0.25 (iter3, 2026-08-17): tight gate assumed lateral drift could be corrected during brake, but 2026-08-17 experiments showed carry has zero yaw authority under load; drop now snaps place_target to robot_y so any offset within table 2 half-extent lands the box; 0.25 keeps a safety cap so obviously-broken runs time out
    carry_brake_dist_max: float = 0.05     # 5 cm radial gate to carry_target_xy: farther than this the two-stage brake keeps walking with vx+wz to reposition; only then switches to pure yaw-align (carry ignores vy so walking toward target with wz steering is the only lateral correction)
    carry_brake_reposition_speed_cap: float = 0.18   # capped below carry_speed_cap so reposition stage is a slow crawl (arms stable), still large enough to overcome walk expert's minimum stride
    drop_max_steps: int = 400              # trained policy hand-off, enters with velocities near zero (see carry_brake)
    drop_done_box_z_max: float = 0.68      # box resting on Table 2 top (0.585 + ~0.10 margin)


def _load_actor_from_state_dict(
    ckpt_path: Path,
    obs_dim: int,
    action_dim: int,
    hidden: tuple[int, ...],
    device: str,
) -> Callable[[dict], torch.Tensor]:
    """Rebuild an rsl-rl MLPModel from a checkpoint without creating a mjlab env.
    Args: ckpt_path: trained .pt; obs_dim: actor input dim; action_dim: actor output dim; hidden: hidden-layer widths; device: torch device.
    Returns: callable taking dict{"actor"} and returning (n, action_dim) deterministic actions.
    """
    import torch.nn as nn
    from rsl_rl.models.mlp_model import MLPModel

    class _WalkNormaliserActor(nn.Module):
        """Wrapper mirroring MjlabOnPolicyRunner's deterministic inference path."""
        def __init__(self, mlp: nn.Sequential, normalizer_mean: torch.Tensor, normalizer_var: torch.Tensor, eps: float = 1e-8):
            """Store the trained MLP and obs normaliser stats.
            Args: mlp: trained mlp; normalizer_mean: running mean; normalizer_var: running variance; eps: variance floor.
            Returns: none.
            """
            super().__init__()
            self.mlp = mlp
            self.register_buffer("mean", normalizer_mean)
            self.register_buffer("var", normalizer_var)
            self.eps = eps

        def forward(self, obs_dict: dict) -> torch.Tensor:
            """Normalise the actor obs and run the MLP.
            Args: obs_dict: dict with key "actor" holding (n, obs_dim) tensor.
            Returns: (n, action_dim) action tensor.
            """
            x = (obs_dict["actor"] - self.mean) / (self.var.sqrt() + self.eps)
            return self.mlp(x)

    blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    actor_sd = blob["actor_state_dict"]

    layers: list[nn.Module] = []
    prev = obs_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers += [nn.Linear(prev, action_dim)]
    mlp = nn.Sequential(*layers).to(device)

    # Copy trunk weights matching the "mlp.X.*" keys emitted by rsl-rl.
    mlp_state = {k.split("mlp.", 1)[1]: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(mlp_state)

    mean = actor_sd.get("obs_normalizer._mean", None)
    var = actor_sd.get("obs_normalizer._var", None)
    if mean is None or var is None:
        # Fallback for older rsl-rl builds that used flat keys.
        mean = actor_sd.get("obs_normalizer.mean", torch.zeros(obs_dim, device=device))
        var = actor_sd.get("obs_normalizer.var", torch.ones(obs_dim, device=device))

    actor = _WalkNormaliserActor(mlp, mean.to(device), var.to(device)).to(device).eval()
    return actor


def _twist_toward(
    base_env,
    target_xy_w: torch.Tensor,
    speed_cap: float,
) -> torch.Tensor:
    """Turn-in-place-then-walk-forward twist toward a world-frame target.
    Args: base_env: shared env; target_xy_w: (n, 2) world-frame target xy; speed_cap: max forward speed (m/s).
    Returns: (n, 3) tensor of (vx_b, vy_b, wz).
    """
    # Isolating rotation and forward motion avoids the crooked side-step gait produced
    # by an earlier vx+vy formulation.
    robot = base_env.scene["robot"]
    root_pos = robot.data.root_link_pos_w[:, :2]
    dxy_w = target_xy_w - root_pos
    dist = torch.linalg.norm(dxy_w, dim=-1, keepdim=True).squeeze(-1) + 1e-6

    root_quat = robot.data.root_link_quat_w
    w, x, y, z = root_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    dxb =  cy * dxy_w[:, 0] + sy * dxy_w[:, 1]
    dyb = -sy * dxy_w[:, 0] + cy * dxy_w[:, 1]
    heading_err = torch.atan2(dyb, dxb + 1e-6)
    heading_align = torch.exp(-2.5 * heading_err.abs())  # smooth [0,1]; 1 when aligned

    forward_speed = speed_cap * torch.clamp(dist / 0.5, 0.0, 1.0) * heading_align
    vx = torch.where(dist > 0.05, forward_speed, torch.zeros_like(dxb))
    vy = torch.zeros_like(vx)  # never side-step
    wz = torch.clamp(1.5 * heading_err, -0.6, 0.6)
    return torch.stack([vx, vy, wz], dim=-1)


def _compute_walk_obs(
    base_env,
    twist_cmd: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """Assemble the walk actor's obs vector from shared env state.
    Args: base_env: shared env; twist_cmd: (n, 3) synthesized velocity command; last_action: (n, 29) previous action.
    Returns: (n, 99) tensor matching walk's actor_terms layout (base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action, twist_command).
    """
    robot = base_env.scene["robot"]
    joint_default = robot.data.default_joint_pos
    joint_pos_rel = robot.data.joint_pos - joint_default
    joint_vel = robot.data.joint_vel
    return torch.cat(
        [
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
            robot.data.projected_gravity_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            twist_cmd,
        ],
        dim=-1,
    )


def _resolve_pick_indices(base_env) -> dict:
    """Cache body/actuator indices used by the pick obs adapter.
    Args: base_env: shared env.
    Returns: dict with keys left_hand, right_hand, trunk, wrist_actuators.
    """
    robot = base_env.scene["robot"]
    body_names = list(robot.body_names)
    act_names = list(robot.actuator_names) if hasattr(robot, "actuator_names") else []
    if not act_names and hasattr(robot, "joint_names"):
        act_names = list(robot.joint_names)  # entity lacks actuator_names; assume 1:1 with joints
    wrist_ids = [i for i, n in enumerate(act_names) if "wrist" in n]
    return {
        "left_hand": body_names.index("left_wrist_yaw_link"),
        "right_hand": body_names.index("right_wrist_yaw_link"),
        "trunk": body_names.index("torso_link"),
        "wrist_actuators": wrist_ids,
    }


def _disable_terminations(env_cfg) -> None:
    """Remove non-fatal termination terms so the episode never resets mid-rollout.
    Args: env_cfg: the phase's ManagerBasedRlEnvCfg.
    Returns: none.
    """
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return
    if isinstance(terms, dict):
        keep = {k: v for k, v in terms.items() if "time_out" in k}  # keep time_out for safety
        terms.clear()
        terms.update(keep)
