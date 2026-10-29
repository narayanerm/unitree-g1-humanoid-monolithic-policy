##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""DAgger fine-tune of the monolithic walk-pick-carry-drop student against the FSM oracle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.fsm_continuous import RolloutConfig, _compute_walk_obs, _disable_terminations, _twist_toward
from src.fsm_continuous_batch import (
    PICK_PHASE_ID,
    WALK_PHASE_ID,
    _snap_pick_commands_batched,
    _unified_obs_per_env,
)
from src.fsm_wpcd_batched import (
    CARRY_PHASE_ID,
    DROP_PHASE_ID,
    POST_DROP_PHASE_ID,
    _compute_carry_obs,
    _compute_drop_obs,
    _make_two_tables_spec_fn,
    _sample_table2_positions_np,
)
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal
from src.phase_gates import (
    GateConfig,
    PhaseState,
    advance_phases_full,
    teleport_done_envs,
)
from src.student_policy import MonolithicPolicyStudent


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = ROOT / "outputs" / "logs" / "g1_monolithic_dagger"


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS for the ETA log line.
    Args: seconds: non-negative duration.
    Returns: zero-padded HH:MM:SS string.
    """
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class DAggerConfig:
    """Hyperparameters for the DAgger fine-tune."""
    num_envs: int = 1024
    rounds: int = 10
    # Budget covers worst-case gate traversal (walk+brake+pick+carry+brake+drop ~725 steps).
    episode_max_steps: int = 1200
    walk_speed_cap: float = 1.0
    carry_speed_cap: float = 0.45
    env_spacing: float = 20.0
    rollout_seed: int = 7
    device: str = "cuda:0"

    gate: GateConfig = field(default_factory=GateConfig)

    epochs_per_round: int = 60
    bc_batch_size: int = 8192
    bc_lr: float = 3e-4
    bc_weight_decay: float = 1e-5
    balance_phases: bool = True
    # Target share per phase in each refit batch; empty phases redistribute proportionally.
    phase_target_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    buffer_window: int = 3

    save_every_rounds: int = 1

    # Cap BC-seed GPU residency; falls back to disk reload above budget.
    bc_seed_gpu_budget_bytes: int = 2 * 1024 ** 3
    # AMP is a loss on sm_75 (overhead beats tensor-core gains); keep off by default.
    use_amp: bool = False
    # Early-stop costs a val pass per epoch; only enable for long refits (>=60 epochs).
    early_stop_patience: int = 0
    early_stop_val_frac: float = 0.05
    early_stop_min_epochs: int = 5
    # Round-0 fit anchors normaliser stats; freezing afterwards prevents input-scale drift.
    freeze_normalizer_after_round_zero: bool = True

    # Scheduled teleport fractions per phase; remainder (1-sum) spawns fresh in walk.
    carry_reset_fraction: float = 0.25
    pick_reset_fraction: float = 0.0
    drop_reset_fraction: float = 0.0
    carry_snapshot_buffer_size: int = 2048
    drop_snapshot_buffer_size: int = 1024
    walk_terminal_buffer_path: str = "outputs/walk_terminal_states.npz"

    # Phase-stratified cap on BC-seed transitions (0 keeps all); for low-VRAM GPUs.
    bc_seed_max_transitions: int = 0


class FSMOracle:
    """Four-expert dispatcher producing per-env actions matching the FSM behaviour."""

    def __init__(
        self,
        base_env,
        num_envs: int,
        device: str,
        walk_speed_cap: float,
        carry_speed_cap: float,
    ):
        """Load the four frozen experts and cache obs-building indices.
        Args: base_env: mjlab env already built; num_envs: parallel env count; device: torch device for all weights; walk_speed_cap: walk expert's forward speed cap in m/s; carry_speed_cap: carry expert's forward speed cap in m/s.
        Returns: none.
        """
        from dataclasses import asdict
        from mjlab.rl import MjlabOnPolicyRunner
        from mjlab.tasks.registry import load_rl_cfg
        from src.expert_registry import BY_NAME, resolve_ckpt
        from src.fsm_continuous import _load_actor_from_state_dict, _resolve_pick_indices
        from src.fsm_wpcd_batched import CARRY_OBS_DIM, DROP_OBS_DIM
        from src.policy_pick import (
            BOX_HALF,
            GRASP_NOMINAL_Y,
            GRASP_NOMINAL_Z,
            TABLE_HEIGHT,
            TABLE_TOP_HALF,
        )
        from src.policy_pick import TASK_ID as PICK_TASK_ID

        self._base_env = base_env
        self._device = device
        self._num_envs = num_envs
        self._walk_speed_cap = float(walk_speed_cap)
        self._carry_speed_cap = float(carry_speed_cap)

        # Pick runner reads obs through base_env's observation manager; no re-wrapping.
        agent_cfg = load_rl_cfg(PICK_TASK_ID)
        from mjlab.rl import RslRlVecEnvWrapper
        pick_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        pick_runner = MjlabOnPolicyRunner(pick_env, asdict(agent_cfg), device=device)
        pick_runner.load(
            str(resolve_ckpt(BY_NAME["pick"])),
            load_cfg={"actor": True}, strict=True, map_location=device,
        )
        self._pick_policy = pick_runner.get_inference_policy(device=device)
        self._pick_env = pick_env

        # Walk/carry/drop as raw MLPs to avoid building three extra mjlab envs.
        self._walk_policy = _load_actor_from_state_dict(
            ckpt_path=resolve_ckpt(BY_NAME["walk"]),
            obs_dim=99, action_dim=29, hidden=(512, 256, 128), device=device,
        )
        self._carry_policy = _load_actor_from_state_dict(
            ckpt_path=resolve_ckpt(BY_NAME["carry"]),
            obs_dim=CARRY_OBS_DIM, action_dim=29, hidden=(512, 256, 128), device=device,
        )
        self._drop_policy = _load_actor_from_state_dict(
            ckpt_path=resolve_ckpt(BY_NAME["place"]),
            obs_dim=DROP_OBS_DIM, action_dim=29, hidden=(512, 256, 128), device=device,
        )

        idx = _resolve_pick_indices(base_env)
        self._wrist_ids = idx["wrist_actuators"]
        self._left_hand_id = idx["left_hand"]
        self._right_hand_id = idx["right_hand"]
        self._trunk_id = idx["trunk"]

        self._grasp_left = torch.tensor(
            [0.0, GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device,
        ).expand(num_envs, 3).contiguous()
        self._grasp_right = torch.tensor(
            [0.0, -GRASP_NOMINAL_Y, GRASP_NOMINAL_Z], device=device,
        ).expand(num_envs, 3).contiguous()

        # Default place-target z used until drop_fire snaps the per-env target.
        self._table_top_z = TABLE_HEIGHT + TABLE_TOP_HALF
        self._box_half = BOX_HALF

    def default_place_target(self, table2_world_xy: torch.Tensor) -> torch.Tensor:
        """Return a (N, 3) initial place-target tensor sitting on top of table 2.
        Args: table2_world_xy: (N, 2) world positions of each env's table 2.
        Returns: (N, 3) place target with z = table top + box half + env origin z.
        """
        n = table2_world_xy.shape[0]
        z = (
            self._table_top_z + self._box_half
            + self._base_env.scene.env_origins[:, 2]
        )
        out = torch.zeros(n, 3, device=table2_world_xy.device)
        out[:, 0] = table2_world_xy[:, 0]
        out[:, 1] = table2_world_xy[:, 1]
        out[:, 2] = z
        return out

    @torch.no_grad()
    def __call__(
        self,
        state: PhaseState,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch each env to its phase expert and return the (N, 29) oracle action.
        Args: state: current PhaseState (phase_ids + locked targets); last_action: (N, 29) previous action produced by the student.
        Returns: (N, 29) oracle action for the current step.
        """
        robot = self._base_env.scene["robot"]
        box = self._base_env.scene["box"]

        box_xy_w = box.data.root_link_pos_w[:, :2]
        action = torch.zeros(self._num_envs, UNIFIED_ACT_DIM, device=self._device)

        in_walk = state.phase_ids == WALK_PHASE_ID
        in_pick = state.phase_ids == PICK_PHASE_ID
        in_carry = state.phase_ids == CARRY_PHASE_ID
        in_drop = state.phase_ids == DROP_PHASE_ID
        in_post_drop = state.phase_ids == POST_DROP_PHASE_ID

        # Walk pursues the box; brake handled by advance_phases_full.
        walk_twist = _twist_toward(self._base_env, box_xy_w, self._walk_speed_cap)
        walk_obs = _compute_walk_obs(self._base_env, walk_twist, last_action)
        walk_action = self._walk_policy({"actor": walk_obs})
        action = torch.where(in_walk.unsqueeze(-1), walk_action, action)

        if in_pick.any():
            pick_obs = self._pick_env.get_observations()
            pick_action = self._pick_policy(pick_obs)
            action = torch.where(in_pick.unsqueeze(-1), pick_action, action)

        if in_carry.any():
            carry_twist = _twist_toward(
                self._base_env, state.approach_target_xy, self._carry_speed_cap,
            )
            carry_obs = _compute_carry_obs(
                self._base_env, carry_twist, last_action,
                self._wrist_ids, self._left_hand_id, self._right_hand_id, self._trunk_id,
            )
            carry_action = self._carry_policy({"actor": carry_obs})
            action = torch.where(in_carry.unsqueeze(-1), carry_action, action)

        if in_drop.any():
            drop_obs = _compute_drop_obs(
                self._base_env, last_action,
                self._wrist_ids, self._left_hand_id, self._right_hand_id, self._trunk_id,
                state.place_target_pos_w, state.place_target_quat_w,
                self._grasp_left, self._grasp_right,
                state.object_target_pos_w, state.object_target_quat_w,
            )
            drop_action = self._drop_policy({"actor": drop_obs})
            action = torch.where(in_drop.unsqueeze(-1), drop_action, action)

        if in_post_drop.any():
            # Hold station post-drop with walk expert at zero twist.
            stand_twist = torch.zeros(self._num_envs, 3, device=self._device)
            stand_obs = _compute_walk_obs(self._base_env, stand_twist, last_action)
            stand_action = self._walk_policy({"actor": stand_obs})
            action = torch.where(in_post_drop.unsqueeze(-1), stand_action, action)

        return action


class CarryStateBuffer:
    """Circular GPU buffer of physics snapshots taken at phase-fire moments."""

    def __init__(self, capacity: int, num_joints: int, device: str, target_phase_id: int = CARRY_PHASE_ID):
        """Allocate the circular buffers; the slot write cursor starts at 0 and wraps.
        Args: capacity: max snapshots retained before oldest overwritten; num_joints: dof count for joint_pos / joint_vel buffers; device: torch device; target_phase_id: phase id to assign on teleport (CARRY or DROP).
        Returns: none.
        """
        self.capacity = capacity
        self.device = device
        self.target_phase_id = target_phase_id
        self.n_filled = 0
        self.cursor = 0
        self.robot_pose = torch.zeros(capacity, 7, device=device)
        self.robot_vel = torch.zeros(capacity, 6, device=device)
        self.joint_pos = torch.zeros(capacity, num_joints, device=device)
        self.joint_vel = torch.zeros(capacity, num_joints, device=device)
        self.box_pose = torch.zeros(capacity, 7, device=device)
        self.box_vel = torch.zeros(capacity, 6, device=device)
        self.approach_xy = torch.zeros(capacity, 2, device=device)
        self.place_target_pos_w = torch.zeros(capacity, 3, device=device)
        self.place_target_quat_w = torch.zeros(capacity, 4, device=device)
        self.place_target_quat_w[:, 0] = 1.0
        self.object_target_pos_w = torch.zeros(capacity, 3, device=device)
        self.object_target_quat_w = torch.zeros(capacity, 4, device=device)
        self.object_target_quat_w[:, 0] = 1.0

    def size(self) -> int:
        """Return the number of snapshots currently stored (<= capacity).
        Args: none.
        Returns: current fill level.
        """
        return self.n_filled

    def add_from_env(self, base_env, state: PhaseState, env_ids: torch.Tensor) -> None:
        """Snapshot physics + locked approach + locked place/object targets for env_ids into the next free slots.
        Args: base_env: mjlab env; state: current PhaseState (reads approach_target_xy, place/object_target_*); env_ids: 1-D long tensor of envs whose state should be captured.
        Returns: none.
        """
        n = int(env_ids.numel())
        if n == 0:
            return
        robot = base_env.scene["robot"]
        box = base_env.scene["box"]
        pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], dim=-1)[env_ids]
        vel = torch.cat([robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w], dim=-1)[env_ids]
        jp = robot.data.joint_pos[env_ids].clone()
        jv = robot.data.joint_vel[env_ids].clone()
        bp = torch.cat([box.data.root_link_pos_w, box.data.root_link_quat_w], dim=-1)[env_ids]
        bv = torch.cat([box.data.root_link_lin_vel_w, box.data.root_link_ang_vel_w], dim=-1)[env_ids]
        at = state.approach_target_xy[env_ids].clone()
        pt_pos = state.place_target_pos_w[env_ids].clone()
        pt_quat = state.place_target_quat_w[env_ids].clone()
        ot_pos = state.object_target_pos_w[env_ids].clone()
        ot_quat = state.object_target_quat_w[env_ids].clone()
        for i in range(n):
            slot = self.cursor
            self.robot_pose[slot] = pose[i]
            self.robot_vel[slot] = vel[i]
            self.joint_pos[slot] = jp[i]
            self.joint_vel[slot] = jv[i]
            self.box_pose[slot] = bp[i]
            self.box_vel[slot] = bv[i]
            self.approach_xy[slot] = at[i]
            self.place_target_pos_w[slot] = pt_pos[i]
            self.place_target_quat_w[slot] = pt_quat[i]
            self.object_target_pos_w[slot] = ot_pos[i]
            self.object_target_quat_w[slot] = ot_quat[i]
            self.cursor = (self.cursor + 1) % self.capacity
            if self.n_filled < self.capacity:
                self.n_filled += 1

    def teleport_into(self, base_env, state: PhaseState, env_ids: torch.Tensor, buf_idx: torch.Tensor) -> None:
        """Overwrite the physics + phase + locked-target state of env_ids with the snapshots at buf_idx.
        Args: base_env: mjlab env; state: PhaseState to mutate (sets phase_ids=target_phase_id and copies approach / place / object targets); env_ids: 1-D long tensor of envs to overwrite; buf_idx: 1-D long tensor of snapshot indices (same length as env_ids).
        Returns: none.
        """
        n = int(env_ids.numel())
        if n == 0 or self.n_filled == 0:
            return
        robot = base_env.scene["robot"]
        box = base_env.scene["box"]
        env_ids_i32 = env_ids.to(torch.int32)
        robot.write_root_link_pose_to_sim(self.robot_pose[buf_idx], env_ids=env_ids_i32)
        robot.write_root_link_velocity_to_sim(self.robot_vel[buf_idx], env_ids=env_ids_i32)
        robot.write_joint_state_to_sim(self.joint_pos[buf_idx], self.joint_vel[buf_idx], env_ids=env_ids_i32)
        box.write_root_link_pose_to_sim(self.box_pose[buf_idx], env_ids=env_ids_i32)
        box.write_root_link_velocity_to_sim(self.box_vel[buf_idx], env_ids=env_ids_i32)
        state.phase_ids[env_ids] = self.target_phase_id
        state.approach_target_xy[env_ids] = self.approach_xy[buf_idx]
        state.place_target_pos_w[env_ids] = self.place_target_pos_w[buf_idx]
        state.place_target_quat_w[env_ids] = self.place_target_quat_w[buf_idx]
        state.object_target_pos_w[env_ids] = self.object_target_pos_w[buf_idx]
        state.object_target_quat_w[env_ids] = self.object_target_quat_w[buf_idx]
        state.brake_counter[env_ids] = 0
        state.pick_stable_counter[env_ids] = 0
        state.in_pick_counter[env_ids] = 0
        state.carry_brake_counter[env_ids] = 0
        state.drop_done_ever[env_ids] = False


class WalkTerminalBuffer:
    """Loader and teleport helper for a pre-generated walk_terminal_states.npz."""

    def __init__(self, buffer_path: str, device: str):
        """Load the walk buffer npz into GPU tensors.
        Args: buffer_path: path to walk_terminal_states.npz (produced by src.collect_buffer with walk policy); device: torch device.
        Returns: none; raises FileNotFoundError if the buffer file does not exist.
        """
        import numpy as np
        self.device = device
        p = Path(buffer_path)
        if not p.exists():
            raise FileNotFoundError(
                f"walk terminal buffer not found at {p.resolve()}; "
                f"run: python -m src.collect_buffer --policy-module src.policy_walk --checkpoint <walk_ckpt> --out {p}"
            )
        arr = np.load(str(p))
        self.joint_pos = torch.from_numpy(arr["joint_pos"]).float().to(device)
        self.joint_vel = torch.from_numpy(arr["joint_vel"]).float().to(device)
        self.root_z = torch.from_numpy(arr["root_z"]).float().to(device)
        self.quat = torch.from_numpy(arr["quat_zy"]).float().to(device)
        self.lin_vel = torch.from_numpy(arr["lin_vel_by"]).float().to(device)
        self.ang_vel = torch.from_numpy(arr["ang_vel_by"]).float().to(device)
        self.n = self.joint_pos.shape[0]

    def size(self) -> int:
        """Return the number of walk-terminal snapshots in the file.
        Args: none.
        Returns: integer count.
        """
        return self.n

    def teleport_into(self, base_env, state: PhaseState, env_ids: torch.Tensor, buf_idx: torch.Tensor, table1_world_xy: torch.Tensor) -> None:
        """Teleport env_ids robots to sampled walk-end poses and schedule pick-phase handoff.
        Args: base_env: mjlab env; state: PhaseState to mutate (sets phase_ids=PICK, zeros per-env counters); env_ids: 1-D long tensor of envs to overwrite; buf_idx: 1-D long tensor of walk-buffer indices (same length as env_ids); table1_world_xy: (num_envs, 2) world XY of the pick table so robot pose stays within reach of the box.
        Returns: none.
        """
        n = int(env_ids.numel())
        if n == 0:
            return
        robot = base_env.scene["robot"]
        env_ids_i32 = env_ids.to(torch.int32)
        env_origins = base_env.scene.env_origins[env_ids]
        # 0.3 m pre-pick stance behind table1 keeps the robot within pick-command reach.
        pick_offset_xy = torch.zeros(n, 2, device=self.device)
        pick_offset_xy[:, 0] = -0.3
        root_xy_w = table1_world_xy[env_ids] + pick_offset_xy
        root_z_w = self.root_z[buf_idx].squeeze(-1) + env_origins[:, 2]
        pose = torch.cat([root_xy_w, root_z_w.unsqueeze(-1), self.quat[buf_idx]], dim=-1)
        vel = torch.cat([self.lin_vel[buf_idx], self.ang_vel[buf_idx]], dim=-1)
        robot.write_root_link_pose_to_sim(pose, env_ids=env_ids_i32)
        robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids_i32)
        robot.write_joint_state_to_sim(self.joint_pos[buf_idx], self.joint_vel[buf_idx], env_ids=env_ids_i32)
        state.phase_ids[env_ids] = PICK_PHASE_ID
        state.brake_counter[env_ids] = 0
        state.pick_stable_counter[env_ids] = 0
        state.in_pick_counter[env_ids] = 0
        state.carry_brake_counter[env_ids] = 0
        state.drop_done_ever[env_ids] = False


def _rollout_round(
    student: MonolithicPolicyStudent,
    oracle: FSMOracle,
    base_env,
    env,
    goal,
    state: PhaseState,
    last_action: torch.Tensor,
    ep_step: torch.Tensor,
    cfg: DAggerConfig,
    rollout_cfg: RolloutConfig,
    table1_world_xy: torch.Tensor,
    table2_world_xy: torch.Tensor,
    seed_offset: int,
    carry_buffer: Optional[CarryStateBuffer] = None,
    drop_buffer: Optional[CarryStateBuffer] = None,
    walk_buffer: Optional[WalkTerminalBuffer] = None,
    enable_phase_resets: bool = False,
) -> dict:
    """Collect one DAgger round: step the env with the student, label with the oracle.
    Args: student: current MonolithicPolicyStudent (eval mode); oracle: FSMOracle; base_env: mjlab env; env: RslRlVecEnvWrapper around base_env; goal: UnifiedGoal struct used in the obs; state: PhaseState (mutated in place); last_action: (N, 29) rolling previous student action (mutated in place); ep_step: (N,) per-env step counter (mutated); cfg: DAggerConfig; rollout_cfg: shared RolloutConfig; table1_world_xy / table2_world_xy: fixed world positions; seed_offset: integer shift for per-reset seeding; carry_buffer: in-memory snapshots at carry_fires (teleport sets phase=CARRY); drop_buffer: in-memory snapshots at drop_fires (teleport sets phase=DROP); walk_buffer: file-backed walk-terminal poses (teleport sets phase=PICK); enable_phase_resets: if True, apply the three per-phase reset fractions at the start of the round.
    Returns: dict with obs/action/phase arrays (on CPU) plus phase-fire counters.
    """
    device = base_env.scene["robot"].data.root_link_pos_w.device
    N = cfg.num_envs
    T = cfg.episode_max_steps

    # Teleport fractions of envs into deeper phases so the oracle labels pick/carry/drop states.
    n_pick_reset = 0
    n_carry_reset = 0
    n_drop_reset = 0
    if enable_phase_resets:
        all_perm = torch.randperm(N, device=device).to(torch.long)
        cursor = 0

        if walk_buffer is not None and walk_buffer.size() > 0 and cfg.pick_reset_fraction > 0.0:
            n_pick_reset = min(int(N * cfg.pick_reset_fraction), N - cursor)
            if n_pick_reset > 0:
                reset_ids = all_perm[cursor:cursor + n_pick_reset]
                buf_idx = torch.randint(0, walk_buffer.size(), (n_pick_reset,), device=device)
                walk_buffer.teleport_into(base_env, state, reset_ids, buf_idx, table1_world_xy)
                last_action[reset_ids] = 0.0
                ep_step[reset_ids] = 0
                cursor += n_pick_reset
                # Refresh pick-cmd buffers so they reflect the teleported pose.
                _snap_pick_commands_batched(base_env, env_ids=reset_ids)

        if carry_buffer is not None and carry_buffer.size() > 0 and cfg.carry_reset_fraction > 0.0:
            n_carry_reset = min(int(N * cfg.carry_reset_fraction), N - cursor)
            if n_carry_reset > 0:
                reset_ids = all_perm[cursor:cursor + n_carry_reset]
                buf_idx = torch.randint(0, carry_buffer.size(), (n_carry_reset,), device=device)
                carry_buffer.teleport_into(base_env, state, reset_ids, buf_idx)
                last_action[reset_ids] = 0.0
                ep_step[reset_ids] = 0
                cursor += n_carry_reset

        if drop_buffer is not None and drop_buffer.size() > 0 and cfg.drop_reset_fraction > 0.0:
            n_drop_reset = min(int(N * cfg.drop_reset_fraction), N - cursor)
            if n_drop_reset > 0:
                reset_ids = all_perm[cursor:cursor + n_drop_reset]
                buf_idx = torch.randint(0, drop_buffer.size(), (n_drop_reset,), device=device)
                drop_buffer.teleport_into(base_env, state, reset_ids, buf_idx)
                last_action[reset_ids] = 0.0
                ep_step[reset_ids] = 0
                cursor += n_drop_reset

    obs_rec: list[torch.Tensor] = []
    act_rec: list[torch.Tensor] = []
    phase_rec: list[torch.Tensor] = []

    pick_fires_total = 0
    carry_fires_total = 0
    drop_fires_total = 0
    drops_done_total = 0
    falls_total = 0
    fell_this_round = torch.zeros(N, dtype=torch.bool, device=device)
    reset_counter_local = seed_offset
    t_start = time.time()

    for t in range(T):
        # Snapshot phase before advance so (obs, action, phase) describe the same step.
        unified = _unified_obs_per_env(
            base_env, state.phase_ids.clamp(max=DROP_PHASE_ID), goal, last_action,
        )

        with torch.no_grad():
            student_action = student(unified)

        fires = advance_phases_full(
            base_env, state, table1_world_xy, table2_world_xy, cfg.gate, rollout_cfg,
        )
        pick_fires_total += int(fires["pick_fires"].sum())
        carry_fires_total += int(fires["carry_fires"].sum())
        drop_fires_total += int(fires["drop_fires"].sum())
        drops_done_total += int(fires["new_drop_done"].sum())
        if fires["pick_fires"].any():
            _snap_pick_commands_batched(
                base_env,
                env_ids=fires["pick_fires"].nonzero(as_tuple=False).squeeze(-1),
            )
        # Capture carry/drop poses so later rounds can teleport straight into those phases.
        if carry_buffer is not None and fires["carry_fires"].any():
            carry_ids = fires["carry_fires"].nonzero(as_tuple=False).squeeze(-1).to(torch.long)
            carry_buffer.add_from_env(base_env, state, carry_ids)
        if drop_buffer is not None and fires["drop_fires"].any():
            drop_ids = fires["drop_fires"].nonzero(as_tuple=False).squeeze(-1).to(torch.long)
            drop_buffer.add_from_env(base_env, state, drop_ids)

        oracle_action = oracle(state, last_action)
        obs_rec.append(unified.detach().cpu())
        act_rec.append(oracle_action.detach().cpu())
        phase_rec.append(state.phase_ids.clamp(max=DROP_PHASE_ID).detach().cpu().to(torch.int8))

        _, _, done, extras = env.step(student_action)
        last_action.copy_(student_action.detach())
        ep_step += 1

        up_align = -base_env.scene["robot"].data.projected_gravity_b[:, 2]
        new_fell = (up_align < 0.3) & ~fell_this_round
        falls_total += int(new_fell.sum())
        fell_this_round |= new_fell

        time_outs = extras.get("time_outs", torch.zeros_like(done).bool()) if isinstance(extras, dict) else torch.zeros_like(done).bool()
        succeeded = fires["new_drop_done"]
        timeout = ep_step >= cfg.episode_max_steps
        terminated = time_outs | succeeded | timeout | fell_this_round
        if terminated.any():
            done_ids = terminated.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
            teleport_done_envs(
                base_env, done_ids, cfg.gate,
                seed_offset=reset_counter_local, table2_world_xy=table2_world_xy,
            )
            reset_counter_local += 1
            state.reset_envs(done_ids)
            last_action[done_ids] = 0.0
            ep_step[done_ids] = 0
            fell_this_round[done_ids] = False
            new_goal = sample_goal(int(done_ids.numel()), device=device, seed=cfg.rollout_seed + reset_counter_local)
            for name in ("object_init_pos_w", "object_init_quat_w", "place_target_pos_w", "place_target_quat_w"):
                tensor = getattr(goal, name)
                tensor[done_ids] = getattr(new_goal, name)

    obs_arr = torch.cat(obs_rec, dim=0).numpy()
    act_arr = torch.cat(act_rec, dim=0).numpy()
    phase_arr = torch.cat(phase_rec, dim=0).numpy()
    wall_s = time.time() - t_start
    return {
        "unified_obs": obs_arr,
        "action": act_arr,
        "phase": phase_arr,
        "pick_fires": pick_fires_total,
        "carry_fires": carry_fires_total,
        "drop_fires": drop_fires_total,
        "drops_done": drops_done_total,
        "falls": falls_total,
        "n_pick_reset": n_pick_reset,
        "n_carry_reset": n_carry_reset,
        "n_drop_reset": n_drop_reset,
        "carry_buffer_size": (carry_buffer.size() if carry_buffer is not None else 0),
        "drop_buffer_size": (drop_buffer.size() if drop_buffer is not None else 0),
        "wall_s": wall_s,
    }


def _pick_amp_dtype(device: str) -> Optional[torch.dtype]:
    """Choose the autocast dtype for this GPU: bfloat16 on sm_80+, fp16 on sm_75-, None on CPU.
    Args: device: torch device string.
    Returns: torch.dtype or None if CPU / AMP not available.
    """
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _train_bc_round(
    bc_seed_tensors: Optional[dict],
    bc_seed_path: Path,
    round_paths: list[Path],
    warmstart_ckpt: Path,
    cfg: DAggerConfig,
    log_dir: Path,
    round_idx: int,
    writer,
    global_epoch_start: int,
) -> Path:
    """Retrain the student by MSE on the aggregated dataset, warm-starting from a checkpoint.
    Args: bc_seed_tensors: dict with 'obs'/'act'/'phase' GPU tensors for the BC seed (opt 1), or None if seed does not fit in bc_seed_gpu_budget_bytes; bc_seed_path: fallback .npz path when bc_seed_tensors is None; round_paths: list of per-round .npz files to concat (newest first in the window); warmstart_ckpt: previous student checkpoint used as init; cfg: DAggerConfig; log_dir: DAgger run dir where this round's checkpoint lands; round_idx: current round number; writer: SummaryWriter for scalar logs; global_epoch_start: offset for the TB x-axis so all rounds share one axis.
    Returns: Path to the retrained checkpoint for this round.
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"

    OUTLIER = 50.0

    # Round data loaded fresh from disk (small, grows with window).
    round_obs_list, round_act_list, round_phase_list = [], [], []
    for p in round_paths:
        blob = np.load(str(p))
        a = blob["action"]
        o = blob["unified_obs"]
        ph = blob["phase"].astype(np.int64)
        keep = np.abs(a).max(axis=1) < OUTLIER
        if (~keep).any():
            print(f"[dagger] filtered {int((~keep).sum())} outlier transitions in {p.name}")
        round_obs_list.append(torch.from_numpy(o[keep]).float().to(device, non_blocking=True))
        round_act_list.append(torch.from_numpy(a[keep]).float().to(device, non_blocking=True))
        round_phase_list.append(torch.from_numpy(ph[keep]).to(device, non_blocking=True))

    # Use the pre-cached GPU seed when available; otherwise reload from disk.
    if bc_seed_tensors is not None:
        seed_obs = bc_seed_tensors["obs"]
        seed_act = bc_seed_tensors["act"]
        seed_phase = bc_seed_tensors["phase"]
    else:
        blob = np.load(str(bc_seed_path))
        a = blob["action"]
        o = blob["unified_obs"]
        ph = blob["phase"].astype(np.int64)
        keep = np.abs(a).max(axis=1) < OUTLIER
        if (~keep).any():
            print(f"[dagger] filtered {int((~keep).sum())} outlier transitions in {bc_seed_path.name}")
        seed_obs = torch.from_numpy(o[keep]).float().to(device, non_blocking=True)
        seed_act = torch.from_numpy(a[keep]).float().to(device, non_blocking=True)
        seed_phase = torch.from_numpy(ph[keep]).to(device, non_blocking=True)

    obs_t = torch.cat([seed_obs] + round_obs_list, dim=0)
    act_t = torch.cat([seed_act] + round_act_list, dim=0)
    phase_t = torch.cat([seed_phase] + round_phase_list, dim=0)
    print(f"[dagger] round {round_idx}: training on {obs_t.shape[0]} transitions "
          f"(seed cached={bc_seed_tensors is not None}, round files={len(round_paths)})", flush=True)

    student = MonolithicPolicyStudent.load(str(warmstart_ckpt), device=device)
    # Freeze normaliser after round 0 so input scale does not drift with student distribution.
    if round_idx == 0 or not cfg.freeze_normalizer_after_round_zero:
        student.normalizer.update(obs_t)
    student.train()

    optim = torch.optim.AdamW(student.parameters(), lr=cfg.bc_lr, weight_decay=cfg.bc_weight_decay)
    loss_fn = nn.MSELoss()

    # Pay the held-out val cost only when early stop is enabled.
    n_all = obs_t.shape[0]
    use_early_stop = cfg.early_stop_patience > 0
    if use_early_stop:
        n_val = max(cfg.bc_batch_size, int(n_all * cfg.early_stop_val_frac))
        n_val = min(n_val, n_all // 4)
        perm = torch.randperm(n_all, device=device)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        tr_obs, tr_act, tr_phase = obs_t[train_idx], act_t[train_idx], phase_t[train_idx]
        va_obs, va_act = obs_t[val_idx], act_t[val_idx]
    else:
        tr_obs, tr_act, tr_phase = obs_t, act_t, phase_t
        va_obs = va_act = None
        n_val = 0
    n_tr = tr_obs.shape[0]

    if cfg.balance_phases:
        counts = torch.bincount(tr_phase, minlength=4).double()
        # Phases with no samples get zero target share before renormalisation.
        target = torch.tensor(cfg.phase_target_weights, dtype=torch.float64, device=device)
        target = target * (counts > 0).double()
        if target.sum() == 0:
            raise RuntimeError("[dagger] all phases have zero samples; cannot build sampler weights")
        target = target / target.sum()
        counts_safe = counts.clamp(min=1.0)
        per_phase_weight = target / counts_safe
        sampler_weights = per_phase_weight[tr_phase]
    else:
        sampler_weights = None

    amp_dtype = _pick_amp_dtype(device) if cfg.use_amp else None
    scaler = torch.amp.GradScaler("cuda") if (amp_dtype == torch.float16) else None

    batches_per_epoch = max(1, n_tr // cfg.bc_batch_size)
    best_val = float("inf")
    no_improve = 0
    stopped_at = cfg.epochs_per_round
    for epoch in range(cfg.epochs_per_round):
        student.train()
        epoch_loss_sum = torch.zeros((), device=device)
        for _ in range(batches_per_epoch):
            if sampler_weights is not None:
                idx = torch.multinomial(sampler_weights, cfg.bc_batch_size, replacement=True)
            else:
                idx = torch.randint(0, n_tr, (cfg.bc_batch_size,), device=device)
            x = tr_obs[idx]
            y = tr_act[idx]
            optim.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = student(x)
                    loss = loss_fn(pred, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    # bf16 skips the grad scaler.
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                    optim.step()
            else:
                pred = student(x)
                loss = loss_fn(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                optim.step()
            epoch_loss_sum = epoch_loss_sum + loss.detach() * cfg.bc_batch_size
        train_loss = (epoch_loss_sum / max(1, batches_per_epoch * cfg.bc_batch_size)).item()

        val_loss = float("nan")
        if use_early_stop:
            student.eval()
            with torch.no_grad():
                v_sum = torch.zeros((), device=device)
                seen = 0
                for s in range(0, n_val, cfg.bc_batch_size):
                    e = min(s + cfg.bc_batch_size, n_val)
                    v_pred = student(va_obs[s:e])
                    v_sum = v_sum + loss_fn(v_pred, va_act[s:e]) * (e - s)
                    seen += (e - s)
                val_loss = (v_sum / max(1, seen)).item()

        if not np.isfinite(train_loss) or (use_early_stop and not np.isfinite(val_loss)):
            raise RuntimeError(
                f"[dagger] round {round_idx} epoch {epoch}: non-finite loss "
                f"(train={train_loss}, val={val_loss}). Set cfg.use_amp=False and retry."
            )

        global_epoch = global_epoch_start + epoch
        writer.add_scalar("DAgger/train_loss", train_loss, global_epoch)
        if use_early_stop:
            writer.add_scalar("DAgger/val_loss", val_loss, global_epoch)
        writer.add_scalar("DAgger/round", round_idx, global_epoch)

        if use_early_stop:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience and epoch + 1 >= cfg.early_stop_min_epochs:
                    stopped_at = epoch + 1
                    print(f"[dagger] round {round_idx} early-stop at epoch {epoch} "
                          f"(no val improvement for {cfg.early_stop_patience} epochs, best_val={best_val:.5f})",
                          flush=True)
                    break

    writer.add_scalar("DAgger/epochs_run", stopped_at, round_idx)
    if use_early_stop:
        writer.add_scalar("DAgger/best_val", best_val, round_idx)

    student.eval()
    ckpt_path = log_dir / f"model_round_{round_idx:02d}.pt"
    student.save(ckpt_path)
    return ckpt_path


def dagger_finetune(
    bc_ckpt: Path | str,
    bc_dataset: Path | str,
    cfg: DAggerConfig = DAggerConfig(),
    log_dir: Optional[Path | str] = None,
) -> Path:
    """Run DAgger: alternate student rollout + oracle labelling + warm-started BC refit.
    Args: bc_ckpt: starting MonolithicPolicyStudent checkpoint (round 0 warm-start); bc_dataset: BC seed dataset .npz (kept permanently in the aggregated buffer); cfg: DAggerConfig; log_dir: output dir, defaults to outputs/logs/g1_monolithic_dagger/<timestamp>.
    Returns: Path to the final-round checkpoint.
    """
    from torch.utils.tensorboard import SummaryWriter

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.rollout_seed))
    torch.cuda.manual_seed_all(int(cfg.rollout_seed))
    np.random.seed(int(cfg.rollout_seed))

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"[dagger] log dir: {log_dir}", flush=True)

    # Pick scene with per-env table 2 (WPCD chain), same layout as fsm_wpcd_batched.
    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.env_spacing = cfg.env_spacing
    table2_rel_xy_np = _sample_table2_positions_np(
        cfg.num_envs, cfg.gate.table2_r_min, cfg.gate.table2_r_max, cfg.rollout_seed,
    )
    env_cfg.scene.spec_fn = _make_two_tables_spec_fn(cfg.num_envs, cfg.env_spacing, table2_rel_xy_np)
    _disable_terminations(env_cfg)
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    env_cfg.episode_length_s = float(cfg.episode_max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    print(f"[dagger] building env (num_envs={cfg.num_envs})...", flush=True)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    _ = env.get_observations()

    origins_t = base_env.scene.env_origins[:, :2]
    table1_world_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    table2_rel_torch = torch.from_numpy(table2_rel_xy_np).to(device)
    table2_world_xy = origins_t + table2_rel_torch

    print("[dagger] loading FSM oracle (walk/pick/carry/drop experts)...", flush=True)
    oracle = FSMOracle(
        base_env=base_env,
        num_envs=cfg.num_envs,
        device=device,
        walk_speed_cap=cfg.walk_speed_cap,
        carry_speed_cap=cfg.carry_speed_cap,
    )

    rollout_cfg = RolloutConfig(seed=cfg.rollout_seed, max_steps=cfg.episode_max_steps)
    rollout_cfg.walk_to_pick_dist = cfg.gate.walk_to_pick_dist

    state = PhaseState(cfg.num_envs, device)
    last_action = torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device)
    goal = sample_goal(cfg.num_envs, device=device, seed=cfg.rollout_seed)
    all_ids = torch.arange(cfg.num_envs, device=device, dtype=torch.long)
    teleport_done_envs(base_env, all_ids, cfg.gate, seed_offset=0, table2_world_xy=table2_world_xy)
    env.step(torch.zeros(cfg.num_envs, UNIFIED_ACT_DIM, device=device))
    _snap_pick_commands_batched(base_env)
    ep_step = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

    # Seed default place-target z so the drop expert has valid obs before the first drop_fire.
    default_place = oracle.default_place_target(table2_world_xy)
    state.place_target_pos_w.copy_(default_place)

    num_joints = base_env.scene["robot"].data.joint_pos.shape[-1]
    carry_buffer = CarryStateBuffer(
        capacity=cfg.carry_snapshot_buffer_size,
        num_joints=num_joints, device=device, target_phase_id=CARRY_PHASE_ID,
    ) if cfg.carry_reset_fraction > 0.0 else None
    drop_buffer = CarryStateBuffer(
        capacity=cfg.drop_snapshot_buffer_size,
        num_joints=num_joints, device=device, target_phase_id=DROP_PHASE_ID,
    ) if cfg.drop_reset_fraction > 0.0 else None
    walk_buffer: Optional[WalkTerminalBuffer] = None
    if cfg.pick_reset_fraction > 0.0:
        try:
            walk_buffer = WalkTerminalBuffer(cfg.walk_terminal_buffer_path, device=device)
        except FileNotFoundError as e:
            print(f"[dagger] pick_reset_fraction={cfg.pick_reset_fraction} but {e}; disabling pick reset", flush=True)
            walk_buffer = None
    if carry_buffer is not None or drop_buffer is not None or walk_buffer is not None:
        total_fraction = cfg.pick_reset_fraction + cfg.carry_reset_fraction + cfg.drop_reset_fraction
        if total_fraction > 1.0:
            raise ValueError(f"pick+carry+drop reset fractions sum to {total_fraction:.3f}, must be <= 1.0")
        print(f"[dagger] phase resets per round: pick={cfg.pick_reset_fraction} ({int(cfg.num_envs * cfg.pick_reset_fraction)} envs), "
              f"carry={cfg.carry_reset_fraction} ({int(cfg.num_envs * cfg.carry_reset_fraction)} envs), "
              f"drop={cfg.drop_reset_fraction} ({int(cfg.num_envs * cfg.drop_reset_fraction)} envs), "
              f"walk={1.0 - total_fraction:.2f} (fresh)",
              flush=True)

    cur_ckpt = Path(bc_ckpt)
    round_npz_paths: list[Path] = []
    bc_seed_path = Path(bc_dataset)
    assert bc_seed_path.exists(), f"BC seed dataset not found: {bc_seed_path}"

    # Pre-load the BC seed to GPU once; fall back to per-round disk reload above budget.
    bc_seed_tensors: Optional[dict] = None
    outlier = 50.0
    blob = np.load(str(bc_seed_path))
    a = blob["action"]
    o = blob["unified_obs"]
    ph = blob["phase"].astype(np.int64)
    keep = np.abs(a).max(axis=1) < outlier
    if (~keep).any():
        print(f"[dagger] filtered {int((~keep).sum())} BC seed outlier transitions")
    a, o, ph = a[keep], o[keep], ph[keep]

    # Phase-stratified subsample for low-VRAM machines.
    if cfg.bc_seed_max_transitions > 0 and len(a) > cfg.bc_seed_max_transitions:
        rng = np.random.default_rng(cfg.rollout_seed)
        counts_src = np.array([int((ph == p).sum()) for p in range(4)])
        totals = counts_src.sum()
        target_per_phase = np.zeros(4, dtype=np.int64)
        for p in range(4):
            target_per_phase[p] = max(1, int(round(cfg.bc_seed_max_transitions * counts_src[p] / totals)))
        # Correct rounding drift against the exact cap.
        delta = int(target_per_phase.sum()) - cfg.bc_seed_max_transitions
        if delta != 0:
            step = -1 if delta > 0 else 1
            for _ in range(abs(delta)):
                target_per_phase[int(target_per_phase.argmax())] += step
        pick_idx = []
        for p in range(4):
            src = np.where(ph == p)[0]
            if len(src) > 0:
                take = min(target_per_phase[p], len(src))
                pick_idx.append(rng.choice(src, size=take, replace=False))
        pick_idx = np.concatenate(pick_idx)
        rng.shuffle(pick_idx)
        a, o, ph = a[pick_idx], o[pick_idx], ph[pick_idx]
        print(f"[dagger] subsampled BC seed to {len(a)} transitions (cap={cfg.bc_seed_max_transitions})", flush=True)

    nbytes = a.nbytes + o.nbytes + ph.nbytes
    if nbytes <= cfg.bc_seed_gpu_budget_bytes:
        bc_seed_tensors = {
            "obs": torch.from_numpy(o).float().to(device, non_blocking=True),
            "act": torch.from_numpy(a).float().to(device, non_blocking=True),
            "phase": torch.from_numpy(ph).to(device, non_blocking=True),
        }
        print(f"[dagger] cached BC seed on GPU: "
              f"{bc_seed_tensors['obs'].shape[0]} transitions, "
              f"{nbytes/1e6:.1f} MB", flush=True)
    else:
        print(f"[dagger] BC seed is {nbytes/1e6:.0f} MB > budget "
              f"{cfg.bc_seed_gpu_budget_bytes/1e6:.0f} MB; will reload per round "
              f"(bump cfg.bc_seed_gpu_budget_bytes if VRAM allows)", flush=True)
    del blob, a, o, ph

    t_run_start = time.time()
    for round_idx in range(cfg.rounds):
        t_round_start = time.time()
        print(f"\n[dagger] round {round_idx + 1}/{cfg.rounds}", flush=True)

        print(f"[dagger] rolling out student ({cfg.num_envs} envs x {cfg.episode_max_steps} steps)...", flush=True)
        student = MonolithicPolicyStudent.load(str(cur_ckpt), device=device)
        student.eval()
        round_stats = _rollout_round(
            student=student,
            oracle=oracle,
            base_env=base_env,
            env=env,
            goal=goal,
            state=state,
            last_action=last_action,
            ep_step=ep_step,
            cfg=cfg,
            rollout_cfg=rollout_cfg,
            table1_world_xy=table1_world_xy,
            table2_world_xy=table2_world_xy,
            seed_offset=1 + round_idx * 1000,
            carry_buffer=carry_buffer,
            drop_buffer=drop_buffer,
            walk_buffer=walk_buffer,
            enable_phase_resets=(round_idx > 0),
        )
        rollout_s = round_stats["wall_s"]
        n_trans = round_stats["unified_obs"].shape[0]
        print(
            f"[dagger] round {round_idx} rollout: {n_trans} transitions in {_fmt_hms(rollout_s)} "
            f"(pick_fires={round_stats['pick_fires']}, carry_fires={round_stats['carry_fires']}, "
            f"drop_fires={round_stats['drop_fires']}, drops_done={round_stats['drops_done']}, "
            f"falls={round_stats['falls']}, resets p={round_stats['n_pick_reset']}/"
            f"c={round_stats['n_carry_reset']}/d={round_stats['n_drop_reset']}, "
            f"bufs c={round_stats['carry_buffer_size']}/d={round_stats['drop_buffer_size']})",
            flush=True,
        )
        writer.add_scalar("Rollout/pick_fires", round_stats["pick_fires"], round_idx)
        writer.add_scalar("Rollout/carry_fires", round_stats["carry_fires"], round_idx)
        writer.add_scalar("Rollout/drop_fires", round_stats["drop_fires"], round_idx)
        writer.add_scalar("Rollout/drops_done", round_stats["drops_done"], round_idx)
        writer.add_scalar("Rollout/falls", round_stats["falls"], round_idx)
        writer.add_scalar("Rollout/transitions", n_trans, round_idx)
        writer.add_scalar("Rollout/n_pick_reset", round_stats["n_pick_reset"], round_idx)
        writer.add_scalar("Rollout/n_carry_reset", round_stats["n_carry_reset"], round_idx)
        writer.add_scalar("Rollout/n_drop_reset", round_stats["n_drop_reset"], round_idx)
        writer.add_scalar("Rollout/carry_buffer_size", round_stats["carry_buffer_size"], round_idx)
        writer.add_scalar("Rollout/drop_buffer_size", round_stats["drop_buffer_size"], round_idx)
        writer.add_scalar("Perf/rollout_s", rollout_s, round_idx)

        round_npz = log_dir / f"round_{round_idx:02d}.npz"
        np.savez_compressed(
            round_npz,
            unified_obs=round_stats["unified_obs"],
            action=round_stats["action"],
            phase=round_stats["phase"],
        )
        round_npz_paths.append(round_npz)
        print(f"[dagger] round {round_idx} saved rollout to {round_npz}", flush=True)

        windowed_rounds = round_npz_paths[-cfg.buffer_window:]
        print(f"[dagger] round {round_idx} refit on BC seed + {len(windowed_rounds)} round file(s): "
              f"{[p.name for p in windowed_rounds]}", flush=True)
        cur_ckpt = _train_bc_round(
            bc_seed_tensors=bc_seed_tensors,
            bc_seed_path=bc_seed_path,
            round_paths=windowed_rounds,
            warmstart_ckpt=cur_ckpt,
            cfg=cfg,
            log_dir=log_dir,
            round_idx=round_idx,
            writer=writer,
            global_epoch_start=round_idx * cfg.epochs_per_round,
        )
        round_s = time.time() - t_round_start
        print(f"[dagger] round {round_idx} wall time: {_fmt_hms(round_s)}; current ckpt: {cur_ckpt}", flush=True)

        if round_idx > 0 and len(round_npz_paths) > cfg.buffer_window:
            # Drop files outside the sliding window to bound disk use.
            to_del = round_npz_paths[:-cfg.buffer_window]
            for p in to_del:
                if p.exists():
                    p.unlink()
            round_npz_paths = round_npz_paths[-cfg.buffer_window:]

    writer.close()
    total_s = time.time() - t_run_start
    print(f"[dagger] done. total wall time: {_fmt_hms(total_s)}. final ckpt: {cur_ckpt}", flush=True)
    return cur_ckpt


def main() -> None:
    """Parse CLI flags and run DAgger starting from a given BC checkpoint.
    Args: reads --bc-ckpt, --bc-dataset, --rounds, --num-envs, --max-steps, --epochs-per-round, --buffer-window, --carry-reset-fraction, --device from argv.
    Returns: none.
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bc-ckpt", required=True)
    ap.add_argument("--bc-dataset", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--epochs-per-round", type=int, default=60)
    ap.add_argument("--buffer-window", type=int, default=3)
    ap.add_argument("--carry-reset-fraction", type=float, default=0.25,
                    help="fraction of envs teleported to a sampled carry pose at round start")
    ap.add_argument("--pick-reset-fraction", type=float, default=0.0,
                    help="fraction of envs teleported to a sampled walk-terminal pose at round start")
    ap.add_argument("--drop-reset-fraction", type=float, default=0.0,
                    help="fraction of envs teleported to a sampled drop-fire snapshot at round start")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = DAggerConfig(
        num_envs=args.num_envs,
        rounds=args.rounds,
        episode_max_steps=args.max_steps,
        epochs_per_round=args.epochs_per_round,
        buffer_window=args.buffer_window,
        carry_reset_fraction=args.carry_reset_fraction,
        pick_reset_fraction=args.pick_reset_fraction,
        drop_reset_fraction=args.drop_reset_fraction,
        device=args.device,
    )
    dagger_finetune(args.bc_ckpt, args.bc_dataset, cfg=cfg)


if __name__ == "__main__":
    main()
