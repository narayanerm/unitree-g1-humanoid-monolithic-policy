##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
##########################################################################################
"""Oracle-driven collector for phase-rich BC data and drop-buffer seed.

Writes two npz files:
    --out-dataset       : (unified_obs, action, phase) to append to the BC seed.
    --out-drop-buffer   : drop-fire snapshots consumed by DAgger drop_reset_fraction.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dagger_finetune import (
    CarryStateBuffer,
    DAggerConfig,
    FSMOracle,
    WalkTerminalBuffer,
    _fmt_hms,
)
from src.fsm_continuous import RolloutConfig, _disable_terminations
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
    _make_two_tables_spec_fn,
    _sample_table2_positions_np,
)
from src.goal_features import UNIFIED_ACT_DIM, UNIFIED_OBS_DIM, sample_goal
from src.phase_gates import GateConfig, PhaseState, advance_phases_full, teleport_done_envs


def main() -> None:
    """Run the oracle-driven rollout and dump the dataset + drop-buffer npz pair."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--rounds", type=int, default=5,
                    help="Oracle rollout rounds; each round applies phase-resets then runs max-steps.")
    ap.add_argument("--out-dataset", default=str(ROOT / "outputs" / "phase_rich_batched.npz"))
    ap.add_argument("--out-drop-buffer", default=str(ROOT / "outputs" / "drop_terminal_states.npz"))
    ap.add_argument("--pick-start-fraction", type=float, default=0.3,
                    help="Fraction of envs teleported to walk-terminal (phase=PICK) at round 0.")
    ap.add_argument("--carry-reset-fraction", type=float, default=0.6,
                    help="Fraction of envs teleported to a captured carry pose at each round start (round>=1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    print(f"[collect] building env (num_envs={args.num_envs})...", flush=True)
    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 20.0
    table2_rel_xy_np = _sample_table2_positions_np(args.num_envs, 1.5, 5.0, args.seed)
    env_cfg.scene.spec_fn = _make_two_tables_spec_fn(args.num_envs, 20.0, table2_rel_xy_np)
    _disable_terminations(env_cfg)
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    env_cfg.episode_length_s = float(args.max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    _ = env.get_observations()

    origins_t = base_env.scene.env_origins[:, :2]
    table1_world_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    table2_rel_torch = torch.from_numpy(table2_rel_xy_np).to(device)
    table2_world_xy = origins_t + table2_rel_torch

    print("[collect] loading FSM oracle...", flush=True)
    oracle = FSMOracle(
        base_env=base_env, num_envs=args.num_envs, device=device,
        walk_speed_cap=1.0, carry_speed_cap=0.45,
    )
    gate_cfg = GateConfig(spawn_r_min=1.5, spawn_r_max=5.0, seed=args.seed)
    rollout_cfg = RolloutConfig(seed=args.seed, max_steps=args.max_steps)
    rollout_cfg.walk_to_pick_dist = gate_cfg.walk_to_pick_dist

    state = PhaseState(args.num_envs, device)
    last_action = torch.zeros(args.num_envs, UNIFIED_ACT_DIM, device=device)
    goal = sample_goal(args.num_envs, device=device, seed=args.seed)
    all_ids = torch.arange(args.num_envs, device=device, dtype=torch.long)
    teleport_done_envs(base_env, all_ids, gate_cfg, seed_offset=0, table2_world_xy=table2_world_xy)
    env.step(torch.zeros(args.num_envs, UNIFIED_ACT_DIM, device=device))
    _snap_pick_commands_batched(base_env)
    ep_step = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    default_place = oracle.default_place_target(table2_world_xy)
    state.place_target_pos_w.copy_(default_place)

    num_joints = base_env.scene["robot"].data.joint_pos.shape[-1]
    carry_buffer = CarryStateBuffer(2048, num_joints, device, target_phase_id=CARRY_PHASE_ID)
    drop_buffer = CarryStateBuffer(2048, num_joints, device, target_phase_id=DROP_PHASE_ID)
    walk_buffer_path = ROOT / "outputs" / "walk_terminal_states.npz"
    walk_buffer = WalkTerminalBuffer(str(walk_buffer_path), device=device) if walk_buffer_path.exists() else None
    if walk_buffer is None:
        print(f"[collect][WARN] {walk_buffer_path} missing; pick-start-fraction disabled.", flush=True)

    obs_rec: list[torch.Tensor] = []
    act_rec: list[torch.Tensor] = []
    phase_rec: list[torch.Tensor] = []
    pf_total = cf_total = df_total = dd_total = falls_total = 0

    t_run = time.time()
    for r in range(args.rounds):
        all_perm = torch.randperm(args.num_envs, device=device).to(torch.long)
        cursor = 0
        if walk_buffer is not None and args.pick_start_fraction > 0:
            n = int(args.num_envs * args.pick_start_fraction)
            if n > 0:
                reset_ids = all_perm[cursor:cursor + n]
                buf_idx = torch.randint(0, walk_buffer.size(), (n,), device=device)
                walk_buffer.teleport_into(base_env, state, reset_ids, buf_idx, table1_world_xy)
                last_action[reset_ids] = 0.0
                ep_step[reset_ids] = 0
                _snap_pick_commands_batched(base_env, env_ids=reset_ids)
                cursor += n
                print(f"[collect] round {r}: {n} envs teleported to PICK via walk_buffer", flush=True)
        if r > 0 and carry_buffer.size() > 0 and args.carry_reset_fraction > 0:
            n = min(int(args.num_envs * args.carry_reset_fraction), args.num_envs - cursor)
            if n > 0:
                reset_ids = all_perm[cursor:cursor + n]
                buf_idx = torch.randint(0, carry_buffer.size(), (n,), device=device)
                carry_buffer.teleport_into(base_env, state, reset_ids, buf_idx)
                last_action[reset_ids] = 0.0
                ep_step[reset_ids] = 0
                cursor += n
                print(f"[collect] round {r}: {n} envs teleported to CARRY (buf size={carry_buffer.size()})", flush=True)

        fell_this_round = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        reset_counter = r * 10_000 + 1
        t_round = time.time()
        for t in range(args.max_steps):
            unified = _unified_obs_per_env(base_env, state.phase_ids.clamp(max=DROP_PHASE_ID), goal, last_action)
            fires = advance_phases_full(base_env, state, table1_world_xy, table2_world_xy, gate_cfg, rollout_cfg)
            pf_total += int(fires["pick_fires"].sum())
            cf_total += int(fires["carry_fires"].sum())
            df_total += int(fires["drop_fires"].sum())
            dd_total += int(fires["new_drop_done"].sum())
            if fires["pick_fires"].any():
                _snap_pick_commands_batched(base_env, env_ids=fires["pick_fires"].nonzero(as_tuple=False).squeeze(-1))
            if fires["carry_fires"].any():
                carry_ids = fires["carry_fires"].nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                carry_buffer.add_from_env(base_env, state, carry_ids)
            if fires["drop_fires"].any():
                drop_ids = fires["drop_fires"].nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                drop_buffer.add_from_env(base_env, state, drop_ids)

            action = oracle(state, last_action)
            obs_rec.append(unified.detach())
            act_rec.append(action.detach())
            phase_rec.append(state.phase_ids.clamp(max=DROP_PHASE_ID).detach().to(torch.int8))

            _, _, done, extras = env.step(action)
            last_action.copy_(action.detach())
            ep_step += 1

            up_align = -base_env.scene["robot"].data.projected_gravity_b[:, 2]
            new_fell = (up_align < 0.3) & ~fell_this_round
            falls_total += int(new_fell.sum())
            fell_this_round |= new_fell

            time_outs = extras.get("time_outs", torch.zeros_like(done).bool()) if isinstance(extras, dict) else torch.zeros_like(done).bool()
            terminated = time_outs | fires["new_drop_done"] | (ep_step >= args.max_steps) | fell_this_round
            if terminated.any():
                done_ids = terminated.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                teleport_done_envs(base_env, done_ids, gate_cfg, seed_offset=reset_counter, table2_world_xy=table2_world_xy)
                reset_counter += 1
                state.reset_envs(done_ids)
                last_action[done_ids] = 0.0
                ep_step[done_ids] = 0
                fell_this_round[done_ids] = False

            if t % 100 == 0:
                print(f"[collect] round {r} step {t:4d}/{args.max_steps} | "
                      f"P={pf_total} C={cf_total} D={df_total} done={dd_total} falls={falls_total} "
                      f"| carry_buf={carry_buffer.size()} drop_buf={drop_buffer.size()}",
                      flush=True)
        print(f"[collect] round {r} wall={_fmt_hms(time.time()-t_round)}; "
              f"carry_buf={carry_buffer.size()} drop_buf={drop_buffer.size()}", flush=True)

    obs_arr = torch.cat(obs_rec, dim=0).cpu().numpy().astype(np.float32)
    act_arr = torch.cat(act_rec, dim=0).cpu().numpy().astype(np.float32)
    phase_arr = torch.cat(phase_rec, dim=0).cpu().numpy().astype(np.int8)
    base_env.close()

    # Filter corrupted samples (|action|>50 from unstable drop expert runs).
    OUTLIER = 50.0
    keep = np.abs(act_arr).max(axis=1) < OUTLIER
    n_drop_outlier = int((~keep).sum())
    if n_drop_outlier > 0:
        print(f"[collect] filtering {n_drop_outlier} outlier transitions", flush=True)
        obs_arr, act_arr, phase_arr = obs_arr[keep], act_arr[keep], phase_arr[keep]

    out_dataset = Path(args.out_dataset)
    out_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dataset), unified_obs=obs_arr, action=act_arr, phase=phase_arr)
    print(f"[collect] wrote dataset {out_dataset} ({obs_arr.shape[0]} transitions, "
          f"{out_dataset.stat().st_size/1e6:.1f} MB)", flush=True)
    for p in range(4):
        n_p = int((phase_arr == p).sum())
        print(f"  phase {p} ({['walk','pick','carry','drop'][p]}): {n_p} ({100*n_p/max(1,len(phase_arr)):.1f}%)")

    if drop_buffer.size() > 0:
        drop_buffer.save_to_npz(args.out_drop_buffer)
        print(f"[collect] wrote drop buffer {args.out_drop_buffer} ({drop_buffer.size()} snapshots)", flush=True)
    else:
        print(f"[collect][WARN] drop_buffer empty; drop seed not written. "
              f"Try more rounds or raising carry_reset_fraction.", flush=True)

    print(f"[collect] total wall={_fmt_hms(time.time()-t_run)} | "
          f"P={pf_total} C={cf_total} D={df_total} done={dd_total} falls={falls_total}", flush=True)


if __name__ == "__main__":
    main()
