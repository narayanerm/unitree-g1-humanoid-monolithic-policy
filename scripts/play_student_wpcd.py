##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
##########################################################################################
"""Closed-loop rollout of the BC student in the two-table wpcd scene.

Mirrors the FSM wpcd env (table 1 + table 2 + robot + box per env) but
replaces every expert policy with the monolithic student. Phase ids are
tracked by the same physics gates used by the PPO loop, so the student
receives the correct one-hot each step and we can report how far the
chain progressed before any failure.

Use this to sanity-check a BC checkpoint before kicking off PPO: if the
student cannot even reach the box without the FSM's A* / brake / twist
scaffolding, PPO has nothing to polish.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsm_continuous import RolloutConfig, _disable_terminations
from src.fsm_continuous_batch import (
    PICK_PHASE_ID,
    WALK_PHASE_ID,
    _random_spawn_annulus,
    _snap_pick_commands_batched,
    _teleport_robot_batch,
    _unified_obs_per_env,
)
from src.fsm_wpcd_batched import (
    CARRY_PHASE_ID,
    DROP_PHASE_ID,
    POST_DROP_PHASE_ID,
    TABLE2_HALF_X,
    TABLE2_HALF_Y,
    _fmt_hms,
    _make_two_tables_spec_fn,
    _resample_spawn_outside_t2,
    _sample_table2_positions_np,
)
from src.goal_features import UNIFIED_ACT_DIM, sample_goal
from src.ppo_finetune import PPOFinetuneConfig, _PhaseState, _advance_phases_full
from src.student_policy import MonolithicPolicyStudent


def main() -> None:
    """Parse CLI flags and run the closed-loop student in the wpcd scene.
    Args: reads --ckpt, --num-envs, --seed, --max-steps, --r-min, --r-max, --t2-r-min, --t2-r-max, --out-video, --device from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="Path to a MonolithicPolicyStudent .pt checkpoint")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1600)
    ap.add_argument("--r-min", type=float, default=1.5)
    ap.add_argument("--r-max", type=float, default=5.0)
    ap.add_argument("--env-spacing", type=float, default=20.0)
    ap.add_argument("--t2-r-min", type=float, default=1.5)
    ap.add_argument("--t2-r-max", type=float, default=5.0)
    ap.add_argument("--out-video", default=str(ROOT / "outputs" / "media" / "play_student_wpcd.mp4"))
    ap.add_argument("--pane-h", type=int, default=540)
    ap.add_argument("--pane-w", type=int, default=960)
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Build env (mirrors fsm_wpcd_batched: 2 tables per env, same solver defaults).
    print(f"[play] loading env (num_envs={args.num_envs}, max_steps={args.max_steps})...", flush=True)
    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = args.env_spacing
    table2_rel_xy_np = _sample_table2_positions_np(args.num_envs, args.t2_r_min, args.t2_r_max, args.seed)
    env_cfg.scene.spec_fn = _make_two_tables_spec_fn(args.num_envs, args.env_spacing, table2_rel_xy_np)
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(args.max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=3.0)
    _ = env.get_observations()

    device_t = base_env.scene["robot"].data.root_link_pos_w.device
    origins_t = base_env.scene.env_origins[:, :2]
    table1_world_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    table2_world_xy = origins_t + torch.from_numpy(table2_rel_xy_np).to(device_t)

    # Load student.
    print(f"[play] loading student from {args.ckpt}...", flush=True)
    student = MonolithicPolicyStudent.load(args.ckpt, device=device)
    student.eval()

    # Spawn + initial goal.
    box_world = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    spawn_xy, spawn_yaw = _random_spawn_annulus(box_world, args.r_min, args.r_max, args.seed, device)
    spawn_xy = _resample_spawn_outside_t2(
        spawn_xy, box_world, table2_world_xy,
        r_min=args.r_min, r_max=args.r_max, safety=1.0, seed=args.seed, device=device,
    )
    cfg_rc = RolloutConfig(seed=args.seed, max_steps=args.max_steps)
    cfg_rc.walk_to_pick_dist = 0.55
    spawn_z = torch.full((args.num_envs,), cfg_rc.spawn_z, device=device_t)
    _teleport_robot_batch(base_env, spawn_xy, spawn_z, spawn_yaw)
    env.step(torch.zeros(args.num_envs, UNIFIED_ACT_DIM, device=device_t))
    _snap_pick_commands_batched(base_env)

    goal = sample_goal(args.num_envs, device=device_t, seed=args.seed)
    state = _PhaseState(args.num_envs, device_t)
    last_action = torch.zeros(args.num_envs, UNIFIED_ACT_DIM, device=device_t)

    # Phase-gate shared config.
    ppo_cfg = PPOFinetuneConfig(num_envs=args.num_envs, episode_max_steps=args.max_steps)

    # Grid video renderer (one pane per env).
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig
    import mediapy
    PHASE_LABELS = {WALK_PHASE_ID: "WALK", PICK_PHASE_ID: "PICK", CARRY_PHASE_ID: "CARRY", DROP_PHASE_ID: "DROP", POST_DROP_PHASE_ID: "DONE"}
    grid_cols = int(np.ceil(np.sqrt(args.num_envs)))
    grid_rows = int(np.ceil(args.num_envs / grid_cols))
    grid_renderers: list = []
    for env_i in range(args.num_envs):
        pane_cfg = ViewerConfig(
            height=args.pane_h, width=args.pane_w,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot", distance=3.5, elevation=-20.0, azimuth=135.0,
            env_idx=env_i, max_extra_envs=0,
            enable_shadows=True, enable_reflections=True,
        )
        r_pane = OffscreenRenderer(model=base_env.sim.mj_model, cfg=pane_cfg, scene=base_env.scene)
        r_pane.initialize()
        grid_renderers.append(r_pane)
    out_mp4 = Path(args.out_video).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    grid_h = grid_rows * args.pane_h
    grid_w = grid_cols * args.pane_w
    writer = mediapy.VideoWriter(str(out_mp4), shape=(grid_h, grid_w), fps=args.fps)
    writer.__enter__()
    try:
        from PIL import Image, ImageDraw, ImageFont
        _pil_ok = True
        _font = ImageFont.load_default(size=max(18, int(args.pane_h * 0.045)))
    except (ImportError, TypeError):
        _pil_ok = False
        _font = None

    # Trajectory stats per env.
    fell = torch.zeros(args.num_envs, dtype=torch.bool, device=device_t)
    max_abs_action = torch.zeros(args.num_envs, device=device_t)
    t_start = time.time()

    for step in range(args.max_steps):
        active = state.phase_ids != POST_DROP_PHASE_ID
        active &= ~fell
        if not bool(active.any()):
            break

        unified = _unified_obs_per_env(base_env, state.phase_ids, goal, last_action)
        with torch.no_grad():
            action = student(unified)
        max_abs_action = torch.maximum(max_abs_action, action.abs().amax(dim=-1))

        _advance_phases_full(base_env, state, table1_world_xy, table2_world_xy, ppo_cfg, cfg_rc)
        _, _, done, extras = env.step(action)
        last_action = action.detach()

        # Fall tracking.
        up_align = -base_env.scene["robot"].data.projected_gravity_b[:, 2]
        new_fell = (up_align < 0.3) & active
        fell = fell | new_fell

        # Progress every 50 steps.
        if step % 50 == 0:
            n_walk = int((state.phase_ids == WALK_PHASE_ID).sum())
            n_pick = int((state.phase_ids == PICK_PHASE_ID).sum())
            n_carry = int((state.phase_ids == CARRY_PHASE_ID).sum())
            n_drop = int((state.phase_ids == DROP_PHASE_ID).sum())
            n_done = int((state.phase_ids == POST_DROP_PHASE_ID).sum())
            n_fell = int(fell.sum())
            pct = 100.0 * (step + 1) / args.max_steps
            elapsed = time.time() - t_start
            eta = (args.max_steps - step - 1) * elapsed / max(1, step + 1)
            print(f"[play] step {step:4d}/{args.max_steps} ({pct:5.1f}%) W={n_walk} P={n_pick} C={n_carry} D={n_drop} DONE={n_done} fell={n_fell} max|a|={float(max_abs_action.max()):.2f} | elapsed={_fmt_hms(elapsed)} eta={_fmt_hms(eta)}", flush=True)

        # Render grid frame.
        panes = []
        for env_i, r_pane in enumerate(grid_renderers):
            r_pane.update(base_env.sim.data)
            img = r_pane.render().copy()
            phase_i = int(state.phase_ids[env_i].item())
            if bool(fell[env_i]):
                status = "FELL"
            else:
                status = PHASE_LABELS.get(phase_i, "?")
            if _pil_ok:
                pil = Image.fromarray(img)
                draw = ImageDraw.Draw(pil)
                label = f"env{env_i}  {status}  t={step}  |a|={float(max_abs_action[env_i]):.1f}"
                bar_h = int(_font.size * 1.6)
                bar_w = int(args.pane_w * 0.75)
                draw.rectangle([(0, 0), (bar_w, bar_h)], fill=(0, 0, 0))
                draw.text((10, int(bar_h * 0.15)), label, fill=(255, 255, 255), font=_font)
                img = np.array(pil)
            panes.append(img)
        while len(panes) < grid_rows * grid_cols:
            panes.append(np.zeros((args.pane_h, args.pane_w, 3), dtype=np.uint8))
        rows_imgs = [np.hstack(panes[r * grid_cols : (r + 1) * grid_cols]) for r in range(grid_rows)]
        writer.add_image(np.vstack(rows_imgs))

    writer.__exit__(None, None, None)
    for r_pane in grid_renderers:
        r_pane.close()
    base_env.close()

    # Final report.
    print()
    print(f"[play] wrote {out_mp4}")
    print(f"[play] per-env trajectory ({args.num_envs} envs, max_steps={args.max_steps}):")
    for i in range(args.num_envs):
        phase_i = int(state.phase_ids[i].item())
        status = "FELL" if bool(fell[i]) else PHASE_LABELS.get(phase_i, "?")
        drop_done = bool(state.drop_done_ever[i])
        print(f"  env{i}: final_phase={status:5s} drop_done={drop_done!s:5s} max|a|={float(max_abs_action[i]):8.2f}")
    print()
    print(f"[play] summary: reached_pick={int((state.phase_ids >= PICK_PHASE_ID).sum())}/{args.num_envs}  reached_carry={int((state.phase_ids >= CARRY_PHASE_ID).sum())}/{args.num_envs}  reached_drop={int((state.phase_ids >= DROP_PHASE_ID).sum())}/{args.num_envs}  drops_done={int(state.drop_done_ever.sum())}/{args.num_envs}  falls={int(fell.sum())}/{args.num_envs}")


if __name__ == "__main__":
    main()
