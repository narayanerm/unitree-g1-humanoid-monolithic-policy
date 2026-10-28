##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
##########################################################################################
"""Record an mp4 of walk.pt: forward twist for `--walk-steps`, then zero twist for `--stop-steps`.
Shows how quickly the policy brakes to a stand after locomotion momentum.
Run: `MUJOCO_GL=egl python3.10 scripts/walk_park_video.py --ckpt experts/walk.pt --out outputs/media/walk_park.mp4 --walk-steps 120 --stop-steps 200 --walk-vx 0.6`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fsm_continuous import _compute_walk_obs, _disable_terminations, _load_actor_from_state_dict
from src.policy_walk import TASK_ID as WALK_TASK_ID, register as register_walk


def main() -> None:
    """Parse CLI args, roll out walk-then-park, write mp4 and print drift stats.
    Args: none (reads argparse from sys.argv).
    Returns: none.
    """
    ap = argparse.ArgumentParser(description="Walk-then-park rollout: forward twist for N steps, then zero.")
    ap.add_argument("--ckpt", default="experts/walk.pt")
    ap.add_argument("--out", default="outputs/media/walk_park.mp4")
    ap.add_argument("--walk-steps", type=int, default=120, help="steps with twist=(walk_vx, 0, 0)")
    ap.add_argument("--stop-steps", type=int, default=200, help="steps with twist=(0, 0, 0) after the walk phase")
    ap.add_argument("--walk-vx", type=float, default=0.6, help="forward base-frame vx (m/s) during the walk phase")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    total_steps = args.walk_steps + args.stop_steps

    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig
    import mediapy

    register_walk()
    env_cfg = load_env_cfg(WALK_TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(total_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    agent_cfg = load_rl_cfg(WALK_TASK_ID)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    walk_policy = _load_actor_from_state_dict(
        ckpt_path=Path(args.ckpt),
        obs_dim=99, action_dim=29, hidden=(512, 256, 128), device=args.device,
    )

    _ = env.get_observations()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viewer_cfg = ViewerConfig(
        height=720, width=1280,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=4.0, elevation=-15.0, azimuth=135.0,
        max_extra_envs=0, env_idx=0,
        enable_shadows=True, enable_reflections=True,
    )
    renderer = OffscreenRenderer(model=base_env.sim.mj_model, cfg=viewer_cfg, scene=base_env.scene)
    renderer.initialize()

    last_action = torch.zeros(1, 29, device=args.device)
    walk_twist = torch.tensor([[args.walk_vx, 0.0, 0.0]], device=args.device)
    zero_twist = torch.zeros(1, 3, device=args.device)

    base_pos_log = []
    stop_start_xy = None
    with mediapy.VideoWriter(str(out_path), shape=(720, 1280), fps=args.fps) as writer:
        for step in range(total_steps):
            phase = "walk" if step < args.walk_steps else "stop"
            twist = walk_twist if phase == "walk" else zero_twist
            walk_obs = _compute_walk_obs(base_env, twist, last_action)
            with torch.no_grad():
                action = walk_policy({"actor": walk_obs})
            env.step(action)
            last_action = action.detach()
            renderer.update(base_env.sim.data)
            writer.add_image(renderer.render().copy())
            if step == args.walk_steps:
                p = base_env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
                stop_start_xy = (float(p[0]), float(p[1]))
            if step % 20 == 0 or step == args.walk_steps:
                p = base_env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
                v = base_env.scene["robot"].data.root_link_lin_vel_b[0].detach().cpu().numpy()
                base_pos_log.append((step, p[0], p[1], p[2], v[0], v[1]))
                print(f"[walk-park] {phase:4s} step {step:3d} base_xy=({p[0]:+.3f},{p[1]:+.3f}) z={p[2]:.3f} vxb={v[0]:+.3f} vyb={v[1]:+.3f}")

    renderer.close()
    base_env.close()

    if base_pos_log:
        x0, y0 = base_pos_log[0][1], base_pos_log[0][2]
        x1, y1 = base_pos_log[-1][1], base_pos_log[-1][2]
        drift = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        print(f"[walk-park] total drift over {total_steps} steps: {drift:.3f} m "
              f"(start xy=({x0:+.3f},{y0:+.3f}) -> end xy=({x1:+.3f},{y1:+.3f}))")
        if stop_start_xy is not None:
            xs, ys = stop_start_xy
            post_stop = ((x1 - xs) ** 2 + (y1 - ys) ** 2) ** 0.5
            print(f"[walk-park] post-stop drift ({args.stop_steps} zero-twist steps): {post_stop:.3f} m "
                  f"(stop_cmd xy=({xs:+.3f},{ys:+.3f}) -> end xy=({x1:+.3f},{y1:+.3f}))")
    print(f"[walk-park] wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
