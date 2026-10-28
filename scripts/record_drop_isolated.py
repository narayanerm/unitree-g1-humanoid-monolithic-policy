##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Record a 2x2 tracking-camera grid of the trained drop policy running in isolation.

Input:  drop checkpoint resolved via expert_registry; CLI flags for out path, device, seed.
Output: mp4 at ``outputs/media/drop_isolated_n4_latest.mp4`` showing per-env drop rollouts.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "outputs" / "media" / "drop_isolated_n4_latest.mp4"


def main() -> None:
    """Roll out the drop policy on num_envs envs and dump a tracking-camera grid mp4.
    Args: reads argparse flags for out path, device, seed, max-steps, num-envs, pane size, cam distance, fps.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--pane-h", type=int, default=540)
    ap.add_argument("--pane-w", type=int, default=960)
    ap.add_argument("--cam-distance", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=40)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig
    import mediapy

    from src.expert_registry import BY_NAME, resolve_ckpt
    from src.policy_drop import TASK_ID as DROP_TASK_ID
    from src.policy_drop import lock_curriculum_at_iter, register as register_drop

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    register_drop()
    # Lock curriculum at the trained iter so place_bias=0 (box starts in hands, not pre-placed on table); default schedule reaches bb=0 at iter >= 4000.
    lock_curriculum_at_iter(4900)
    print("[drop-iso] curriculum locked at iter 4900 (place_bias=0, box starts in hands)")
    env_cfg = load_env_cfg(DROP_TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = float(args.max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(DROP_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=args.device)
    runner.load(str(resolve_ckpt(BY_NAME["place"])), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    _ = env.get_observations()

    out_mp4 = Path(args.out).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    grid_cols = int(np.ceil(np.sqrt(args.num_envs)))
    grid_rows = int(np.ceil(args.num_envs / grid_cols))
    pane_h, pane_w = args.pane_h, args.pane_w
    renderers = []
    for env_i in range(args.num_envs):
        cfg = ViewerConfig(
            height=pane_h, width=pane_w,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=args.cam_distance,
            elevation=-20.0, azimuth=135.0,
            env_idx=env_i, max_extra_envs=0,
            enable_shadows=True, enable_reflections=True,
        )
        r = OffscreenRenderer(model=base_env.sim.mj_model, cfg=cfg, scene=base_env.scene)
        r.initialize()
        renderers.append(r)

    writer = mediapy.VideoWriter(
        str(out_mp4),
        shape=(grid_rows * pane_h, grid_cols * pane_w),
        fps=args.fps,
    )
    writer.__enter__()

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_ok = True
        font_size = max(18, int(pane_h * 0.045))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
    except ImportError:
        pil_ok = False

    box = base_env.scene["box"]
    robot = base_env.scene["robot"]
    up_align = torch.zeros(args.num_envs, device=args.device)
    fallen = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

    print(f"[drop-iso] rolling out {args.num_envs} envs for {args.max_steps} steps, seed={args.seed}")
    for step in range(args.max_steps):
        obs_dict = env.get_observations()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        with torch.no_grad():
            action = policy(obs_dict)
        env.step(action)

        box_z = box.data.root_link_pos_w[:, 2]
        root_z = robot.data.root_link_pos_w[:, 2]
        up_align = -robot.data.projected_gravity_b[:, 2]
        fell_now = (up_align < 0.25) | ((root_z < 0.15) & (up_align < 0.7))
        fallen |= fell_now

        if step % 20 == 0:
            print(f"[drop-iso] step {step:3d} box_z={box_z.detach().cpu().tolist()} "
                  f"up={up_align.detach().cpu().tolist()} fallen={fallen.sum().item()}")

        panes = []
        for env_i, r_pane in enumerate(renderers):
            r_pane.update(base_env.sim.data)
            img = r_pane.render().copy()
            status = "FELL" if bool(fallen[env_i]) else "DROP"
            bz = float(box_z[env_i].item())
            if pil_ok:
                pil = Image.fromarray(img)
                draw = ImageDraw.Draw(pil)
                label = f"env{env_i} {status} t={step} box_z={bz:.2f}"
                bar_h = int(font_size * 1.6)
                draw.rectangle([(0, 0), (int(pane_w * 0.70), bar_h)], fill=(0, 0, 0))
                draw.text((10, int(bar_h * 0.15)), label, fill=(255, 255, 255), font=font)
                img = np.array(pil)
            panes.append(img)
        rows = [np.hstack(panes[r * grid_cols : (r + 1) * grid_cols]) for r in range(grid_rows)]
        writer.add_image(np.vstack(rows))

    writer.__exit__(None, None, None)
    for r in renderers:
        r.close()
    base_env.close()

    print(f"[drop-iso] wrote {out_mp4}")
    print(f"[drop-iso] final: fallen={fallen.sum().item()}/{args.num_envs}")


if __name__ == "__main__":
    main()
