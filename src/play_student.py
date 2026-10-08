##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/humanoid-loco-manipulation                   #
##########################################################################################
"""Closed-loop rollout of the monolithic student in the pick scene.

Loads a ``MonolithicPolicyStudent`` checkpoint, rolls it out for N episodes with
different spawns and goals, prints a pass/fail table, and writes an mp4 of the
first episode with a tracking camera on the robot.

The FSM phase fed to the student mirrors ``fsm_continuous_batch``:
``phase=walk`` while far from the box, flips to ``phase=pick`` once the walk
brake predicate fires (close + slow + upright). No expert weights are used: the
student's own action closes the loop.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.fsm_continuous import RolloutConfig, _disable_terminations
from src.fsm_continuous_batch import (
    PICK_PHASE_ID,
    WALK_PHASE_ID,
    _random_spawn_annulus,
    _teleport_robot_batch,
    _unified_obs_per_env,
)
from src.goal_features import UNIFIED_ACT_DIM, sample_goal
from src.student_policy import MonolithicPolicyStudent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "outputs" / "logs" / "g1_monolithic"
DEFAULT_VIDEO = ROOT / "outputs" / "media" / "student_rollout.mp4"


@dataclass
class EpisodeResult:
    """Outcome of one rollout."""
    seed: int
    steps: int
    reached_pick: bool
    lifted: bool
    fell: bool
    min_dist: float


def _find_latest_ckpt(root: Path) -> Path:
    """Return the newest ``model_best.pt`` under ``outputs/logs/g1_monolithic/*``.
    Args: root: directory containing timestamped run subfolders.
    Returns: Path to ``model_best.pt`` in the most recent run.
    """
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"no runs under {root}")
    ck = runs[-1] / "model_best.pt"
    if not ck.exists():
        raise FileNotFoundError(f"no model_best.pt in {runs[-1]}")
    return ck


def _rollout_episode(
    base_env,
    env,
    student: MonolithicPolicyStudent,
    cfg: RolloutConfig,
    seed: int,
    spawn_r_min: float,
    spawn_r_max: float,
    device: str,
    writer=None,
    renderer=None,
) -> EpisodeResult:
    """Reset the scene, spawn the robot, and roll the student until success, fall, or timeout.
    Args: base_env: mjlab env (num_envs==1); env: RslRlVecEnvWrapper around it; student: loaded policy; cfg: shared rollout thresholds; seed: RNG seed for spawn+goal; spawn_r_min/max: annulus radii around the box; device: torch device; writer: optional mediapy VideoWriter; renderer: optional OffscreenRenderer.
    Returns: ``EpisodeResult`` summarizing the episode.
    """
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

    base_env.reset()  # restore joints to default and rerandomise box; without this the fallen pose of the previous episode leaks into the next spawn

    goal = sample_goal(1, device=device, seed=seed)

    box_xy = base_env.scene["box"].data.root_link_pos_w[:, :2].clone()
    spawn_xy, spawn_yaw = _random_spawn_annulus(box_xy, spawn_r_min, spawn_r_max, seed, device)
    spawn_z = torch.full((1,), cfg.spawn_z, device=device)
    _teleport_robot_batch(base_env, spawn_xy, spawn_z, spawn_yaw)
    env.step(torch.zeros(1, UNIFIED_ACT_DIM, device=device))  # flush teleport

    phase_ids = torch.zeros(1, dtype=torch.long, device=device)
    last_action = torch.zeros(1, UNIFIED_ACT_DIM, device=device)
    min_dist = float("inf")
    reached_pick = False
    lifted = False
    fell = False
    steps = 0
    brake_counter = 0

    for step in range(cfg.max_steps):
        robot = base_env.scene["robot"]
        box = base_env.scene["box"]

        up_align = float(-robot.data.projected_gravity_b[0, 2])
        root_z = float(robot.data.root_link_pos_w[0, 2])
        box_z = float(box.data.root_link_pos_w[0, 2])
        dist = float(torch.linalg.norm(
            box.data.root_link_pos_w[0, :2] - robot.data.root_link_pos_w[0, :2]
        ))
        base_vx = float(robot.data.root_link_lin_vel_b[0, 0].abs())
        min_dist = min(min_dist, dist)

        if up_align < 0.25 or (root_z < 0.15 and up_align < 0.7):
            fell = True
            steps = step
            break

        upright = up_align > cfg.upright_min
        in_brake = int(phase_ids[0]) == WALK_PHASE_ID and dist <= cfg.walk_to_pick_dist
        brake_counter = brake_counter + 1 if in_brake else 0
        if (
            int(phase_ids[0]) == WALK_PHASE_ID
            and upright
            and dist <= cfg.walk_to_pick_dist
            and base_vx < cfg.walk_to_pick_speed_max
            and brake_counter >= cfg.walk_to_pick_brake_steps
        ):
            phase_ids[0] = PICK_PHASE_ID
            reached_pick = True

        unified = _unified_obs_per_env(base_env, phase_ids, goal, last_action)
        with torch.no_grad():
            action = student(unified)

        env.step(action)
        last_action = action.detach()
        steps = step + 1

        if writer is not None and renderer is not None:
            renderer.update(base_env.sim.data)
            writer.add_image(renderer.render().copy())

        if box_z > cfg.pick_to_carry_box_z and upright:
            lifted = True
            break

    return EpisodeResult(
        seed=seed,
        steps=steps,
        reached_pick=reached_pick,
        lifted=lifted,
        fell=fell,
        min_dist=min_dist,
    )


def run(
    ckpt_path: Optional[Path],
    num_episodes: int,
    base_seed: int,
    device: str,
    spawn_r_min: float,
    spawn_r_max: float,
    max_steps: int,
    video_path: Optional[Path],
    video_hw: tuple[int, int],
    video_fps: int,
    video_episodes: int,
) -> list[EpisodeResult]:
    """Run ``num_episodes`` rollouts, print a summary table, optionally record the first ``video_episodes``.
    Args: ckpt_path: student .pt (auto-discovered under ``outputs/logs/g1_monolithic`` if None); num_episodes: rollouts; base_seed: seed for episode 0 (increments per episode); device: torch device; spawn_r_min/max: annulus radii; max_steps: per-episode step cap; video_path: mp4 destination (skips render if None); video_hw: (h, w); video_fps: framerate; video_episodes: how many leading episodes to concatenate into the mp4.
    Returns: list of ``EpisodeResult``.
    """
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from src.policy_pick import TASK_ID as PICK_TASK_ID
    from src.policy_pick import register as register_pick

    ckpt_path = ckpt_path or _find_latest_ckpt(DEFAULT_CKPT)
    print(f"[play] student ckpt: {ckpt_path}")

    register_pick()
    env_cfg = load_env_cfg(PICK_TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    _disable_terminations(env_cfg)
    env_cfg.episode_length_s = float(max_steps) * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    agent_cfg = load_rl_cfg(PICK_TASK_ID)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    _ = env.get_observations()  # prime the manager buffers

    student = MonolithicPolicyStudent.load(ckpt_path, device=device)
    student.eval()

    cfg = RolloutConfig(max_steps=max_steps, device=device)
    cfg.walk_to_pick_dist = 0.55  # matches fsm_continuous_batch widened envelope

    writer = None
    renderer = None
    if video_path is not None:
        from mjlab.viewer.offscreen_renderer import OffscreenRenderer
        from mjlab.viewer.viewer_config import ViewerConfig
        import mediapy

        h, w = video_hw
        viewer_cfg = ViewerConfig(
            height=h, width=w,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
            distance=4.0, elevation=-15.0, azimuth=135.0,
            max_extra_envs=0, env_idx=0,
            enable_shadows=True, enable_reflections=True,
        )
        renderer = OffscreenRenderer(model=base_env.sim.mj_model, cfg=viewer_cfg, scene=base_env.scene)
        renderer.initialize()
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = mediapy.VideoWriter(str(video_path), shape=(h, w), fps=video_fps)
        writer.__enter__()

    results: list[EpisodeResult] = []
    for i in range(num_episodes):
        seed = base_seed + i
        record = writer is not None and i < video_episodes
        res = _rollout_episode(
            base_env, env, student, cfg,
            seed=seed, spawn_r_min=spawn_r_min, spawn_r_max=spawn_r_max,
            device=device,
            writer=writer if record else None,
            renderer=renderer if record else None,
        )
        results.append(res)
        tag = "LIFT" if res.lifted else ("FELL" if res.fell else ("REACH" if res.reached_pick else "MISS"))
        print(
            f"[play] ep {i:02d} seed={seed} steps={res.steps:4d} "
            f"min_dist={res.min_dist:.2f} phase_pick={res.reached_pick} "
            f"lifted={res.lifted} fell={res.fell}  [{tag}]"
        )

    if writer is not None:
        writer.__exit__(None, None, None)
        print(f"[play] video: {video_path}")
    if renderer is not None:
        renderer.close()
    base_env.close()

    n = len(results)
    lifted = sum(r.lifted for r in results)
    reached = sum(r.reached_pick for r in results)
    fell = sum(r.fell for r in results)
    print(
        f"\n[play] summary: {n} eps | lift={lifted}/{n} ({100 * lifted / n:.0f}%) | "
        f"reached_pick={reached}/{n} ({100 * reached / n:.0f}%) | fell={fell}/{n} ({100 * fell / n:.0f}%)"
    )
    return results


def _main() -> None:
    """Parse CLI flags and run the student rollout(s).
    Args: none, reads --ckpt, --episodes, --base-seed, --device, --r-min, --r-max, --max-steps, --video-path, --no-video, --video-episodes, --video-hw, --video-fps from argv.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="Path to model_best.pt (default: latest under outputs/logs/g1_monolithic).")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--video-path", default=str(DEFAULT_VIDEO))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-episodes", type=int, default=3, help="How many leading episodes to include in the mp4.")
    ap.add_argument("--video-hw", type=int, nargs=2, default=(720, 1280), metavar=("H", "W"))
    ap.add_argument("--video-fps", type=int, default=20)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    run(
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        num_episodes=args.episodes,
        base_seed=args.base_seed,
        device=args.device,
        spawn_r_min=args.r_min,
        spawn_r_max=args.r_max,
        max_steps=args.max_steps,
        video_path=None if args.no_video else Path(args.video_path),
        video_hw=tuple(args.video_hw),
        video_fps=args.video_fps,
        video_episodes=args.video_episodes,
    )


if __name__ == "__main__":
    _main()
