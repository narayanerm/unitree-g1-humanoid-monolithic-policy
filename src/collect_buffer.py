##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Terminal-state buffer collector: roll out any trained policy and dump snapshots to .npz.

Input:  trained checkpoint (`--checkpoint`), target policy module (`--policy-module`), output path (`--out`).
Output: .npz keyed by SNAPSHOT_ENTITIES declared on the module (robot always, box optional).

Contract expected on the policy module:
    TASK_ID: str
    register() -> None
    SNAPSHOT_ENTITIES: list[str]   # optional, default ["robot"]; supported: "robot", "box"
    SNAPSHOT_YAW_ALIGN: bool       # optional, default True
    snapshot_predicate(env) -> BoolTensor[num_envs]  # optional; default is random horizon 50 to 400
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def collect(
    policy_module: str,
    checkpoint: str | Path,
    out_path: str | Path,
    num_envs: int = 2048,
    n_snapshots: int = 20_000,
    min_steps: int = 50,
    max_steps: int = 400,
    device: str = "cuda:0",
    seed: int = 0,
) -> Path:
    """Roll out `policy_module`'s trained policy and store yaw-aligned snapshots to `.npz`.
    Args: policy_module: dotted path (e.g. 'src.policy_walk'); checkpoint: trained .pt; out_path: output .npz; num_envs: parallel envs; n_snapshots: target sample count; min_steps/max_steps: random horizon bounds (used when no snapshot_predicate); device: torch device; seed: RNG seed.
    Returns: Path to the written .npz.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul

    torch.manual_seed(seed)
    np.random.seed(seed)

    mod = importlib.import_module(policy_module)
    mod.register()
    task_id = mod.TASK_ID
    entities = list(getattr(mod, "SNAPSHOT_ENTITIES", ["robot"]))
    yaw_align = bool(getattr(mod, "SNAPSHOT_YAW_ALIGN", True))
    predicate = getattr(mod, "snapshot_predicate", None)

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task_id)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    step_cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
    next_snap = torch.randint(min_steps, max_steps + 1, (num_envs,), device=device)

    buf: dict[str, list[torch.Tensor]] = defaultdict(list)
    saved = 0
    tag = f"[collect_buffer:{policy_module.rsplit('.', 1)[-1]}]"
    print(f"{tag} collecting {n_snapshots} states across {num_envs} envs "
          f"(entities={entities}, yaw_align={yaw_align}, "
          f"trigger={'predicate' if predicate else f'random {min_steps}-{max_steps}'})")

    while saved < n_snapshots:
        with torch.no_grad():
            action = policy(obs)
        obs, _, dones, _ = env.step(action)
        step_cnt += 1
        if dones.any():
            step_cnt[dones.bool()] = 0

        if predicate is not None:
            ready_mask = predicate(base_env) & (~dones.bool())
        else:
            ready_mask = (step_cnt >= next_snap) & (~dones.bool())
        ready = ready_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(ready) > 0:
            _snapshot(base_env, ready, entities, yaw_align, buf,
                      quat_mul, quat_apply, quat_from_euler_xyz)
            saved += len(ready)
            step_cnt[ready] = 0
            if predicate is None:
                next_snap[ready] = torch.randint(min_steps, max_steps + 1, (len(ready),), device=device)
            if saved // num_envs > (saved - len(ready)) // num_envs:
                print(f"  {saved}/{n_snapshots}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{k: torch.cat(v)[:n_snapshots].cpu().numpy() for k, v in buf.items()},
    )
    base_env.close()
    print(f"{tag} wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def _snapshot(base_env, env_ids, entities, yaw_align, buf,
              quat_mul, quat_apply, quat_from_euler_xyz) -> None:
    """Append yaw-aligned snapshots for the requested entities to `buf`.
    Args: base_env: ManagerBasedRlEnv; env_ids: envs to snapshot; entities: subset of {"robot","box"}; yaw_align: rotate world tensors into robot body-yaw frame; buf: dict of tensor lists to extend; quat_*: mjlab math helpers.
    Returns: none.
    """
    robot = base_env.scene["robot"]
    robot_pose = robot.data.root_link_pose_w[env_ids]
    robot_lin_vel = robot.data.root_link_lin_vel_w[env_ids]
    robot_ang_vel = robot.data.root_link_ang_vel_w[env_ids]

    root_pos = robot_pose[:, 0:3]
    root_quat = robot_pose[:, 3:7]

    if yaw_align:
        yaw = _yaw_from_quat(root_quat)
        zero = torch.zeros_like(yaw)
        q_yaw_inv = quat_from_euler_xyz(zero, zero, -yaw)
        def rot_vec(v: torch.Tensor) -> torch.Tensor:
            return quat_apply(q_yaw_inv, v)
        def rot_quat(q: torch.Tensor) -> torch.Tensor:
            return quat_mul(q_yaw_inv, q)
    else:
        def rot_vec(v: torch.Tensor) -> torch.Tensor:
            return v
        def rot_quat(q: torch.Tensor) -> torch.Tensor:
            return q

    if "robot" in entities:
        buf["root_z"].append(root_pos[:, 2:3].clone())
        buf["quat_zy"].append(rot_quat(root_quat).clone())
        buf["lin_vel_by"].append(rot_vec(robot_lin_vel).clone())
        buf["ang_vel_by"].append(rot_vec(robot_ang_vel).clone())
        buf["joint_pos"].append(robot.data.joint_pos[env_ids].clone())
        buf["joint_vel"].append(robot.data.joint_vel[env_ids].clone())

    if "box" in entities:
        box = base_env.scene["box"]
        box_pose = box.data.root_link_pose_w[env_ids]
        box_pos_w = box_pose[:, 0:3]
        box_quat_w = box_pose[:, 3:7]
        buf["box_pos_by"].append(rot_vec(box_pos_w - root_pos).clone())
        buf["box_quat_by"].append(rot_quat(box_quat_w).clone())
        buf["box_lin_vel_by"].append(rot_vec(box.data.root_link_lin_vel_w[env_ids]).clone())
        buf["box_ang_vel_by"].append(rot_vec(box.data.root_link_ang_vel_w[env_ids]).clone())


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract the yaw angle from a wxyz quaternion.
    Args: q: quaternion tensor with last dim = 4.
    Returns: yaw angle tensor with one fewer dim.
    """
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _main() -> None:
    """CLI entrypoint: parse args and invoke collect.
    Args: none.
    Returns: none.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy-module", required=True,
                    help="dotted path, e.g. src.policy_walk")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--n-snapshots", type=int, default=20_000)
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    collect(
        policy_module=args.policy_module,
        checkpoint=args.checkpoint,
        out_path=args.out,
        num_envs=args.num_envs,
        n_snapshots=args.n_snapshots,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
