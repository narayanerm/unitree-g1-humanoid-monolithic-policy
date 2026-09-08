##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Shared utilities for the notebooks: checkpoint discovery, TB parsing, math helpers.

Input:  paths under ``outputs/logs/`` produced by rsl_rl / TensorBoard.
Output: pathlib.Path to a checkpoint or TB dir, (steps, values) arrays, or MuJoCo (model, data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def find_latest_checkpoint(log_root: str | Path, experiment_hint: str = "") -> Path | None:
    """Locate the most recently modified model_*.pt beneath log_root.
    Args: log_root: search root; experiment_hint: exact path-segment match (avoids `g1_carry` grabbing `g1_carry_slosh`).
    Returns: Path to the newest checkpoint, or None if nothing matched.
    """
    root = Path(log_root).expanduser()
    all_ckpts = sorted(root.glob("**/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if experiment_hint:
        matching = [p for p in all_ckpts if experiment_hint in p.parts]
        return matching[-1] if matching else None
    return all_ckpts[-1] if all_ckpts else None


def checkpoint_obs_dim(ckpt_path: str | Path) -> int:
    """Return the actor's input dimension baked into a saved checkpoint.
    Args: ckpt_path: path to a ``model_*.pt`` file.
    Returns: input dim (int) inferred from the first MLP layer weight.
    """
    import torch
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["actor_state_dict"]
    first_w = next(v for k, v in sd.items() if k.endswith(".mlp.0.weight") or k.endswith("0.weight"))
    return int(first_w.shape[1])


def parse_tb_reward(log_dir: str | Path, tag: str = "Train/mean_reward") -> tuple[list[int], list[float]]:
    """Read scalar values for a tag from a TensorBoard event file.
    Args: log_dir: directory holding events file; tag: scalar tag name.
    Returns: tuple (steps list, values list); empty on failure.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            scalars = ea.Scalars(tag)
            return [s.step for s in scalars], [s.value for s in scalars]
    except Exception:
        pass
    return [], []


def find_latest_tb_log(log_root: str | Path, experiment_hint: str = "") -> Path | None:
    """Locate the directory holding the newest TensorBoard event file.
    Args: log_root: search root; experiment_hint: substring filter on path.
    Returns: Path to the parent directory, or None if no events file was found.
    """
    root = Path(log_root).expanduser()
    pattern = "**/events.out.tfevents.*"
    all_events = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if experiment_hint:
        matching = [p for p in all_events if experiment_hint in str(p)]
        events = matching or all_events
    else:
        events = all_events
    return events[-1].parent if events else None


def load_scene_model():
    """Build the G1 humanoid with a groundplane and initialise MuJoCo state.
    Args: none.
    Returns: tuple (mjModel, mjData) after one mj_forward.
    """
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    spec = get_spec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.05),
        rgba=(0.3, 0.35, 0.4, 1.0),
        condim=3,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def export_policy_to_onnx(
    checkpoint_path: str | Path,
    task_id: str,
    out_path: str | Path,
    obs_group: str = "actor",
) -> Path:
    """Export a trained rsl-rl actor to a deployable .onnx with embedded metadata.
    Args: checkpoint_path: trained actor .pt; task_id: mjlab task; out_path: .onnx destination; obs_group: obs group key to export.
    Returns: Path to the written .onnx file.
    """
    import json

    import torch
    import torch.nn as nn

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.scripts.train import TrainConfig
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from dataclasses import asdict

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Throwaway env so the runner can size the actor to real obs/action dims.
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg.agent), "/tmp/onnx_export_runner", device)

    # Trunk-frozen-diagnostic checkpoints replaced mlp.6 with Sequential; mirror the swap so shapes match.
    sd = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if any(k.startswith("actor.mlp.6.0.") for k in sd.get("actor_state_dict", {})) or \
       any(k.startswith("mlp.6.0.")       for k in sd.get("actor_state_dict", {})):
        old_head = runner.alg.actor.mlp[6]
        in_dim, out_dim = old_head.in_features, old_head.out_features
        runner.alg.actor.mlp[6] = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ELU(), nn.Linear(64, out_dim),
        ).to(device=old_head.weight.device, dtype=old_head.weight.dtype)
        print("[export] Detected trunk-frozen-diagnostic architecture; head swapped to Seq(Linear,ELU,Linear).")

    runner.load(str(checkpoint_path), map_location=device)
    actor = runner.alg.actor.eval()

    obs_dim = int(env.get_observations()[0][obs_group].shape[-1])
    action_dim = int(env.num_actions)

    # Wraps raw obs -> normalize -> MLP -> deterministic action.
    class Deployable(nn.Module):
        """Fuse observation normaliser and actor MLP into a single deployable module."""

        def __init__(self, actor_ref):
            """Capture normaliser stats and MLP from a trained actor.
            Args: actor_ref: trained actor exposing obs_normalizer and mlp.
            Returns: none.
            """
            super().__init__()
            self.register_buffer("norm_mean", actor_ref.obs_normalizer._mean.detach().clone())
            self.register_buffer("norm_std", actor_ref.obs_normalizer._std.detach().clone())
            self.eps = float(actor_ref.obs_normalizer.eps)
            self.mlp = actor_ref.mlp

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """Normalise raw observations and produce the deterministic action.
            Args: obs: raw observation tensor.
            Returns: action tensor from the MLP mean output.
            """
            x = (obs - self.norm_mean) / (self.norm_std + self.eps)
            return self.mlp(x)

    deployable = Deployable(actor).to(device).eval()
    dummy = torch.zeros(1, obs_dim, device=device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deployable,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    # Embed metadata (values must be strings) so the deploy side reads joint order and timing.
    import onnx
    m = onnx.load(str(out_path))
    env_cfg = cfg.env
    joint_names = list(env.unwrapped.scene["robot"].joint_names)
    policy_hz = 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)
    meta = {
        "task_id":       task_id,
        "checkpoint":    str(Path(checkpoint_path).resolve()),
        "obs_dim":       str(obs_dim),
        "action_dim":    str(action_dim),
        "joint_names":   ",".join(joint_names),
        "sim_timestep":  str(env_cfg.sim.mujoco.timestep),
        "decimation":    str(env_cfg.decimation),
        "policy_hz":     f"{policy_hz:.4f}",
        "obs_group":     obs_group,
    }
    for k, v in meta.items():
        prop = m.metadata_props.add()
        prop.key, prop.value = k, v
    onnx.save(m, str(out_path))

    env.close()
    print(f"Exported: {out_path}  ({obs_dim}-dim obs -> {action_dim}-dim action, {policy_hz:.1f} Hz)")
    return out_path


def smooth01(x: np.ndarray | float) -> np.ndarray | float:
    """Smoothstep clamp: map [0, 1] to [0, 1] with zero endpoint derivatives.
    Args: x: scalar or array to clamp and smooth.
    Returns: smoothed value of the same shape.
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: np.ndarray | float, b: np.ndarray | float, t: float) -> np.ndarray | float:
    """Smoothstep-blended interpolation between a and b.
    Args: a: start value; b: end value; t: parameter in [0, 1].
    Returns: interpolated value shaped like a/b.
    """
    return a + (b - a) * smooth01(t)
