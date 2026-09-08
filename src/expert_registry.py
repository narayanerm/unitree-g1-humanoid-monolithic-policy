##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""Registry of the four expert policies driven by the FSM controller."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
EXPERT_CKPT_ROOT = ROOT / "experts"
TERMINAL_BUFFER_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class ExpertPhase:
    """One atomic segment of the chain (walk / pick / carry / place)."""
    phase_id: int
    name: str
    policy_module: str
    task_id: str
    ckpt_filename: str
    experiment_hint: str
    max_steps: int
    terminal_buffer_out: Optional[str]


PHASES: tuple[ExpertPhase, ...] = (
    ExpertPhase(0, "walk",  "src.policy_walk",  "Mjlab-Velocity-Flat-Unitree-G1-Custom", "walk.pt", "g1_walk", 400, "walk_terminal_states.npz"),
    ExpertPhase(1, "pick",  "src.policy_pick",  "Mjlab-G1-Pick",  "pick.pt",  "g1_pick",  400, "pick_terminal_states.npz"),
    ExpertPhase(2, "carry", "src.policy_carry", "Mjlab-G1-Carry", "carry.pt", "g1_carry", 400, "carry_terminal_states.npz"),
    ExpertPhase(3, "place", "src.policy_drop",  "Mjlab-G1-Drop",  "place.pt", "g1_drop",  400, None),
)

BY_NAME: dict[str, ExpertPhase] = {p.name: p for p in PHASES}
PHASE_ONE_HOT_DIM: int = len(PHASES)


def resolve_ckpt(phase: ExpertPhase) -> Path:
    """Locate where the staging script drops this phase's weights.
    Args: phase: the PHASES entry.
    Returns: Path under EXPERT_CKPT_ROOT (may not exist yet).
    """
    return EXPERT_CKPT_ROOT / phase.ckpt_filename


def resolve_terminal_buffer(phase: ExpertPhase) -> Optional[Path]:
    """Locate the .npz this phase writes to warm-start the successor.
    Args: phase: the PHASES entry.
    Returns: Path or None if the phase does not seed a successor.
    """
    if phase.terminal_buffer_out is None:
        return None
    return TERMINAL_BUFFER_DIR / phase.terminal_buffer_out


def load_expert(
    phase: ExpertPhase,
    num_envs: int,
    device: str = "cuda:0",
    override_env_cfg: Optional[Callable] = None,
):
    """Instantiate the phase env, load the actor, and return an inference handle.
    Args: phase: PHASES entry; num_envs: parallel envs; device: torch device; override_env_cfg: optional callable mutating the cfg in place (e.g. to point at a warm-start buffer).
    Returns: tuple (env, base_env, inference_policy, agent_cfg).
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    mod = importlib.import_module(phase.policy_module)
    mod.register()

    env_cfg = load_env_cfg(phase.task_id, play=True)
    env_cfg.scene.num_envs = num_envs
    if override_env_cfg is not None:
        override_env_cfg(env_cfg)

    agent_cfg = load_rl_cfg(phase.task_id)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(phase.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    ckpt = resolve_ckpt(phase)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"[expert_registry] missing checkpoint for phase={phase.name!r} at {ckpt}. "
            f"Run scripts/stage_expert_checkpoints.sh first."
        )
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return env, base_env, policy, agent_cfg


def all_checkpoints_present() -> tuple[bool, list[Path]]:
    """Report whether every expert checkpoint has been staged locally.
    Args: none.
    Returns: (True, []) if complete, else (False, list of missing Paths).
    """
    missing = [resolve_ckpt(p) for p in PHASES if not resolve_ckpt(p).exists()]
    return (len(missing) == 0, missing)
