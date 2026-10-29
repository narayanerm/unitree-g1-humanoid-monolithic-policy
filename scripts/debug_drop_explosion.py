##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/itandroids-academic                         #
##########################################################################################
"""Live diagnostic for the carry-to-drop expert explosion.

Runs the real build_dataset_wpcd with 8 envs and 400 steps, with _compute_drop_obs
monkey-patched to log the exact 147-D obs values at every call. Also wraps the
drop policy forward to log the action magnitude. Isolates whether the explosion
is in the obs builder, in the state the FSM hands over, or in the mjlab buffer
sync at drop_fire.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from src.expert_registry import BY_NAME, resolve_ckpt
import src.fsm_wpcd_batched as fsm_mod


def label_dim(d: int) -> str:
    """Human-readable name for a drop-obs dim index.
    Args: d: dim index 0..146.
    Returns: short label.
    """
    if d < 3: return f"lin_vel_b[{d}]"
    if d < 6: return f"ang_vel_b[{d-3}]"
    if d < 9: return f"proj_grav_b[{d-6}]"
    if d < 38: return f"joint_pos_rel[{d-9}]"
    if d < 67: return f"joint_vel[{d-38}]"
    if d < 96: return f"last_action[{d-67}]"
    if d < 105: return f"hand_to_box[{d-96}]"
    if d < 111: return f"wrist_force[{d-105}]"
    if d < 114: return f"box_rel[{d-111}]"
    if d < 117: return f"box_lin[{d-114}]"
    if d < 120: return f"box_ang[{d-117}]"
    if d < 124: return f"box_quat[{d-120}]"
    if d < 127: return f"box_size[{d-124}]"
    if d < 130: return f"grasp_L[{d-127}]"
    if d < 133: return f"grasp_R[{d-130}]"
    if d < 136: return f"object_target_pos_rel[{d-133}]"
    if d < 140: return f"object_target_quat[{d-136}]"
    if d < 143: return f"place_target_rel[{d-140}]"
    return f"place_target_quat_w[{d-143}]"


def main() -> None:
    """Monkey-patch the drop obs builder and run the real wpcd FSM with 8 envs.
    Args: none.
    Returns: none.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load drop normalizer stats up front for inline comparison.
    blob = torch.load(str(resolve_ckpt(BY_NAME["place"])), map_location=device, weights_only=False)
    drop_mean = blob["actor_state_dict"]["obs_normalizer._mean"].squeeze(0).to(device)
    drop_std = blob["actor_state_dict"]["obs_normalizer._std"].squeeze(0).to(device)

    orig_compute_drop_obs = fsm_mod._compute_drop_obs
    orig_load_actor = fsm_mod._load_actor_from_state_dict
    drop_call_count = [0]

    def logged_compute_drop_obs(*args, **kwargs):
        """Call the real drop obs builder; log the result on the first few calls.
        Args: forwarded to _compute_drop_obs.
        Returns: the same (N, 147) tensor the real builder returns.
        """
        drop_obs = orig_compute_drop_obs(*args, **kwargs)
        if drop_call_count[0] < 3:
            drop_call_count[0] += 1
            # Only log envs with non-trivial state (phase is DROP for them).
            print(f"\n### DROP OBS CALL #{drop_call_count[0]} ###")
            print(f"  shape={tuple(drop_obs.shape)}, device={drop_obs.device}")
            for env_i in range(min(2, drop_obs.shape[0])):
                o = drop_obs[env_i]
                normalized = (o - drop_mean) / (drop_std + 0.01)
                print(f"  env{env_i}: |o|max={float(o.abs().max()):.4f}  |o|mean={float(o.abs().mean()):.4f}  "
                      f"|normed|max={float(normalized.abs().max()):.4f}")
                if o.abs().max() > 100:
                    print(f"    EXPLODED. top-10 obs dims by |o|:")
                    top = o.abs().argsort(descending=True)[:10].tolist()
                    for d in top:
                        print(f"      dim {d:>4d}  raw={float(o[d]):+15.4f}  mean={float(drop_mean[d]):+10.4f}  std={float(drop_std[d]):10.4e}  normed={float(normalized[d]):+12.4f}  {label_dim(d)}")
        return drop_obs

    def logged_load_actor(ckpt_path, obs_dim, action_dim, hidden, device):
        """Load an expert actor; wrap the drop expert to log its first few outputs.
        Args: forwarded to _load_actor_from_state_dict.
        Returns: callable actor; drop actor is wrapped to log.
        """
        actor = orig_load_actor(ckpt_path=ckpt_path, obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, device=device)
        if "place" in str(ckpt_path):
            drop_act_count = [0]
            def wrapped(obs_dict):
                """Forward the obs through the drop actor; log |a| on the first few calls.
                Args: obs_dict: observation dict handed to the real actor.
                Returns: action tensor from the wrapped actor.
                """
                a = actor(obs_dict)
                if drop_act_count[0] < 3:
                    drop_act_count[0] += 1
                    print(f"  DROP ACTOR CALL #{drop_act_count[0]}: shape={tuple(a.shape)}  |a|max={float(a.abs().max()):.6e}  |a|mean={float(a.abs().mean()):.6e}")
                return a
            return wrapped
        return actor

    fsm_mod._compute_drop_obs = logged_compute_drop_obs
    fsm_mod._load_actor_from_state_dict = logged_load_actor

    print("[debug] calling build_dataset_wpcd(num_envs=8, max_steps=800)...")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp_npz = Path(tf.name)
    try:
        stats = fsm_mod.build_dataset_wpcd(
            num_envs=8,
            device=device,
            out_path=tmp_npz,
            seed=42,
            max_steps=800,
            spawn_r_min=0.8,
            spawn_r_max=1.5,
            env_spacing=20.0,
            table2_r_min=1.5,
            table2_r_max=3.0,
            verbose=True,
        )
    finally:
        if tmp_npz.exists():
            tmp_npz.unlink()
    print(f"\n[debug] stats: {stats}")


if __name__ == "__main__":
    main()
