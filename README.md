# unitree-g1-humanoid-monolithic-policy

*Single goal-conditioned MLP student for the Unitree G1, trained by behavioural cloning on FSM-generated loco-manipulation trajectories.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/framework-torch%20%2B%20mjlab-lightgrey.svg)](https://github.com/mujocolab/mjlab)

- One MLP replaces the four-expert FSM: `Linear(115) -> ELU -> ... -> Linear(29)`, phase-conditioned via a 4-dim one-hot in the observation.
- Data generation and BC training are fully local. Expert checkpoints are staged into `experts/`; no runtime dependency on any sibling folder.
- MLP topology matches the sibling rsl-rl actors, so a later PPO fine-tune can warm-start from the trained student.

## Getting Started

### Prerequisites
- Python 3.10 exactly.
- A CUDA-capable GPU for dataset generation (developed on B300 SXM6).
- `uv` package manager (`pip install uv`).
- Trained walk and pick checkpoints in the sibling repos.

### Install
```bash
git clone https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy.git
cd unitree-g1-humanoid-monolithic-policy
uv pip install -r requirements.txt
bash scripts/stage_expert_checkpoints.sh
```

The staging script sources the newest `model_*.pt` from the sibling walk and pick repos into `experts/`.

### Run
Open `01_walk_pick.ipynb` and execute top-to-bottom. Notebook flags at the top:

| Flag | Purpose |
|---|---|
| `DEBUG_LIGHT` | 4 episodes, 3 epochs; roughly two minutes end-to-end. |
| `REUSE_DATASET` | Skip data generation if `outputs/walk_pick_sequential.npz` exists. |
| `REUSE_TRAINING` | Skip BC if a checkpoint exists under `outputs/logs/g1_monolithic/`. |

Regenerate the dataset from the CLI:
```bash
python -m src.fsm_continuous_sequential \
    --episodes 64 --base-seed 100 --max-steps 800 \
    --r-min 0.15 --r-max 5.0 \
    --out outputs/walk_pick_sequential.npz
```

Train the student from the CLI:
```bash
python -m src.bc_train --dataset outputs/walk_pick_sequential.npz --epochs 100
```

TensorBoard: `tensorboard --logdir outputs/logs/g1_monolithic`.

## Method

The pipeline has two stages. First, a walk-to-pick FSM chain runs the frozen experts in mjlab with fully randomised robot spawn (distance to table in the annulus `[r_min, r_max]`, angle around the table uniform in `[0, 2*pi)`, initial yaw uniform in `[-pi, pi)`) and records `(unified_obs, action, phase)` for every transition. Episodes run with `num_envs=1` under subprocess isolation to work around an mjlab context leak that would otherwise OOM around episode 60. Second, `bc_train.py` runs MSE behavioural cloning with a `WeightedRandomSampler` for phase balance (walk usually produces more transitions than pick; without balancing the student collapses to a walking-only policy). The unified observation layout (`goal_features.py`) is 96 dims of proprioception plus 15 dims of egocentric goal plus a 4-dim phase one-hot, totalling 115 dims.

## Related repositories

- [`unitree-g1-humanoid-walk`](../unitree-g1-humanoid-walk) - expert 1 in the FSM used for data generation.
- [`unitree-g1-humanoid-pick`](../unitree-g1-humanoid-pick) - expert 2 in the FSM used for data generation.
- [`unitree-g1-humanoid-carry`](../unitree-g1-humanoid-carry) - expert 3 (loaded by the registry, currently not exercised in the walk-to-pick pipeline).
- [`unitree-g1-humanoid-place`](../unitree-g1-humanoid-place) - expert 4 (loaded by the registry, currently not exercised in the walk-to-pick pipeline).
- [`unitree-g1-humanoid-fsm`](../unitree-g1-humanoid-fsm) - alternative FSM-only pipeline that outputs a `unified_dataset.npz` on the same schema.

## Citation

```bibtex
@misc{medeiros2026g1mono,
  author = {Ribeiro Medeiros, Narayane},
  title  = {unitree-g1-humanoid-monolithic-policy: single MLP loco-manipulation student for the Unitree G1},
  year   = {2026},
  howpublished = {\url{https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy}}
}
```

## Author

**Narayane Ribeiro Medeiros** - ITAndroids Humanoid Team, Instituto Tecnologico de Aeronautica.
[GitHub](https://github.com/narayanerm) - narayane.rm@gmail.com

## License

Released under the [MIT License](LICENSE).
