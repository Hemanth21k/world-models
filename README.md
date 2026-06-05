# world-models

> Unified interface for testing and extending world models for Physical AI.

A research repository providing a common interface for working with world model 
architectures across the latent-predictive, generative, hybrid, and 
vision-language-action (VLA) paradigms. Designed to make it easier to evaluate, 
compare, and build on top of existing world models for Physical AI research — 
spanning embodied perception, action-conditioned prediction, and robot autonomy.

## Status

🚧 Early development. Currently integrating:

- **GR00T-H** (`vla/`) — healthcare VLA. **Fine-tuned and evaluated on the TUM SonATA
  robotic-ultrasound dataset** (Franka + ultrasound probe). Weights on
  [🤗 Hemanth21k/GR00T-H-N1.7-TUM-SonATA-Franka](https://huggingface.co/Hemanth21k/GR00T-H-N1.7-TUM-SonATA-Franka);
  full reproducible pipeline in [`experiments/groot_h_tum_sonata/`](experiments/groot_h_tum_sonata/).
- **V-JEPA 2** (`latent_predictive/`) — video world model, latent-space prediction.

Planned next:

- V-JEPA 2-AC (action-conditioned planning)
- Cosmos-Predict (generative world-model rollouts on the same dataset)
- Additional VLA baselines (OpenVLA, π0)

## Demo

[![GR00T-H-N1.7 on TUM SonATA — unified demo](docs/assets/demo_2050_open_loop.gif)](docs/assets/demo_2050_open_loop.mp4)

GR00T-H-N1.7 fine-tuned on TUM SonATA (open-loop): probe positioning → transverse-plane
traversal. One frame shows the live cameras (third-person / wrist / ultrasound), predicted
vs. ground-truth action curves, position + orientation tracking error, the commanded Franka
motion (GT green-ghost vs. predicted orange), and the EEF path. **▶ [Full video](docs/assets/demo_2050_open_loop.mp4).**

## Repository structure

```
world-models/
├── adapters/            # Thin wrappers from shared contracts to model code
├── latent_predictive/   # JEPA-family: predict future latents
│   └── vjepa2/          # submodule — Meta FAIR V-JEPA 2 / 2.1
├── generative/          # Cosmos, DiT-based: predict pixels / tokens
├── hybrid/              # Latent prediction + reconstruction (PAN, Dreamer-style)
├── vla/                 # Vision-language-action models (output actions)
│   └── GR00T-H/         # submodule — NVIDIA GR00T-H (healthcare robotics)
├── eval/                # Shared benchmarks and evaluation utilities
├── experiments/         # Self-contained run scripts per model × dataset
│   └── groot_h_tum_sonata/
├── patches/             # Isolated compatibility patches for upstream projects
├── research/            # Public, cleaned research extensions and examples
├── datasets/            # Raw data — gitignored (set DATASET_DIR env var)
├── docs/                # Architecture notes, design decisions
└── LICENSE
```

The taxonomy is organized by **what the model predicts**: future latents, 
future pixels, both, or actions — the cleanest axis for cross-family comparison.

## Research workflow

Treat this repository as the public platform layer: Docker images, model
adapters, dataset adapters, evaluation contracts, metrics, and reproducible
example experiments. Keep messy, unpublished, or private research in a separate
repo that depends on `world-models`.

See [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md) for the recommended
private-research setup and [docs/EXPERIMENT_CONTRACTS.md](docs/EXPERIMENT_CONTRACTS.md)
for the shared experiment contract approach.

## Installation

```bash
git clone --recurse-submodules https://github.com/Hemanth21k/world-models.git
cd world-models
```

All models run inside Docker. See [`docker/`](docker/) for the unified setup.

## Docker

All containers share the `world-models` image repository and are identified by tag:

```
world-models:groot-h        # deployment — GR00T-H source baked in
world-models:groot-h-dev    # development — deps only, source mounted live
world-models:vjepa2         # deployment — V-JEPA 2 source baked in
world-models:vjepa2-dev     # development
```

Every model follows the same two-tag convention defined in [`docker/docker-compose.yml`](docker/docker-compose.yml):

| Tag suffix | Use case |
|------------|----------|
| `<model>` | **Deployment** — source baked in, fully self-contained, no mounts needed |
| `<model>-dev` | **Development** — deps only, source folder mounted live from the repo |

```bash
# List all available model tags
bash docker/docker_run.sh list

# Build one image or all at once
bash docker/docker_run.sh build groot-h
bash docker/docker_run.sh build           # builds every world-models:* image

# Open a shell (one model at a time)
bash docker/docker_run.sh shell groot-h          # deployment
bash docker/docker_run.sh shell groot-h-dev      # development
# → inside the dev container, run once to install in editable mode:
#   pip install -e . --no-deps

# Verify GPUs (defaults to first deployment service if no tag given)
bash docker/docker_run.sh gpu-check
bash docker/docker_run.sh gpu-check vjepa2

# Run any command inside a container
DATASET_DIR=/data bash docker/docker_run.sh run vjepa2 python app/main.py
```

Adding a new model requires only a new stage in [`docker/Dockerfile`](docker/Dockerfile) and a new service in [`docker/docker-compose.yml`](docker/docker-compose.yml). `docker_run.sh` picks it up automatically.

## Quick start: GR00T-H inference on TUM SonATA Franka

```bash
# 1. Build the deployment image (~10-20 min first time)
bash experiments/groot_h_tum_sonata/01_build_image.sh

# 2. Download weights (requires HuggingFace token + accepted license)
HF_TOKEN=<your_token> python experiments/groot_h_tum_sonata/02_download_weights.py

# 3. Run inference (set DATASET_DIR to your local sonata_all path)
DATASET_DIR=/path/to/sonata_all \
bash experiments/groot_h_tum_sonata/03_run_inference.sh
```

Predicted vs ground-truth action plots appear in `outputs/groot_h_tum_sonata/`.

## GR00T-H × TUM SonATA — workflow & architecture

The `experiments/groot_h_tum_sonata/` pipeline takes the dataset through fine-tuning,
evaluation, and visualization. What's available now:

```
TUM SonATA dataset  (3 cameras · joints · force/torque · language, LeRobot format)
      │
      ▼
04_prep_stats ─► 05_finetune (GR00T-H-N1.7, 6×A6000) ─► checkpoint ─► 06_upload_to_hf ─► 🤗 HF model
      │                                  (loss 1.62 → 0.026)
      ▼
fine-tuned model
      │
      ├─► 09_eval ──────────► metrics   (XYZ L2 + geodesic orientation + zero-motion
      │                                   baseline; 482 test episodes; multi-GPU sharded)
      │
      ├─► 07_demo_video ────► demo_*.mp4 (3 cameras + GT-vs-pred EEF trajectory +
      │                                   orientation triad + error-over-time; open-loop & rollout)
      │
      └─► 12_robot_rollout ─► arm video  (predicted EEF pose → IK → Franka playback)   [in progress]
```

**Model architecture (as run)** — a System-2 VLM conditions a System-1 diffusion action
head. Verified from the fine-tuned checkpoint config; see
[`render_architecture.py`](experiments/groot_h_tum_sonata/render_architecture.py) and
[`docs/assets/architecture.png`](docs/assets/architecture.png):

```
3 camera views (256×256) ┐
language instruction     ├─► Qwen3-VL  (System 2)            ─► VL tokens
─────────────────────────┘   nvidia/Cosmos-Reason2-2B            (B, S≈230, 2048)
                              features @ layer 16                      │
                                                                       │ cross-attention
robot state (7D joints + 6D F/T + EEF ref) ─► state token ──┐          ▼
noised action chunk (50 × 6D) ──────────────► action tokens ├─► Diffusion Transformer
                                                            │   (System 1, flow-matching DiT,
                                                            │    32 layers, cross-attn ⇄ self-attn,
                                                            │    × 4 denoising steps)
                                                            ▼
                                   EEF pose action chunk (50 steps × 6D)
                                   REL_XYZ_ROT6D → (x, y, z, roll, pitch, yaw),  targets t+1 … t+50
                                                            │
                                                            ▼
                                          Franka Panda + ultrasound probe
```

Notes: the backbone is **NVIDIA Cosmos-Reason2-2B** (a Qwen3-VL model); conditioning is
**single-frame / Markovian** (no observation history — `delta_indices=[0]`); the action head
is a **flow-matching** DiT (4 denoising steps, stochastic — evaluation is seeded).

### Results (test split, 482 episodes)

Per-step error of the predicted vs. ground-truth EEF action, swept over the re-inference
horizon `H`. Position = XYZ L2; orientation = geodesic angle; baseline = zero-motion
(hold last pose). Seeded for reproducibility — see
[`09_eval.py`](experiments/groot_h_tum_sonata/09_eval.py) /
[`11_merge_eval.py`](experiments/groot_h_tum_sonata/11_merge_eval.py).

| Mode | H | Pos (cm) | Rot (°) | Baseline pos (cm) | Baseline rot (°) |
|------|---|----------|---------|-------------------|------------------|
| Open-loop | 1  | **0.09** | 1.07 | 0.19 | 0.53 |
| Open-loop | 8  | 0.37 | 1.47 | 0.84 | 2.37 |
| Open-loop | 16 | 0.64 | 2.15 | 1.55 | 4.39 |
| Open-loop | 50 | 1.61 | 4.92 | 4.24 | 12.70 |
| Rollout   | 16 | 4.72 | 14.14 | 1.55 | 4.39 |

- **Open-loop** (true state each step): sub-cm to ~6 mm, beating the zero-motion baseline
  ~2–2.6× on position; degrades gracefully as re-inference gets sparser (`H` larger).
- **Rollout** (predicted EEF pose fed back as the reference state): errors compound (~4 cm /
  ~14°). This rollout is **hybrid** — only the EEF pose is fed back; cameras and the rest of
  the state still come from the dataset (a true closed loop needs a world model).

(Full sweep — all horizons, both modes, std/median/max — in `eval_results_merged.json`.)

**Splits** (from `meta/info.json`): train `0–1676`, val `1677–1914`, **test `1915–2396`** — disjoint;
the eval runs only on the held-out test episodes. Note the test set is *in-distribution* (same
phantoms / task families, unseen episodes), so this measures held-out imitation accuracy rather
than out-of-distribution generalization.

## Scope

This repository focuses on world models for **Physical AI**: models that learn 
to perceive, predict, and act in the physical world. In scope:

- Joint-embedding predictive architectures (JEPA family)
- Generative video / world foundation models
- Hybrid latent-generative architectures
- Vision-language-action models for embodied control

Out of scope:

- Pure video generation models without predictive or embodied goals
- Spatial / 3D scene models (NeRF, Gaussian splatting)
- Full RL infrastructure (this is not a model-based RL framework)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
add a new model and how to credit AI assistant contributions.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

### Third-party components

This repository's Apache 2.0 license covers **only the code in this repository**.
External models, weights, and datasets carry their own licenses.

| Component | License |
|-----------|---------|
| GR00T-H weights | NVIDIA OneWay Noncommercial |
| V-JEPA 2 code | CC-BY-NC 4.0 |
| Open-H Embodiment dataset | Per-institution terms |

### Contributing terms

By submitting a pull request you agree your contribution will be licensed under
Apache 2.0.

## Acknowledgements

This work was conducted at the **Quantitative Bio Imaging Lab (QBIL)** at
**The University of Texas at Dallas**. Computing resources were provided by the QBIL GPU cluster.

Research reported here was supported in part by the National Cancer Institute of the National
Institutes of Health under Award Numbers **R01CA288379** and **R01CA204254**, and by the
Cancer Prevention and Research Institute of Texas (CPRIT) under Award Number **RP240289**.
The content is solely the responsibility of the authors and does not necessarily represent the
official views of the National Institutes of Health.

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{world_models_2026,
  author = {Pasupuleti, Hemanth},
  title  = {world-models: Unified interface for testing and extending world 
            model architectures for Physical AI},
  year   = {2026},
  url    = {https://github.com/Hemanth21k/world-models}
}
```

## Disclaimer

This is research code provided "as is", without warranty of any kind. It is not
intended for clinical, surgical, or any safety-critical deployment. Validate
thoroughly before deploying any output of this code in a physical system.
