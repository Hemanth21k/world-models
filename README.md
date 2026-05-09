# world-models

> Unified interface for testing and extending world models for Physical AI.

A research repository providing a common interface for working with world model 
architectures across the latent-predictive, generative, hybrid, and 
vision-language-action (VLA) paradigms. Designed to make it easier to evaluate, 
compare, and build on top of existing world models for Physical AI research — 
spanning embodied perception, action-conditioned prediction, and robot autonomy.

## Status

🚧 Early development. Currently integrating:

- **V-JEPA 2** (`latent_predictive/`) — video world model, latent-space prediction
- **GR00T-H** (`vla/`) — healthcare VLA via the Open-H Embodiment dataset

Planned next:

- V-JEPA 2-AC (action-conditioned planning)
- Cosmos-Predict (generative)
- Additional VLA baselines (OpenVLA, π0)

## Repository structure

```
world-models/
├── latent_predictive/   # JEPA-family: predict future latents
│   └── vjepa2/          # submodule — Meta FAIR V-JEPA 2 / 2.1
├── generative/          # Cosmos, DiT-based: predict pixels / tokens
├── hybrid/              # Latent prediction + reconstruction (PAN, Dreamer-style)
├── vla/                 # Vision-language-action models (output actions)
│   └── GR00T-H/         # submodule — NVIDIA GR00T-H (healthcare robotics)
├── eval/                # Shared benchmarks and evaluation utilities
├── experiments/         # Self-contained run scripts per model × dataset
│   └── groot_h_tum_sonata/
├── datasets/            # Raw data — gitignored (set DATASET_DIR env var)
├── docs/                # Architecture notes, design decisions
└── LICENSE
```

The taxonomy is organized by **what the model predicts**: future latents, 
future pixels, both, or actions — the cleanest axis for cross-family comparison.

## Installation

```bash
git clone --recurse-submodules https://github.com/Hemanth21k/world-models.git
cd world-models
```

All models run inside Docker. See [`docker/`](docker/) for the unified setup.

## Docker

Every model in the repo follows the same two-target convention defined in [`docker/docker-compose.yml`](docker/docker-compose.yml):

| Target | Use case |
|--------|----------|
| `<model>` | **Deployment** — source baked in, fully self-contained |
| `<model>-dev` | **Development** — deps only, source mounted live from repo |

```bash
# See all available model services
bash docker/docker_run.sh list

# Build any model (deployment or dev)
bash docker/docker_run.sh build <model>
bash docker/docker_run.sh build all

# Shell into any model
bash docker/docker_run.sh shell <model>          # deployment
bash docker/docker_run.sh shell <model>-dev      # development
# → inside the dev container, run once:
#   pip install -e . --no-deps

# Verify GPUs
bash docker/docker_run.sh gpu-check [model]

# Run any command inside a model container
bash docker/docker_run.sh run <model> python my_script.py
```

Adding a new model just requires a new stage in [`docker/Dockerfile`](docker/Dockerfile) and a new service in [`docker/docker-compose.yml`](docker/docker-compose.yml). The `docker_run.sh` script works with any service name automatically.

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
