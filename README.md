# WorldModelling

An open-source project for running inference and experiments on foundation world models, starting with healthcare robotics and video prediction.

## Models

| Model | Source | Description |
|-------|--------|-------------|
| [GR00T-H](GR00T-H/) | NVIDIA Medtech | VLA post-trained on 16 surgical robot embodiments |
| [V-JEPA 2](vjepa2/) | Meta FAIR | Self-supervised video world model with action-conditioned planning |

Both models are tracked as git submodules. Clone with:
```bash
git clone --recurse-submodules <repo-url>
```

## Experiments

Each experiment lives in `experiments/` and is self-contained: a build script, a weight-download script, and a run script.

| Experiment | Model | Dataset |
|------------|-------|---------|
| [groot_h_tum_sonata](experiments/groot_h_tum_sonata/) | GR00T-H | TUM SonATA Franka (ultrasound sonography) |

## Requirements

- Docker with NVIDIA Container Toolkit
- GPU(s) with 12 GB+ VRAM
- HuggingFace account with accepted model licenses
