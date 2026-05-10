# Research Workflow

`world-models` is intended to be the public platform layer: Docker images,
model adapters, dataset adapters, evaluation contracts, metrics, and reproducible
example experiments.

Use a separate private repository for messy or unpublished research work. Treat
that repository like a lab notebook: commit drafts, ablations, failed attempts,
private notes, local paths, and paper-specific scripts there.

## Recommended Setup

Keep the public platform and private research as sibling repositories:

```text
workspace/
├── world-models/
└── private-groot-openh-research/
```

The private repo can install the public repo in editable mode:

```bash
pip install -e ../world-models
```

or include it as a git submodule:

```text
private-groot-openh-research/
├── world-models/
├── configs/
├── notebooks/
├── experiments/
├── scripts/
└── README.md
```

## What Belongs Where

Put reusable, public infrastructure in `world-models`:

- Docker images and run helpers
- model adapters
- dataset adapters
- evaluation contracts
- metrics
- smoke tests
- cleaned, reproducible public experiments

Keep private or unstable work in a separate research repo:

- unpublished model ideas
- paper-specific training scripts
- exploratory notebooks
- private datasets and local paths
- failed attempts and scratch logs
- ablation sweeps
- draft result analysis

## Submodule Policy

Upstream model folders are treated as read-only by default:

```text
latent_predictive/vjepa2/
vla/GR00T-H/
```

Do not use those directories as the main place for research changes. Prefer
wrapping them from platform code or a private research repo.

If an upstream model needs a small compatibility fix, keep the change isolated
and document it under `patches/<model>/`. If the change becomes broadly useful,
submit it upstream or keep it as a clearly named platform patch.

## Upstreaming Back Into world-models

When private research produces reusable infrastructure, upstream only the clean
part:

1. Move common code into adapters, datasets, eval contracts, or metrics.
2. Add a minimal reproducible experiment under `experiments/`.
3. Keep private data paths, notebooks, and unpublished analysis out of the public
   repo.
4. Update `README.md`, `CONTRIBUTING.md`, and license notes when adding a model,
   dataset, or dependency.
