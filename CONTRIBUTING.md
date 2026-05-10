# Contributing to world-models

## Adding a new model

1. Place the model (or register it as a git submodule) under its paradigm folder:
   - `latent_predictive/` — JEPA-family, predict future latents
   - `generative/` — pixel / token predictors (Cosmos, DiT-based)
   - `hybrid/` — latent + reconstruction (PAN, Dreamer-style)
   - `vla/` — vision-language-action models

2. Add an experiment under `experiments/<model>_<dataset>/` following the
   three-script pattern:
   - `01_build_image.sh` — build Docker image
   - `02_download_weights.py` — fetch weights from HuggingFace or equivalent
   - `03_run_inference.sh` — run inference inside the container

3. Update the experiment table in `README.md`.

4. Update the third-party license table in `README.md` with the model's license.

## Platform vs research code

`world-models` should stay usable as a public platform repo. Put reusable
infrastructure here:

- Docker images and run helpers
- model adapters
- dataset adapters
- evaluation contracts
- metrics
- smoke tests
- cleaned, reproducible public experiments

Keep messy, unpublished, or private research work in a separate repository that
depends on `world-models`. See [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md).

## Upstream model changes

Treat upstream model folders as read-only by default:

- `latent_predictive/vjepa2/`
- `vla/GR00T-H/`

Prefer adapters, experiment scripts, or private research repos over modifying
submodule code directly. If an upstream compatibility change is required, keep
it small, document it under `patches/<model>/`, and explain why it cannot live
in the platform layer.

## Experiment contracts

Experiments should declare a task contract instead of inventing a one-off output
layout. The first supported contracts are expected to be:

- `action_prediction`
- `video_prediction`
- `latent_prediction`
- `action_conditioned_prediction`

See [docs/EXPERIMENT_CONTRACTS.md](docs/EXPERIMENT_CONTRACTS.md) for the
recommended manifest, artifact layout, and adapter boundary.

## Crediting AI contributions

If an AI assistant (e.g. Claude, GPT-4, Gemini) materially contributed to your
PR — writing code, designing APIs, drafting docs — please record it:

**In your commit messages**, add a `Co-Authored-By` trailer:
```
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Co-Authored-By: GPT-4o (OpenAI) <noreply@openai.com>
```

**In `CONTRIBUTORS.md`**, add a row to the AI contributions table:
```markdown
| Model name (Provider) | What it did | YYYY-MM |
```

This keeps the project honest about how it was built and helps others
understand the provenance of design decisions.

## License

By submitting a pull request you agree your contribution will be licensed under
[Apache 2.0](LICENSE). No CLA required.
