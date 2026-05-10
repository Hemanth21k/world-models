# Experiment Contracts

Different world models should not be forced into one fake universal API.
Instead, each experiment declares a task contract. A contract standardizes the
outer experiment shape while allowing model-specific internals.

## Contract Layers

Each experiment should define:

- **Task type**: what is being evaluated, such as action prediction, video
  prediction, latent prediction, or action-conditioned prediction.
- **Input schema**: the normalized sample shape passed into a model adapter.
- **Output schema**: the typed prediction artifact returned by the adapter.
- **Metrics**: task-appropriate measurements and visualizations.
- **Artifacts**: a consistent output directory layout.

## Artifact Layout

Every experiment should write outputs under a single experiment directory:

```text
outputs/<experiment_id>/
├── manifest.yaml
├── predictions/
├── metrics.json
├── plots/
└── logs/
```

`manifest.yaml` should capture enough information to rerun or interpret the
experiment:

```yaml
model: groot-h
dataset: open-h/tum_sonata
task: action_prediction
split: test
weights: /weights/GR00T-H
inputs:
  modalities: [image, proprioception, language]
outputs:
  type: action_sequence
  horizon: 50
metrics:
  - action_mse
  - action_mae
  - per_dimension_plot
```

## Initial Task Contracts

Start with the smallest useful set:

| Contract | Input | Output | Example model |
|----------|-------|--------|---------------|
| `action_prediction` | episode observations | action sequence | GR00T-H |
| `video_prediction` | video context frames | future frames or tokens | Cosmos-style models |
| `latent_prediction` | encoded context | future latents | V-JEPA 2 |
| `action_conditioned_prediction` | observations plus candidate actions | predicted future state | V-JEPA 2-AC |

## Adapter Boundary

Model adapters translate between the contract and model-specific code:

```python
class ModelAdapter:
    def load(self, config):
        ...

    def predict(self, batch):
        ...

    def format_output(self, prediction):
        ...
```

Dataset adapters translate raw datasets into contract samples:

```python
{
    "episode_id": "...",
    "timestep": 120,
    "observations": {
        "image": "...",
        "proprio": "...",
        "language": "..."
    },
    "targets": {
        "actions": "..."
    },
    "metadata": {}
}
```

The contract should stay small. Add fields only when at least one real
experiment needs them.
