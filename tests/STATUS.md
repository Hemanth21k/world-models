# Model Verification Status

Legend: ✅ verified · ⏳ not yet tested · ❌ known issue

## GR00T-H (`world-models:groot-h`)

| Check | Status | Notes |
|-------|--------|-------|
| Docker image builds | ✅ | 2026-05-09 — all 4 images (groot-h, groot-h-dev, vjepa2, vjepa2-dev) |
| GPU visible inside container | ✅ | 2026-05-10 — 6× NVIDIA RTX A6000 visible |
| Package imports (`import gr00t`) | ✅ | 2026-05-10 — `Gr00tPolicy` and `EmbodimentTag.TUM_SONATA_FRANKA` import |
| Inference runs end-to-end | ⏳ | TUM SonATA Franka, traj 0–2 |

## V-JEPA 2 (`world-models:vjepa2`)

| Check | Status | Notes |
|-------|--------|-------|
| Docker image builds | ✅ | 2026-05-09 — built alongside GR00T-H |
| GPU visible inside container | ✅ | 2026-05-10 — 6× NVIDIA RTX A6000 visible |
| Package imports (`import app`) | ✅ | 2026-05-10 — `app` and `src.models` import |
| Inference runs end-to-end | ⏳ | |

---

_Update this file after running `bash tests/smoke/<model>.sh`._
