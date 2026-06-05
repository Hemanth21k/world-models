# Project Roadmap

Kanban board for `world-models` — updated as work progresses.
Add new items under **Backlog**, move them left as they advance.

---

## Board

### Done ✅

| Item | Notes |
|------|-------|
| Docker infrastructure | GR00T-H + V-JEPA 2 images build and run on 6× A6000 |
| GPU visibility smoke test | 6× RTX A6000 confirmed inside containers |
| Package import smoke test | `Gr00tPolicy`, `EmbodimentTag.TUM_SONATA_FRANKA` import |
| Upgrade to GR00T-H-N1.7 | Submodule bumped to d8369db; new Cosmos-Reason2-2B backbone |
| N1.7 weights download | `weights/GR00T-H-N1.7/` on disk |
| Dataset prep (episodes.jsonl, tasks.jsonl, temporal_stats.json) | Generated from SonATA parquet files |
| Fine-tuning pipeline | `04_prep_stats.sh` + `05_finetune.sh`; 6 GPUs, batch 192, 20k steps |
| GR00T-H-N1.7 fine-tune on TUM SonATA Franka | Loss 1.62 → 0.026; checkpoint-16000 (backbone: NVIDIA Cosmos-Reason2-2B / Qwen3-VL; single-frame conditioning) |
| Demo video (Level 1+2) | `07_demo_video.py` — 3 cameras + GT-vs-pred EEF trajectory + orientation triad + error-over-time; open-loop & rollout; fixed action/GT time alignment |
| Quantitative evaluation on test split | `09_eval.py` (8-shard multi-GPU) over **482** test episodes. XYZ L2 + geodesic orientation + zero-motion baseline; reproducible (`--seed`). Open-loop ≈ **0.09 cm @ h=1**, 0.64 cm @ h=16 (beats baseline ~2–2.6×); rollout ≈ 4 cm. `11_merge_eval.py` |
| Robot rollout (Level 3) | `12_robot_rollout.py` — PyBullet Franka playback of predicted EEF poses; GT (green ghost) vs predicted (orange); self-calibrated probe→flange tool transform; GT-seeded IK |
| Unified demo (cache → compositor) | `14_rollout_cache.py` → `15_demo_compose.py` (`16_run_unified_demo.sh`): one frame with cameras + action curves + tracking error + robot view + EEF-path minimap. Decoupled so layout iterates without re-running the model |
| HuggingFace model release | `Hemanth21k/GR00T-H-N1.7-TUM-SonATA-Franka` live |
| GitHub repo public | `Hemanth21k/world-models` — full reproducible pipeline |
| Bug fix: N1.7 unexpected key crash | Patched `setup.py`; documented in `patches/groot_h/`; fork at `Hemanth21k/GR00T-H` |
| CITATION.cff with QBIL / NIH / CPRIT funding | R01CA288379, R01CA204254, RP240289 |

---

### In Progress 🔄

| Item | Notes |
|------|-------|
| README showcase | Add eval results table + demo GIF + as-run architecture to README |
| PR to NVIDIA upstream | Fix for `dropout_prob_by_embodiment` unexpected key; patch ready in `patches/groot_h/` |

---

### To Do 📋

| Item | Priority | Notes |
|------|----------|-------|
| Upload final checkpoint-20000 to HF | High | Run `python experiments/groot_h_tum_sonata/06_upload_to_hf.py` after training |
| Inference end-to-end test (fine-tuned model) | High | Update `03_run_inference.sh` to use fine-tuned weights; verify output plots |
| Open PR to NVIDIA-Medtech/GR00T-H | High | 2-line fix, already in fork `Hemanth21k/GR00T-H` |
| Zero-shot GR00T-H-N1.7 baseline | Medium | Same eval harness on the un-finetuned model (note: new-embodiment projector is untrained) to quantify the fine-tuning gain |
| V-JEPA 2 smoke test | Medium | Image builds ✅, imports ✅, inference not yet tested |
| Generative world-model rollout | Low | Close the visual loop: Cosmos-Predict (pixels) or V-JEPA 2-AC (latent) action-conditioned rollouts on SonATA — replaces the hybrid rollout's GT frames |
| Multi-embodiment fine-tune | Low | Add other Open-H embodiments (e.g. JHU dVRK) to the training config |
| Adapter layer for new embodiments | Low | Fill `adapters/` directory with reusable dataset adapters |
| Paper / technical report | Low | Methods, results, comparison with zero-shot GR00T-H-N1.7 baseline |

---

### Backlog 💡

| Item | Notes |
|------|-------|
| Zero-shot GR00T-H-N1.7 baseline on SonATA | Compare against fine-tuned model quantitatively |
| WandB sweep for LR / batch size | The 8e-4 LR caused a spike; find optimal schedule |
| Tune LLM backbone on SonATA | `--tune-llm` run for higher-quality model; needs full 49GB per GPU |
| CI smoke test in GitHub Actions | Run `tests/smoke/groot_h.sh` on PR |
| TensorRT export of fine-tuned weights | Use `scripts/deployment/export_onnx_n1d7.py` for faster inference |
| Deployment on robot hardware | Real Franka + ROS integration |

---

_Last updated: 2026-06-04. Move items by editing this file._
