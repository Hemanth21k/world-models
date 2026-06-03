"""
Evaluation: GR00T-H-N1.7 fine-tuned on TUM SonATA Franka (test split).

Sweeps action_horizon (re-inference frequency) across [1, 4, 8, 16, 50] for both
open_loop and rollout modes, computing per-episode XYZ L2 error then aggregating
across all test episodes.

Metric:
  At each step t, error_t = ||pred_xyz_t - gt_xyz_t||₂  (metres)
  Episode error = mean(error_t)  over all steps in the episode
  Final metric  = statistics over all test episodes

Report: table of mean ± std, median, max — saved as JSON + printed.

Usage (inside groot-h-dev container):
    cd /workspace/groot_h
    python /workspace/scripts/09_eval.py \\
        --model-path /checkpoints \\
        --dataset-path /data/sonata_all \\
        --output-dir /eval_out

    # Faster — skip horizon=1 and rollout (useful while training is still running)
    python /workspace/scripts/09_eval.py \\
        --model-path /checkpoints \\
        --dataset-path /data/sonata_all \\
        --output-dir /eval_out \\
        --horizons 4 8 16 50 \\
        --modes open_loop \\
        --max-episodes 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import open_h.embodiments
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


# ──────────────────────────────────────────────────────────────────────────────

def latest_checkpoint(model_path: str) -> str:
    p = Path(model_path)
    if (p / "config.json").exists():
        return str(p)
    ckpts = sorted([d for d in p.iterdir() if d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {model_path}")
    return str(ckpts[-1])


def load_policy(ckpt_path: str, device: int = 0) -> Gr00tPolicy:
    print(f"Loading policy: {ckpt_path}  (device={device})")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.TUM_SONATA_FRANKA,
        model_path=ckpt_path,
        device=device,
    )
    return policy


def build_observation(traj, step: int, modality_cfg: dict,
                      eef_override=None) -> tuple[dict, str]:
    video_keys = modality_cfg["video"].modality_keys
    state_keys = modality_cfg["state"].modality_keys
    lang_key   = modality_cfg["language"].modality_keys[0]

    video_dict = {
        cam: np.array(traj[f"video.{cam}"].iloc[step])[None, None, ...].astype(np.uint8)
        for cam in video_keys
    }
    state_dict = {}
    for key in state_keys:
        if key == "eef_pose" and eef_override is not None:
            val = eef_override.astype(np.float32)
        else:
            val = np.array(traj[f"state.{key}"].iloc[step]).flatten().astype(np.float32)
        state_dict[key] = val[None, None, :]

    instruction = str(traj[f"language.{lang_key}"].iloc[step])
    return {"video": video_dict, "state": state_dict,
            "language": {lang_key: [[instruction]]}}, instruction


def evaluate_episode(
    policy: Gr00tPolicy,
    traj,
    modality_cfg: dict,
    action_horizon: int,
    mode: str,
) -> dict:
    """
    Run inference on one episode and return per-step XYZ L2 errors.

    Returns:
        {
          "errors":        list[float]  — per-step L2 errors (metres)
          "episode_mean":  float
          "episode_median": float
          "episode_max":   float
          "n_steps":       int
          "n_inferences":  int
        }
    """
    policy.reset()
    gt_raw = np.stack([np.array(traj["action.eef_pose"].iloc[i]).astype(np.float32)
                       for i in range(len(traj))])   # (T, 6)

    errors        = []
    cached_pred   = np.zeros((50, 6))  # model always outputs 50 steps
    pred_eef_6d   = gt_raw[0].copy()
    n_inferences  = 0

    for step in range(len(traj)):
        eef_for_state = None if mode == "open_loop" else pred_eef_6d
        obs, _ = build_observation(traj, step, modality_cfg, eef_for_state)

        if step % action_horizon == 0:
            action_chunk, _ = policy.get_action(obs)
            pred_raw = np.array(action_chunk.get("eef_pose", np.zeros((1, 50, 6))))
            if pred_raw.ndim == 3:
                pred_raw = pred_raw[0]   # (50, 6)
            # Pad if model returns fewer than 50 steps
            if len(pred_raw) < 50:
                pad = np.tile(pred_raw[-1:], (50 - len(pred_raw), 1))
                pred_raw = np.concatenate([pred_raw, pad], axis=0)
            cached_pred = pred_raw
            n_inferences += 1

        chunk_offset = step % action_horizon
        pred_step = cached_pred[min(chunk_offset, len(cached_pred) - 1)]

        # XYZ L2 error
        err = float(np.linalg.norm(pred_step[:3] - gt_raw[step][:3]))
        errors.append(err)

        # Rollout: update predicted EEF state
        if mode == "rollout":
            pred_eef_6d = pred_step.copy()

    err_arr = np.array(errors)
    return {
        "errors":         errors,
        "episode_mean":   float(err_arr.mean()),
        "episode_median": float(np.median(err_arr)),
        "episode_max":    float(err_arr.max()),
        "n_steps":        len(errors),
        "n_inferences":   n_inferences,
    }


def run_evaluation(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    test_episode_ids: list[int],
    horizons: list[int],
    modes: list[str],
    output_dir: Path,
) -> dict:
    """
    Sweep over horizons × modes, evaluate all test episodes, save and return results.
    """
    results = {}  # results[mode][horizon] = {per_episode: [...], stats: {...}}

    total_combos = len(modes) * len(horizons)
    combo_idx = 0

    for mode in modes:
        results[mode] = {}
        for horizon in horizons:
            combo_idx += 1
            key = f"{mode}_h{horizon}"
            print(f"\n[{combo_idx}/{total_combos}] mode={mode}  horizon={horizon}"
                  f"  episodes={len(test_episode_ids)}")

            per_episode = []
            t0 = time.time()

            for ep_idx, ep_id in enumerate(test_episode_ids):
                traj = loader[ep_id]
                ep_result = evaluate_episode(
                    policy, traj, loader.modality_configs, horizon, mode
                )
                per_episode.append({
                    "episode_id": ep_id,
                    **{k: v for k, v in ep_result.items() if k != "errors"},
                })

                # Live progress
                elapsed = time.time() - t0
                rate = (ep_idx + 1) / elapsed
                eta = (len(test_episode_ids) - ep_idx - 1) / max(rate, 1e-6)
                print(
                    f"  ep {ep_idx+1:4d}/{len(test_episode_ids)}"
                    f"  mean={ep_result['episode_mean']*100:.2f}cm"
                    f"  ETA {eta/60:.1f}min",
                    end="\r",
                )

            # Aggregate across episodes
            ep_means = np.array([e["episode_mean"] for e in per_episode])
            stats = {
                "mean_of_means":   float(ep_means.mean()),
                "std_of_means":    float(ep_means.std()),
                "median_of_means": float(np.median(ep_means)),
                "max_of_means":    float(ep_means.max()),
                "min_of_means":    float(ep_means.min()),
                "n_episodes":      len(per_episode),
            }
            results[mode][horizon] = {"per_episode": per_episode, "stats": stats}

            elapsed = time.time() - t0
            print(f"\n  Done in {elapsed/60:.1f} min  "
                  f"→ mean={stats['mean_of_means']*100:.2f} ± "
                  f"{stats['std_of_means']*100:.2f} cm  "
                  f"median={stats['median_of_means']*100:.2f} cm  "
                  f"max={stats['max_of_means']*100:.2f} cm")

            # Save incrementally (so partial results survive if interrupted)
            out_file = output_dir / "eval_results.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

    return results


def print_table(results: dict) -> None:
    """Print a formatted summary table."""
    modes   = list(results.keys())
    horizon_set = sorted({int(h) for m in modes for h in results[m].keys()})

    print("\n" + "=" * 72)
    print("GR00T-H-N1.7  TUM SonATA Franka  Evaluation (test split)")
    print("Metric: per-step XYZ L2 error  →  mean across episodes  (cm)")
    print("=" * 72)

    for mode in modes:
        label = "Open-loop (GT state)" if mode == "open_loop" else "Rollout (pred state)"
        print(f"\n{label}")
        print(f"  {'Horizon':>8} | {'Mean (cm)':>10} | {'± Std':>8} | "
              f"{'Median (cm)':>12} | {'Max (cm)':>9}")
        print("  " + "-" * 56)
        for h in horizon_set:
            if h not in results[mode]:
                continue
            s = results[mode][h]["stats"]
            print(f"  {h:>8} | {s['mean_of_means']*100:>10.2f} | "
                  f"{s['std_of_means']*100:>7.2f} | "
                  f"{s['median_of_means']*100:>12.2f} | "
                  f"{s['max_of_means']*100:>9.2f}")

    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",   default=os.environ.get("MODEL_PATH",  "/checkpoints"))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_DIR", "/data/sonata_all"))
    parser.add_argument("--output-dir",   default=os.environ.get("EVAL_OUT",    "/eval_out"))
    parser.add_argument("--horizons",  nargs="+", type=int, default=[1, 4, 8, 16, 50])
    parser.add_argument("--modes",     nargs="+", default=["open_loop", "rollout"],
                        choices=["open_loop", "rollout"])
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit test episodes (useful for quick sanity checks)")
    parser.add_argument("--device",       type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = latest_checkpoint(args.model_path)
    policy = load_policy(ckpt, device=args.device)

    modality_configs = MODALITY_CONFIGS[EmbodimentTag.TUM_SONATA_FRANKA.value]
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
    )

    # Test split: episodes 1915-2396 (from info.json splits)
    import json as _json
    info = _json.load(open(Path(args.dataset_path) / "meta/info.json"))
    test_range = info["splits"]["test"]   # e.g. "1915:2397"
    start, end = map(int, test_range.split(":"))
    test_ids = list(range(start, end))
    if args.max_episodes:
        test_ids = test_ids[:args.max_episodes]

    print(f"Test episodes: {len(test_ids)}  ({test_range})")
    print(f"Horizons: {args.horizons}")
    print(f"Modes: {args.modes}")

    results = run_evaluation(
        policy=policy,
        loader=loader,
        test_episode_ids=test_ids,
        horizons=args.horizons,
        modes=args.modes,
        output_dir=output_dir,
    )

    print_table(results)
    print(f"\nFull results saved to: {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
