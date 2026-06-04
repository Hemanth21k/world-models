"""
Merge parallel eval results into one table.

Works for either parallelisation scheme used by 09_eval.py:
  · combo-sharding   — one subdir per (horizon, mode), all episodes
  · episode-sharding — every subdir holds the full sweep on a slice of episodes
    (--num-shards / --shard-index)

It concatenates the per-episode records across ALL subdirs for each (mode, horizon)
and recomputes the aggregate stats from scratch, so overlapping/disjoint episode
sets across shards are combined correctly (no double counting as long as shards are
disjoint, which strided sharding guarantees).

Usage:
    python 11_merge_eval.py --eval-dir outputs/groot_h_tum_sonata_eval
"""
import argparse, json
from pathlib import Path
import numpy as np


def recompute_stats(per_episode: list) -> dict:
    """Aggregate per-episode records into the stats block (mirrors 09_eval.py)."""
    ep_means   = np.array([e["episode_mean"]               for e in per_episode])
    rot_means  = np.array([e.get("rot_mean_deg", 0.0)      for e in per_episode])
    base_means = np.array([e.get("base_mean", 0.0)         for e in per_episode])
    brot_means = np.array([e.get("base_rot_mean_deg", 0.0) for e in per_episode])
    return {
        "mean_of_means":     float(ep_means.mean()),
        "std_of_means":      float(ep_means.std()),
        "median_of_means":   float(np.median(ep_means)),
        "max_of_means":      float(ep_means.max()),
        "min_of_means":      float(ep_means.min()),
        "n_episodes":        len(per_episode),
        "rot_mean_deg":      float(rot_means.mean()),
        "rot_std_deg":       float(rot_means.std()),
        "base_mean":         float(base_means.mean()),
        "base_rot_mean_deg": float(brot_means.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()

    base = Path(args.eval_dir)

    # Concatenate per-episode records across every subdir, keyed by (mode, horizon).
    # Dedup by episode_id so accidental overlap between shards can't double-count.
    pooled: dict = {}   # pooled[mode][horizon] = {episode_id: record}
    n_dirs = 0
    for subdir in sorted(p for p in base.iterdir() if p.is_dir()):
        result_file = subdir / "eval_results.json"
        if not result_file.exists():
            continue
        n_dirs += 1
        data = json.load(open(result_file))
        for mode, horizons in data.items():
            pooled.setdefault(mode, {})
            for h, v in horizons.items():
                h = int(h)
                pooled[mode].setdefault(h, {})
                eps = v.get("per_episode", [])
                for rec in eps:
                    pooled[mode][h][rec["episode_id"]] = rec
                print(f"  {subdir.name:28s} mode={mode:9s} h={h:<3} "
                      f"+{len(eps)} eps")

    if n_dirs == 0:
        print(f"No eval_results.json found under {base}")
        return

    # Recompute stats per (mode, horizon)
    merged = {}
    for mode, hmap in pooled.items():
        merged[mode] = {}
        for h, ep_dict in hmap.items():
            merged[mode][h] = recompute_stats(list(ep_dict.values()))

    out = base / "eval_results_merged.json"
    json.dump(merged, open(out, "w"), indent=2)

    # Print table
    modes = sorted(merged.keys())
    all_h = sorted({h for m in modes for h in merged[m]})

    print("\n" + "=" * 86)
    print("GR00T-H-N1.7  TUM SonATA Franka  Evaluation (test split)")
    print("Pos: XYZ L2 (cm) | Rot: geodesic angle (deg) | Base: zero-motion baseline")
    print("=" * 86)
    for mode in modes:
        label = "Open-loop (GT state)" if mode == "open_loop" else "Rollout (pred state)"
        print(f"\n{label}")
        print(f"  {'H':>4} | {'Pos cm':>7} | {'±Std':>6} | {'Median':>7} | {'Max':>6}"
              f" | {'Rot°':>6} | {'Base cm':>8} | {'Base°':>6} | N")
        print("  " + "-" * 74)
        for h in all_h:
            if h not in merged.get(mode, {}):
                print(f"  {h:>4} | {'—':>7} | {'—':>6} | {'—':>7} | {'—':>6}"
                      f" | {'—':>6} | {'—':>8} | {'—':>6} | —")
                continue
            s = merged[mode][h]
            print(f"  {h:>4} | {s['mean_of_means']*100:>7.2f} | "
                  f"{s['std_of_means']*100:>6.2f} | "
                  f"{s['median_of_means']*100:>7.2f} | "
                  f"{s['max_of_means']*100:>6.2f} | "
                  f"{s.get('rot_mean_deg', float('nan')):>6.2f} | "
                  f"{s.get('base_mean', float('nan'))*100:>8.2f} | "
                  f"{s.get('base_rot_mean_deg', float('nan')):>6.2f} | "
                  f"{s['n_episodes']}")
    print("=" * 86)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
