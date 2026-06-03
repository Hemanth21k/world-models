"""
Demo video: GR00T-H-N1.7 fine-tuned inference on TUM SonATA Franka.

Generates an MP4 showing per-step model inference alongside the episode:
  - Row 1 : Three live camera views (TPV | Wrist | Ultrasound)
  - Row 2 : Predicted vs ground-truth action (xyz, 3 dims) scrolling over time
  - Row 3 : 3-D scatter of GT EEF trajectory + predicted future horizon

Usage (inside groot-h-dev container):
    cd /workspace/groot_h
    python /workspace/scripts/07_demo_video.py \\
        --model-path /outputs \\
        --dataset-path /data/sonata_all \\
        --traj-ids 1916 1920 1925 \\
        --output-dir /outputs/demo_videos

Env overrides (for the shell launch script):
    MODEL_PATH   - path to fine-tuned checkpoint directory
    DATASET_DIR  - path to sonata_all LeRobot dataset
    OUTPUTS_DIR  - where to write demo videos
    TRAJ_IDS     - space-separated test trajectory IDs
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import open_h.embodiments  # registers Open-H embodiment configs
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.policy import Gr00tPolicy


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ACTION_LABELS = ["Δx (m)", "Δy (m)", "Δz (m)", "r1", "r2", "r3", "r4", "r5", "r6"]
CAMERA_TITLES = ["Third-person", "Wrist", "Ultrasound"]
CAMERA_KEYS   = ["tpv_camera", "wrist_camera", "ultrasound"]


def load_policy(model_path: str) -> Gr00tPolicy:
    """Load fine-tuned policy. Picks the latest checkpoint if path is a dir."""
    p = Path(model_path)
    checkpoints = sorted(
        [d for d in p.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    ) if p.is_dir() and not (p / "config.json").exists() else []
    ckpt = str(checkpoints[-1]) if checkpoints else model_path
    print(f"Loading policy from: {ckpt}")
    return Gr00tPolicy(
        model_path=ckpt,
        embodiment_tag="TUM_SONATA_FRANKA",
        denoising_steps=4,
    )


def build_obs(traj, step: int, loader: LeRobotEpisodeLoader) -> dict:
    """Build observation dict for one inference step."""
    from copy import deepcopy
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.policy.gr00t_policy import parse_observation_gr00t

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action", None)
    data_point = extract_step_data(traj, step, modality_configs, EmbodimentTag.TUM_SONATA_FRANKA)

    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for lang_key in loader.modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text
    return parse_observation_gr00t(obs, loader.modality_configs)


def get_camera_frame(traj, step: int, cam_key: str) -> np.ndarray:
    """Return the current frame (H,W,3) uint8 for a given camera."""
    col = f"video.{cam_key}"
    frames = np.array(traj[col].iloc[step])   # (T, H, W, C)
    return frames[-1].astype(np.uint8)         # most recent frame


def parse_predicted_xyz(action_chunk: dict) -> np.ndarray:
    """Extract horizon × 3 xyz deltas from the policy output."""
    eef = action_chunk.get("action.eef_pose")
    if eef is None:
        return np.zeros((50, 3))
    return np.array(eef)[:, :3]   # (horizon, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Per-step figure renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(
    cameras: list[np.ndarray],        # 3 × (H, W, 3)
    gt_xyz: np.ndarray,               # (T_so_far, 3) absolute EEF xyz history
    pred_xyz_delta: np.ndarray,       # (horizon, 3) predicted relative xyz
    current_eef: np.ndarray,          # (3,) absolute xyz at current step
    gt_action_history: np.ndarray,    # (T_so_far, 3) GT xyz actions so far
    pred_action_history: np.ndarray,  # (T_so_far, 3) pred xyz actions so far
    step: int,
    total_steps: int,
    instruction: str,
) -> np.ndarray:
    """Render one composite frame and return it as (H, W, 3) uint8."""

    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")

    # ── title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f'GR00T-H-N1.7 · TUM SonATA Franka  │  Step {step}/{total_steps}\n"{instruction}"',
        color="white", fontsize=11, y=0.98,
    )

    # ── Row 1: cameras (3 columns) ─────────────────────────────────────────
    for i, (img, title) in enumerate(zip(cameras, CAMERA_TITLES)):
        ax = fig.add_subplot(3, 4, i + 1)
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.axis("off")

    # ── Row 2: pred vs GT action curves (xyz only) ─────────────────────────
    dim_labels = ["Δx", "Δy", "Δz"]
    colors_gt   = ["#4ade80", "#60a5fa", "#f472b6"]
    colors_pred = ["#22c55e", "#3b82f6", "#ec4899"]
    T = len(gt_action_history)
    xs = np.arange(T)

    for i in range(3):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        ax.set_facecolor("#0f0f23")
        if T > 1:
            ax.plot(xs, gt_action_history[:, i],
                    color=colors_gt[i], lw=1.5, label="GT", alpha=0.9)
            ax.plot(xs, pred_action_history[:, i],
                    color=colors_pred[i], lw=1.5, label="Pred", linestyle="--", alpha=0.9)
        ax.set_title(dim_labels[i], color="white", fontsize=9, pad=2)
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        if i == 0:
            ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                      loc="upper left")

    # ── Row 3: 3-D EEF trajectory ──────────────────────────────────────────
    ax3d = fig.add_subplot(3, 4, (9, 12), projection="3d")
    ax3d.set_facecolor("#0f0f23")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_edgecolor("#333355")

    # GT trajectory so far
    if len(gt_xyz) > 1:
        ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
                  color="#4ade80", lw=1.5, alpha=0.8, label="GT path")
        ax3d.scatter(*gt_xyz[-1], color="#4ade80", s=40, zorder=5)

    # Predicted future horizon
    if pred_xyz_delta is not None and len(pred_xyz_delta) > 0:
        horizon_pts = current_eef + np.cumsum(pred_xyz_delta, axis=0)
        ax3d.plot(horizon_pts[:, 0], horizon_pts[:, 1], horizon_pts[:, 2],
                  color="#60a5fa", lw=1.2, linestyle="--", alpha=0.8, label="Pred horizon")
        ax3d.scatter(*horizon_pts[0], color="#60a5fa", s=30, zorder=5)

    ax3d.set_title("EEF Trajectory", color="white", fontsize=9, pad=4)
    ax3d.tick_params(colors="gray", labelsize=6)
    ax3d.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Render to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# Per-trajectory demo
# ──────────────────────────────────────────────────────────────────────────────

def run_trajectory_demo(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    output_dir: Path,
    action_horizon: int = 16,
    max_steps: int = 300,
    fps: int = 10,
):
    print(f"\n── Trajectory {traj_id} ──")
    traj = loader[traj_id]
    traj_len = len(traj)
    steps = min(max_steps, traj_len)
    instruction = str(traj["task"].iloc[0]) if "task" in traj.columns else ""

    # Pre-collect GT absolute EEF positions (xyz from raw action column)
    gt_abs_xyz = np.stack([np.array(a)[:3] for a in traj["action"]])[:steps]

    gt_action_history   = []
    pred_action_history = []
    gt_xyz_trace        = []

    frame_dir = output_dir / f"frames_traj{traj_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    step_counts = list(range(0, steps, action_horizon))

    for frame_idx, step in enumerate(step_counts):
        print(f"  step {step}/{steps}", end="\r")

        # Camera frames
        cameras = [get_camera_frame(traj, step, k) for k in CAMERA_KEYS]

        # Inference
        obs = build_obs(traj, step, loader)
        action_chunk, _ = policy.get_action(obs)

        # Predicted xyz deltas for the horizon
        pred_xyz_delta = parse_predicted_xyz(action_chunk)   # (horizon, 3)

        # Current absolute EEF from state pass-through
        current_eef = np.array(traj["action"].iloc[step])[:3]

        # GT action at this step (raw xyz, first 3 dims of 6D EEF pose)
        gt_action_xyz = np.array(traj["action"].iloc[step])[:3]

        # For curves: show the delta between consecutive GT absolute poses
        if len(gt_abs_xyz) > step + 1:
            gt_delta = gt_abs_xyz[step + 1] - gt_abs_xyz[step]
        else:
            gt_delta = np.zeros(3)
        pred_delta_step0 = pred_xyz_delta[0] if len(pred_xyz_delta) > 0 else np.zeros(3)

        gt_action_history.append(gt_delta)
        pred_action_history.append(pred_delta_step0)
        gt_xyz_trace.append(current_eef)

        frame = render_frame(
            cameras=cameras,
            gt_xyz=np.array(gt_xyz_trace),
            pred_xyz_delta=pred_xyz_delta,
            current_eef=current_eef,
            gt_action_history=np.array(gt_action_history),
            pred_action_history=np.array(pred_action_history),
            step=step,
            total_steps=steps,
            instruction=instruction,
        )

        frame_path = frame_dir / f"frame_{frame_idx:05d}.png"
        plt.imsave(str(frame_path), frame)

    # Compile to mp4 with ffmpeg
    video_path = output_dir / f"demo_traj{traj_id}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(video_path),
    ], check=True, capture_output=True)
    print(f"\n  Saved: {video_path}")
    return video_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/outputs"))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_DIR", "/data/sonata_all"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUTS_DIR", "/outputs/demo_videos"))
    parser.add_argument("--traj-ids", nargs="+", type=int,
                        default=list(map(int, os.environ.get("TRAJ_IDS", "1916 1920 1925").split())))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.model_path)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        embodiment_tag=EmbodimentTag.TUM_SONATA_FRANKA,
    )

    for traj_id in args.traj_ids:
        run_trajectory_demo(
            policy=policy,
            loader=loader,
            traj_id=traj_id,
            output_dir=output_dir,
            action_horizon=args.action_horizon,
            max_steps=args.max_steps,
            fps=args.fps,
        )

    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
