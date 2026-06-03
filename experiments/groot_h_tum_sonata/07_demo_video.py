"""
Demo video: GR00T-H-N1.7 fine-tuned inference on TUM SonATA Franka.

Renders per-step model inference as an MP4:
  - Row 1 : Three live camera views (TPV | Wrist | Ultrasound)
  - Row 2 : Predicted vs GT xyz delta curves scrolling over time
  - Row 3 : 3-D EEF trajectory (GT trace + predicted future horizon)

Usage (inside groot-h-dev container):
    cd /workspace/groot_h
    python /workspace/scripts/07_demo_video.py \\
        --model-path /checkpoints \\
        --dataset-path /data/sonata_all \\
        --traj-ids 1916 1920 \\
        --output-dir /demo_out
"""

import argparse
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import open_h.embodiments  # registers TUM_SONATA_FRANKA
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


CAMERA_TITLES = ["Third-person view", "Wrist camera", "Ultrasound"]
CAMERA_KEYS   = ["tpv_camera", "wrist_camera", "ultrasound"]


# ──────────────────────────────────────────────────────────────────────────────

def latest_checkpoint(model_path: str) -> str:
    p = Path(model_path)
    if (p / "config.json").exists():
        return str(p)
    ckpts = sorted(
        [d for d in p.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    return str(ckpts[-1])


def load_policy(ckpt_path: str, device: int = 0) -> Gr00tPolicy:
    print(f"Loading policy from: {ckpt_path}  (device={device})")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.TUM_SONATA_FRANKA,
        model_path=ckpt_path,
        device=device,
    )
    policy.reset()
    return policy


def build_observation(traj, step: int, modality_cfg: dict) -> tuple[dict, str]:
    """Build the video/state/language observation dict for one inference step.

    N1.7 format:
      video:    {cam: (B=1, T=1, H, W, C) uint8}
      state:    {key: (B=1, T=1, D) float32}
      language: {lang_key: [["instruction"]]}
    """
    video_keys = modality_cfg["video"].modality_keys
    state_keys = modality_cfg["state"].modality_keys
    lang_key   = modality_cfg["language"].modality_keys[0]

    video_dict = {
        cam: np.array(traj[f"video.{cam}"].iloc[step])[np.newaxis, np.newaxis, ...].astype(np.uint8)
        for cam in video_keys
    }  # each: (1, 1, H, W, C)

    state_dict = {
        key: np.array(traj[f"state.{key}"].iloc[step]).flatten().astype(np.float32)[np.newaxis, np.newaxis, :]
        for key in state_keys
    }  # each: (1, 1, D)

    instruction = str(traj[f"language.{lang_key}"].iloc[step])
    lang_dict   = {lang_key: [[instruction]]}

    return {"video": video_dict, "state": state_dict, "language": lang_dict}, instruction


def render_frame(
    cameras: list[np.ndarray],
    gt_xyz_trace: np.ndarray,
    pred_xyz_delta: np.ndarray,
    current_eef: np.ndarray,
    gt_hist: np.ndarray,
    pred_hist: np.ndarray,
    step: int,
    total_steps: int,
    instruction: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(20, 11), dpi=90)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f'GR00T-H-N1.7 · TUM SonATA Franka   step {step}/{total_steps}\n"{instruction}"',
        color="white", fontsize=11, y=0.99,
    )

    # Row 1: 3 cameras
    for i, (img, title) in enumerate(zip(cameras, CAMERA_TITLES)):
        ax = fig.add_subplot(3, 4, i + 1)
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.axis("off")

    # Row 2: pred vs GT delta xyz curves
    dim_labels = ["Δx (m)", "Δy (m)", "Δz (m)"]
    clr_gt   = ["#4ade80", "#60a5fa", "#f472b6"]
    clr_pred = ["#16a34a", "#2563eb", "#db2777"]
    T = len(gt_hist)
    xs = np.arange(T)
    for i in range(3):
        ax = fig.add_subplot(3, 4, 4 + i + 1)
        ax.set_facecolor("#0f0f23")
        if T > 0:
            ax.plot(xs, gt_hist[:, i],   color=clr_gt[i],   lw=1.5, label="GT",   alpha=0.9)
            ax.plot(xs, pred_hist[:, i], color=clr_pred[i], lw=1.5, label="Pred",
                    linestyle="--", alpha=0.9)
        ax.set_title(dim_labels[i], color="white", fontsize=9, pad=2)
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        if i == 0:
            ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="upper left")

    # Row 3: 3-D EEF trajectory
    ax3d = fig.add_subplot(3, 4, (9, 12), projection="3d")
    ax3d.set_facecolor("#0f0f23")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333355")

    if len(gt_xyz_trace) > 1:
        ax3d.plot(gt_xyz_trace[:, 0], gt_xyz_trace[:, 1], gt_xyz_trace[:, 2],
                  color="#4ade80", lw=1.5, alpha=0.8, label="GT path")
        ax3d.scatter(*gt_xyz_trace[-1], color="#4ade80", s=40, zorder=5)

    if pred_xyz_delta is not None and len(pred_xyz_delta) > 0 and current_eef is not None:
        future = current_eef + np.cumsum(pred_xyz_delta[:, :3], axis=0)
        ax3d.plot(future[:, 0], future[:, 1], future[:, 2],
                  color="#60a5fa", lw=1.2, linestyle="--", alpha=0.8, label="Pred horizon")

    ax3d.set_title("EEF Trajectory", color="white", fontsize=9, pad=4)
    ax3d.tick_params(colors="gray", labelsize=6)
    ax3d.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[..., :3]  # (H, W, 3) RGB
    plt.close(fig)
    return buf


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
    policy.reset()
    traj = loader[traj_id]
    steps = min(max_steps, len(traj))
    modality_cfg = loader.modality_configs

    # GT absolute EEF xyz from action.eef_pose column (first 3 dims = xyz)
    gt_abs_xyz = np.stack([np.array(traj["action.eef_pose"].iloc[i])[:3] for i in range(steps)])

    gt_hist, pred_hist, gt_xyz_trace = [], [], []

    frame_dir = output_dir / f"frames_{traj_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    step_counts = list(range(0, steps, action_horizon))
    frame_idx = 0

    for step in step_counts:
        print(f"  step {step:4d}/{steps}", end="\r", flush=True)

        obs, instruction = build_observation(traj, step, modality_cfg)

        # Camera frames for display
        cameras = [
            obs["video"][cam][0, 0] if cam in obs["video"]   # (H, W, C)
            else np.zeros((480, 640, 3), dtype=np.uint8)
            for cam in CAMERA_KEYS
        ]

        # Inference
        action_chunk, _ = policy.get_action(obs)

        # action_chunk: {"eef_pose": (B=1, horizon, 6)} — squeeze batch, take xyz (first 3 dims)
        pred_eef = np.array(action_chunk.get("eef_pose", np.zeros((1, action_horizon, 6))))
        if pred_eef.ndim == 3:
            pred_eef = pred_eef[0]       # (horizon, 6)
        pred_xyz_delta = pred_eef        # (horizon, 6), first 3 used for 3D plot

        # Current absolute EEF position
        current_eef = gt_abs_xyz[step]
        gt_xyz_trace.append(current_eef)

        # GT and pred deltas for the curve plot
        if step + 1 < len(gt_abs_xyz):
            gt_delta = gt_abs_xyz[step + 1] - gt_abs_xyz[step]
        else:
            gt_delta = np.zeros(3)
        pred_delta_0 = pred_xyz_delta[0, :3] if len(pred_xyz_delta) > 0 else np.zeros(3)  # (3,)

        gt_hist.append(gt_delta)
        pred_hist.append(pred_delta_0)

        frame = render_frame(
            cameras=cameras,
            gt_xyz_trace=np.array(gt_xyz_trace),
            pred_xyz_delta=pred_xyz_delta,
            current_eef=current_eef,
            gt_hist=np.array(gt_hist),
            pred_hist=np.array(pred_hist),
            step=step,
            total_steps=steps,
            instruction=instruction,
        )
        plt.imsave(str(frame_dir / f"frame_{frame_idx:05d}.png"), frame)
        frame_idx += 1

    video_path = output_dir / f"demo_traj{traj_id}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(video_path),
    ], check=True, capture_output=True)
    print(f"\n  Saved → {video_path}")
    return video_path


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",   default=os.environ.get("MODEL_PATH",   "/checkpoints"))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_DIR",  "/data/sonata_all"))
    parser.add_argument("--output-dir",   default=os.environ.get("OUTPUTS_DIR",  "/demo_out"))
    parser.add_argument("--traj-ids", nargs="+", type=int,
                        default=list(map(int, os.environ.get("TRAJ_IDS", "1916 1920").split())))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-steps",      type=int, default=300)
    parser.add_argument("--fps",            type=int, default=10)
    parser.add_argument("--device",         type=int, default=0)
    args = parser.parse_args()

    ckpt = latest_checkpoint(args.model_path)
    policy = load_policy(ckpt, device=args.device)

    modality_configs = MODALITY_CONFIGS[EmbodimentTag.TUM_SONATA_FRANKA.value]
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\nAll videos → {output_dir}")


if __name__ == "__main__":
    main()
