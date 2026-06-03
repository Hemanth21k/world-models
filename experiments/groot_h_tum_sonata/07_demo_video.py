"""
Demo video: GR00T-H-N1.7 fine-tuned inference on TUM SonATA Franka.

Two inference modes:
  --mode open_loop  : GT observation at every step (best-case action prediction quality)
  --mode rollout    : predicted EEF pose fed back as next state (tests closed-loop behaviour)

Layout (3 rows × 5 cols):
  Row 0 cols 0-2 : cameras (TPV | Wrist | Ultrasound)
  Row 1 cols 0-2 : GT vs Pred xyz curves
  Row 2 cols 0-1 : GT vs Pred orientation curves (roll | pitch | yaw)
  Row 2 col  2   : running position error (L2)
  Cols 3-4 all rows : 3-D EEF trajectory with faded prediction history

Usage inside groot-h-dev container:
    cd /workspace/groot_h
    python /workspace/scripts/07_demo_video.py \\
        --model-path /checkpoints --dataset-path /data/sonata_all \\
        --traj-ids 1916 --mode both --output-dir /demo_out
"""

import argparse
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

import open_h.embodiments
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


CAMERA_TITLES   = ["Third-person view", "Wrist camera", "Ultrasound"]
CAMERA_KEYS     = ["tpv_camera", "wrist_camera", "ultrasound"]
XYZ_LABELS      = ["x (m)", "y (m)", "z (m)"]
RPY_LABELS      = ["roll (rad)", "pitch (rad)", "yaw (rad)"]
CLR_GT          = ["#4ade80", "#60a5fa", "#f472b6"]
CLR_PRED        = ["#16a34a", "#2563eb", "#db2777"]
CLR_RPY_GT      = ["#fbbf24", "#a78bfa", "#fb923c"]
CLR_RPY_PRED    = ["#d97706", "#7c3aed", "#ea580c"]


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
    policy.reset()
    return policy


def build_observation(traj, step: int, modality_cfg: dict,
                       eef_override: np.ndarray | None = None) -> tuple[dict, str]:
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
    return {"video": video_dict, "state": state_dict, "language": {lang_key: [[instruction]]}}, instruction


# ──────────────────────────────────────────────────────────────────────────────

def render_frame(
    cameras: list[np.ndarray],
    gt_action_hist: np.ndarray,       # (T, 6) GT absolute EEF pose over time
    pred_action_hist: np.ndarray,     # (T, 6) pred first-step over time
    gt_eef_path: np.ndarray,          # (T, 3) GT xyz path
    pred_horizon_history: list,       # list of (step, horizon_pts (16,3)) — anchored
    current_eef: np.ndarray,          # (3,)
    error_hist: np.ndarray,           # (T,) per-step L2 xyz error
    step: int,
    total_steps: int,
    instruction: str,
    mode: str,
) -> np.ndarray:

    fig = plt.figure(figsize=(24, 12), dpi=85)
    fig.patch.set_facecolor("#1a1a2e")
    mode_label = "Open-loop (GT state)" if mode == "open_loop" else "Rollout (pred state)"
    fig.suptitle(
        f'GR00T-H-N1.7 · TUM SonATA Franka  [{mode_label}]  step {step}/{total_steps}'
        f'\n"{instruction}"',
        color="white", fontsize=11, y=0.99,
    )

    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.04, right=0.97, top=0.90, bottom=0.05,
    )

    # ── Row 0: cameras ─────────────────────────────────────────────────────
    for i, (img, title) in enumerate(zip(cameras, CAMERA_TITLES)):
        ax = fig.add_subplot(gs[0, i])
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.axis("off")

    T = len(gt_action_hist)
    xs = np.arange(T)

    # ── Row 1: xyz action curves ────────────────────────────────────────────
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor("#0f0f23")
        if T > 0:
            ax.plot(xs, gt_action_hist[:, i],   color=CLR_GT[i],   lw=1.5, label="GT",   alpha=0.9)
            ax.plot(xs, pred_action_hist[:, i], color=CLR_PRED[i], lw=1.5, label="Pred",
                    linestyle="--", alpha=0.9)
        ax.set_title(f"action {XYZ_LABELS[i]}", color="white", fontsize=9, pad=2)
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        if i == 0:
            ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="best")

    # ── Row 2, cols 0-1: RPY orientation curves ─────────────────────────────
    rpy_titles = ["Roll (dim 3)", "Pitch (dim 4)", "Yaw (dim 5)"]
    for i in range(3):
        dim = i + 3
        ax = fig.add_subplot(gs[2, i] if i < 2 else gs[2, 2])
        ax.set_facecolor("#0f0f23")
        if T > 0 and gt_action_hist.shape[1] > dim:
            ax.plot(xs, gt_action_hist[:, dim],   color=CLR_RPY_GT[i],   lw=1.5,
                    label="GT",   alpha=0.9)
            ax.plot(xs, pred_action_hist[:, dim], color=CLR_RPY_PRED[i], lw=1.5,
                    label="Pred", linestyle="--", alpha=0.9)
        ax.set_title(rpy_titles[i], color="white", fontsize=9, pad=2)
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        if i == 0:
            ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="best")

    # ── Row 2, col 2: running error plot → replace with error panel ─────────
    # (Col 2 used for Yaw above — put error in a text box on the 3D panel instead)

    # ── Cols 3-4, all rows: 3-D EEF trajectory ─────────────────────────────
    ax3d = fig.add_subplot(gs[0:3, 3:5], projection="3d")
    ax3d.set_facecolor("#0f0f23")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333355")
    ax3d.tick_params(colors="gray", labelsize=7)

    # GT path
    if len(gt_eef_path) > 1:
        ax3d.plot(gt_eef_path[:, 0], gt_eef_path[:, 1], gt_eef_path[:, 2],
                  color="#4ade80", lw=2, alpha=0.9, label="GT path", zorder=3)
    if len(gt_eef_path) > 0:
        ax3d.scatter(*gt_eef_path[-1], color="#4ade80", s=60, zorder=5)

    # Prediction history: past chunks = solid, current chunk = dashed
    n_hist = len(pred_horizon_history)
    for hi, (h_step, h_pts) in enumerate(pred_horizon_history):
        is_current = (hi == n_hist - 1)
        age = n_hist - hi                          # 1 = most recent, n_hist = oldest
        alpha = 0.85 if is_current else max(0.08, 0.55 * (1.0 - age / max(n_hist, 1)))
        lw    = 2.0  if is_current else max(0.6, 1.2 * (1.0 - age / max(n_hist, 1)))
        ls    = "--" if is_current else "-"        # dashed = future prediction, solid = past
        ax3d.plot(h_pts[:, 0], h_pts[:, 1], h_pts[:, 2],
                  color="#60a5fa", lw=lw, linestyle=ls, alpha=alpha,
                  label="Pred (current)" if is_current else ("Pred (past)" if hi == n_hist - 2 else None))
    if n_hist > 0 and len(pred_horizon_history[-1][1]) > 0:
        ax3d.scatter(*pred_horizon_history[-1][1][-1], color="#60a5fa", s=40, zorder=5)

    # Current EEF pose + error text box
    if current_eef is not None:
        mean_err = float(np.mean(error_hist)) if len(error_hist) > 0 else 0.0
        curr_err = float(error_hist[-1]) if len(error_hist) > 0 else 0.0
        pose_txt = (
            f"Current EEF\n"
            f"  x: {current_eef[0]:+.4f} m\n"
            f"  y: {current_eef[1]:+.4f} m\n"
            f"  z: {current_eef[2]:+.4f} m\n"
            f"\nXYZ Error (L2)\n"
            f"  step:  {curr_err*100:.2f} cm\n"
            f"  mean:  {mean_err*100:.2f} cm"
        )
        ax3d.text2D(0.02, 0.98, pose_txt,
                    transform=ax3d.transAxes,
                    color="white", fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f23",
                              alpha=0.85, edgecolor="#333355"))

    ax3d.set_xlabel("x (m)", color="gray", fontsize=8, labelpad=2)
    ax3d.set_ylabel("y (m)", color="gray", fontsize=8, labelpad=2)
    ax3d.set_zlabel("z (m)", color="gray", fontsize=8, labelpad=2)
    ax3d.set_title("EEF Trajectory", color="white", fontsize=10, pad=6)
    ax3d.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                loc="upper right", framealpha=0.7)

    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return buf


# ──────────────────────────────────────────────────────────────────────────────

def run_trajectory_demo(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    output_dir: Path,
    mode: str = "open_loop",
    action_horizon: int = 16,
    max_steps: int = 9999,
    fps: int = 15,
):
    print(f"\n── Trajectory {traj_id}  [{mode}] ──")
    policy.reset()
    traj = loader[traj_id]
    steps = min(max_steps, len(traj))
    modality_cfg = loader.modality_configs

    gt_raw     = np.stack([np.array(traj["action.eef_pose"].iloc[i]).astype(np.float32)
                            for i in range(steps)])   # (T, 6)
    gt_abs_xyz = gt_raw[:, :3]                        # (T, 3)

    gt_action_hist      = []
    pred_action_hist    = []
    gt_eef_path         = []
    pred_horizon_history = []   # list of (step, (16,3)) anchored predicted paths
    error_hist          = []

    current_eef_xyz = gt_abs_xyz[0].copy()
    pred_eef_6d     = gt_raw[0].copy()
    cached_pred     = np.zeros((action_horizon, 6))
    cached_instruction = ""

    frame_dir = output_dir / f"frames_{traj_id}_{mode}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for step in range(steps):
        print(f"  step {step:4d}/{steps}", end="\r", flush=True)

        eef_for_state = None if mode == "open_loop" else pred_eef_6d
        obs, instruction = build_observation(traj, step, modality_cfg, eef_for_state)
        if instruction:
            cached_instruction = instruction

        cameras = [
            obs["video"][cam][0, 0] if cam in obs["video"]
            else np.zeros((480, 640, 3), dtype=np.uint8)
            for cam in CAMERA_KEYS
        ]

        # Inference at start of each chunk
        if step % action_horizon == 0:
            action_chunk, _ = policy.get_action(obs)
            pred_raw = np.array(action_chunk.get("eef_pose", np.zeros((1, action_horizon, 6))))
            if pred_raw.ndim == 3:
                pred_raw = pred_raw[0]
            cached_pred = pred_raw

            # Store anchored near-horizon (first 16 steps) for history plot
            if len(gt_eef_path) > 0 or step == 0:
                near = cached_pred[:16, :3].copy()
                anchor_offset = gt_abs_xyz[step] - near[0]
                anchored = near + anchor_offset
                pred_horizon_history.append((step, anchored))

        chunk_offset   = step % action_horizon
        pred_remaining = cached_pred[chunk_offset:]
        pred_step0     = pred_remaining[0]

        gt_action_hist.append(gt_raw[step])
        pred_action_hist.append(pred_step0)
        gt_eef_path.append(gt_abs_xyz[step])

        # Per-step XYZ L2 error
        xyz_err = np.linalg.norm(pred_step0[:3] - gt_raw[step][:3])
        error_hist.append(xyz_err)

        # Rollout state update
        if mode == "rollout":
            current_eef_xyz = pred_step0[:3].copy()
            pred_eef_6d = pred_step0.copy()
        else:
            current_eef_xyz = gt_abs_xyz[step].copy()

        frame = render_frame(
            cameras=cameras,
            gt_action_hist=np.array(gt_action_hist),
            pred_action_hist=np.array(pred_action_hist),
            gt_eef_path=np.array(gt_eef_path),
            pred_horizon_history=pred_horizon_history,
            current_eef=current_eef_xyz,
            error_hist=np.array(error_hist),
            step=step,
            total_steps=steps,
            instruction=cached_instruction,
            mode=mode,
        )
        plt.imsave(str(frame_dir / f"frame_{step:05d}.png"), frame)

    # Print summary
    err_arr = np.array(error_hist)
    print(f"\n  XYZ L2 error — mean: {err_arr.mean()*100:.2f} cm  "
          f"median: {np.median(err_arr)*100:.2f} cm  "
          f"max: {err_arr.max()*100:.2f} cm")

    video_path = output_dir / f"demo_traj{traj_id}_{mode}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(video_path),
    ], check=True, capture_output=True)
    print(f"  Saved → {video_path}")
    return video_path, err_arr


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",     default=os.environ.get("MODEL_PATH",   "/checkpoints"))
    parser.add_argument("--dataset-path",   default=os.environ.get("DATASET_DIR",  "/data/sonata_all"))
    parser.add_argument("--output-dir",     default=os.environ.get("OUTPUTS_DIR",  "/demo_out"))
    parser.add_argument("--traj-ids",  nargs="+", type=int,
                        default=list(map(int, os.environ.get("TRAJ_IDS", "1916 1920").split())))
    parser.add_argument("--mode",           default="open_loop",
                        choices=["open_loop", "rollout", "both"])
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-steps",      type=int, default=9999)
    parser.add_argument("--fps",            type=int, default=15)
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

    modes = ["open_loop", "rollout"] if args.mode == "both" else [args.mode]

    for traj_id in args.traj_ids:
        for mode in modes:
            _, errors = run_trajectory_demo(
                policy=policy, loader=loader, traj_id=traj_id,
                output_dir=output_dir, mode=mode,
                action_horizon=args.action_horizon,
                max_steps=args.max_steps, fps=args.fps,
            )

    print(f"\nAll videos → {output_dir}")


if __name__ == "__main__":
    main()
