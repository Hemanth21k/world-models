"""
Stage 2 of the unified demo: compose the cached rollout (from 14_rollout_cache.py)
into one video. NO model, NO dataset, NO GPU — just the .npz + PyBullet (CPU) for
the arm + matplotlib. Re-run freely to iterate on the layout.

Layout (cameras bigger; robot is the hero; abstract 3D EEF demoted to a minimap):

  ┌─────────┬─────────┬─────────┬───────────────┐
  │  TPV    │  Wrist  │   US    │               │
  ├─────────┼─────────┼─────────┤   ROBOT VIEW  │
  │  act x  │  act y  │  act z  │  GT ghost vs  │
  ├─────────┼─────────┼─────────┤   predicted   │
  │  roll   │  pitch  │  yaw    │               │
  ├─────────┴─────────┴─────────┼───────────────┤
  │   tracking error over time  │  EEF path     │
  └─────────────────────────────┴───────────────┘

Usage (groot-h-dev container, pybullet importable):
    python 15_demo_compose.py --cache /demo_out/cache/rollout_1916_open_loop.npz \
        --output-dir /demo_out --fps 15
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pybullet as pb
import pybullet_data

CAMERA_TITLES = ["Third-person view", "Wrist camera", "Ultrasound"]
XYZ_LABELS = ["x (m)", "y (m)", "z (m)"]
CLR_GT = ["#4ade80", "#60a5fa", "#f472b6"]
CLR_PRED = ["#16a34a", "#2563eb", "#db2777"]
CLR_RPY_GT = ["#fbbf24", "#a78bfa", "#fb923c"]
CLR_RPY_PRED = ["#d97706", "#7c3aed", "#ea580c"]
CLR_GT_PATH = "#4ade80"
CLR_PRED_TRACE = "#f59e0b"
GT_RGB, PRED_RGB, GHOST_ALPHA = [0.25, 0.80, 0.42], [0.96, 0.55, 0.10], 0.45
FS_SUP, FS_INSTR, FS_TITLE, FS_LABEL, FS_TICK, FS_LEG, FS_TEXT = 17, 13, 13, 11, 10, 11, 12


# ── PyBullet robot rendering (from 12_robot_rollout) ────────────────────────

def load_arm():
    return pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)


def arm_links(robot):
    rev = [j for j in range(pb.getNumJoints(robot))
           if pb.getJointInfo(robot, j)[2] == pb.JOINT_REVOLUTE][:7]
    return rev


def set_joints(robot, rev, q):
    for i, j in enumerate(rev):
        pb.resetJointState(robot, j, float(q[i]))


def paint(robot, rgb, alpha):
    c = list(rgb) + [alpha]
    for j in range(-1, pb.getNumJoints(robot)):
        pb.changeVisualShape(robot, j, rgbaColor=c)


def _pass(view, proj, w, h, body):
    img = pb.getCameraImage(w, h, view, proj, renderer=pb.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
    seg = np.reshape(img[4], (h, w))
    mask = (seg >= 0) & ((seg & ((1 << 24) - 1)) == body)
    return rgb, mask


def render_robot(gt_arm, pred_arm, rev, view, proj, w, h, qgt, qpred):
    set_joints(gt_arm, rev, qgt); set_joints(pred_arm, rev, qpred)
    paint(pred_arm, PRED_RGB, 1.0); paint(gt_arm, GT_RGB, 0.0)
    rgb_pred, m_pred = _pass(view, proj, w, h, pred_arm)
    paint(gt_arm, GT_RGB, 1.0); paint(pred_arm, PRED_RGB, 0.0)
    rgb_gt, m_gt = _pass(view, proj, w, h, gt_arm)
    out = rgb_pred.copy()
    ghost = m_gt & (~m_pred)
    out[ghost] = (GHOST_ALPHA * rgb_gt[ghost] + (1 - GHOST_ALPHA) * out[ghost]).astype(np.uint8)
    return out


def style_2d(ax, title, xlabel="step"):
    ax.set_facecolor("#0f0f23")
    ax.set_title(title, color="white", fontsize=FS_TITLE, pad=4)
    ax.set_xlabel(xlabel, color="gray", fontsize=FS_TICK)
    ax.tick_params(colors="gray", labelsize=FS_TICK)
    ax.grid(True, color="#262647", lw=0.6, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")


# ── main composite ──────────────────────────────────────────────────────────

def compose(cache_path, out_dir, fps, robot_w, robot_h):
    d = np.load(cache_path, allow_pickle=True)
    gt_pose, pred_pose = d["gt_pose"], d["pred_pose"]
    gt_joints, pred_joints = d["gt_joints"], d["pred_joints"]
    pos_err, rot_err = d["pos_err"], d["rot_err"]
    cams = [d["cam_tpv"], d["cam_wrist"], d["cam_us"]]
    instr = d["instructions"]
    meta = d["meta"].item()
    T = len(gt_pose)
    traj_id, mode = meta["traj_id"], meta["mode"]
    mode_label = "Open-loop (GT state)" if mode == "open_loop" else "Rollout (pred state)"

    # PyBullet scene (robot panel)
    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    gt_arm, pred_arm = load_arm(), load_arm()
    rev = arm_links(gt_arm)
    ctr = gt_pose[:, :3].mean(0)
    # Portrait panel: aim higher + closer + level-ish so the arm fills the frame
    # instead of leaving empty floor at the bottom.
    tgt = [ctr[0] * 0.6, ctr[1] * 0.6, 0.52]
    view = pb.computeViewMatrix([tgt[0] + 1.0, tgt[1] - 1.0, tgt[2] + 0.42], tgt, [0, 0, 1])
    proj = pb.computeProjectionMatrixFOV(46, robot_w / robot_h, 0.05, 5.0)

    # Fixed 3D minimap bounds
    bnd = []
    for k in range(3):
        lo, hi = gt_pose[:, k].min(), gt_pose[:, k].max()
        pad = max((hi - lo) * 0.15, 1e-3)
        bnd.append((lo - pad, hi + pad))

    frame_dir = out_dir / f"udemo_frames_{traj_id}_{mode}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(T)

    for step in range(T):
        print(f"  compose {step:4d}/{T}", end="\r", flush=True)
        rgb_robot = render_robot(gt_arm, pred_arm, rev, view, proj, robot_w, robot_h,
                                 gt_joints[step], pred_joints[step])

        fig = plt.figure(figsize=(24, 15), dpi=80)   # 1920×1200
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(f"GR00T-H-N1.7  ·  TUM SonATA Franka  ·  {mode_label}  ·  step {step}/{T}",
                     color="white", fontsize=FS_SUP, fontweight="bold", y=0.99)
        fig.text(0.32, 0.962, f'"{instr[step]}"', color="#cbd5e1", fontsize=FS_INSTR,
                 style="italic", ha="center", va="top")

        gs = gridspec.GridSpec(4, 5, figure=fig, height_ratios=[1.7, 0.9, 0.9, 1.95],
                               hspace=0.36, wspace=0.22,
                               left=0.025, right=0.99, top=0.935, bottom=0.045)

        # Row 0: cameras (bigger)
        for i in range(3):
            ax = fig.add_subplot(gs[0, i]); ax.imshow(cams[i][step]); ax.axis("off")
            color = "#22d3ee" if i == 2 else "white"   # emphasize ultrasound
            ax.set_title(CAMERA_TITLES[i], color=color, fontsize=FS_TITLE, pad=5,
                         fontweight="bold" if i == 2 else "normal")

        # Rows 1-2: action curves (xyz, rpy)
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(xs[:step + 1], gt_pose[:step + 1, i], color=CLR_GT[i], lw=2, label="GT")
            ax.plot(xs[:step + 1], pred_pose[:step + 1, i], color=CLR_PRED[i], lw=2,
                    ls="--", label="Pred")
            style_2d(ax, f"action {XYZ_LABELS[i]}")
            if i == 0:
                ax.legend(fontsize=FS_LEG, facecolor="#1a1a2e", labelcolor="white", loc="best")
        rpy = ["action roll (rad)", "action pitch (rad)", "action yaw (rad)"]
        for i in range(3):
            ax = fig.add_subplot(gs[2, i]); dim = i + 3
            ax.plot(xs[:step + 1], gt_pose[:step + 1, dim], color=CLR_RPY_GT[i], lw=2, label="GT")
            ax.plot(xs[:step + 1], pred_pose[:step + 1, dim], color=CLR_RPY_PRED[i], lw=2,
                    ls="--", label="Pred")
            style_2d(ax, rpy[i])
            if i == 0:
                ax.legend(fontsize=FS_LEG, facecolor="#1a1a2e", labelcolor="white", loc="best")

        # Row 3 cols 0-2: tracking error over time
        axE = fig.add_subplot(gs[3, 0:3])
        axE.plot(xs[:step + 1], pos_err[:step + 1] * 100, color="#22d3ee", lw=2, label="XYZ (cm)")
        axE.set_ylabel("XYZ error (cm)", color="#22d3ee", fontsize=FS_LABEL)
        axE.tick_params(axis="y", colors="#22d3ee", labelsize=FS_TICK)
        axE2 = axE.twinx()
        axE2.plot(xs[:step + 1], rot_err[:step + 1], color="#f472b6", lw=2, label="Orientation (°)")
        axE2.set_ylabel("Orientation error (°)", color="#f472b6", fontsize=FS_LABEL)
        axE2.tick_params(axis="y", colors="#f472b6", labelsize=FS_TICK)
        for sp in axE2.spines.values():
            sp.set_edgecolor("#333355")
        style_2d(axE, "Tracking error over time")

        # Rows 0-2 cols 3-4: ROBOT (hero)
        axR = fig.add_subplot(gs[0:3, 3:5]); axR.imshow(rgb_robot); axR.axis("off")
        axR.set_title("Robot — GT (green ghost) vs predicted (orange)",
                      color="white", fontsize=FS_TITLE + 1, pad=6)
        axR.text(0.015, 0.985,
                 f"probe pos err: {pos_err[step]*100:5.2f} cm\n"
                 f"probe ori err: {rot_err[step]:5.2f}°",
                 transform=axR.transAxes, color="white", fontsize=FS_TEXT, va="top",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f23",
                           alpha=0.8, edgecolor="#333355"))

        # Row 3 cols 3-4: EEF path minimap
        ax3 = fig.add_subplot(gs[3, 3:5], projection="3d")
        ax3.set_facecolor("#0f0f23")
        for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor("#333355")
        ax3.set_xlim(*bnd[0]); ax3.set_ylim(*bnd[1]); ax3.set_zlim(*bnd[2])
        ax3.view_init(elev=22, azim=-60)
        try:
            ax3.set_box_aspect(None, zoom=1.35)   # enlarge the cube within the axes
        except TypeError:
            pass
        ax3.plot(gt_pose[:step + 1, 0], gt_pose[:step + 1, 1], gt_pose[:step + 1, 2],
                 color=CLR_GT_PATH, lw=2, label="GT")
        ax3.plot(pred_pose[:step + 1, 0], pred_pose[:step + 1, 1], pred_pose[:step + 1, 2],
                 color=CLR_PRED_TRACE, lw=2, label="Pred")
        ax3.scatter(*gt_pose[step, :3], color=CLR_GT_PATH, s=40)
        ax3.tick_params(colors="gray", labelsize=7)
        ax3.set_title("EEF path", color="white", fontsize=FS_TITLE, pad=2)
        ax3.legend(fontsize=FS_LEG - 1, facecolor="#1a1a2e", labelcolor="white", loc="upper right")

        fig.savefig(str(frame_dir / f"frame_{step:05d}.png"), facecolor="#1a1a2e")
        plt.close(fig)
    pb.disconnect()

    video = out_dir / f"unified_demo_traj{traj_id}_{mode}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                    "-i", str(frame_dir / "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video)],
                   check=True, capture_output=True)
    print(f"\n  Saved → {video}")
    return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, help="rollout .npz file(s)")
    ap.add_argument("--output-dir", default="/demo_out")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--robot-w", type=int, default=820)
    ap.add_argument("--robot-h", type=int, default=900)
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for c in args.cache:
        compose(Path(c), out_dir, args.fps, args.robot_w, args.robot_h)
    print(f"\nAll unified demos → {out_dir}")


if __name__ == "__main__":
    main()
