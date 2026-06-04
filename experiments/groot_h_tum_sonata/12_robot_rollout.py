"""
Robot rollout: animate the Franka executing GR00T-H-N1.7's commanded EEF poses.

Standalone PyBullet video — GT arm (translucent "ghost") overlaid with the
predicted arm (solid). The gap between them is the model's error, shown as a
moving robot.

How it works
------------
- GT arm   : recorded joint angles → set directly (exact, no IK).
- Pred arm : model predicts the ultrasound-probe pose (action.eef_pose). We
             self-calibrate the rigid probe→flange tool transform from the GT
             (FK(joints) vs recorded pose), map the predicted probe pose back to
             a `panda_grasptarget` target, and solve IK (seeded for smoothness).
- Same action/GT time alignment as 07_demo_video / 09_eval (delta_indices 1..50:
  the chunk inferred at step s forecasts rows s+1..s+50; row r uses index r-s-1).

Headless: uses PyBullet's TinyRenderer (CPU), so no GPU is needed for rendering;
the GPU is only used by the policy. Needs `pybullet` (see 13_run_robot_rollout.sh).

Usage (inside groot-h-dev container, with pybullet importable):
    python 12_robot_rollout.py --model-path /checkpoints --dataset-path /data/sonata_all \
        --output-dir /demo_out --traj-ids 1916 --mode open_loop --max-steps 200
"""
import argparse
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation

import open_h.embodiments
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

CAMERA_KEYS = ["tpv_camera", "wrist_camera", "ultrasound"]

# Franka Panda joint limits / rest pose (for stable IK)
JL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JU = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
JR = [u - l for l, u in zip(JL, JU)]
REST = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# GT = green ghost, predicted = solid orange. Rendered in separate passes and
# composited with masks (no transparency z-fighting / flicker).
GT_RGB   = [0.25, 0.80, 0.42]    # green  (ground truth, shown as faint ghost)
PRED_RGB = [0.96, 0.55, 0.10]    # orange (predicted, solid, drawn on top)
GHOST_ALPHA = 0.45               # how strongly the GT ghost shows over the floor


# ── policy / data helpers (mirror 07_demo_video) ────────────────────────────

def latest_checkpoint(model_path: str) -> str:
    p = Path(model_path)
    if (p / "config.json").exists():
        return str(p)
    ckpts = sorted([d for d in p.iterdir() if d.name.startswith("checkpoint-")],
                   key=lambda d: int(d.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {model_path}")
    return str(ckpts[-1])


def build_observation(traj, step, modality_cfg, eef_override=None):
    video_keys = modality_cfg["video"].modality_keys
    state_keys = modality_cfg["state"].modality_keys
    lang_key = modality_cfg["language"].modality_keys[0]
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


# ── PyBullet scene ──────────────────────────────────────────────────────────

def load_arm():
    return pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)


def paint(robot, rgb, alpha):
    c = list(rgb) + [alpha]
    for j in range(-1, pb.getNumJoints(robot)):
        pb.changeVisualShape(robot, j, rgbaColor=c)


def arm_links(robot):
    rev = [j for j in range(pb.getNumJoints(robot))
           if pb.getJointInfo(robot, j)[2] == pb.JOINT_REVOLUTE][:7]
    grasp = next((j for j in range(pb.getNumJoints(robot))
                  if "grasptarget" in pb.getJointInfo(robot, j)[12].decode()), rev[-1])
    return rev, grasp


def set_joints(robot, rev, q):
    for i, j in enumerate(rev):
        pb.resetJointState(robot, j, float(q[i]))


def fk(robot, link):
    ls = pb.getLinkState(robot, link, computeForwardKinematics=True)
    return list(ls[4]), list(ls[5])   # pos, orn(quat)


def calibrate_tool(robot, rev, grasp, joints, action, n=60):
    """Estimate the rigid grasptarget→probe transform from GT frames."""
    idxs = np.linspace(0, len(joints) - 1, min(n, len(joints))).astype(int)
    pos_t, quat_t = [], []
    for t in idxs:
        set_joints(robot, rev, joints[t][:7])
        gp, gq = fk(robot, grasp)
        probe_q = pb.getQuaternionFromEuler(action[t, 3:6].tolist())
        inv_p, inv_q = pb.invertTransform(gp, gq)
        tp, tq = pb.multiplyTransforms(inv_p, inv_q, action[t, :3].tolist(), probe_q)
        pos_t.append(tp); quat_t.append(tq)
    pos = np.mean(pos_t, axis=0)
    quat = Rotation.from_quat(np.array(quat_t)).mean().as_quat()   # average rotation
    return pos.tolist(), quat.tolist()


def probe_to_grasp(probe_pos, probe_eul, tool_pos, tool_quat):
    """Map a desired probe pose to the grasptarget target pose (undo tool xform)."""
    probe_q = pb.getQuaternionFromEuler(list(probe_eul))
    inv_p, inv_q = pb.invertTransform(tool_pos, tool_quat)
    return pb.multiplyTransforms(list(probe_pos), probe_q, inv_p, inv_q)


def solve_ik(robot, grasp, rev, target_pos, target_orn, seed):
    set_joints(robot, rev, seed)            # seed IK with previous solution
    q = pb.calculateInverseKinematics(
        robot, grasp, target_pos, target_orn,
        lowerLimits=JL, upperLimits=JU, jointRanges=JR, restPoses=list(seed),
        maxNumIterations=120, residualThreshold=1e-4)
    return np.array(q[:7])


def _pass(view, proj, w, h, body):
    """Render the scene and return (rgb, mask-of-`body`)."""
    img = pb.getCameraImage(w, h, view, proj, renderer=pb.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
    seg = np.reshape(img[4], (h, w))
    mask = (seg >= 0) & ((seg & ((1 << 24) - 1)) == body)
    return rgb, mask


def render_overlay(gt_arm, pred_arm, view, proj, w, h):
    """Two-pass composite: solid predicted arm on top, GT as a faint ghost.
    Each arm is rendered alone (the other hidden via alpha=0), so coincident
    surfaces never z-fight -> no flicker."""
    paint(pred_arm, PRED_RGB, 1.0); paint(gt_arm, GT_RGB, 0.0)
    rgb_pred, m_pred = _pass(view, proj, w, h, pred_arm)
    paint(gt_arm, GT_RGB, 1.0); paint(pred_arm, PRED_RGB, 0.0)
    rgb_gt, m_gt = _pass(view, proj, w, h, gt_arm)
    out = rgb_pred.copy()                       # floor + solid predicted arm
    ghost = m_gt & (~m_pred)                    # GT visible only where pred doesn't cover
    out[ghost] = (GHOST_ALPHA * rgb_gt[ghost] +
                  (1 - GHOST_ALPHA) * out[ghost]).astype(np.uint8)
    return out


# ── main rollout ────────────────────────────────────────────────────────────

def run(policy, loader, traj_id, out_dir, mode, action_horizon, max_steps, fps):
    print(f"\n── Robot rollout: traj {traj_id} [{mode}] ──")
    policy.reset()
    traj = loader[traj_id]
    steps = min(max_steps, len(traj))
    mcfg = loader.modality_configs

    gt_raw = np.stack([np.array(traj["action.eef_pose"].iloc[i]).astype(np.float32)
                       for i in range(steps)])              # (T,6) GT probe pose
    gt_joints = np.stack([np.array(traj["state.joint_angles"].iloc[i]).astype(np.float32)
                          for i in range(steps)])           # (T,7)
    gt_R = Rotation.from_euler("xyz", gt_raw[:, 3:6])

    # PyBullet scene: ghost GT arm + solid predicted arm
    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("plane.urdf")
    gt_arm = load_arm()
    pred_arm = load_arm()
    rev, grasp = arm_links(gt_arm)
    rev_p, grasp_p = arm_links(pred_arm)

    tool_pos, tool_quat = calibrate_tool(gt_arm, rev, grasp, gt_joints, gt_raw)
    print(f"  tool offset (grasp→probe): pos {np.round(tool_pos,4)} "
          f"quat {np.round(tool_quat,3)}")

    # Fixed camera framing the workspace (centered on mean GT probe xyz)
    # Aim between the base (origin) and the probe region so the whole arm is
    # centered (aiming at the probe alone pushes the base low/left).
    ctr = gt_raw[:, :3].mean(0)
    tgt = [ctr[0] * 0.55, ctr[1] * 0.55, 0.40]
    view = pb.computeViewMatrix(cameraEyePosition=[tgt[0] + 1.25, tgt[1] - 1.25, tgt[2] + 0.75],
                                cameraTargetPosition=tgt, cameraUpVector=[0, 0, 1])
    W, H = 960, 720
    proj = pb.computeProjectionMatrixFOV(fov=52, aspect=W / H, nearVal=0.05, farVal=5.0)

    frame_dir = out_dir / f"robot_frames_{traj_id}_{mode}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cached, anchor, pred_eef = None, -1, gt_raw[0].copy()
    pos_err = rot_err = 0.0
    instr = ""

    for step in range(steps):
        print(f"  step {step:4d}/{steps}", end="\r", flush=True)
        obs, ins = build_observation(traj, step, mcfg,
                                     None if mode == "open_loop" else pred_eef)
        if ins:
            instr = ins

        # current-row prediction (aligned), then re-plan (mirror 07/09)
        if cached is None:
            pred_step = gt_raw[step].copy()
        else:
            idx = min(step - anchor - 1, len(cached) - 1)
            pred_step = gt_raw[step].copy() if idx < 0 else cached[idx]

        # GT arm (exact) + predicted arm (IK from predicted probe pose).
        # Seed IK with the current GT joints so the redundant elbow resolves near
        # GT's configuration → visible arm divergence reflects true EEF-pose error,
        # not arbitrary null-space choices.
        set_joints(gt_arm, rev, gt_joints[step][:7])
        gpos, gorn = probe_to_grasp(pred_step[:3], pred_step[3:6], tool_pos, tool_quat)
        qpred = solve_ik(pred_arm, grasp_p, rev_p, gpos, gorn, gt_joints[step][:7])
        set_joints(pred_arm, rev_p, qpred)

        pos_err = float(np.linalg.norm(pred_step[:3] - gt_raw[step][:3]))
        rot_err = float(np.degrees(
            (Rotation.from_euler("xyz", pred_step[3:6]).inv() * gt_R[step]).magnitude()))

        if mode == "rollout":
            pred_eef = pred_step.copy()
        if step % action_horizon == 0:
            chunk, _ = policy.get_action(obs)
            pr = np.array(chunk.get("eef_pose", np.zeros((1, 50, 6))))
            cached = pr[0] if pr.ndim == 3 else pr
            anchor = step

        rgb = render_overlay(gt_arm, pred_arm, view, proj, W, H)

        # caption with matplotlib (keeps demo styling)
        fig = plt.figure(figsize=(W / 100, H / 100 + 1.1), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(f"GR00T-H-N1.7 · TUM SonATA Franka · robot rollout "
                     f"[{'Open-loop' if mode=='open_loop' else 'Rollout'}] · step {step}/{steps}",
                     color="white", fontsize=12, fontweight="bold", y=0.985)
        fig.text(0.5, 0.93, f'"{instr}"', color="#cbd5e1", fontsize=9.5,
                 style="italic", ha="center", va="top")
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.88]); ax.imshow(rgb); ax.axis("off")
        ax.text(0.015, 0.97,
                f"GT = green (ghost)   ·   predicted = orange (solid)\n"
                f"probe pos err: {pos_err*100:5.2f} cm\n"
                f"probe ori err: {rot_err:5.2f}°",
                transform=ax.transAxes, color="white", fontsize=11, va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f23",
                          alpha=0.8, edgecolor="#333355"))
        fig.savefig(str(frame_dir / f"frame_{step:05d}.png"), facecolor="#1a1a2e")
        plt.close(fig)

    pb.disconnect()
    video = out_dir / f"robot_rollout_traj{traj_id}_{mode}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                    "-i", str(frame_dir / "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video)],
                   check=True, capture_output=True)
    print(f"\n  Saved → {video}")
    return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/checkpoints"))
    ap.add_argument("--dataset-path", default=os.environ.get("DATASET_DIR", "/data/sonata_all"))
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUTS_DIR", "/demo_out"))
    ap.add_argument("--traj-ids", nargs="+", type=int, default=[1916])
    ap.add_argument("--mode", default="open_loop", choices=["open_loop", "rollout", "both"])
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=9999)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.TUM_SONATA_FRANKA,
                         model_path=latest_checkpoint(args.model_path), device=args.device)
    policy.reset()
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=MODALITY_CONFIGS[EmbodimentTag.TUM_SONATA_FRANKA.value])

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["open_loop", "rollout"] if args.mode == "both" else [args.mode]
    for tid in args.traj_ids:
        for m in modes:
            run(policy, loader, tid, out_dir, m, args.action_horizon, args.max_steps, args.fps)
    print(f"\nAll robot rollouts → {out_dir}")


if __name__ == "__main__":
    main()
