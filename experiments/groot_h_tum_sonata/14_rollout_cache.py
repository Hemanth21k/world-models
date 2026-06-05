"""
Stage 1 of the unified demo: run GR00T-H-N1.7 inference + IK ONCE per trajectory
and cache everything the compositor needs into a self-contained .npz.

This decouples the expensive part (policy + IK, needs GPU) from rendering, so the
compositor (15_demo_compose.py) can re-render any layout in seconds without ever
touching the model.

Cached per step: GT/predicted EEF pose, GT/predicted joint angles (predicted via
IK), XYZ + geodesic-orientation errors, the 3 camera frames (downsampled), the
instruction, and the calibrated probe tool transform.

Same action/GT time alignment as 07_demo_video / 09_eval.

Usage (groot-h-dev container, pybullet importable):
    python 14_rollout_cache.py --model-path /checkpoints --dataset-path /data/sonata_all \
        --output-dir /demo_out --traj-ids 1916 --mode both
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation

import open_h.embodiments
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

CAMERA_KEYS = ["tpv_camera", "wrist_camera", "ultrasound"]
JL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JU = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
JR = [u - l for l, u in zip(JL, JU)]


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


def arm_links(robot):
    rev = [j for j in range(pb.getNumJoints(robot))
           if pb.getJointInfo(robot, j)[2] == pb.JOINT_REVOLUTE][:7]
    grasp = next((j for j in range(pb.getNumJoints(robot))
                  if "grasptarget" in pb.getJointInfo(robot, j)[12].decode()), rev[-1])
    return rev, grasp


def set_joints(robot, rev, q):
    for i, j in enumerate(rev):
        pb.resetJointState(robot, j, float(q[i]))


def calibrate_tool(robot, rev, grasp, joints, action, n=60):
    idxs = np.linspace(0, len(joints) - 1, min(n, len(joints))).astype(int)
    pos_t, quat_t = [], []
    for t in idxs:
        set_joints(robot, rev, joints[t][:7])
        ls = pb.getLinkState(robot, grasp, computeForwardKinematics=True)
        probe_q = pb.getQuaternionFromEuler(action[t, 3:6].tolist())
        inv_p, inv_q = pb.invertTransform(ls[4], ls[5])
        tp, tq = pb.multiplyTransforms(inv_p, inv_q, action[t, :3].tolist(), probe_q)
        pos_t.append(tp); quat_t.append(tq)
    return (np.mean(pos_t, 0).tolist(),
            Rotation.from_quat(np.array(quat_t)).mean().as_quat().tolist())


def probe_to_grasp(probe_pos, probe_eul, tool_pos, tool_quat):
    probe_q = pb.getQuaternionFromEuler(list(probe_eul))
    inv_p, inv_q = pb.invertTransform(tool_pos, tool_quat)
    return pb.multiplyTransforms(list(probe_pos), probe_q, inv_p, inv_q)


def solve_ik(robot, grasp, rev, target_pos, target_orn, seed):
    set_joints(robot, rev, seed)
    q = pb.calculateInverseKinematics(
        robot, grasp, target_pos, target_orn,
        lowerLimits=JL, upperLimits=JU, jointRanges=JR, restPoses=list(seed),
        maxNumIterations=120, residualThreshold=1e-4)
    return np.array(q[:7], dtype=np.float32)


def cache_episode(policy, loader, traj_id, out_dir, mode, action_horizon, max_steps, cam_w):
    print(f"\n── Caching rollout: traj {traj_id} [{mode}] ──")
    policy.reset()
    traj = loader[traj_id]
    steps = min(max_steps, len(traj))
    mcfg = loader.modality_configs

    gt_pose = np.stack([np.array(traj["action.eef_pose"].iloc[i]).astype(np.float32)
                        for i in range(steps)])
    gt_joints = np.stack([np.array(traj["state.joint_angles"].iloc[i]).astype(np.float32)
                          for i in range(steps)])[:, :7]
    gt_R = Rotation.from_euler("xyz", gt_pose[:, 3:6])

    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    arm = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    rev, grasp = arm_links(arm)
    tool_pos, tool_quat = calibrate_tool(arm, rev, grasp, gt_joints, gt_pose)

    cam_h = int(round(cam_w * 3 / 4))
    cams = {c: np.zeros((steps, cam_h, cam_w, 3), np.uint8) for c in CAMERA_KEYS}
    pred_pose = np.zeros((steps, 6), np.float32)
    pred_joints = np.zeros((steps, 7), np.float32)
    pos_err = np.zeros(steps, np.float32)
    rot_err = np.zeros(steps, np.float32)
    instr = []

    cached, anchor, pred_eef, last_instr = None, -1, gt_pose[0].copy(), ""
    for step in range(steps):
        print(f"  step {step:4d}/{steps}", end="\r", flush=True)
        obs, ins = build_observation(traj, step, mcfg,
                                     None if mode == "open_loop" else pred_eef)
        last_instr = ins or last_instr
        instr.append(last_instr)
        for c in CAMERA_KEYS:
            fr = obs["video"][c][0, 0] if c in obs["video"] else np.zeros((480, 640, 3), np.uint8)
            cams[c][step] = np.asarray(Image.fromarray(fr).resize((cam_w, cam_h)))

        if cached is None:
            ps = gt_pose[step].copy()
        else:
            idx = min(step - anchor - 1, len(cached) - 1)
            ps = gt_pose[step].copy() if idx < 0 else cached[idx]
        pred_pose[step] = ps

        gpos, gorn = probe_to_grasp(ps[:3], ps[3:6], tool_pos, tool_quat)
        pred_joints[step] = solve_ik(arm, grasp, rev, gpos, gorn, gt_joints[step])
        pos_err[step] = np.linalg.norm(ps[:3] - gt_pose[step][:3])
        rot_err[step] = np.degrees(
            (Rotation.from_euler("xyz", ps[3:6]).inv() * gt_R[step]).magnitude())

        if mode == "rollout":
            pred_eef = ps.copy()
        if step % action_horizon == 0:
            chunk, _ = policy.get_action(obs)
            pr = np.array(chunk.get("eef_pose", np.zeros((1, 50, 6))))
            cached = pr[0] if pr.ndim == 3 else pr
            anchor = step
    pb.disconnect()

    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"rollout_{traj_id}_{mode}.npz"
    np.savez_compressed(
        out, gt_pose=gt_pose, pred_pose=pred_pose, gt_joints=gt_joints,
        pred_joints=pred_joints, pos_err=pos_err, rot_err=rot_err,
        cam_tpv=cams["tpv_camera"], cam_wrist=cams["wrist_camera"],
        cam_us=cams["ultrasound"], instructions=np.array(instr, dtype=object),
        tool_pos=np.array(tool_pos), tool_quat=np.array(tool_quat),
        meta=np.array({"traj_id": traj_id, "mode": mode,
                       "action_horizon": action_horizon}, dtype=object))
    print(f"\n  cached → {out}  ({out.stat().st_size/1e6:.1f} MB)  "
          f"pos {pos_err.mean()*100:.2f} cm  rot {rot_err.mean():.2f}°")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/checkpoints"))
    ap.add_argument("--dataset-path", default=os.environ.get("DATASET_DIR", "/data/sonata_all"))
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUTS_DIR", "/demo_out"))
    ap.add_argument("--traj-ids", nargs="+", type=int, default=[1916])
    ap.add_argument("--mode", default="open_loop", choices=["open_loop", "rollout", "both"])
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=9999)
    ap.add_argument("--cam-width", type=int, default=384, help="cached camera frame width")
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
            cache_episode(policy, loader, tid, out_dir, m,
                          args.action_horizon, args.max_steps, args.cam_width)
    print(f"\nAll rollout caches → {out_dir/'cache'}")


if __name__ == "__main__":
    main()
