#!/usr/bin/env python3
import argparse
import json
import os

import cv2
import mujoco
import numpy as np

from fastumi_gui import clamp_to_ctrl, quat_inv, quat_mul, quat_to_axis_angle, quat_to_rot


FR3_JOINTS = [
    ("fr3_joint1", -0.1258342),
    ("fr3_joint2", 0.72059298),
    ("fr3_joint3", 0.15445824),
    ("fr3_joint4", -1.61090684),
    ("fr3_joint5", -0.12259634),
    ("fr3_joint6", 2.35071445),
    ("fr3_joint7", 1.62955177),
]


def build_context(mjcf_path):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "live_fingers_actuator")
    if act_id < 0:
        raise RuntimeError("actuator 'live_fingers_actuator' not found in MJCF")

    ctx = {
        "model": model,
        "data": data,
        "renderer": mujoco.Renderer(model, height=480, width=640),
        "act": act_id,
        "steps": 12,
        "base_qadr": None,
        "ref_body_id": None,
        "fr3_qpos_adrs": [],
        "fr3_dof_ids": [],
        "fr3_act_ids": [],
        "fr3_ranges": [],
        "default_fr3_qpos": [val for _, val in FR3_JOINTS],
        "ik_iters": 6,
        "ik_damping": 0.1,
        "ik_step": 0.15,
        "ik_posture_gain": 0.08,
    }

    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "live_base_mount_free")
    if base_jid >= 0:
        ctx["base_qadr"] = model.jnt_qposadr[base_jid]

    ref_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
    if ref_bid >= 0:
        ctx["ref_body_id"] = ref_bid

    for idx, (name, val) in enumerate(FR3_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = val
        ctx["fr3_qpos_adrs"].append(adr)
        ctx["fr3_dof_ids"].append(model.jnt_dofadr[jid])
        ctx["fr3_ranges"].append(tuple(model.jnt_range[jid]))
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            data.ctrl[aid] = FR3_JOINTS[idx][1]
            ctx["fr3_act_ids"].append(aid)

    cam = mujoco.MjvCamera()
    cam.azimuth = 90.0
    cam.elevation = -25.0
    cam.distance = 2.5
    cam.lookat[:] = np.array([0.0, 0.0, 0.8])
    ctx["camera"] = cam
    mujoco.mj_forward(model, data)
    return ctx


def follow_gripper(ctx, target_pos, target_quat):
    ref_body_id = ctx["ref_body_id"]
    if ref_body_id is None:
        return
    model = ctx["model"]
    data = ctx["data"]
    dof_ids = ctx["fr3_dof_ids"]
    qpos_adrs = ctx["fr3_qpos_adrs"]
    act_ids = ctx["fr3_act_ids"]
    ranges = ctx["fr3_ranges"]
    if not dof_ids or not qpos_adrs:
        return

    q_ref = np.array(ctx["default_fr3_qpos"], dtype=float)
    for _ in range(ctx["ik_iters"]):
        mujoco.mj_forward(model, data)
        cur_quat = np.array(data.xquat[ref_body_id])
        cur_pos = np.array(data.xpos[ref_body_id], dtype=float)
        pos_err = np.array(target_pos, dtype=float) - cur_pos
        rot_err = quat_mul(target_quat, quat_inv(cur_quat))
        ang_err = quat_to_axis_angle(rot_err)
        err = np.concatenate([pos_err, ang_err])
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ref_body_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        JT = J.T
        A = J @ JT + (ctx["ik_damping"] * ctx["ik_damping"]) * np.eye(6)
        J_pinv = JT @ np.linalg.solve(A, np.eye(6))
        dq = J_pinv @ err
        q_cur = np.array([data.qpos[qadr] for qadr in qpos_adrs], dtype=float)
        dq_posture = ctx["ik_posture_gain"] * (q_ref - q_cur)
        null_proj = np.eye(len(qpos_adrs)) - (J_pinv @ J)
        dq = dq + null_proj @ dq_posture
        scale = max(1.0, np.max(np.abs(dq)) / ctx["ik_step"])
        dq = dq / scale
        for i, qadr in enumerate(qpos_adrs):
            data.qpos[qadr] += dq[i]
            lo, hi = ranges[i]
            data.qpos[qadr] = float(max(lo, min(hi, data.qpos[qadr])))

    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = data.qpos[qpos_adrs[i]]


def open_video_writer(path, size, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        return writer, path
    alt_path = os.path.splitext(path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(alt_path, fourcc, fps, size)
    if writer.isOpened():
        return writer, alt_path
    raise RuntimeError("failed to open video writer for {}".format(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record_dir")
    parser.add_argument(
        "--mjcf",
        default=os.path.join(os.path.dirname(__file__), "assets", "fr3_robotiq_scene.xml"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite joints.json and video.mp4 instead of writing *_recomputed outputs.",
    )
    args = parser.parse_args()

    record_dir = os.path.abspath(args.record_dir)
    cartesian_path = os.path.join(record_dir, "cartesian.json")
    undistort_path = os.path.join(record_dir, "video_undistort.mp4")
    if not os.path.isfile(cartesian_path):
        raise FileNotFoundError(cartesian_path)
    if not os.path.isfile(undistort_path):
        raise FileNotFoundError(undistort_path)

    with open(cartesian_path, "r", encoding="utf-8") as f:
        cartesian = json.load(f)
    frames = cartesian.get("frames", [])

    ctx = build_context(args.mjcf)
    cap = cv2.VideoCapture(undistort_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    left_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if left_w <= 0 or left_h <= 0:
        raise RuntimeError("invalid undistort video size")
    video_target = "video.mp4" if args.overwrite else "video_recomputed.mp4"
    writer, out_video_path = open_video_writer(
        os.path.join(record_dir, video_target),
        (left_w + 640, max(left_h, 480)),
        fps,
    )

    out_joint_frames = []
    frame_total = min(len(frames), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(frames))
    for idx in range(frame_total):
        ok, left_frame = cap.read()
        if not ok:
            break
        frame = frames[idx]
        pose = frame.get("目标位姿")
        clamp = frame.get("夹爪开合")
        target_pos = None
        target_quat = None
        if pose is not None:
            target_pos = tuple(pose["p"])
            target_quat = tuple(pose["q"])
            if ctx["base_qadr"] is not None:
                qadr = ctx["base_qadr"]
                ctx["data"].qpos[qadr : qadr + 3] = target_pos
                ctx["data"].qpos[qadr + 3 : qadr + 7] = target_quat
            follow_gripper(ctx, target_pos, target_quat)
        if clamp is not None:
            ctx["data"].ctrl[ctx["act"]] = clamp_to_ctrl(clamp)
        for _ in range(ctx["steps"]):
            mujoco.mj_step(ctx["model"], ctx["data"])

        out_joint_frames.append(
            {
                "时间戳": frame.get("时间戳"),
                "关节": [float(ctx["data"].qpos[adr]) for adr in ctx["fr3_qpos_adrs"]],
                "夹爪开合": clamp,
            }
        )

        ctx["renderer"].update_scene(ctx["data"], camera=ctx["camera"])
        mj_rgb = ctx["renderer"].render()
        mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)
        if left_frame.shape[0] != mj_bgr.shape[0]:
            left_frame = cv2.resize(left_frame, (left_frame.shape[1], mj_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        combo = np.concatenate([left_frame, mj_bgr], axis=1)
        writer.write(combo)

    cap.release()
    writer.release()

    out_joint_payload = {
        "start_ts": cartesian.get("start_ts"),
        "frames": out_joint_frames,
    }
    out_joint_name = "joints.json" if args.overwrite else "joints_recomputed.json"
    out_joint_path = os.path.join(record_dir, out_joint_name)
    with open(out_joint_path, "w", encoding="utf-8") as f:
        json.dump(out_joint_payload, f, ensure_ascii=False, indent=2)

    print("wrote", out_joint_path)
    print("wrote", out_video_path)
    print("frames", len(out_joint_frames))


if __name__ == "__main__":
    main()
