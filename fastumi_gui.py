#!/usr/bin/env python3
import argparse
import json
import math
import os
import signal
import shutil
import subprocess
import re
import struct
import tempfile
import wave
import sys
import threading
import time

import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else (
    os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
)
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")
_OFFICIAL_SCRIPT = os.path.abspath(os.path.join(_BASE_DIR, "start_official.sh"))


def _resolve_config_path(filename):
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(_CONFIG_DIR, filename)
    legacy_path = os.path.join(_BASE_DIR, filename)
    if os.path.isfile(config_path):
        return config_path
    if os.path.isfile(legacy_path):
        try:
            shutil.move(legacy_path, config_path)
            return config_path
        except Exception:
            return legacy_path
    return config_path
def _resolve_qt_plugin_path():
    candidates = []
    for base in sys.path:
        candidates.append(os.path.join(base, "PyQt5", "Qt5", "plugins"))
        candidates.append(os.path.join(base, "PyQt5", "Qt", "plugins"))
        candidates.append(os.path.join(base, "PySide2", "Qt5", "plugins"))
        candidates.append(os.path.join(base, "PySide2", "Qt", "plugins"))
    candidates.append("/usr/lib/x86_64-linux-gnu/qt5/plugins")
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


_QT_PLUGIN_PATH = _resolve_qt_plugin_path()
if _QT_PLUGIN_PATH:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _QT_PLUGIN_PATH
    os.environ["QT_PLUGIN_PATH"] = _QT_PLUGIN_PATH
    os.environ["QT_QPA_PLATFORM"] = "xcb"

if sys.version_info[0] < 3:
    raise SystemExit("Use python3 to run this script.")

cv2 = None
rospy = None
CvBridge = None
Image = None
rostopic = None
AnyMsg = None
Clamp = None
PoseStampedConfidence = None
mujoco = None

from ros_env_helper import ensure_ros_env, start_ros_backend


DEFAULT_SERIAL = "250801DR48FP25002287"
ROT_QUAT_MATRIX = [
    [0.71918939, 0.06225367, -0.69198852],
    [0.02706709, 0.98700960, 0.15848403],
    [0.69433250, -0.14789961, 0.70456402],
]
LIVE_GRIPPER_OFFSET_LOCAL = (0.0, 0.0, 0.1)
LIVE_GRIPPER_ROT_EULER = (0.0, 0.0, 0.0)
GRIPPER_AXIS_OFFSET = 0.0927
TARGET_GRIPPER_POSE = {
    "p": (0.690702, 0.000908656, 0.105101),
    "q": (0.924869, -0.380058, 0.0122284, 0.0048999),
}


def _start_official_backend():
    script = _OFFICIAL_SCRIPT
    if not os.path.isfile(script):
        return False
    try:
        subprocess.run(["bash", script, "--start-only"], check=False)
        return _xv_sdk_running()
    except Exception:
        return False


def _xv_sdk_running():
    try:
        if subprocess.run(
            ["pgrep", "-f", "roslaunch .*xv_sdk.launch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return True
        if subprocess.run(
            ["pgrep", "-f", "xv_sdk"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return True
    except Exception:
        return False
    return False


def _list_image_topics():
    try:
        published = rospy.get_published_topics()
    except Exception:
        return []
    return [t for t, ttype in published if ttype.endswith("sensor_msgs/Image")]


def _preferred_candidates(serial):
    candidates = []
    if serial:
        candidates.append("/xv_sdk/{}/color_camera/image".format(serial))
    candidates += [
        "/xv_sdk/color_camera/image",
        "/color_camera/image",
        "color_camera/image",
    ]
    return candidates


def resolve_topic(explicit_topic):
    if explicit_topic:
        return explicit_topic
    serial = os.environ.get("XV_DEVICE_SERIAL", "").strip() or DEFAULT_SERIAL
    candidates = _preferred_candidates(serial)
    published_topics = _list_image_topics()
    for t in candidates:
        if t in published_topics:
            return t
    fallback = [
        t for t in published_topics if "color" in t or "rgb" in t or "camera" in t
    ]
    if fallback:
        return fallback[0]
    return candidates[0]


def resolve_named_topic(topic_candidates):
    published_all = {t for t, _ in rospy.get_published_topics()}
    for t in topic_candidates:
        if t in published_all:
            return t
    return topic_candidates[0]


def resolve_msg_class(topic):
    try:
        msg_class, real_topic, _ = rostopic.get_topic_class(
            topic, blocking=False
        )
        return msg_class, real_topic or topic
    except Exception:
        return None, topic


def resolve_slam_topic(serial):
    visual_candidates = [
        "/xv_sdk/{}/slam/visual_pose".format(serial),
        "/xv_sdk/slam/visual_pose",
        "/slam/visual_pose",
    ]
    pose_candidates = [
        "/xv_sdk/{}/slam/pose".format(serial),
        "/xv_sdk/slam/pose",
        "/slam/pose",
    ]
    visual_topic = resolve_named_topic(visual_candidates)
    visual_class, visual_real = resolve_msg_class(visual_topic)
    if visual_class is not None:
        return visual_real, visual_class
    pose_topic = resolve_named_topic(pose_candidates)
    pose_class, pose_real = resolve_msg_class(pose_topic)
    return pose_real, pose_class


def format_pose(msg):
    pose = msg.poseMsg.pose
    return "slam conf={:.3f} pos=({:.3f},{:.3f},{:.3f}) quat=({:.3f},{:.3f},{:.3f},{:.3f})".format(
        msg.confidence,
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inv(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    inv = 1.0 / n
    return (w * inv, -x * inv, -y * inv, -z * inv)


def quat_to_euler_deg(q):
    w, x, y, z = q
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def euler_deg_to_quat(roll, pitch, yaw):
    cr = math.cos(math.radians(roll) * 0.5)
    sr = math.sin(math.radians(roll) * 0.5)
    cp = math.cos(math.radians(pitch) * 0.5)
    sp = math.sin(math.radians(pitch) * 0.5)
    cy = math.cos(math.radians(yaw) * 0.5)
    sy = math.sin(math.radians(yaw) * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def quat_to_rot(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def quat_to_axis_angle(q):
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-6:
        return np.zeros(3)
    axis = np.array([x, y, z], dtype=float) / s
    return axis * angle


def rot_to_quat(R):
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    return (w / n, x / n, y / n, z / n)


def apply_rot_matrix_to_quat(q, R):
    Rq = quat_to_rot(q)
    Rn = [
        [
            R[0][0] * Rq[0][0] + R[0][1] * Rq[1][0] + R[0][2] * Rq[2][0],
            R[0][0] * Rq[0][1] + R[0][1] * Rq[1][1] + R[0][2] * Rq[2][1],
            R[0][0] * Rq[0][2] + R[0][1] * Rq[1][2] + R[0][2] * Rq[2][2],
        ],
        [
            R[1][0] * Rq[0][0] + R[1][1] * Rq[1][0] + R[1][2] * Rq[2][0],
            R[1][0] * Rq[0][1] + R[1][1] * Rq[1][1] + R[1][2] * Rq[2][1],
            R[1][0] * Rq[0][2] + R[1][1] * Rq[1][2] + R[1][2] * Rq[2][2],
        ],
        [
            R[2][0] * Rq[0][0] + R[2][1] * Rq[1][0] + R[2][2] * Rq[2][0],
            R[2][0] * Rq[0][1] + R[2][1] * Rq[1][1] + R[2][2] * Rq[2][1],
            R[2][0] * Rq[0][2] + R[2][1] * Rq[1][2] + R[2][2] * Rq[2][2],
        ],
    ]
    return rot_to_quat(Rn)


def apply_offset(base_pos, base_quat, t_offset, rpy_deg):
    r_quat = euler_deg_to_quat(*rpy_deg)
    R = quat_to_rot(r_quat)
    x, y, z = base_pos
    tx = R[0][0] * x + R[0][1] * y + R[0][2] * z + t_offset[0]
    ty = R[1][0] * x + R[1][1] * y + R[1][2] * z + t_offset[1]
    tz = R[2][0] * x + R[2][1] * y + R[2][2] * z + t_offset[2]
    return (tx, ty, tz), quat_mul(r_quat, base_quat)


def clamp_to_ctrl(value):
    v = max(0.0, float(value))
    if v <= 88.0:
        vmax = 88.0
    elif v <= 100.0:
        vmax = 100.0
    else:
        vmax = 255.0
    v = min(v, vmax)
    t = 1.0 - (v / vmax)
    return t * 255.0


def _fmt_vec(vals, prec):
    parts = []
    for v in vals:
        try:
            fv = float(v)
            parts.append("{:.{p}f}".format(fv, p=prec))
        except (TypeError, ValueError):
            parts.append(str(v))
    return "(" + ",".join(parts) + ")"


def _build_status_line(
    latest_slam,
    latest_clamp,
    latest_tf,
    offset,
    arm_pose=None,
    ee_pose=None,
    live_pose=None,
    tf_pose=None,
):
    if latest_slam["conf"] is not None:
        p = _fmt_vec(latest_slam["pos"], 3)
        q = _fmt_vec(latest_slam["quat"], 3)
        slam_part = "slam c={:.2f} p={} q={}".format(latest_slam["conf"], p, q)
    else:
        slam_part = "slam: n/a"
    clamp_part = "clamp={:.2f}".format(latest_clamp["value"])
    parts = [slam_part, clamp_part]
    if tf_pose is not None:
        adj_pos, adj_quat = tf_pose
        p = _fmt_vec(adj_pos, 3)
        q = _fmt_vec(adj_quat, 3)
        er, ep, ey = quat_to_euler_deg(adj_quat)
        e = _fmt_vec((er, ep, ey), 1)
        tf_part = "对齐tf p={} e={} q={}".format(p, e, q)
        parts.append(tf_part)
    elif latest_tf["pos"] is not None and latest_tf["quat"] is not None:
        adj_pos, adj_quat = apply_offset(
            latest_tf["pos"], latest_tf["quat"], offset["t"], offset["rpy"]
        )
        p = _fmt_vec(adj_pos, 3)
        q = _fmt_vec(adj_quat, 3)
        er, ep, ey = quat_to_euler_deg(adj_quat)
        e = _fmt_vec((er, ep, ey), 1)
        tf_part = "对齐tf p={} e={} q={}".format(p, e, q)
        parts.append(tf_part)
    if arm_pose is not None:
        arm_pos, arm_quat = arm_pose
        p = _fmt_vec(arm_pos, 3)
        q = _fmt_vec(arm_quat, 3)
        arm_part = "机械臂夹爪 p={} q={}".format(p, q)
        parts.append(arm_part)
    if ee_pose is not None:
        ee_pos, ee_quat = ee_pose
        p = _fmt_vec(ee_pos, 3)
        q = _fmt_vec(ee_quat, 3)
        ee_part = "机械臂末端 p={} q={}".format(p, q)
        parts.append(ee_part)
    if live_pose is not None:
        live_pos, live_quat = live_pose
        p = _fmt_vec(live_pos, 3)
        q = _fmt_vec(live_quat, 3)
        live_part = "独立夹爪(对齐) p={} q={}".format(p, q)
        parts.append(live_part)
    return " | ".join(parts)


def _import_qt():
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets
    except Exception:
        from PySide2 import QtCore, QtGui, QtWidgets
        return QtCore, QtGui, QtWidgets


def _list_realsense_serials():
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        return [], str(exc)
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        return serials, None
    except Exception as exc:
        return [], str(exc)


class RealSenseReader:
    def __init__(self, serial, shared, key, lock):
        self.serial = serial
        self.shared = shared
        self.key = key
        self.lock = lock
        self._stop_event = threading.Event()
        self._thread = None
        self._pipeline = None

    def start(self):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frames = None
                if hasattr(self._pipeline, "poll_for_frames"):
                    frames = self._pipeline.poll_for_frames()
                    if not frames:
                        time.sleep(0.005)
                        continue
                else:
                    frames = self._pipeline.wait_for_frames()
                color_frame = frames.get_color_frame() if frames else None
                if not color_frame:
                    continue
                img = np.asanyarray(color_frame.get_data())
                with self.lock:
                    self.shared["latest_rs"][self.key] = img
            except Exception:
                time.sleep(0.05)
        self._stop_pipeline()

    def _stop_pipeline(self):
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        except Exception:
            pass
        self._pipeline = None

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop_pipeline()


def build_gui_classes(QtCore, QtGui, QtWidgets):
    class MujocoView(QtWidgets.QLabel):
        def __init__(self, parent):
            super().__init__(parent)
            self._last_pos = None
            self.setMouseTracking(True)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)

        def mousePressEvent(self, event):
            self._last_pos = event.pos()
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):
            self._last_pos = None
            super().mouseReleaseEvent(event)

        def mouseMoveEvent(self, event):
            if self._last_pos is None:
                return
            dx = event.x() - self._last_pos.x()
            dy = event.y() - self._last_pos.y()
            if event.buttons() & QtCore.Qt.LeftButton:
                handler = self.window()
                if hasattr(handler, "adjust_camera_orbit"):
                    handler.adjust_camera_orbit(dx, dy)
            elif event.buttons() & QtCore.Qt.RightButton:
                handler = self.window()
                if hasattr(handler, "adjust_camera_pan"):
                    handler.adjust_camera_pan(dx, dy)
            self._last_pos = event.pos()
            super().mouseMoveEvent(event)

        def wheelEvent(self, event):
            handler = self.window()
            if hasattr(handler, "adjust_camera_zoom"):
                handler.adjust_camera_zoom(event.angleDelta().y())
            super().wheelEvent(event)

    class SliderRow(QtWidgets.QWidget):
        def __init__(self, name, minv, maxv, init, suffix):
            super().__init__()
            self.label = QtWidgets.QLabel(name)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(minv, maxv)
            self.slider.setValue(init)
            self.value = QtWidgets.QLabel()
            layout = QtWidgets.QHBoxLayout(self)
            layout.addWidget(self.label)
            layout.addWidget(self.slider, 1)
            layout.addWidget(self.value)
            self._suffix = suffix
            self.slider.valueChanged.connect(self._update_value)
            self._update_value(self.slider.value())

        def _update_value(self, val):
            self.value.setText("{}{}".format(val, self._suffix))

    class FloatSliderRow(QtWidgets.QWidget):
        def __init__(self, name, minv, maxv, init, scale, fmt, editable=False):
            super().__init__()
            self.label = QtWidgets.QLabel(name)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(minv, maxv)
            self.slider.setValue(init)
            self.value = QtWidgets.QLabel()
            self.editor = QtWidgets.QLineEdit()
            self.editor.setFixedWidth(60)
            self.editor.hide()
            layout = QtWidgets.QHBoxLayout(self)
            layout.addWidget(self.label)
            layout.addWidget(self.slider, 1)
            layout.addWidget(self.value)
            layout.addWidget(self.editor)
            self._scale = scale
            self._fmt = fmt
            self._editable = editable
            self.slider.valueChanged.connect(self._update_value)
            self.editor.editingFinished.connect(self._apply_editor)
            self._update_value(self.slider.value())

        def _update_value(self, val):
            text = self._fmt.format(val * self._scale)
            self.value.setText(text)
            if self._editable:
                self.editor.setText(text)
                self.editor.show()

        def _apply_editor(self):
            if not self._editable:
                return
            raw = self.editor.text().strip()
            if not raw:
                return
            try:
                val = float(raw)
            except ValueError:
                return
            slider_val = int(round(val / self._scale))
            slider_val = max(self.slider.minimum(), min(self.slider.maximum(), slider_val))
            self.slider.setValue(slider_val)

    class FastUMIGui(QtWidgets.QMainWindow):
        def __init__(self, args, shared):
            super().__init__()
            self.args = args
            self.shared = shared
            self.calib = args.calib
            self.undistort_enabled = bool(self.calib) and not args.no_undistort
            self.undistort_balance = float(getattr(args, "undistort_balance", 0.0))
            self.undistort_fov_scale = float(getattr(args, "undistort_fov_scale", 0.2))
            self.undistort_maps = None
            self.undistort_size = None
            self.crop_center_y = 0.5
            self.recording = False
            self.record_start_ts = None
            self.record_dir = None
            self.record_writer = None
            self.record_raw_writer = None
            self.record_undistort_writer = None
            self.record_rs1_writer = None
            self.record_rs2_writer = None
            self.record_frames = []
            self.joint_frames = []
            self.cartesian_frames = []
            self.ros_pose_frames = []
            self.extra_sensor_frames = {
                "rgbd": [],
                "tof": [],
                "realsense1_depth": [],
                "realsense2_depth": [],
            }
            self.extra_sensor_last_ids = {
                "rgbd": None,
                "tof": None,
                "realsense1_depth": None,
                "realsense2_depth": None,
            }
            self.record_frame_size = None
            self.record_raw_size = None
            self.record_undistort_size = None
            self.record_rs1_size = None
            self.record_rs2_size = None
            self.last_record_dir = None
            self.offset_path = _resolve_config_path("pose_offset.json")
            self.display_offset_path = _resolve_config_path("gripper_display_offset.json")
            self.task_state_path = _resolve_config_path("last_task.json")
            self.offset = {"t": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)}
            self.follow_enabled = False
            self.default_fr3_qpos = None
            self.live_offset_local = np.array(LIVE_GRIPPER_OFFSET_LOCAL, dtype=float)
            self.live_rot_offset = euler_deg_to_quat(*LIVE_GRIPPER_ROT_EULER)
            self.gripper_axis_offset = np.array([0.0, 0.0, GRIPPER_AXIS_OFFSET], dtype=float)
            self.gripper_rot_offset = euler_deg_to_quat(0.0, 0.0, 0.0)
            self.last_aligned_pose = None
            self.arm_visible = True
            self.rs_serials = list(self.shared.get("rs_serials", []))
            self._sound_start = None
            self._sound_stop = None
            self._sound_dir = None
            self.task_name_edit = None
            self._mouse_remote_enabled = True
            self._mouse_press_button = None
            self._mouse_press_pos = None
            self._mouse_press_time = None
            self._init_ui()
            self._load_offset()
            self._load_display_offset()
            self._load_task_name()
            self._refresh_record_indicator()
            self._init_mujoco()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._update_ui)
            self.timer.start(33)

        def _init_ui(self):
            self.setWindowTitle("FastUMI Integrated GUI")
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            main_layout = QtWidgets.QVBoxLayout(central)
            main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            main_layout.addWidget(main_splitter)
            top_panel = QtWidgets.QWidget()
            top_layout = QtWidgets.QVBoxLayout(top_panel)

            self.view_panel = QtWidgets.QWidget()
            view_layout = QtWidgets.QGridLayout(self.view_panel)
            self.raw_label = QtWidgets.QLabel("Raw image...")
            self.raw_label.setAlignment(QtCore.Qt.AlignCenter)
            self.raw_label.setMinimumSize(320, 200)
            self.raw_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.undistort_label = QtWidgets.QLabel("Undistorted image...")
            self.undistort_label.setAlignment(QtCore.Qt.AlignCenter)
            self.undistort_label.setMinimumSize(320, 200)
            self.undistort_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.mujoco_label = MujocoView(self)
            self.mujoco_label.setAlignment(QtCore.Qt.AlignCenter)
            self.mujoco_label.setMinimumSize(320, 240)
            self.mujoco_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.rs1_label = QtWidgets.QLabel("RealSense 1...")
            self.rs1_label.setAlignment(QtCore.Qt.AlignCenter)
            self.rs1_label.setMinimumSize(320, 200)
            self.rs1_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.rs2_label = QtWidgets.QLabel("RealSense 2...")
            self.rs2_label.setAlignment(QtCore.Qt.AlignCenter)
            self.rs2_label.setMinimumSize(320, 200)
            self.rs2_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.record_state_label = QtWidgets.QLabel("REC: OFF")
            self.record_state_label.setAlignment(QtCore.Qt.AlignCenter)
            self.record_state_label.setMinimumSize(220, 120)
            self.record_state_label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            view_layout.addWidget(self.raw_label, 0, 0)
            view_layout.addWidget(self.undistort_label, 0, 1)
            view_layout.addWidget(self.mujoco_label, 0, 2)
            view_layout.addWidget(self.rs1_label, 1, 0)
            view_layout.addWidget(self.rs2_label, 1, 1)
            view_layout.addWidget(self.record_state_label, 1, 2)
            view_layout.setColumnStretch(0, 1)
            view_layout.setColumnStretch(1, 1)
            view_layout.setColumnStretch(2, 1)
            view_layout.setRowStretch(0, 1)
            view_layout.setRowStretch(1, 1)
            top_layout.addWidget(self.view_panel, 3)

            self.control_panel = QtWidgets.QWidget()
            control_layout = QtWidgets.QGridLayout(self.control_panel)
            self.tx_row = SliderRow("tx_mm", 0, 10000, 5000, "")
            self.ty_row = SliderRow("ty_mm", 0, 10000, 5000, "")
            self.tz_row = SliderRow("tz_mm", 0, 10000, 5000, "")
            self.rx_row = SliderRow("rx_d10", 0, 3600, 1800, "")
            self.ry_row = SliderRow("ry_d10", 0, 3600, 1800, "")
            self.rz_row = SliderRow("rz_d10", 0, 3600, 1800, "")
            self.balance_row = FloatSliderRow(
                "undistort_balance", 0, 100, int(self.undistort_balance * 100), 0.01, "{:.2f}"
            )
            self.fov_row = FloatSliderRow(
                "undistort_fov", 0, 120, int(self.undistort_fov_scale * 100), 0.01, "{:.2f}"
            )
            self.crop_center_row = FloatSliderRow(
                "crop_center_y", 0, 100, int(self.crop_center_y * 100), 0.01, "{:.2f}"
            )
            self.grip_rx_row = FloatSliderRow("grip_rx_deg", -1800, 1800, 0, 0.1, "{:.1f}")
            self.grip_ry_row = FloatSliderRow("grip_ry_deg", -1800, 1800, 0, 0.1, "{:.1f}")
            self.grip_rz_row = FloatSliderRow("grip_rz_deg", -1800, 1800, 0, 0.1, "{:.1f}")
            control_layout.addWidget(self.tx_row, 0, 0)
            control_layout.addWidget(self.ty_row, 0, 1)
            control_layout.addWidget(self.tz_row, 0, 2)
            control_layout.addWidget(self.rx_row, 1, 0)
            control_layout.addWidget(self.ry_row, 1, 1)
            control_layout.addWidget(self.rz_row, 1, 2)
            control_layout.addWidget(self.balance_row, 2, 0)
            control_layout.addWidget(self.fov_row, 2, 1)
            control_layout.addWidget(self.crop_center_row, 2, 2)
            control_layout.addWidget(self.grip_rx_row, 3, 0)
            control_layout.addWidget(self.grip_ry_row, 3, 1)
            control_layout.addWidget(self.grip_rz_row, 3, 2)
            top_layout.addWidget(self.control_panel)

            button_layout = QtWidgets.QHBoxLayout()
            self.record_start_btn = QtWidgets.QPushButton("Start Recording")
            self.record_stop_btn = QtWidgets.QPushButton("Stop Recording")
            self.save_offset_btn = QtWidgets.QPushButton("Save Offset")
            self.align_btn = QtWidgets.QPushButton("Align Gripper")
            self.auto_calib_btn = QtWidgets.QPushButton("Auto Calib Display")
            self.toggle_arm_btn = QtWidgets.QPushButton("Hide Arm")
            self.follow_btn = QtWidgets.QPushButton("Follow Gripper")
            self.follow_btn.setCheckable(True)
            self.reset_btn = QtWidgets.QPushButton("Reset Arm")
            self.toggle_sliders_btn = QtWidgets.QPushButton("Hide Sliders")
            self.record_stop_btn.setEnabled(False)
            task_label = QtWidgets.QLabel("Task:")
            self.task_name_edit = QtWidgets.QLineEdit()
            self.task_name_edit.setPlaceholderText("task name")
            self.task_name_edit.setFixedWidth(180)
            self.open_record_btn = QtWidgets.QPushButton("Open Recordings")
            self._set_record_indicator(False, play_sound=False)
            button_layout.addWidget(self.record_start_btn)
            button_layout.addWidget(self.record_stop_btn)
            button_layout.addWidget(task_label)
            button_layout.addWidget(self.task_name_edit)
            button_layout.addWidget(self.open_record_btn)
            button_layout.addWidget(self.save_offset_btn)
            button_layout.addWidget(self.align_btn)
            button_layout.addWidget(self.auto_calib_btn)
            button_layout.addWidget(self.toggle_arm_btn)
            button_layout.addWidget(self.follow_btn)
            button_layout.addWidget(self.reset_btn)
            button_layout.addWidget(self.toggle_sliders_btn)
            top_layout.addLayout(button_layout)

            self.status_text = QtWidgets.QTextEdit()
            self.status_text.setReadOnly(True)
            self.status_text.setMinimumHeight(80)
            main_splitter.addWidget(top_panel)
            main_splitter.addWidget(self.status_text)
            main_splitter.setStretchFactor(0, 4)
            main_splitter.setStretchFactor(1, 1)
            main_splitter.setCollapsible(0, False)
            QtCore.QTimer.singleShot(0, lambda: main_splitter.setSizes([800, 200]))

            self.record_start_btn.clicked.connect(self._start_recording)
            self.record_stop_btn.clicked.connect(self._stop_recording)
            self.save_offset_btn.clicked.connect(self._save_offset)
            self.align_btn.clicked.connect(self._align_gripper)
            self.auto_calib_btn.clicked.connect(self._auto_calib_display)
            self.toggle_arm_btn.clicked.connect(self._toggle_arm_visibility)
            self.follow_btn.toggled.connect(self._toggle_follow)
            self.reset_btn.clicked.connect(self._reset_arm)
            self.open_record_btn.clicked.connect(self._open_record_folder)
            self.toggle_sliders_btn.clicked.connect(self._toggle_sliders)
            self.task_name_edit.textChanged.connect(self._refresh_record_indicator)
            self.control_panel.setVisible(False)
            self.toggle_sliders_btn.setText("Show Sliders")
            self.record_toggle_shortcut = QtWidgets.QShortcut(
                QtGui.QKeySequence(QtCore.Qt.Key_Space), self
            )
            self.record_toggle_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            self.record_toggle_shortcut.setAutoRepeat(False)
            self.record_toggle_shortcut.activated.connect(self._toggle_recording)
            self._init_sound()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.installEventFilter(self)

        def eventFilter(self, obj, event):
            if not self._mouse_remote_enabled:
                return super().eventFilter(obj, event)
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
                    self._mouse_press_button = event.button()
                    self._mouse_press_pos = event.globalPos()
                    self._mouse_press_time = time.time()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if (
                    self._mouse_press_button in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton)
                    and event.button() == self._mouse_press_button
                    and self._mouse_press_pos is not None
                ):
                    delta = event.globalPos() - self._mouse_press_pos
                    long_press_sec = 2.0
                    held = 0.0
                    if self._mouse_press_time is not None:
                        held = time.time() - self._mouse_press_time
                    if delta.manhattanLength() <= 6:
                        if self._mouse_press_button == QtCore.Qt.RightButton:
                            if held >= long_press_sec:
                                self._trigger_reset_align_follow()
                            else:
                                self._toggle_recording()
                            self._mouse_press_button = None
                            self._mouse_press_pos = None
                            self._mouse_press_time = None
                            return True
                        if self._mouse_press_button == QtCore.Qt.MiddleButton:
                            self._delete_last_recording()
                            self._mouse_press_button = None
                            self._mouse_press_pos = None
                            self._mouse_press_time = None
                            return True
                self._mouse_press_button = None
                self._mouse_press_pos = None
                self._mouse_press_time = None
            return super().eventFilter(obj, event)

        def _init_sound(self):
            qt_multimedia = None
            try:
                from PyQt5 import QtMultimedia as _QtMultimedia
                qt_multimedia = _QtMultimedia
            except Exception:
                try:
                    from PySide2 import QtMultimedia as _QtMultimedia
                    qt_multimedia = _QtMultimedia
                except Exception:
                    qt_multimedia = None
            if qt_multimedia is None:
                return
            try:
                self._sound_dir = tempfile.mkdtemp(prefix="fastumi_sound_")
                start_path = os.path.join(self._sound_dir, "rec_start.wav")
                stop_path = os.path.join(self._sound_dir, "rec_stop.wav")
                self._write_tone(start_path, 880.0, 0.09)
                self._write_tone(stop_path, 520.0, 0.09)
                self._sound_start = qt_multimedia.QSoundEffect()
                self._sound_start.setSource(QtCore.QUrl.fromLocalFile(start_path))
                self._sound_start.setVolume(0.5)
                self._sound_stop = qt_multimedia.QSoundEffect()
                self._sound_stop.setSource(QtCore.QUrl.fromLocalFile(stop_path))
                self._sound_stop.setVolume(0.5)
            except Exception:
                self._sound_start = None
                self._sound_stop = None

        def _write_tone(self, path, freq_hz, duration_s, rate=44100):
            frames = int(rate * duration_s)
            amp = 0.35
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                for i in range(frames):
                    sample = int(amp * 32767.0 * math.sin(2.0 * math.pi * freq_hz * i / rate))
                    wf.writeframes(struct.pack("<h", sample))

        def _play_sound(self, kind):
            if kind == "start" and self._sound_start is not None:
                self._sound_start.stop()
                self._sound_start.play()
                return
            if kind == "stop" and self._sound_stop is not None:
                self._sound_stop.stop()
                self._sound_stop.play()
                return
            if kind == "start":
                QtWidgets.QApplication.beep()
            else:
                QtWidgets.QApplication.beep()
                QtCore.QTimer.singleShot(120, QtWidgets.QApplication.beep)

        def _current_task_dir(self):
            base_dir = os.path.abspath(self.args.record_dir)
            raw_task = self.task_name_edit.text() if self.task_name_edit is not None else ""
            task_name = self._sanitize_task_name(raw_task)
            return os.path.join(base_dir, task_name)

        def _max_record_index(self, task_dir):
            max_idx = 0
            try:
                for entry in os.listdir(task_dir):
                    m = re.match(r"^recording_(\d+)$", entry)
                    if m:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
            except Exception:
                pass
            return max_idx

        def _record_indicator_index(self, recording):
            if recording and self.record_dir:
                m = re.search(r"recording_(\d+)$", self.record_dir)
                if m:
                    return int(m.group(1))
            return self._max_record_index(self._current_task_dir())

        def _refresh_record_indicator(self):
            self._set_record_indicator(self.recording, play_sound=False)

        def _set_record_indicator(self, recording, play_sound=True):
            idx = self._record_indicator_index(recording)
            text = (
                "<div style='text-align:center;'>"
                "REC: {}<br><span style='font-size:56px; font-weight:900;'>{:03d}</span>"
                "</div>"
            ).format("ON" if recording else "OFF", idx)
            if recording:
                if hasattr(self, "record_state_label"):
                    self.record_state_label.setText(text)
                    self.record_state_label.setStyleSheet(
                        "color: #ffffff; background-color: #c62828; "
                        "font-weight: 900; font-size: 28px; "
                        "padding: 6px 14px; border: 3px solid #7f0000; border-radius: 8px;"
                    )
                if play_sound:
                    self._play_sound("start")
            else:
                if hasattr(self, "record_state_label"):
                    self.record_state_label.setText(text)
                    self.record_state_label.setStyleSheet(
                        "color: #666666; background-color: #f1f1f1; "
                        "font-weight: 800; font-size: 24px; "
                        "padding: 6px 14px; border: 2px solid #bdbdbd; border-radius: 8px;"
                    )
                if play_sound:
                    self._play_sound("stop")

        def _trigger_reset_align_follow(self):
            self._reset_arm()
            self._align_gripper()
            if not self.follow_btn.isChecked():
                self.follow_btn.setChecked(True)

        def _sanitize_task_name(self, raw):
            name = (raw or "").strip()
            if not name:
                return "default"
            name = name.replace("/", "_").replace("\\", "_")
            name = re.sub(r"\s+", "_", name)
            name = re.sub(r"[^\w\-一-鿿]+", "_", name)
            name = name.strip("_")
            return name or "default"

        def _next_record_index(self, task_dir):
            return self._max_record_index(task_dir) + 1

        def _open_record_folder(self):
            if self.recording and self.record_dir:
                target = self.record_dir
            elif self.last_record_dir:
                target = self.last_record_dir
            else:
                target = os.path.abspath(self.args.record_dir)
            if not os.path.isdir(target):
                self.statusBar().showMessage("Recording folder not found: {}".format(target))
                return
            try:
                subprocess.Popen(["xdg-open", target])
                self.statusBar().showMessage("Opened {}".format(target))
            except Exception as exc:
                self.statusBar().showMessage("Open folder failed: {}".format(exc))

        def _load_task_name(self):
            if self.task_name_edit is None:
                return
            if not os.path.isfile(self.task_state_path):
                return
            try:
                with open(self.task_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = str(data.get("task", "")).strip()
                if name:
                    self.task_name_edit.setText(name)
            except Exception:
                pass

        def _save_task_name(self):
            if self.task_name_edit is None:
                return
            name = self.task_name_edit.text().strip()
            try:
                with open(self.task_state_path, "w", encoding="utf-8") as f:
                    json.dump({"task": name}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _load_offset(self):
            if not os.path.isfile(self.offset_path):
                return
            try:
                with open(self.offset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tx, ty, tz = data.get("t", [0.0, 0.0, 0.0])
                rx, ry, rz = data.get("rpy", [0.0, 0.0, 0.0])
                undistort = data.get("undistort", {})
                balance = float(undistort.get("balance", self.undistort_balance))
                fov_scale = float(undistort.get("fov_scale", self.undistort_fov_scale))
                crop_center_y = float(data.get("crop_center_y", self.crop_center_y))
                self.tx_row.slider.setValue(int(tx * 1000 + 5000))
                self.ty_row.slider.setValue(int(ty * 1000 + 5000))
                self.tz_row.slider.setValue(int(tz * 1000 + 5000))
                self.rx_row.slider.setValue(int(rx * 10 + 1800))
                self.ry_row.slider.setValue(int(ry * 10 + 1800))
                self.rz_row.slider.setValue(int(rz * 10 + 1800))
                self.balance_row.slider.setValue(int(balance * 100))
                self.fov_row.slider.setValue(int(fov_scale * 100))
                self.crop_center_row.slider.setValue(int(max(0.0, min(1.0, crop_center_y)) * 100))
            except Exception:
                pass

        def _save_offset(self):
            tx = (self.tx_row.slider.value() - 5000) / 1000.0
            ty = (self.ty_row.slider.value() - 5000) / 1000.0
            tz = (self.tz_row.slider.value() - 5000) / 1000.0
            rx = (self.rx_row.slider.value() - 1800) / 10.0
            ry = (self.ry_row.slider.value() - 1800) / 10.0
            rz = (self.rz_row.slider.value() - 1800) / 10.0
            balance = self.balance_row.slider.value() / 100.0
            fov_scale = self.fov_row.slider.value() / 100.0
            crop_center_y = self.crop_center_row.slider.value() / 100.0
            try:
                with open(self.offset_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "t": [tx, ty, tz],
                            "rpy": [rx, ry, rz],
                            "undistort": {"balance": balance, "fov_scale": fov_scale},
                            "crop_center_y": crop_center_y,
                        },
                        f,
                    )
            except Exception:
                pass

        def _load_display_offset(self):
            if not os.path.isfile(self.display_offset_path):
                return
            try:
                with open(self.display_offset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rx, ry, rz = data.get("rpy", [0.0, 0.0, 0.0])
                self.grip_rx_row.slider.setValue(int(rx * 10))
                self.grip_ry_row.slider.setValue(int(ry * 10))
                self.grip_rz_row.slider.setValue(int(rz * 10))
            except Exception:
                pass

        def _save_display_offset(self):
            rx = self.grip_rx_row.slider.value() * 0.1
            ry = self.grip_ry_row.slider.value() * 0.1
            rz = self.grip_rz_row.slider.value() * 0.1
            payload = {"rpy": [rx, ry, rz]}
            try:
                with open(self.display_offset_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass

        def _toggle_sliders(self):
            visible = self.control_panel.isVisible()
            self.control_panel.setVisible(not visible)
            self.toggle_sliders_btn.setText("Show Sliders" if visible else "Hide Sliders")

        def _align_gripper(self):
            if self.mj["model"] is None or self.mj["data"] is None:
                self.statusBar().showMessage("MuJoCo not ready")
                return
            with self.shared["pose_lock"]:
                latest_tf = dict(self.shared["latest_tf"])
            base_pos = latest_tf.get("pos")
            base_quat = latest_tf.get("quat")
            if base_pos is None or base_quat is None:
                self.statusBar().showMessage("No live gripper pose yet")
                return
            base_pos, base_quat = self._apply_live_offset(base_pos, base_quat)
            ref_body_id = self.mj.get("ref_body_id")
            if ref_body_id is None:
                self.statusBar().showMessage("Reference gripper body not found")
                return
            mujoco.mj_forward(self.mj["model"], self.mj["data"])
            ref_quat = tuple(self.mj["data"].xquat[ref_body_id])
            ref_pos = np.array(self.mj["data"].xpos[ref_body_id], dtype=float)
            ref_offset = self.mj.get("ref_offset_local")
            if ref_offset is not None:
                R_ref = np.array(quat_to_rot(ref_quat), dtype=float)
                ref_pos = ref_pos + R_ref.dot(ref_offset)
            ref_pos = tuple(ref_pos)
            try:
                r_quat = quat_mul(ref_quat, quat_inv(base_quat))
            except ValueError:
                self.statusBar().showMessage("Invalid live gripper rotation")
                return
            R = quat_to_rot(r_quat)
            bx, by, bz = base_pos
            tx = ref_pos[0] - (R[0][0] * bx + R[0][1] * by + R[0][2] * bz)
            ty = ref_pos[1] - (R[1][0] * bx + R[1][1] * by + R[1][2] * bz)
            tz = ref_pos[2] - (R[2][0] * bx + R[2][1] * by + R[2][2] * bz)
            rx, ry, rz = quat_to_euler_deg(r_quat)

            def wrap_angle_deg(val):
                return (val + 180.0) % 360.0 - 180.0

            rx = wrap_angle_deg(rx)
            ry = wrap_angle_deg(ry)
            rz = wrap_angle_deg(rz)

            def set_slider(slider, value):
                slider.setValue(int(max(slider.minimum(), min(slider.maximum(), value))))

            set_slider(self.tx_row.slider, round(tx * 1000 + 5000))
            set_slider(self.ty_row.slider, round(ty * 1000 + 5000))
            set_slider(self.tz_row.slider, round(tz * 1000 + 5000))
            set_slider(self.rx_row.slider, round(rx * 10 + 1800))
            set_slider(self.ry_row.slider, round(ry * 10 + 1800))
            set_slider(self.rz_row.slider, round(rz * 10 + 1800))
            self.statusBar().showMessage("Aligned live gripper to arm gripper")

        def _toggle_follow(self, checked):
            if checked and (self.mj["model"] is None or self.mj["data"] is None):
                self.follow_btn.setChecked(False)
                self.statusBar().showMessage("MuJoCo not ready")
                return
            self.follow_enabled = checked
            self.follow_btn.setText("Stop Follow" if checked else "Follow Gripper")
            self.statusBar().showMessage("Follow enabled" if checked else "Follow disabled")

        def _follow_gripper(self, target_pos, target_quat):
            if self.mj["ref_body_id"] is None:
                return
            model = self.mj["model"]
            data = self.mj["data"]
            dof_ids = self.mj["fr3_dof_ids"]
            qpos_adrs = self.mj["fr3_qpos_adrs"]
            act_ids = self.mj["fr3_act_ids"]
            ranges = self.mj["fr3_ranges"]
            if not dof_ids or not qpos_adrs:
                return
            iters = self.mj.get("ik_iters", 6)
            damping = self.mj.get("ik_damping", 0.1)
            max_step = self.mj.get("ik_step", 0.15)
            for _ in range(iters):
                mujoco.mj_forward(model, data)
                cur_quat = np.array(data.xquat[self.mj["ref_body_id"]])
                cur_pos = np.array(data.xpos[self.mj["ref_body_id"]], dtype=float)
                ref_offset = self.mj.get("ref_offset_local")
                if ref_offset is not None:
                    R_ref = np.array(quat_to_rot(cur_quat), dtype=float)
                    offset_world = R_ref.dot(ref_offset)
                    cur_pos = cur_pos + offset_world
                pos_err = np.array(target_pos) - cur_pos
                rot_err = quat_mul(target_quat, quat_inv(cur_quat))
                ang_err = quat_to_axis_angle(rot_err)
                err = np.concatenate([pos_err, ang_err])
                if np.linalg.norm(err) < 1e-4:
                    break
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, self.mj["ref_body_id"])
                Jp = jacp[:, dof_ids]
                Jr = jacr[:, dof_ids]
                if ref_offset is not None:
                    Jp = Jp.copy()
                    Jp += np.cross(Jr.T, offset_world).T
                J = np.vstack([Jp, Jr])
                JT = J.T
                A = J @ JT + (damping * damping) * np.eye(6)
                dq = JT @ np.linalg.solve(A, err)
                scale = max(1.0, np.max(np.abs(dq)) / max_step)
                dq = dq / scale
                for i, qadr in enumerate(qpos_adrs):
                    data.qpos[qadr] += dq[i]
                    lo, hi = ranges[i]
                    data.qpos[qadr] = float(max(lo, min(hi, data.qpos[qadr])))
            for i, aid in enumerate(act_ids):
                data.ctrl[aid] = data.qpos[qpos_adrs[i]]

        def _reset_arm(self):
            if self.mj["model"] is None or self.mj["data"] is None:
                self.statusBar().showMessage("MuJoCo not ready")
                return
            if self.default_fr3_qpos is None:
                self.statusBar().showMessage("Default arm pose not set")
                return
            self.follow_enabled = False
            self.follow_btn.setChecked(False)
            model = self.mj["model"]
            data = self.mj["data"]
            for i, qadr in enumerate(self.mj["fr3_qpos_adrs"]):
                data.qpos[qadr] = self.default_fr3_qpos[i]
            for i, aid in enumerate(self.mj["fr3_act_ids"]):
                data.ctrl[aid] = self.default_fr3_qpos[i]
            mujoco.mj_forward(model, data)
            self.statusBar().showMessage("Arm reset to default pose")

        def _toggle_arm_visibility(self):
            model = self.mj.get("model")
            arm_geom_ids = self.mj.get("arm_geom_ids", [])
            if model is None or not arm_geom_ids:
                self.statusBar().showMessage("Arm geoms not ready")
                return
            self.arm_visible = not self.arm_visible
            if self.arm_visible:
                if self.mj.get("arm_geom_rgba") is not None:
                    model.geom_rgba[arm_geom_ids] = self.mj["arm_geom_rgba"]
                self.toggle_arm_btn.setText("Hide Arm")
            else:
                model.geom_rgba[arm_geom_ids, 3] = 0.0
                self.toggle_arm_btn.setText("Show Arm")
            self.statusBar().showMessage("Arm visible" if self.arm_visible else "Arm hidden")

        def _auto_calib_display(self):
            if self.last_aligned_pose is None:
                self.statusBar().showMessage("No aligned pose yet")
                return
            target_pos = np.array(TARGET_GRIPPER_POSE["p"], dtype=float)
            target_quat = tuple(TARGET_GRIPPER_POSE["q"])
            pos, quat = self.last_aligned_pose
            rot_offset = quat_mul(quat_inv(quat), target_quat)
            rot_offset = rot_to_quat(quat_to_rot(rot_offset))
            disp_quat = quat_mul(quat, rot_offset)
            R_disp = np.array(quat_to_rot(disp_quat), dtype=float)
            local_offset = R_disp.T.dot(target_pos - np.array(pos, dtype=float)) - self.gripper_axis_offset
            rx, ry, rz = quat_to_euler_deg(rot_offset)

            def set_slider(slider, value):
                slider.setValue(int(max(slider.minimum(), min(slider.maximum(), value))))

            set_slider(self.grip_rx_row.slider, round(rx * 10))
            set_slider(self.grip_ry_row.slider, round(ry * 10))
            set_slider(self.grip_rz_row.slider, round(rz * 10))
            self._save_display_offset()
            self.statusBar().showMessage("Display rotation calibrated")

        def _apply_live_offset(self, pos, quat):
            adj_quat = quat_mul(quat, self.live_rot_offset)
            adj_pos = np.array(pos, dtype=float)
            if self.live_offset_local is not None:
                R = np.array(quat_to_rot(adj_quat), dtype=float)
                adj_pos = adj_pos + R.dot(self.live_offset_local)
            return tuple(adj_pos), adj_quat

        def _apply_axis_offset(self, pos, quat):
            if self.gripper_axis_offset is None:
                return pos, quat
            R = np.array(quat_to_rot(quat), dtype=float)
            adj_pos = np.array(pos, dtype=float) + R.dot(self.gripper_axis_offset)
            adj_quat = quat_mul(quat, self.gripper_rot_offset)
            return tuple(adj_pos), adj_quat

        def _init_mujoco(self):
            self.mj = {
                "model": None,
                "data": None,
                "renderer": None,
                "act": None,
                "steps": 12,
                "base_qadr": None,
                "camera": None,
                "ref_body_id": None,
                "ee_site_id": None,
                "ref_offset_local": np.array([0.0, 0.0, 0.0], dtype=float),
                "fr3_qpos_adrs": [],
                "fr3_dof_ids": [],
                "fr3_act_ids": [],
                "fr3_ranges": [],
                "ik_iters": 6,
                "ik_damping": 0.1,
                "ik_step": 0.15,
                "arm_geom_ids": [],
                "arm_geom_rgba": None,
            }
            if self.args.no_mujoco:
                self.mujoco_label.setText("MuJoCo disabled")
                return
            model = mujoco.MjModel.from_xml_path(self.args.mjcf)
            data = mujoco.MjData(model)
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, "live_fingers_actuator"
            )
            if act_id < 0:
                raise RuntimeError("actuator 'live_fingers_actuator' not found in MJCF")
            self.mj["model"] = model
            self.mj["data"] = data
            self.mj["act"] = act_id
            base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "live_base_mount_free")
            if base_jid >= 0:
                self.mj["base_qadr"] = model.jnt_qposadr[base_jid]
            ref_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
            if ref_bid >= 0:
                self.mj["ref_body_id"] = ref_bid
            ee_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            if ee_sid >= 0:
                self.mj["ee_site_id"] = ee_sid
            self.mj["renderer"] = mujoco.Renderer(model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.azimuth = 90.0
            cam.elevation = -25.0
            cam.distance = 2.5
            cam.lookat[:] = np.array([0.0, 0.0, 0.8])
            self.mj["camera"] = cam
            fr3_joints = [
                ("fr3_joint1", -0.1258342),
                ("fr3_joint2", 0.72059298),
                ("fr3_joint3", 0.15445824),
                ("fr3_joint4", -1.61090684),
                ("fr3_joint5", -0.12259634),
                ("fr3_joint6", 2.35071445),
                ("fr3_joint7", 1.62955177),
            ]
            fr3_actuators = [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ]
            self.default_fr3_qpos = [val for _, val in fr3_joints]
            for name, val in fr3_joints:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    adr = model.jnt_qposadr[jid]
                    data.qpos[adr] = val
                    self.mj["fr3_qpos_adrs"].append(adr)
                    self.mj["fr3_dof_ids"].append(model.jnt_dofadr[jid])
                    self.mj["fr3_ranges"].append(tuple(model.jnt_range[jid]))
            for idx, name in enumerate(fr3_actuators):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    data.ctrl[aid] = fr3_joints[idx][1]
                    self.mj["fr3_act_ids"].append(aid)
            arm_root_ids = []
            fr3_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_base")
            if fr3_bid >= 0:
                arm_root_ids.append(fr3_bid)
            rob_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base_mount")
            if rob_bid >= 0:
                arm_root_ids.append(rob_bid)

            arm_body_ids = set()
            if arm_root_ids:
                for bid in range(model.nbody):
                    parent = model.body_parentid[bid]
                    seen = set()
                    while parent != -1 and parent not in seen:
                        seen.add(parent)
                        if parent in arm_root_ids:
                            arm_body_ids.add(bid)
                            break
                        parent = model.body_parentid[parent]
                arm_body_ids.update(arm_root_ids)

            arm_geom_ids = []
            if arm_body_ids:
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] in arm_body_ids:
                        arm_geom_ids.append(gid)
            self.mj["arm_geom_ids"] = arm_geom_ids
            if arm_geom_ids:
                self.mj["arm_geom_rgba"] = model.geom_rgba[arm_geom_ids].copy()

        def adjust_camera_orbit(self, dx, dy):
            cam = self.mj.get("camera")
            if cam is None:
                return
            cam.azimuth += dx * 0.3
            cam.elevation += dy * 0.3

        def adjust_camera_zoom(self, delta):
            cam = self.mj.get("camera")
            if cam is None:
                return
            factor = 0.95 if delta > 0 else 1.05
            cam.distance = max(0.1, cam.distance * factor)

        def adjust_camera_pan(self, dx, dy):
            cam = self.mj.get("camera")
            if cam is None:
                return
            scale = 0.002 * cam.distance
            cam.lookat[0] -= dx * scale
            cam.lookat[1] += dy * scale

        def _start_recording(self):
            base_dir = os.path.abspath(self.args.record_dir)
            raw_task = self.task_name_edit.text() if self.task_name_edit is not None else ""
            task_name = self._sanitize_task_name(raw_task)
            task_dir = os.path.join(base_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            idx = self._next_record_index(task_dir)
            self.record_dir = os.path.join(task_dir, "recording_{:03d}".format(idx))
            os.makedirs(self.record_dir, exist_ok=True)
            self.record_frames = []
            self.record_writer = None
            self.record_raw_writer = None
            self.record_undistort_writer = None
            self.record_frame_size = None
            self.record_raw_size = None
            self.record_undistort_size = None
            self.record_rs1_writer = None
            self.record_rs2_writer = None
            self.record_rs1_size = None
            self.record_rs2_size = None
            self.joint_frames = []
            self.cartesian_frames = []
            self.ros_pose_frames = []
            self.record_start_ts = time.time()
            self.recording = True
            self._set_record_indicator(True)
            if not self.follow_enabled:
                self.follow_btn.setChecked(True)
            self.record_start_btn.setEnabled(False)
            self.record_stop_btn.setEnabled(True)
            self.statusBar().showMessage("Recording to {}".format(self.record_dir))

        def _stop_recording(self):
            self.recording = False
            self.record_start_btn.setEnabled(True)
            self.record_stop_btn.setEnabled(False)
            self._set_record_indicator(False)
            if self.record_writer is not None:
                self.record_writer.release()
                self.record_writer = None
            if self.record_raw_writer is not None:
                self.record_raw_writer.release()
                self.record_raw_writer = None
            if self.record_undistort_writer is not None:
                self.record_undistort_writer.release()
                self.record_undistort_writer = None
            if self.record_rs1_writer is not None:
                self.record_rs1_writer.release()
                self.record_rs1_writer = None
            if self.record_rs2_writer is not None:
                self.record_rs2_writer.release()
                self.record_rs2_writer = None
            joint_path = os.path.join(self.record_dir, "joints.json")
            joint_payload = {
                "start_ts": self.record_start_ts,
                "frames": self.joint_frames,
            }
            cartesian_path = os.path.join(self.record_dir, "cartesian.json")
            cartesian_payload = {
                "start_ts": self.record_start_ts,
                "frames": self.cartesian_frames,
            }
            ros_pose_path = os.path.join(self.record_dir, "ros_pose.json")
            ros_pose_payload = {
                "start_ts": self.record_start_ts,
                "frames": self.ros_pose_frames,
            }
            try:
                with open(joint_path, "w", encoding="utf-8") as f:
                    json.dump(joint_payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            try:
                with open(cartesian_path, "w", encoding="utf-8") as f:
                    json.dump(cartesian_payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            try:
                with open(ros_pose_path, "w", encoding="utf-8") as f:
                    json.dump(ros_pose_payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            if self.record_dir:
                self.last_record_dir = self.record_dir
            self.statusBar().showMessage("Recording saved to {}".format(self.record_dir))

        def _toggle_recording(self):
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording()

        def _delete_last_recording(self):
            if self.recording:
                self.statusBar().showMessage("Stop recording before deleting last recording")
                return
            last_dir = self.last_record_dir
            if not last_dir:
                self.statusBar().showMessage("No previous recording to delete")
                return
            base_dir = os.path.abspath(self.args.record_dir)
            last_dir = os.path.abspath(last_dir)
            base_prefix = base_dir + os.sep
            if not (last_dir == base_dir or last_dir.startswith(base_prefix)):
                self.statusBar().showMessage("Refuse to delete outside record dir: {}".format(last_dir))
                return
            if not os.path.isdir(last_dir):
                self.statusBar().showMessage("Recording not found: {}".format(last_dir))
                return
            try:
                shutil.rmtree(last_dir)
                if self.record_dir == last_dir:
                    self.record_dir = None
                self.last_record_dir = None
                self._refresh_record_indicator()
                self.statusBar().showMessage("Deleted recording {}".format(last_dir))
            except Exception as exc:
                self.statusBar().showMessage("Delete failed: {}".format(exc))

        def _ensure_writer(self, frame):
            if self.record_writer is not None:
                return
            h, w = frame.shape[:2]
            self.record_frame_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(self.record_dir, "video.mp4")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(self.record_dir, "video.avi")
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            self.record_writer = writer

        def _ensure_raw_writer(self, frame):
            if self.record_raw_writer is not None:
                return
            h, w = frame.shape[:2]
            self.record_raw_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(self.record_dir, "video_raw.mp4")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(self.record_dir, "video_raw.avi")
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            self.record_raw_writer = writer

        def _ensure_undistort_writer(self, frame):
            if self.record_undistort_writer is not None:
                return
            h, w = frame.shape[:2]
            self.record_undistort_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(self.record_dir, "video_undistort.mp4")
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(self.record_dir, "video_undistort.avi")
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            self.record_undistort_writer = writer

        def _ensure_rs_writer(self, idx, frame):
            if idx == 1:
                if self.record_rs1_writer is not None:
                    return
                writer_attr = "record_rs1_writer"
                size_attr = "record_rs1_size"
                base_name = "realsense1"
            else:
                if self.record_rs2_writer is not None:
                    return
                writer_attr = "record_rs2_writer"
                size_attr = "record_rs2_size"
                base_name = "realsense2"
            h, w = frame.shape[:2]
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(self.record_dir, "{}.mp4".format(base_name))
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, size)
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = os.path.join(self.record_dir, "{}.avi".format(base_name))
                writer = cv2.VideoWriter(video_path, fourcc, 30.0, size)
            setattr(self, writer_attr, writer)
            setattr(self, size_attr, size)

        def _scale_calib(self, K, size, calib_size):
            if size == calib_size:
                return K
            sx = size[0] / float(calib_size[0])
            sy = size[1] / float(calib_size[1])
            K_scaled = K.copy()
            K_scaled[0, 0] *= sx
            K_scaled[1, 1] *= sy
            K_scaled[0, 2] *= sx
            K_scaled[1, 2] *= sy
            return K_scaled

        def _undistort(self, frame):
            if not self.undistort_enabled or self.calib is None:
                return frame
            h, w = frame.shape[:2]
            size = (w, h)
            if self.undistort_maps is None or self.undistort_size != size:
                K = np.array(self.calib["K"], dtype=np.float64)
                D = np.array(self.calib["D"], dtype=np.float64)
                calib_size = tuple(self.calib.get("image_size", size))
                K = self._scale_calib(K, size, calib_size)
                balance = float(self.calib.get("balance", self.undistort_balance))
                fov_scale = float(self.calib.get("fov_scale", self.undistort_fov_scale))
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, size, np.eye(3), balance=balance, fov_scale=fov_scale
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), newK, size, cv2.CV_16SC2
                )
                self.undistort_maps = (map1, map2)
                self.undistort_size = size
            map1, map2 = self.undistort_maps
            return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        def _crop_to_aspect(self, frame, target_size):
            target_w, target_h = target_size
            h, w = frame.shape[:2]
            target_ratio = target_w / float(target_h)
            current_ratio = w / float(h)
            if abs(current_ratio - target_ratio) < 1e-6:
                return frame
            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                x0 = max(0, (w - new_w) // 2)
                return frame[:, x0 : x0 + new_w]
            new_h = int(w / target_ratio)
            center = max(0.0, min(1.0, float(self.crop_center_y)))
            y0 = int((h - new_h) * center)
            y0 = max(0, min(h - new_h, y0))
            return frame[y0 : y0 + new_h, :]

        def _crop_and_resize(self, frame, target_size):
            cropped = self._crop_to_aspect(frame, target_size)
            return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

        def _scale_to_label_no_upscale(self, frame, label):
            target = label.size()
            tw, th = target.width(), target.height()
            if tw <= 0 or th <= 0:
                return frame
            h, w = frame.shape[:2]
            if w == 0 or h == 0:
                return frame
            scale = min(1.0, tw / float(w), th / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        def _record_sample(
            self,
            frame,
            raw_frame,
            undistort_frame,
            data,
            rs1_frame=None,
            rs2_frame=None,
            joint_data=None,
            cartesian_data=None,
            ros_pose_data=None,
        ):
            if frame is not None:
                self._ensure_writer(frame)
                if self.record_writer is not None:
                    if frame.shape[1::-1] != self.record_frame_size:
                        frame = cv2.resize(frame, self.record_frame_size, interpolation=cv2.INTER_AREA)
                    self.record_writer.write(frame)
            if raw_frame is not None:
                self._ensure_raw_writer(raw_frame)
                if self.record_raw_writer is not None:
                    if raw_frame.shape[1::-1] != self.record_raw_size:
                        raw_frame = cv2.resize(raw_frame, self.record_raw_size, interpolation=cv2.INTER_AREA)
                    self.record_raw_writer.write(raw_frame)
            if undistort_frame is not None:
                self._ensure_undistort_writer(undistort_frame)
                if self.record_undistort_writer is not None:
                    if undistort_frame.shape[1::-1] != self.record_undistort_size:
                        undistort_frame = cv2.resize(
                            undistort_frame, self.record_undistort_size, interpolation=cv2.INTER_AREA
                        )
                    self.record_undistort_writer.write(undistort_frame)
            if rs1_frame is not None:
                self._ensure_rs_writer(1, rs1_frame)
                if self.record_rs1_writer is not None:
                    if rs1_frame.shape[1::-1] != self.record_rs1_size:
                        rs1_frame = cv2.resize(rs1_frame, self.record_rs1_size, interpolation=cv2.INTER_AREA)
                    self.record_rs1_writer.write(rs1_frame)
            if rs2_frame is not None:
                self._ensure_rs_writer(2, rs2_frame)
                if self.record_rs2_writer is not None:
                    if rs2_frame.shape[1::-1] != self.record_rs2_size:
                        rs2_frame = cv2.resize(rs2_frame, self.record_rs2_size, interpolation=cv2.INTER_AREA)
                    self.record_rs2_writer.write(rs2_frame)
            if data is not None:
                self.record_frames.append(data)
            if joint_data is not None:
                self.joint_frames.append(joint_data)
            if cartesian_data is not None:
                self.cartesian_frames.append(cartesian_data)
            if ros_pose_data is not None:
                self.ros_pose_frames.append(ros_pose_data)

        def _update_ui(self):
            if rospy.is_shutdown():
                self.close()
                return
            tx = (self.tx_row.slider.value() - 5000) / 1000.0
            ty = (self.ty_row.slider.value() - 5000) / 1000.0
            tz = (self.tz_row.slider.value() - 5000) / 1000.0
            rx = (self.rx_row.slider.value() - 1800) / 10.0
            ry = (self.ry_row.slider.value() - 1800) / 10.0
            rz = (self.rz_row.slider.value() - 1800) / 10.0
            balance = self.balance_row.slider.value() / 100.0
            fov_scale = self.fov_row.slider.value() / 100.0
            grip_rx = self.grip_rx_row.slider.value() * 0.1
            grip_ry = self.grip_ry_row.slider.value() * 0.1
            grip_rz = self.grip_rz_row.slider.value() * 0.1
            self.crop_center_y = self.crop_center_row.slider.value() / 100.0
            self.offset["t"] = (tx, ty, tz)
            self.offset["rpy"] = (rx, ry, rz)
            self.gripper_rot_offset = euler_deg_to_quat(grip_rx, grip_ry, grip_rz)
            if balance != self.undistort_balance or fov_scale != self.undistort_fov_scale:
                self.undistort_balance = balance
                self.undistort_fov_scale = fov_scale
                self.undistort_maps = None
            with self.shared["lock"]:
                frame = self.shared["latest"]["img"]
            with self.shared["rs_lock"]:
                rs1_frame = self.shared["latest_rs"].get("cam1")
                rs2_frame = self.shared["latest_rs"].get("cam2")
            with self.shared["pose_lock"]:
                latest_tf = dict(self.shared["latest_tf"])
            latest_slam = dict(self.shared["latest_slam"])
            latest_clamp = dict(self.shared["latest_clamp"])
            live_pose = None
            tf_pose = None
            if latest_tf.get("pos") is not None and latest_tf.get("quat") is not None:
                adj_pos, adj_quat = self._apply_live_offset(latest_tf["pos"], latest_tf["quat"])
                latest_tf["pos"] = adj_pos
                latest_tf["quat"] = adj_quat
                aligned_pos, aligned_quat = apply_offset(
                    latest_tf["pos"], latest_tf["quat"], self.offset["t"], self.offset["rpy"]
                )
                self.last_aligned_pose = (aligned_pos, aligned_quat)
                tf_pos, tf_quat = self._apply_axis_offset(aligned_pos, aligned_quat)
                tf_pose = (tf_pos, tf_quat)
                live_pose = {"pos": list(latest_tf["pos"]), "quat": list(latest_tf["quat"])}
            arm_pose = None
            ee_pose = None
            arm_joint = None
            if self.mj["data"] is not None and self.mj["fr3_qpos_adrs"]:
                arm_joint = [float(self.mj["data"].qpos[adr]) for adr in self.mj["fr3_qpos_adrs"]]
            aligned_pose = tf_pose
            joint_payload = {
                "时间戳": time.time(),
                "关节": arm_joint,
                "夹爪开合": latest_clamp.get("value") if isinstance(latest_clamp, dict) else None,
            }
            cartesian_payload = {
                "时间戳": time.time(),
                "目标位姿": {"p": list(aligned_pos), "q": list(aligned_quat)} if tf_pose else None,
                "夹爪开合": latest_clamp.get("value") if isinstance(latest_clamp, dict) else None,
            }
            ros_pose_payload = {
                "时间戳": time.time(),
                "末端姿态": (
                    {"p": list(latest_slam["pos"]), "q": list(latest_slam["quat"])}
                    if latest_slam.get("pos") is not None and latest_slam.get("quat") is not None
                    else None
                ),
                "置信度": latest_slam.get("conf"),
                "夹爪开合": latest_clamp.get("value") if isinstance(latest_clamp, dict) else None,
                "夹爪时间戳": latest_clamp.get("ts") if isinstance(latest_clamp, dict) else None,
            }

            record_frame = None
            raw_frame = None
            undistort_frame = None
            if frame is not None:
                raw_frame = frame.copy()
                raw_disp = self._scale_to_label_no_upscale(raw_frame, self.raw_label)
                raw_rgb = cv2.cvtColor(raw_disp, cv2.COLOR_BGR2RGB)
                raw_qimg = QtGui.QImage(
                    raw_rgb.data,
                    raw_rgb.shape[1],
                    raw_rgb.shape[0],
                    raw_rgb.strides[0],
                    QtGui.QImage.Format_RGB888,
                )
                raw_pix = QtGui.QPixmap.fromImage(raw_qimg)
                self.raw_label.setPixmap(raw_pix)
                frame = self._undistort(frame)
                frame = self._crop_and_resize(frame, (640, 480))
                undistort_frame = frame.copy()
                undistort_disp = self._scale_to_label_no_upscale(frame, self.undistort_label)
                rgb = cv2.cvtColor(undistort_disp, cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(
                    rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888
                )
                pix = QtGui.QPixmap.fromImage(qimg)
                self.undistort_label.setPixmap(pix)
                record_frame = frame.copy()

            if rs1_frame is not None:
                rs1_disp = self._scale_to_label_no_upscale(rs1_frame, self.rs1_label)
                rs1_rgb = cv2.cvtColor(rs1_disp, cv2.COLOR_BGR2RGB)
                rs1_qimg = QtGui.QImage(
                    rs1_rgb.data,
                    rs1_rgb.shape[1],
                    rs1_rgb.shape[0],
                    rs1_rgb.strides[0],
                    QtGui.QImage.Format_RGB888,
                )
                rs1_pix = QtGui.QPixmap.fromImage(rs1_qimg)
                self.rs1_label.setPixmap(rs1_pix)
            if rs2_frame is not None:
                rs2_disp = self._scale_to_label_no_upscale(rs2_frame, self.rs2_label)
                rs2_rgb = cv2.cvtColor(rs2_disp, cv2.COLOR_BGR2RGB)
                rs2_qimg = QtGui.QImage(
                    rs2_rgb.data,
                    rs2_rgb.shape[1],
                    rs2_rgb.shape[0],
                    rs2_rgb.strides[0],
                    QtGui.QImage.Format_RGB888,
                )
                rs2_pix = QtGui.QPixmap.fromImage(rs2_qimg)
                self.rs2_label.setPixmap(rs2_pix)

            mj_image = None
            if self.mj["renderer"] is not None:
                target_pos = None
                target_quat = None
                if self.mj["base_qadr"] is not None:
                    tf_pos = latest_tf["pos"]
                    tf_quat = latest_tf["quat"]
                    if tf_pos is not None and tf_quat is not None:
                        adj_pos, adj_quat = apply_offset(
                            tf_pos, tf_quat, self.offset["t"], self.offset["rpy"]
                        )
                        qadr = self.mj["base_qadr"]
                        self.mj["data"].qpos[qadr : qadr + 3] = adj_pos
                        self.mj["data"].qpos[qadr + 3 : qadr + 7] = adj_quat
                        target_pos, target_quat = adj_pos, adj_quat
                if self.mj["ref_body_id"] is not None:
                    mujoco.mj_forward(self.mj["model"], self.mj["data"])
                    arm_pos = tuple(self.mj["data"].xpos[self.mj["ref_body_id"]])
                    arm_quat = tuple(self.mj["data"].xquat[self.mj["ref_body_id"]])
                    arm_pose = self._apply_axis_offset(arm_pos, arm_quat)
                if self.mj["ee_site_id"] is not None:
                    mujoco.mj_forward(self.mj["model"], self.mj["data"])
                    ee_pos = tuple(self.mj["data"].site_xpos[self.mj["ee_site_id"]])
                    ee_quat = None
                    if hasattr(self.mj["data"], "site_xquat"):
                        ee_quat = tuple(self.mj["data"].site_xquat[self.mj["ee_site_id"]])
                    else:
                        mat = self.mj["data"].site_xmat[self.mj["ee_site_id"]]
                        quat = np.zeros(4, dtype=float)
                        mujoco.mju_mat2Quat(quat, mat)
                        ee_quat = tuple(quat)
                    ee_pose = (ee_pos, ee_quat)
                if self.follow_enabled and target_pos is not None and target_quat is not None:
                    self._follow_gripper(target_pos, target_quat)
                self.mj["data"].ctrl[self.mj["act"]] = clamp_to_ctrl(latest_clamp["value"])
                for _ in range(self.mj["steps"]):
                    mujoco.mj_step(self.mj["model"], self.mj["data"])
                cam = self.mj["camera"]
                self.mj["renderer"].update_scene(self.mj["data"], camera=cam)
                mj_image = self.mj["renderer"].render()
                if mj_image is not None:
                    mj_image = cv2.cvtColor(mj_image, cv2.COLOR_RGB2BGR)
                    mj_disp = cv2.resize(mj_image, (640, 480), interpolation=cv2.INTER_AREA)
                    mj_show = self._scale_to_label_no_upscale(mj_image, self.mujoco_label)
                    rgb = cv2.cvtColor(mj_show, cv2.COLOR_BGR2RGB)
                    qimg = QtGui.QImage(
                        rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888
                    )
                    pix = QtGui.QPixmap.fromImage(qimg)
                    self.mujoco_label.setPixmap(pix)
                    if record_frame is not None:
                        record_frame = np.hstack([record_frame, mj_disp])
                    else:
                        record_frame = mj_disp.copy()

            status_line = _build_status_line(
                latest_slam,
                latest_clamp,
                latest_tf,
                self.offset,
                arm_pose,
                ee_pose,
                live_pose,
                tf_pose,
            )
            self.status_text.setPlainText(status_line)

            if self.recording:
                self._record_sample(
                    record_frame,
                    raw_frame,
                    undistort_frame,
                    None,
                    rs1_frame,
                    rs2_frame,
                    joint_payload,
                    cartesian_payload,
                    ros_pose_payload,
                )

        def closeEvent(self, event):
            self._save_task_name()
            if self.recording:
                self._stop_recording()
            event.accept()

    return MujocoView, SliderRow, FastUMIGui


def main():
    ensure_ros_env(sys.argv, sys.executable)
    started_official = _start_official_backend()
    if not started_official:
        start_ros_backend()
    global cv2, rospy, CvBridge, Image, rostopic, AnyMsg, Clamp, PoseStampedConfidence, mujoco
    import cv2 as _cv2
    import rospy as _rospy
    import rostopic as _rostopic
    from cv_bridge import CvBridge as _CvBridge
    from rospy.msg import AnyMsg as _AnyMsg
    from sensor_msgs.msg import Image as _Image
    from std_srvs.srv import Trigger as _Trigger
    from xv_sdk.msg import Clamp as _Clamp
    from xv_sdk.msg import PoseStampedConfidence as _PoseStampedConfidence
    from xv_sdk.srv import SaveMapAndSwitchCslam as _SaveMapAndSwitchCslam
    from xv_sdk.srv import LoadMapAndSwithcCslam as _LoadMapAndSwithcCslam
    cv2 = _cv2
    rospy = _rospy
    rostopic = _rostopic
    CvBridge = _CvBridge
    Image = _Image
    Trigger = _Trigger
    AnyMsg = _AnyMsg
    Clamp = _Clamp
    PoseStampedConfidence = _PoseStampedConfidence
    SaveMapAndSwitchCslam = _SaveMapAndSwitchCslam
    LoadMapAndSwithcCslam = _LoadMapAndSwithcCslam

    parser = argparse.ArgumentParser(description="FastUMI integrated GUI")
    parser.add_argument("--topic", default="", help="Override image topic")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--no-mujoco", action="store_true", help="Disable MuJoCo render")
    parser.add_argument("--no-realsense", action="store_true", help="Disable RealSense capture")
    parser.add_argument("--rs1-serial", default="243722070232", help="RealSense camera 1 serial number")
    parser.add_argument("--rs2-serial", default="243722073691", help="RealSense camera 2 serial number")
    parser.add_argument(
        "--mjcf",
        default=os.path.join(os.path.dirname(__file__), "assets", "fr3_robotiq_scene.xml"),
        help="Path to MJCF model",
    )
    parser.add_argument("--map-name", default="001_room_207", help="Map name to load/save")
    parser.add_argument(
        "--map-dir",
        default=os.path.join(os.path.dirname(__file__), "map"),
        help="Directory to store maps",
    )
    parser.add_argument(
        "--no-map-load",
        action="store_true",
        help="Disable map auto-load on startup",
    )
    parser.add_argument(
        "--save-map",
        action="store_true",
        help="Save map on exit (if file does not already exist)",
    )
    parser.add_argument(
        "--force-save-map",
        action="store_true",
        help="Overwrite existing map on exit",
    )
    parser.add_argument(
        "--record-dir",
        default=os.path.join(os.path.dirname(__file__), "recordings"),
        help="Directory to store recordings",
    )
    parser.add_argument(
        "--calib-file",
        default=_resolve_config_path("calib_fisheye.json"),
        help="Path to calibration JSON for undistortion",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Disable undistortion even if calibration exists",
    )
    parser.add_argument(
        "--undistort-balance",
        type=float,
        default=0.0,
        help="Undistort balance (0=crop more, 1=keep more FOV)",
    )
    parser.add_argument(
        "--undistort-fov-scale",
        type=float,
        default=0.2,
        help="Undistort fov scale (<1 crops more, >1 keeps more FOV)",
    )
    args = parser.parse_args()

    if not args.no_mujoco:
        try:
            import mujoco as _mujoco
            mujoco = _mujoco
        except Exception as exc:
            raise RuntimeError("mujoco import failed: {}".format(exc))

    calib = None
    if args.calib_file and os.path.isfile(args.calib_file):
        try:
            with open(args.calib_file, "r", encoding="utf-8") as f:
                calib = json.load(f)
        except Exception:
            calib = None
    args.calib = calib

    rospy.init_node("fastumi_gui", anonymous=True, log_level=rospy.WARN)

    rs_serials = []
    if not args.no_realsense:
        rs_serials = [s for s in [args.rs1_serial, args.rs2_serial] if s]
        if len(rs_serials) < 2:
            found, err = _list_realsense_serials()
            if err:
                rospy.logwarn("RealSense discovery failed: %s", err)
            else:
                for serial in found:
                    if serial not in rs_serials:
                        rs_serials.append(serial)
        if len(rs_serials) >= 2:
            rs_serials = rs_serials[:2]
        else:
            if rs_serials:
                rospy.logwarn(
                    "Need two RealSense cameras, found %d. RealSense display disabled.",
                    len(rs_serials),
                )
            rs_serials = []

    topic = resolve_topic(args.topic)
    serial = os.environ.get("XV_DEVICE_SERIAL", "").strip() or DEFAULT_SERIAL
    clamp_candidates = [
        "/xv_sdk/{}/clamp/Data".format(serial),
        "/xv_sdk/clamp/Data",
        "/clamp/Data",
    ]
    slam_topic, slam_class = resolve_slam_topic(serial)
    clamp_topic = resolve_named_topic(clamp_candidates)

    clamp_start_candidates = [
        "/xv_sdk/{}/clamp/start".format(serial),
        "/xv_sdk/clamp/start",
    ]
    clamp_start = resolve_named_topic(clamp_start_candidates)
    clamp_stop_candidates = [
        "/xv_sdk/{}/clamp/stop".format(serial),
        "/xv_sdk/clamp/stop",
    ]
    clamp_stop = resolve_named_topic(clamp_stop_candidates)

    map_dir = os.path.abspath(args.map_dir)
    map_name = args.map_name.strip()
    map_path = os.path.join(map_dir, "{}.cslam".format(map_name))
    os.makedirs(map_dir, exist_ok=True)
    load_map = not args.no_map_load

    bridge = CvBridge()
    shared = {
        "latest": {"img": None},
        "lock": threading.Lock(),
        "rs_lock": threading.Lock(),
        "pose_lock": threading.Lock(),
        "latest_clamp": {"value": 0.0, "ts": None},
        "latest_slam": {"conf": None, "pos": None, "quat": None},
        "latest_tf": {"pos": None, "quat": None},
        "latest_rs": {"cam1": None, "cam2": None},
        "rs_serials": [],
    }
    shared["rs_serials"] = list(rs_serials)
    align_q = (0.70710678, -0.70710678, 0.0, 0.0)

    rs_workers = []
    if rs_serials:
        for idx, rs_serial in enumerate(rs_serials):
            key = "cam1" if idx == 0 else "cam2"
            worker = RealSenseReader(rs_serial, shared, key, shared["rs_lock"])
            try:
                worker.start()
            except Exception as exc:
                rospy.logwarn("RealSense %d start failed (%s): %s", idx + 1, rs_serial, exc)
                for running in rs_workers:
                    running.stop()
                rs_workers = []
                shared["rs_serials"] = []
                break
            rs_workers.append(worker)

    def cb(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr("cv_bridge conversion failed: %s", exc)
            return
        with shared["lock"]:
            shared["latest"]["img"] = img

    def slam_cb(msg):
        if isinstance(msg, PoseStampedConfidence):
            pose = msg.poseMsg.pose
            with shared["pose_lock"]:
                shared["latest_slam"]["conf"] = msg.confidence
                shared["latest_slam"]["pos"] = (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                )
                shared["latest_slam"]["quat"] = (
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                )
                x, y, z = shared["latest_slam"]["pos"]
                shared["latest_tf"]["pos"] = (x, z, -y)
                shared["latest_tf"]["quat"] = quat_mul(align_q, shared["latest_slam"]["quat"])

    def clamp_cb(msg):
        if isinstance(msg, Clamp):
            shared["latest_clamp"]["value"] = msg.data
            shared["latest_clamp"]["ts"] = msg.timestamp

    rospy.Subscriber(topic, Image, cb, queue_size=1)

    if slam_class is None:
        rospy.Subscriber(
            slam_topic, AnyMsg,
            lambda m: None,
            queue_size=1,
        )
    else:
        rospy.Subscriber(
            slam_topic, slam_class,
            slam_cb,
            queue_size=1,
        )

    clamp_class, clamp_real = resolve_msg_class(clamp_topic)
    if clamp_class is None:
        rospy.Subscriber(
            clamp_real, AnyMsg,
            lambda m: None,
            queue_size=1,
        )
    else:
        rospy.Subscriber(
            clamp_real, clamp_class,
            clamp_cb,
            queue_size=1,
        )

    try:
        rospy.wait_for_service(clamp_stop, timeout=2.0)
        clamp_stop_srv = rospy.ServiceProxy(clamp_stop, Trigger)
        clamp_stop_srv()
    except Exception:
        pass
    try:
        rospy.wait_for_service(clamp_start, timeout=3.0)
        clamp_start_srv = rospy.ServiceProxy(clamp_start, Trigger)
        clamp_start_srv()
    except Exception:
        pass

    slam_save_candidates = [
        "/xv_sdk/{}/slam/save_map_cslam".format(serial),
        "/xv_sdk/slam/save_map_cslam",
    ]
    slam_load_candidates = [
        "/xv_sdk/{}/slam/load_map_cslam".format(serial),
        "/xv_sdk/slam/load_map_cslam",
    ]

    def _call_map_service(candidates, srv_type, filename, timeout=3.0):
        last_exc = None
        for name in candidates:
            try:
                rospy.wait_for_service(name, timeout=timeout)
                srv = rospy.ServiceProxy(name, srv_type)
                return srv(filename)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        raise RuntimeError("No map service available")

    if load_map and os.path.isfile(map_path):
        try:
            _call_map_service(slam_load_candidates, LoadMapAndSwithcCslam, map_path)
        except Exception as exc:
            rospy.logwarn("Load map failed: %s", exc)

    QtCore, QtGui, QtWidgets = _import_qt()
    if _QT_PLUGIN_PATH and os.path.isdir(_QT_PLUGIN_PATH):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _QT_PLUGIN_PATH
        os.environ["QT_PLUGIN_PATH"] = _QT_PLUGIN_PATH
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        QtCore.QCoreApplication.setLibraryPaths([_QT_PLUGIN_PATH])
    MujocoView, SliderRow, FastUMIGui = build_gui_classes(QtCore, QtGui, QtWidgets)
    app = QtWidgets.QApplication(sys.argv)
    window = FastUMIGui(args, shared)
    window.resize(1400, 900)
    window.show()

    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)
    app.exec_()

    for worker in rs_workers:
        worker.stop()
    rs_workers = []

    if args.save_map and not rospy.is_shutdown():
        if os.path.isfile(map_path) and not args.force_save_map:
            rospy.loginfo("Map exists, skip save: %s", map_path)
        else:
            try:
                _call_map_service(slam_save_candidates, SaveMapAndSwitchCslam, map_path)
            except Exception as exc:
                rospy.logwarn("Save map failed: %s", exc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
