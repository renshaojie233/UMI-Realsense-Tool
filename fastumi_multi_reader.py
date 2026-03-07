#!/usr/bin/env python3
import argparse
import os
import sys
import time
import threading
import shutil
import math
import json
import signal

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
_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else (
    os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
)
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")
ROT_QUAT_MATRIX = [
    [0.71918939, 0.06225367, -0.69198852],
    [0.02706709, 0.98700960, 0.15848403],
    [0.69433250, -0.14789961, 0.70456402],
]


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
    # Clamp data range varies by firmware (commonly 0-88, 0-100, or 0-255). Map to ctrlrange 0-255, inverted.
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


def wait_for_first_frame(topic, timeout_s=5):
    try:
        rospy.wait_for_message(topic, Image, timeout=timeout_s)
        return True
    except Exception:
        return False


def main():
    debug = "--debug" in sys.argv
    if debug:
        print("fastumi_multi_reader starting", flush=True)
    ensure_ros_env(sys.argv, sys.executable)
    start_ros_backend()
    if debug:
        print("ros env ready", flush=True)
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
    # mujoco is optional

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
    if debug:
        print("imports ok", flush=True)
    parser = argparse.ArgumentParser(description="FastUMI multi-topic reader")
    parser.add_argument("--topic", default="", help="Override image topic")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--no-mujoco", action="store_true", help="Disable MJCF gripper viewer")
    parser.add_argument(
        "--no-mujoco-viewer",
        action="store_true",
        help="Disable interactive MuJoCo window",
    )
    parser.add_argument(
        "--mujoco-render",
        action="store_true",
        help="Enable MuJoCo offscreen rendering window",
    )
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
    args = parser.parse_args()

    if not args.no_mujoco:
        try:
            import mujoco as _mujoco
            mujoco = _mujoco
        except Exception as exc:
            raise RuntimeError("mujoco import failed: {}".format(exc))

    if args.debug:
        print("init_node", flush=True)
    rospy.init_node("fastumi_multi_reader", anonymous=True, log_level=rospy.WARN)
    if args.debug:
        print("init_node ok", flush=True)

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
    latest = {"img": None}
    lock = threading.Lock()
    pose_lock = threading.Lock()
    frame_counter = {"count": 0}
    latest_text = {"slam": "slam: n/a", "clamp": "clamp: n/a"}
    latest_clamp = {"value": 0.0, "ts": None}
    latest_slam = {"conf": None, "pos": None, "quat": None}

    mj = {
        "model": None,
        "data": None,
        "renderer": None,
        "act": None,
        "steps": 12,
        "base_qadr": None,
        "viewer": None,
    }
    if not args.no_mujoco:
        model = mujoco.MjModel.from_xml_path(args.mjcf)
        data = mujoco.MjData(model)
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "live_fingers_actuator")
        if act_id < 0:
            raise RuntimeError("actuator 'live_fingers_actuator' not found in MJCF")
        mj["model"] = model
        mj["data"] = data
        mj["act"] = act_id
        base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "live_base_mount_free")
        if base_jid >= 0:
            mj["base_qadr"] = model.jnt_qposadr[base_jid]
        if args.mujoco_render:
            try:
                mj["renderer"] = mujoco.Renderer(model, height=480, width=640)
            except Exception as exc:
                raise RuntimeError("mujoco renderer failed: {}".format(exc))
        if not args.no_mujoco_viewer:
            try:
                import mujoco.viewer
                mj["viewer"] = mujoco.viewer.launch_passive(
                    model, data, show_left_ui=False, show_right_ui=False
                )
            except Exception as exc:
                if args.debug:
                    rospy.logwarn("MuJoCo viewer failed: %s", exc)

        fr3_joints = [
            ("fr3_joint1", 0.129181),
            ("fr3_joint2", 0.713713),
            ("fr3_joint3", -0.108329),
            ("fr3_joint4", -1.71892),
            ("fr3_joint5", 0.105498),
            ("fr3_joint6", 2.37552),
            ("fr3_joint7", 1.50851),
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
        for name, val in fr3_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                adr = model.jnt_qposadr[jid]
                data.qpos[adr] = val
        for idx, name in enumerate(fr3_actuators):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = fr3_joints[idx][1]

    def cb(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr("cv_bridge conversion failed: %s", exc)
            return
        with lock:
            latest["img"] = img
            frame_counter["count"] += 1

    if args.debug:
        rospy.loginfo("Subscribing to %s", topic)
        rospy.loginfo("SLAM topic: %s", slam_topic)
        rospy.loginfo("Clamp topic: %s", clamp_topic)
        rospy.loginfo("Clamp start service: %s", clamp_start)

    try:
        rospy.wait_for_service(clamp_stop, timeout=2.0)
        clamp_stop_srv = rospy.ServiceProxy(clamp_stop, Trigger)
        clamp_stop_srv()
    except Exception:
        pass
    try:
        rospy.wait_for_service(clamp_start, timeout=3.0)
        clamp_start_srv = rospy.ServiceProxy(clamp_start, Trigger)
        resp = clamp_start_srv()
        if args.debug:
            rospy.loginfo("Clamp start: %s %s", resp.success, resp.message)
    except Exception as exc:
        if args.debug:
            rospy.logwarn("Clamp start service failed: %s", exc)

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
            resp = _call_map_service(slam_load_candidates, LoadMapAndSwithcCslam, map_path)
            if args.debug:
                rospy.loginfo("Load map: %s %s", resp.success, resp.message)
        except Exception as exc:
            rospy.logwarn("Load map failed: %s", exc)
    elif load_map and args.debug:
        rospy.loginfo("Map not found, skip load: %s", map_path)
    rospy.Subscriber(topic, Image, cb, queue_size=1)

    clamp_class, clamp_real = resolve_msg_class(clamp_topic)
    latest_pose = {"pos": None, "quat": None}
    latest_tf = {"pos": None, "quat": None}
    align_q = (0.70710678, -0.70710678, 0.0, 0.0)  # -90deg about X

    def slam_cb(msg):
        if isinstance(msg, PoseStampedConfidence):
            latest_text["slam"] = format_pose(msg)
            pose = msg.poseMsg.pose
            with pose_lock:
                latest_pose["pos"] = (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                )
                latest_pose["quat"] = (
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                )
                latest_slam["conf"] = msg.confidence
                latest_slam["pos"] = latest_pose["pos"]
                latest_slam["quat"] = latest_pose["quat"]
                x, y, z = latest_pose["pos"]
                latest_tf["pos"] = (x, z, -y)
                latest_tf["quat"] = quat_mul(align_q, latest_pose["quat"])
        else:
            latest_text["slam"] = "slam: {}".format(msg)

    if slam_class is None:
        rospy.Subscriber(
            slam_topic, AnyMsg,
            lambda m: latest_text.__setitem__("slam", "slam bytes={}".format(len(m._buff))),
            queue_size=1,
        )
    else:
        rospy.Subscriber(
            slam_topic, slam_class,
            slam_cb,
            queue_size=1,
        )

    def clamp_cb(msg):
        if isinstance(msg, Clamp):
            latest_text["clamp"] = "clamp data={:.4f} ts={:.3f}".format(msg.data, msg.timestamp)
            latest_clamp["value"] = msg.data
            latest_clamp["ts"] = msg.timestamp
        else:
            latest_text["clamp"] = "clamp: {}".format(msg)

    if clamp_class is None:
        rospy.Subscriber(
            clamp_real, AnyMsg,
            lambda m: latest_text.__setitem__("clamp", "clamp bytes={}".format(len(m._buff))),
            queue_size=1,
        )
    else:
        rospy.Subscriber(
            clamp_real, clamp_class,
            clamp_cb,
            queue_size=1,
        )
    cv2.namedWindow("FastUMI Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pose Adjust", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Adjust", 420, 240)
    cv2.createTrackbar("tx_mm", "Pose Adjust", 5000, 10000, lambda v: None)
    cv2.createTrackbar("ty_mm", "Pose Adjust", 5000, 10000, lambda v: None)
    cv2.createTrackbar("tz_mm", "Pose Adjust", 5000, 10000, lambda v: None)
    cv2.createTrackbar("rx_d10", "Pose Adjust", 1800, 3600, lambda v: None)
    cv2.createTrackbar("ry_d10", "Pose Adjust", 1800, 3600, lambda v: None)
    cv2.createTrackbar("rz_d10", "Pose Adjust", 1800, 3600, lambda v: None)
    cv2.createTrackbar("save", "Pose Adjust", 0, 1, lambda v: None)
    try:
        cv2.startWindowThread()
    except Exception:
        pass
    rate = rospy.Rate(30)
    offset = {"t": (0.0, 0.0, 0.0), "rpy": (0.0, 0.0, 0.0)}
    offset_path = _resolve_config_path("pose_offset.json")
    if os.path.isfile(offset_path):
        try:
            with open(offset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tx, ty, tz = data.get("t", [0.0, 0.0, 0.0])
            rx, ry, rz = data.get("rpy", [0.0, 0.0, 0.0])
            cv2.setTrackbarPos("tx_mm", "Pose Adjust", int(tx * 1000 + 5000))
            cv2.setTrackbarPos("ty_mm", "Pose Adjust", int(ty * 1000 + 5000))
            cv2.setTrackbarPos("tz_mm", "Pose Adjust", int(tz * 1000 + 5000))
            cv2.setTrackbarPos("rx_d10", "Pose Adjust", int(rx * 10 + 1800))
            cv2.setTrackbarPos("ry_d10", "Pose Adjust", int(ry * 10 + 1800))
            cv2.setTrackbarPos("rz_d10", "Pose Adjust", int(rz * 10 + 1800))
        except Exception:
            pass
    start_time = time.time()
    warned = False
    last_count = 0
    last_len = 0
    stop_flag = {"stop": False}
    saved_map = False

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not rospy.is_shutdown() and not stop_flag["stop"]:
            frame = None
            with lock:
                frame = latest["img"]
                count = frame_counter["count"]
            now = time.time()
            display = None
            if frame is not None:
                h, w = frame.shape[:2]
                display = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            elif not warned and (now - start_time) >= 5.0:
                available = _list_image_topics()
                rospy.logwarn("No image received on %s yet.", topic)
                if available:
                    rospy.logwarn("Available image topics: %s", ", ".join(available))
                warned = True
            if count != last_count:
                def _fmt_vec(vals, prec):
                    return "(" + ",".join("{:.{p}f}".format(v, p=prec) for v in vals) + ")"

                def _build_line(prec_pos, prec_quat, prec_euler, include_ts):
                    if latest_slam["conf"] is not None:
                        p = _fmt_vec(latest_slam["pos"], prec_pos)
                        q = _fmt_vec(latest_slam["quat"], prec_quat)
                        slam_part = "slam c={:.2f} p={} q={}".format(latest_slam["conf"], p, q)
                    else:
                        slam_part = latest_text["slam"]
                    clamp_part = "clamp={:.2f}".format(latest_clamp["value"])
                    if include_ts and latest_clamp["ts"] is not None:
                        clamp_part += " t={:.3f}".format(latest_clamp["ts"])
                    parts = [slam_part, clamp_part]
                    if latest_tf["pos"] is not None and latest_tf["quat"] is not None:
                        adj_pos, adj_quat = apply_offset(
                            latest_tf["pos"], latest_tf["quat"], offset["t"], offset["rpy"]
                        )
                        p = _fmt_vec(adj_pos, prec_pos)
                        q = _fmt_vec(adj_quat, prec_quat)
                        er, ep, ey = quat_to_euler_deg(adj_quat)
                        e = _fmt_vec((er, ep, ey), prec_euler)
                        rq = apply_rot_matrix_to_quat(adj_quat, ROT_QUAT_MATRIX)
                        rq = _fmt_vec(rq, prec_quat)
                        tf_part = "tf p={} e={} q={} rq={}".format(p, e, q, rq)
                        parts.append(tf_part)
                    return " | ".join(parts)

                width = shutil.get_terminal_size((120, 20)).columns
                max_len = max(10, width - 1)
                line = None
                for prec_pos, prec_quat, prec_euler, include_ts in (
                    (3, 3, 1, True),
                    (3, 3, 1, False),
                    (2, 2, 1, False),
                    (2, 2, 0, False),
                    (1, 1, 0, False),
                ):
                    candidate = _build_line(prec_pos, prec_quat, prec_euler, include_ts)
                    if len(candidate) <= max_len:
                        line = candidate
                        break
                if line is None:
                    line = _build_line(1, 1, 0, False)
                if len(line) < max_len:
                    line = line + (" " * (max_len - len(line)))
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
                last_len = len(line)
                last_count = count
            tx = (cv2.getTrackbarPos("tx_mm", "Pose Adjust") - 5000) / 1000.0
            ty = (cv2.getTrackbarPos("ty_mm", "Pose Adjust") - 5000) / 1000.0
            tz = (cv2.getTrackbarPos("tz_mm", "Pose Adjust") - 5000) / 1000.0
            rx = (cv2.getTrackbarPos("rx_d10", "Pose Adjust") - 1800) / 10.0
            ry = (cv2.getTrackbarPos("ry_d10", "Pose Adjust") - 1800) / 10.0
            rz = (cv2.getTrackbarPos("rz_d10", "Pose Adjust") - 1800) / 10.0
            offset["t"] = (tx, ty, tz)
            offset["rpy"] = (rx, ry, rz)
            if cv2.getTrackbarPos("save", "Pose Adjust") == 1:
                try:
                    with open(offset_path, "w", encoding="utf-8") as f:
                        json.dump({"t": [tx, ty, tz], "rpy": [rx, ry, rz]}, f)
                    cv2.setTrackbarPos("save", "Pose Adjust", 0)
                except Exception:
                    pass
            mj_image = None
            if mj["model"] is not None:
                if mj["base_qadr"] is not None:
                    with pose_lock:
                        tf_pos = latest_tf["pos"]
                        tf_quat = latest_tf["quat"]
                    if tf_pos is not None and tf_quat is not None:
                        adj_pos, adj_quat = apply_offset(
                            tf_pos, tf_quat, offset["t"], offset["rpy"]
                        )
                        qadr = mj["base_qadr"]
                        mj["data"].qpos[qadr : qadr + 3] = adj_pos
                        mj["data"].qpos[qadr + 3 : qadr + 7] = adj_quat
                mj["data"].ctrl[mj["act"]] = clamp_to_ctrl(latest_clamp["value"])
                for _ in range(mj["steps"]):
                    mujoco.mj_step(mj["model"], mj["data"])
            if mj.get("viewer") is not None:
                try:
                    if mj["viewer"].is_running():
                        mj["viewer"].sync()
                    else:
                        mj["viewer"] = None
                except Exception:
                    mj["viewer"] = None
            if mj["renderer"] is not None:
                mj["renderer"].update_scene(mj["data"])
                mj_image = mj["renderer"].render()
                if mj_image is not None:
                    mj_image = cv2.cvtColor(mj_image, cv2.COLOR_RGB2BGR)
            if display is not None:
                cv2.imshow("FastUMI Image", display)
                cv2.waitKey(1)
            if mj_image is not None and args.mujoco_render:
                cv2.imshow("FastUMI MuJoCo", mj_image)
                cv2.waitKey(1)
            rate.sleep()
    except KeyboardInterrupt:
        if args.save_map and not rospy.is_shutdown():
            if os.path.isfile(map_path) and not args.force_save_map:
                rospy.loginfo("Map exists, skip save: %s", map_path)
            else:
                try:
                    resp = _call_map_service(slam_save_candidates, SaveMapAndSwitchCslam, map_path)
                    rospy.loginfo("Save map: %s %s", resp.success, resp.message)
                    saved_map = True
                except Exception as exc:
                    rospy.logwarn("Save map failed: %s", exc)
        raise
    finally:
        if args.save_map and not saved_map and not rospy.is_shutdown():
            if os.path.isfile(map_path) and not args.force_save_map:
                rospy.loginfo("Map exists, skip save: %s", map_path)
            else:
                try:
                    resp = _call_map_service(slam_save_candidates, SaveMapAndSwitchCslam, map_path)
                    rospy.loginfo("Save map: %s %s", resp.success, resp.message)
                except Exception as exc:
                    rospy.logwarn("Save map failed: %s", exc)
        if mj["renderer"] is not None:
            mj["renderer"] = None
        if mj.get("viewer") is not None:
            try:
                mj["viewer"].close()
            except Exception:
                pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
