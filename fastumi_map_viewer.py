#!/usr/bin/env python3
import argparse
import os
import sys
import time

import rospy
import rostopic
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from sensor_msgs import point_cloud2

from ros_env_helper import ensure_ros_env, start_ros_backend


DEFAULT_SERIAL = "250801DR48FP25002287"


def resolve_named_topic(topic_candidates):
    published_all = {t for t, _ in rospy.get_published_topics()}
    for t in topic_candidates:
        if t in published_all:
            return t
    return topic_candidates[0]


def resolve_map_topic(serial):
    candidates = [
        "/xv_sdk/{}/slam/map_points".format(serial),
        "/xv_sdk/slam/map_points",
        "/slam/map_points",
        "/xv_sdk/{}/slam/trajectory".format(serial),
        "/xv_sdk/slam/trajectory",
        "/slam/trajectory",
    ]
    return resolve_named_topic(candidates)


def load_map(service_candidates, srv_type, filename, timeout=3.0):
    last_exc = None
    for name in service_candidates:
        try:
            rospy.wait_for_service(name, timeout=timeout)
            srv = rospy.ServiceProxy(name, srv_type)
            return srv(filename)
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("No map service available")


def points_from_msg(msg, max_points):
    pts = []
    for i, p in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
        pts.append(p)
        if max_points and i >= max_points - 1:
            break
    return pts


def points_from_path(msg, max_points):
    pts = []
    for i, pose in enumerate(msg.poses):
        pos = pose.pose.position
        pts.append((pos.x, pos.y, pos.z))
        if max_points and i >= max_points - 1:
            break
    return pts


def main():
    ensure_ros_env(sys.argv, sys.executable)
    start_ros_backend()

    parser = argparse.ArgumentParser(description="FastUMI map viewer")
    parser.add_argument("--map-name", default="001_room_207", help="Map name to load")
    parser.add_argument(
        "--map-dir",
        default=os.path.join(os.path.dirname(__file__), "map"),
        help="Directory to store maps",
    )
    parser.add_argument("--save-map", action="store_true", help="Save map before viewing")
    parser.add_argument("--no-load", action="store_true", help="Do not load map on startup")
    parser.add_argument("--no-gui", action="store_true", help="Print stats only")
    parser.add_argument("--topic", default="", help="Override map topic")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale points for visualization")
    parser.add_argument("--max-points", type=int, default=200000, help="Max points to visualize")
    parser.add_argument("--timeout", type=float, default=5.0, help="Wait seconds for point cloud")
    args = parser.parse_args()

    serial = os.environ.get("XV_DEVICE_SERIAL", "").strip() or DEFAULT_SERIAL
    map_dir = os.path.abspath(args.map_dir)
    map_path = os.path.join(map_dir, "{}.cslam".format(args.map_name))
    os.makedirs(map_dir, exist_ok=True)

    rospy.init_node("fastumi_map_viewer", anonymous=True, log_level=rospy.WARN)

    from xv_sdk.srv import SaveMapAndSwitchCslam, LoadMapAndSwithcCslam

    slam_save_candidates = [
        "/xv_sdk/{}/slam/save_map_cslam".format(serial),
        "/xv_sdk/slam/save_map_cslam",
    ]
    slam_load_candidates = [
        "/xv_sdk/{}/slam/load_map_cslam".format(serial),
        "/xv_sdk/slam/load_map_cslam",
    ]

    if args.save_map:
        try:
            resp = load_map(slam_save_candidates, SaveMapAndSwitchCslam, map_path)
            rospy.loginfo("Save map: %s %s", resp.success, resp.message)
        except Exception as exc:
            rospy.logwarn("Save map failed: %s", exc)

    if not args.no_load and os.path.isfile(map_path):
        try:
            resp = load_map(slam_load_candidates, LoadMapAndSwithcCslam, map_path)
            rospy.loginfo("Load map: %s %s", resp.success, resp.message)
        except Exception as exc:
            rospy.logwarn("Load map failed: %s", exc)

    topic = args.topic or resolve_map_topic(serial)
    rospy.loginfo("Map topic: %s", topic)
    msg_class, real_topic, _ = rostopic.get_topic_class(topic, blocking=False)
    if real_topic:
        topic = real_topic

    if args.no_gui:
        try:
            if msg_class is Path:
                msg = rospy.wait_for_message(topic, Path, timeout=args.timeout)
                pts = points_from_path(msg, args.max_points)
                print("trajectory points: {}".format(len(pts)))
            else:
                msg = rospy.wait_for_message(topic, PointCloud2, timeout=args.timeout)
                pts = points_from_msg(msg, args.max_points)
                print("map_points received: {}".format(len(pts)))
        except Exception as exc:
            print("no map_points received: {}".format(exc))
        return

    try:
        import open3d as o3d
    except Exception as exc:
        raise SystemExit("open3d import failed: {}".format(exc))

    latest = {"pts": None}

    def cb(msg):
        if isinstance(msg, Path):
            latest["pts"] = points_from_path(msg, args.max_points)
        else:
            latest["pts"] = points_from_msg(msg, args.max_points)

    if msg_class is Path:
        rospy.Subscriber(topic, Path, cb, queue_size=1)
    else:
        rospy.Subscriber(topic, PointCloud2, cb, queue_size=1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FastUMI Map", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.background_color = [0.05, 0.05, 0.07]
    opt.point_size = 2.0
    view_set = False

    def _update_view(pts):
        nonlocal view_set
        if not pts or view_set:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        cz = (min(zs) + max(zs)) * 0.5
        max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if max_range < 1e-3:
            max_range = 1.0
        vc = vis.get_view_control()
        vc.set_lookat([cx, cy, cz])
        vc.set_front([0.0, -1.0, -0.5])
        vc.set_up([0.0, 0.0, 1.0])
        vc.set_zoom(0.7)
        view_set = True

    try:
        if msg_class is Path:
            msg = rospy.wait_for_message(topic, Path, timeout=2.0)
            pts = points_from_path(msg, args.max_points)
        else:
            msg = rospy.wait_for_message(topic, PointCloud2, timeout=2.0)
            pts = points_from_msg(msg, args.max_points)
        if pts:
            pts = [(x * args.scale, y * args.scale, z * args.scale) for x, y, z in pts]
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([0.9, 0.8, 0.2])
            vis.update_geometry(pcd)
            _update_view(pts)
    except Exception:
        pass

    try:
        while not rospy.is_shutdown():
            if latest["pts"]:
                pts = [(x * args.scale, y * args.scale, z * args.scale) for x, y, z in latest["pts"]]
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color([0.9, 0.8, 0.2])
                vis.update_geometry(pcd)
                _update_view(pts)
                latest["pts"] = None
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
