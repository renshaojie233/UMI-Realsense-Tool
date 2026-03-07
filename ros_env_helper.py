#!/usr/bin/env python3
import atexit
import os
import signal
import socket
import subprocess
import time


def build_ros_env():
    cmd = r"""
    if [ -f /opt/ros/noetic/setup.bash ]; then
      source /opt/ros/noetic/setup.bash
    elif [ -f /opt/ros/melodic/setup.bash ]; then
      source /opt/ros/melodic/setup.bash
    fi

    if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
      source "$HOME/catkin_ws/devel/setup.bash"
    fi
    if [ -f "$HOME/ros_ws/devel/setup.bash" ]; then
      source "$HOME/ros_ws/devel/setup.bash"
    fi

    env -0
    """
    output = subprocess.check_output(["bash", "-lc", cmd])
    env = os.environ.copy()
    for kv in output.split(b"\0"):
        if not kv:
            continue
        key, _, val = kv.partition(b"=")
        env[key.decode("utf-8")] = val.decode("utf-8")
    return normalize_master_uri(env)


def _localhost_unresolvable():
    try:
        socket.getaddrinfo("localhost", 11311)
        return False
    except socket.gaierror:
        return True


def normalize_master_uri(env):
    master_uri = env.get("ROS_MASTER_URI", "")
    if "localhost" in master_uri and _localhost_unresolvable():
        env["ROS_MASTER_URI"] = master_uri.replace("localhost", "127.0.0.1")
    return env


def _needs_ros_refresh():
    if not os.environ.get("ROS_VERSION"):
        return True
    ros_pkg_path = os.environ.get("ROS_PACKAGE_PATH", "")
    home = os.path.expanduser("~")
    catkin_src = os.path.join(home, "catkin_ws", "src")
    if catkin_src and catkin_src not in ros_pkg_path:
        return True
    if "localhost" in os.environ.get("ROS_MASTER_URI", "") and _localhost_unresolvable():
        return True
    return False


def ensure_ros_env(argv, executable):
    if not _needs_ros_refresh():
        normalize_master_uri(os.environ)
        return
    env = build_ros_env()
    os.execvpe(executable, [executable] + argv, env)


def _ros_master_running(env):
    try:
        subprocess.check_output(
            ["rosnode", "list"],
            stderr=subprocess.STDOUT,
            timeout=2,
            env=env,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False


def _rosnode_exists(env, name):
    try:
        output = subprocess.check_output(
            ["rosnode", "list"],
            stderr=subprocess.STDOUT,
            timeout=2,
            env=env,
        ).decode("utf-8", errors="ignore")
    except Exception:
        return False
    for line in output.splitlines():
        if line.strip() == name:
            return True
    return False


def _start_process(cmd, env):
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )


def start_ros_backend():
    env = build_ros_env()
    procs = []

    if _rosnode_exists(env, "/xv_sdk"):
        return env

    if not _ros_master_running(env):
        roscore = _start_process(["roscore"], env)
        procs.append(roscore)
        time.sleep(2)

    if not _rosnode_exists(env, "/xv_sdk"):
        xv_proc = _start_process(["roslaunch", "xv_sdk", "xv_sdk.launch"], env)
        procs.append(xv_proc)
        deadline = time.time() + 10
        while time.time() < deadline:
            if _rosnode_exists(env, "/xv_sdk"):
                break
            time.sleep(0.5)

    def _cleanup():
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    atexit.register(_cleanup)
    return env
