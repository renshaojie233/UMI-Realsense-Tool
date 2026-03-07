# FastUMI GUI

This repository contains the standalone GUI part of the FastUMI workflow.

## Included

- `fastumi_gui.py`: main GUI entrypoint
- `ros_env_helper.py`: ROS environment/bootstrap helper
- `assets/fr3_robotiq_scene.xml`: MuJoCo scene
- `assets/franka_fr3/`: FR3 mesh assets used by the MuJoCo scene
- `assets/robotiq_2f85/assets/`: Robotiq mesh assets used by the MuJoCo scene
- `config/`: GUI calibration and local state files
- `start_official.sh`: local helper to start `xv_sdk` in `--start-only` mode

## Preview

GUI screenshot:

![FastUMI GUI](2026-03-07%2017-03-28%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

Recorded usage video:

- [Download demo video from Releases](https://github.com/renshaojie233/UMI-Realsense-Tool/releases/download/v0.1.0/track0-2026-03-07_17.01.16.mp4)

## External prerequisites

This repository is self-contained for the GUI code and MuJoCo assets, but it still depends on external system/runtime components.

Required:

- Ubuntu Linux
- ROS 1: `noetic` is the expected default, `melodic` is also supported by the helper scripts
- a catkin workspace that provides `xv_sdk`
- Python 3
- an XV device and its ROS topics, if you want live data

Required ROS environment:

- `/opt/ros/noetic/setup.bash` or `/opt/ros/melodic/setup.bash`
- `~/catkin_ws/devel/setup.bash` or `~/ros_ws/devel/setup.bash`

Required Python packages:

- `numpy`
- `opencv-python`
- `mujoco`
- `pyrealsense2`
- `rospy`
- `cv_bridge`
- `PyQt5` or `PySide2`

Typical setup example:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
python3 -m venv ~/venvs/fastumi
source ~/venvs/fastumi/bin/activate
pip install numpy opencv-python mujoco pyrealsense2 PyQt5
```

Notes:

- `rospy` and `cv_bridge` are usually installed from the ROS side rather than plain `pip`.
- If `xv_sdk` is missing from your catkin workspace, the GUI can open, but live XV topics will not be available.
- If `mujoco` is missing, the arm visualization and IK-related GUI features will fail.

## Start

```bash
cd UMI_tool
./run_gui.sh
```

You can override the virtualenv location if needed:

```bash
VENV_PATH=/path/to/venv ./run_gui.sh
```

## Usage

- After the GUI opens, press and hold the right mouse button for about 3 seconds. The live pose will be aligned to the MuJoCo gripper pose as a simple calibration step, similar to teleoperation initialization.
- Click the right mouse button once to start recording. Click it again to stop recording.
- Recorded data is saved under the folder named by the current `Task` field in the GUI.
- See the release demo video above for an example workflow.

## Notes

- `start_official.sh` in this repository is intended for the GUI bootstrap path. If the monitor-menu scripts are not present, it will only start the device and exit.
- Generated runtime files such as recordings, offsets, logs, and maps should usually not be committed.
