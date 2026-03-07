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

- [track0-2026-03-07_17.01.16.mp4](track0-2026-03-07_17.01.16.mp4)

## External prerequisites

This folder is self-contained at the file-path level for the GUI itself, but it still depends on system/runtime components that are not vendored here:

- ROS 1 (`/opt/ros/noetic` or `/opt/ros/melodic`)
- a catkin workspace that provides `xv_sdk`
- Python packages such as `rospy`, `cv_bridge`, `opencv-python`, `numpy`, `mujoco`, `pyrealsense2`, and either `PyQt5` or `PySide2`
- an XV device / ROS topics, if you want live data

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
- See `track0-2026-03-07_17.01.16.mp4` for an example workflow.

## Notes

- `start_official.sh` in this repository is intended for the GUI bootstrap path. If the monitor-menu scripts are not present, it will only start the device and exit.
- Generated runtime files such as recordings, offsets, logs, and maps should usually not be committed.
