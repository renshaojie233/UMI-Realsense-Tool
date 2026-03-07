#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/venvs/fastumi}"
ROS_RUNTIME_DIR="${ROS_RUNTIME_DIR:-/tmp/fastumi_ros}"

cd "$SCRIPT_DIR"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/noetic/setup.bash
elif [[ -f /opt/ros/melodic/setup.bash ]]; then
  # shellcheck source=/dev/null
  source /opt/ros/melodic/setup.bash
else
  echo "ROS setup.bash not found under /opt/ros." >&2
  exit 1
fi

if [[ -f "$HOME/catkin_ws/devel/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/catkin_ws/devel/setup.bash"
elif [[ -f "$HOME/ros_ws/devel/setup.bash" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/ros_ws/devel/setup.bash"
else
  echo "No catkin workspace setup.bash found in ~/catkin_ws or ~/ros_ws." >&2
  exit 1
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
else
  echo "Virtualenv not found: $VENV_PATH" >&2
  echo "Set VENV_PATH if your environment lives elsewhere." >&2
  exit 1
fi

mkdir -p "$ROS_RUNTIME_DIR/log"
export ROS_HOME="$ROS_RUNTIME_DIR"
export ROS_LOG_DIR="$ROS_RUNTIME_DIR/log"

exec python fastumi_gui.py "$@"
