#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LOG_FILE:-/tmp/fastumi_xv_sdk.log}"

usage() {
  cat <<'EOF'
Usage:
  bash start_official.sh            # start device (if needed) + open main menu
  bash start_official.sh <SN>       # start device (if needed) + open single-device menu
  bash start_official.sh --start-only   # start device (if needed) then exit
EOF
}

source_ros_env() {
  if [ -f /opt/ros/noetic/setup.bash ]; then
    # shellcheck source=/dev/null
    source /opt/ros/noetic/setup.bash
  elif [ -f /opt/ros/melodic/setup.bash ]; then
    # shellcheck source=/dev/null
    source /opt/ros/melodic/setup.bash
  fi

  if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "$HOME/catkin_ws/devel/setup.bash"
  fi
  if [ -f "$HOME/ros_ws/devel/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "$HOME/ros_ws/devel/setup.bash"
  fi
}

device_running() {
  if pgrep -f "/opt/ros/.*/bin/roslaunch xv_sdk xv_sdk.launch" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "/lib/xv_sdk/xv_sdk( |$)" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

rosnode_has_xv_sdk() {
  if ! command -v rosnode >/dev/null 2>&1; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rosnode list 2>/dev/null | rg -q '^/xv_sdk$'
  else
    rosnode list 2>/dev/null | grep -q '^/xv_sdk$'
  fi
}

rosnode_xv_sdk_alive() {
  if ! command -v rosnode >/dev/null 2>&1; then
    return 1
  fi
  timeout 3 rosnode ping -c 1 /xv_sdk >/dev/null 2>&1
}

xv_topics_ready() {
  if ! command -v rostopic >/dev/null 2>&1; then
    return 1
  fi
  local pattern='^/xv_sdk/[^/]+/(slam/pose|color_camera/image|clamp/Data)$|^/(xv_sdk/(slam/pose|color_camera/image|clamp/Data)|slam/pose|color_camera/image|clamp/Data)$'
  if command -v rg >/dev/null 2>&1; then
    rostopic list 2>/dev/null | rg -q "$pattern"
  else
    rostopic list 2>/dev/null | grep -Eq "$pattern"
  fi
}

wait_for_xv_topics() {
  local tries="${1:-15}"
  local delay="${2:-2}"
  local i
  for ((i=1; i<=tries; i++)); do
    if rosnode_xv_sdk_alive && xv_topics_ready; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

cleanup_stale_rosnode() {
  if ! command -v rosnode >/dev/null 2>&1; then
    return 0
  fi
  timeout 5 bash -lc "printf 'y\n' | rosnode cleanup" >/dev/null 2>&1 || true
}

start_device_if_needed() {
  if device_running; then
    source_ros_env
    if rosnode_has_xv_sdk && rosnode_xv_sdk_alive && xv_topics_ready; then
      echo "Device already running."
      return 0
    fi
    echo "Stale xv_sdk process detected; restarting."
    pkill -f "/opt/ros/.*/bin/roslaunch xv_sdk xv_sdk.launch" >/dev/null 2>&1 || true
    pkill -f "/lib/xv_sdk/xv_sdk( |$)" >/dev/null 2>&1 || true
    cleanup_stale_rosnode
    sleep 1
  fi

  echo "Starting device: roslaunch xv_sdk xv_sdk.launch"
  source_ros_env
  nohup roslaunch xv_sdk xv_sdk.launch >"$LOG_FILE" 2>&1 &
  echo "Device started in background. Log: $LOG_FILE"
  if wait_for_xv_topics 20 2; then
    echo "Device topics are ready."
  else
    echo "Error: xv_sdk started, but no UMI device topics became ready." >&2
    echo "Check the UMI device USB connection/power and then retry." >&2
    return 1
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  local no_menu=false
  local serial=""
  for arg in "$@"; do
    case "$arg" in
      --start-only|--no-menu)
        no_menu=true
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        if [[ -z "$serial" ]]; then
          serial="$arg"
        else
          usage
          exit 1
        fi
        ;;
    esac
  done

  source_ros_env
  start_device_if_needed

  if [[ "$no_menu" == true ]]; then
    exit 0
  fi

  if [[ ! -f "$SCRIPT_DIR/fastumi_monitor_menu.sh" || ! -f "$SCRIPT_DIR/single_fastumi_monitor_menu.sh" ]]; then
    echo "Device started. Monitor menu scripts are not bundled in this repository."
    exit 0
  fi

  if [[ -n "$serial" ]]; then
    cd "$SCRIPT_DIR"
    bash "$SCRIPT_DIR/single_fastumi_monitor_menu.sh" "$serial"
  else
    cd "$SCRIPT_DIR"
    bash "$SCRIPT_DIR/fastumi_monitor_menu.sh"
  fi
}

main "$@"
