#!/usr/bin/env python3
import argparse
import math
import numpy as np


def quat_to_rot(q, order):
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
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


def make_T(pos, quat, order):
    R = quat_to_rot(quat, order)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(pos, dtype=float)
    return T


def main():
    parser = argparse.ArgumentParser(
        description="Compute transform T_target<-current from two poses of same object."
    )
    parser.add_argument(
        "--target-pos",
        nargs=3,
        type=float,
        default=[0.640215, 0.0313728, 0.0664657],
    )
    parser.add_argument(
        "--target-quat",
        nargs=4,
        type=float,
        default=[0.917124, -0.397332, -0.0161157, 0.027419],
    )
    parser.add_argument(
        "--target-quat-order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
    )
    parser.add_argument(
        "--current-pos",
        nargs=3,
        type=float,
        default=[-0.989, 0.707, 0.105],
    )
    parser.add_argument(
        "--current-quat",
        nargs=4,
        type=float,
        default=[-0.009, -0.074, 0.997, 0.016],
    )
    parser.add_argument(
        "--current-quat-order",
        choices=["wxyz", "xyzw"],
        default="xyzw",
    )
    parser.add_argument(
        "--current-euler",
        nargs=3,
        type=float,
        default=[178.1, -0.9, -171.5],
        help="Fallback current euler deg if you want to compare.",
    )
    args = parser.parse_args()

    T_target = make_T(args.target_pos, args.target_quat, args.target_quat_order)
    T_current = make_T(args.current_pos, args.current_quat, args.current_quat_order)

    T_target_from_current = T_target @ np.linalg.inv(T_current)

    print("T_target_from_current:")
    with np.printoptions(precision=6, suppress=True):
        print(T_target_from_current)

    # Optional: show quaternion from current euler for sanity.
    wxyz = euler_deg_to_quat(*args.current_euler)
    print("current_euler_as_quat_wxyz:", [round(v, 6) for v in wxyz])


if __name__ == "__main__":
    main()
