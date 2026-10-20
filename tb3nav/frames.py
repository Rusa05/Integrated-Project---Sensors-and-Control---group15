"""Frame definitions and transforms (see docs/frames.md for the diagram).

  W  world        : room corner, x east, y north, z up (fixed)
  B  base_link    : midpoint of the wheel axle on the floor, x forward, y left, z up
  C  camera optical (OpenCV convention): z forward along B x, x right (= -B y), y down (= -B z)
  L  lidar        : LDS-01 optical centre, axes parallel to B
  M  marker i     : marker centre, x along the wall, y up, z = outward normal (into the room)

All extrinsics are taken from models/turtlebot3_waffle_pi_sensors.xml.
"""
import numpy as np

# ---- extrinsics (B <- sensor) -----------------------------------------------------
P_B_C = np.array([0.0855, 0.0, 0.107])
R_B_C = np.array([[0.0, 0.0, 1.0],     # columns are the C axes expressed in B
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
P_B_L = np.array([-0.049, 0.0, 0.172])


def wrap(a):
    """Wrap an angle (or array) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rot2(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def se2_compose(a, b):
    """a ⊕ b for SE(2) poses (x, y, th)."""
    x, y, th = a
    p = np.array([x, y]) + rot2(th) @ np.asarray(b[:2])
    return np.array([p[0], p[1], wrap(th + b[2])])


def se2_inverse(a):
    x, y, th = a
    p = -rot2(th).T @ np.array([x, y])
    return np.array([p[0], p[1], -th])


def marker_in_base(rvec, tvec):
    """Marker pose from solvePnP (C <- M) expressed in B.

    Returns (p_B, yaw_B): marker centre in B (3-vector) and the yaw of the marker's outward
    normal (its +z axis) in the B x-y plane.
    """
    import cv2
    R_C_M, _ = cv2.Rodrigues(rvec)
    p_B = R_B_C @ np.asarray(tvec).reshape(3) + P_B_C
    z_B = R_B_C @ R_C_M[:, 2]
    return p_B, float(np.arctan2(z_B[1], z_B[0]))


def quat_to_yaw(q):
    """MuJoCo quaternion (w, x, y, z) -> yaw about world z."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
