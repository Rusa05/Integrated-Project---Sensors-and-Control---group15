"""Robot-side sensor processing. Inputs are raw SensorPacket fields only (no ground truth).

  * Odometry      : encoder ticks -> wheel rotations (dphi_left, dphi_right)
  * MarkerSensor  : camera image -> ArUco detections -> marker pose in base_link
                    -> measurement z = [range, bearing, relative yaw]
  * scan_to_points: LiDAR ranges -> 2D points in base_link
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import world as W
from .config import Config, camera_intrinsics
from .frames import P_B_L, marker_in_base, wrap


class Odometry:
    """Turns cumulative encoder ticks into per-step wheel rotations (dphi_left, dphi_right)."""

    def __init__(self, cfg: Config):
        self.tpr = cfg.robot.ticks_per_rev
        self.prev = None

    def update(self, ticks):
        ticks = np.asarray(ticks)
        if self.prev is None:
            self.prev = ticks
            return np.zeros(2)
        dphi = (ticks - self.prev) * 2 * np.pi / self.tpr
        self.prev = ticks
        return dphi  # [rad]


@dataclass
class MarkerObs:
    id: int
    z: np.ndarray            # [range, bearing, relative_yaw] in base_link (best IPPE solution)
    p_B: np.ndarray          # marker centre in base_link (x, y, z)
    corners: np.ndarray      # 4x2 pixel corners
    reproj_err: float        # RMS reprojection error of the best solution [px]
    z_alt: np.ndarray | None = None   # the second IPPE solution (planar-pose ambiguity)
    ambiguity: float = 0.0            # reproj_err_best / reproj_err_alt (1 = fully ambiguous)


class MarkerNoiseModel:
    """Calibrated ArUco error model, transferred to any tag size by pinhole similarity.

    A tag of size s at range r projects exactly like the reference tag (size s_ref) at range
    r * s_ref / s, scaled by k = s / s_ref. Hence metric errors scale by k and angular errors
    do not:  range: k * f_ref(r / k),  bearing / yaw: f_ref(r / k).
    """

    def __init__(self, cfg: Config):
        self.e = cfg.ekf

    def k(self, size):
        return size / self.e.ref_size

    def bias(self, r, size):
        k = self.k(size); c1, c2 = self.e.range_bias_poly
        return k * (c1 * (r / k) + c2 * (r / k) ** 2)

    def sigma(self, r, size, view=None):
        """Std of [range, bearing, yaw]. `view` = angle between the line of sight and the tag
        normal (0 = head-on); yaw is much noisier head-on (planar-PnP ambiguity)."""
        k, e = self.k(size), self.e
        req = r / k
        s_front = e.r_yaw_frontal[0] + e.r_yaw_frontal[1] * req
        s_obl = e.r_yaw_oblique[0] + e.r_yaw_oblique[1] * req
        if view is None:
            s_yaw = s_front                              # conservative
        else:
            s_yaw = s_obl + (s_front - s_obl) * np.exp(-0.5 * (view / 0.3) ** 2)
        return np.array([k * e.r_range[0] + e.r_range[1] * r,
                         e.r_bearing[0] + e.r_bearing[1] * req, s_yaw])

    @staticmethod
    def view_angle(z):
        """Angle between the line of sight and the tag normal, from the measurement itself."""
        return float(wrap(z[2] - z[1] - np.pi))

    def max_range(self, size):
        return self.k(size) * self.e.max_range

    def yaw_max_range(self, size):
        return self.k(size) * self.e.yaw_max_range


class MarkerSensor:
    """ArUco detection + IPPE square PnP, expressed as a range/bearing/yaw measurement."""

    def __init__(self, cfg: Config, sizes: dict | None = None, use_calibration=True):
        self.K, self.dist = camera_intrinsics(calib=use_calibration)
        self.sizes = sizes or {}
        self.noise = MarkerNoiseModel(cfg)
        self.use_bias = use_calibration
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(W.ARUCO_DICT), params)
        self.min_side_px = 12.0   # reject tags too small to give a usable pose

    @staticmethod
    def _obj(size):
        s = size / 2
        # Order matches ArUco corners: TL, TR, BR, BL in the marker frame (x right, y up).
        return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

    def detect(self, image) -> list[MarkerObs]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            c = c.reshape(4, 2)
            side = np.mean(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1))
            if side < self.min_side_px:
                continue
            if int(i) not in self.sizes and self.sizes:
                continue                     # not in the map: ignore
            size = self.sizes.get(int(i), W.MARKER_SIZE)
            # IPPE gives both solutions of the planar-target ambiguity, sorted by error.
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self._obj(size), c, self.K, self.dist,
                                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n == 0:
                continue
            zs = [self._to_z(rv, tv, size) for rv, tv in zip(rvecs, tvecs)]
            errs = np.asarray(errs).ravel()
            amb = float(errs[0] / errs[1]) if n > 1 and errs[1] > 1e-9 else 0.0
            out.append(MarkerObs(int(i), zs[0][0], zs[0][1], c, float(errs[0]),
                                 zs[1][0] if n > 1 else None, amb))
        return out

    def _to_z(self, rvec, tvec, size):
        p_B, yaw_B = marker_in_base(rvec, tvec)
        rng = float(np.hypot(p_B[0], p_B[1]))
        if self.use_bias:
            rng -= self.noise.bias(rng, size)                  # calibrated bias correction
        brg = float(np.arctan2(p_B[1], p_B[0]))
        # relative yaw: direction of the marker normal relative to the robot heading
        return np.array([rng, brg, wrap(yaw_B)]), p_B


def scan_to_points(scan, angles):
    """LiDAR scan -> Nx2 points in base_link (finite returns only)."""
    ok = np.isfinite(scan)
    r, a = scan[ok], angles[ok]
    return np.stack([r * np.cos(a) + P_B_L[0], r * np.sin(a) + P_B_L[1]], 1)


def front_clearance(scan, angles, half_width=0.17, look=0.6):
    """Smallest forward distance to any LiDAR point inside the robot's swept corridor."""
    if scan is None:
        return np.inf
    pts = scan_to_points(scan, angles)
    m = (pts[:, 0] > 0.0) & (np.abs(pts[:, 1]) < half_width) & (pts[:, 0] < look)
    return float(pts[m, 0].min()) if np.any(m) else np.inf
