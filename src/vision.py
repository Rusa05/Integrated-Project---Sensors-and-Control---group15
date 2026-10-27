"""ArUco measurement: image -> range, bearing and relative heading of each known tag, in base_link.

Frame chain (all rotations are fixed by the robot's build, not by its pose):

  tag      ArUco frame from solvePnP: x right, y up, z out of the tag face.
  optical  OpenCV camera frame: x right, y down, z forward (the viewing direction).
  camera   MuJoCo camera frame: x right, y up, z backward (camera looks along -z).
           optical = camera rotated 180 deg about x, i.e. R_cam_opt = diag(1, -1, -1).
  base     base_link: x forward, y left, z up, origin at the wheel-axle centre.
           The camera pose in base comes from the camera mount in the robot model
           (model.cam_pos / cam_quat are body-relative extrinsics, not the robot pose).
  map      world frame. Each tag's map pose is in the known map, src/config.py TAGS.

IPPE ambiguity: a square seen nearly head-on has two pose solutions that fit the image almost
equally well (they differ mainly in the tag's apparent tilt, so in the relative yaw). Both are
returned; the EKF keeps the one that agrees with its prior (src/ekf.py).
"""
from dataclasses import dataclass
import math
import cv2
import mujoco
import numpy as np

from src import config

R_CAM_OPT = np.diag([1.0, -1.0, -1.0])


@dataclass
class MarkerMeasurement:
    id: int                 # tag id (key into config.TAGS)
    size: float             # m, tag size used for the pose solve
    range: float            # m, horizontal distance base origin -> tag centre
    bearing: float          # rad, direction to tag centre in base frame (+ = left)
    rel_yaw: float          # rad, tag normal heading in base frame (= tag yaw - theta)
    p_base: np.ndarray      # tag centre in base frame (m)
    side_px: float          # mean tag side length in pixels (proxy for measurement quality)
    reproj_px: float        # RMS reprojection error of the chosen solvePnP solution
    corners: np.ndarray     # 4x2 detected corners (TL, TR, BR, BL)
    alt: tuple = None       # (range, bearing, rel_yaw) of the other IPPE solution, if any
    t: float = None         # capture time of the camera frame (set by the robot interface)


def camera_extrinsics(model):
    """(R_base_opt, t_base_cam) of front_camera from the robot model."""
    cam = model.camera('front_camera')
    R_base_cam = np.zeros(9)
    mujoco.mju_quat2Mat(R_base_cam, cam.quat)
    return R_base_cam.reshape(3, 3) @ R_CAM_OPT, cam.pos.copy()


class MarkerDetector:
    def __init__(self, K, R_base_opt, t_base_cam, tags=config.TAGS, dist=None):
        self.K, self.dist = K, dist
        self.R_base_opt, self.t_base_cam = R_base_opt, t_base_cam
        self.tags = tags
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT), params)

    @classmethod
    def from_sensors(cls, sensors):
        R, t = camera_extrinsics(sensors.model)
        return cls(sensors.camera_intrinsics(), R, t)

    def _to_base(self, rvec, tvec):
        R_opt_tag, _ = cv2.Rodrigues(rvec)
        p = self.R_base_opt @ np.asarray(tvec).flatten() + self.t_base_cam
        n = self.R_base_opt @ R_opt_tag[:, 2]
        return p, math.hypot(p[0], p[1]), math.atan2(p[1], p[0]), math.atan2(n[1], n[0])

    def measure_all(self, grey):
        """All known tags in the image, as a list of MarkerMeasurement (empty if none)."""
        corners, ids, _ = self.detector.detectMarkers(grey)
        if ids is None:
            return []
        out = []
        for c, tid in zip(corners, ids.flatten().tolist()):
            if tid not in self.tags:
                continue                                  # not part of the known map
            size = self.tags[tid].size
            h = size / 2
            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
            c = c[0].astype(np.float32)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            errs = np.array(errs).flatten()
            order = np.argsort(errs)
            p, rng, brg, yaw = self._to_base(rvecs[order[0]], tvecs[order[0]])
            alt = None
            if len(order) > 1:
                _, rng2, brg2, yaw2 = self._to_base(rvecs[order[1]], tvecs[order[1]])
                alt = (rng2, brg2, yaw2)
            out.append(MarkerMeasurement(
                id=tid, size=size, range=rng, bearing=brg, rel_yaw=yaw, p_base=p,
                side_px=float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))),
                reproj_px=float(errs[order[0]]), corners=c, alt=alt))
        return out

    def measure(self, grey, tag_id=config.MARKER_ID):
        """The measurement of one tag (default: the main dock tag), or None."""
        return next((m for m in self.measure_all(grey) if m.id == tag_id), None)
