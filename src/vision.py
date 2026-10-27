"""ArUco marker measurement: image -> range, bearing and relative heading in base_link (Part B).

Frame chain (all rotations are fixed by the robot's build, not by its pose):

  marker   ArUco frame from solvePnP: x right, y up, z out of the marker face.
  optical  OpenCV camera frame: x right, y down, z forward (the viewing direction).
  camera   MuJoCo camera frame: x right, y up, z backward (camera looks along -z).
           optical = camera rotated 180 deg about x, i.e. R_cam_opt = diag(1, -1, -1).
  base     base_link: x forward, y left, z up, origin at the wheel-axle centre.
           The camera pose in base comes from the camera mount in scene.xml
           (model.cam_pos / cam_quat are body-relative extrinsics, not the robot pose).
  map      world frame. The marker's map pose is the known landmark in src/config.py.
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
    range: float            # m, horizontal distance base origin -> marker centre
    bearing: float          # rad, direction to marker centre in base frame (+ = left)
    rel_yaw: float          # rad, marker normal heading in base frame (= marker_yaw - theta)
    p_base: np.ndarray      # marker centre in base frame (m)
    side_px: float          # mean marker side length in pixels (proxy for measurement quality)
    reproj_px: float        # RMS reprojection error of the chosen solvePnP solution
    corners: np.ndarray     # 4x2 detected corners (TL, TR, BR, BL)
    t: float = None         # capture time of the camera frame (set by the robot interface)


def camera_extrinsics(model):
    """(R_base_opt, t_base_cam) of front_camera from the robot model."""
    cam = model.camera('front_camera')
    R_base_cam = np.zeros(9)
    mujoco.mju_quat2Mat(R_base_cam, cam.quat)
    return R_base_cam.reshape(3, 3) @ R_CAM_OPT, cam.pos.copy()


class MarkerDetector:
    def __init__(self, K, R_base_opt, t_base_cam, marker_id=config.MARKER_ID,
                 marker_size=config.MARKER_SIZE, dist=None):
        self.K, self.dist = K, dist
        self.R_base_opt, self.t_base_cam = R_base_opt, t_base_cam
        self.marker_id = marker_id
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT), params)
        h = marker_size / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    @classmethod
    def from_sensors(cls, sensors):
        R, t = camera_extrinsics(sensors.model)
        return cls(sensors.camera_intrinsics(), R, t)

    def measure(self, grey):
        """Returns a MarkerMeasurement, or None if the marker is not visible."""
        corners, ids, _ = self.detector.detectMarkers(grey)
        if ids is None:
            return None
        ids = ids.flatten().tolist()
        if self.marker_id not in ids:
            return None
        c = corners[ids.index(self.marker_id)][0].astype(np.float32)
        # IPPE returns both planar-pose solutions; keep the one that reprojects best.
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, c, self.K, self.dist,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        best = int(np.argmin(np.array(errs).flatten()))
        R_opt_marker, _ = cv2.Rodrigues(rvecs[best])
        p_base = self.R_base_opt @ tvecs[best].flatten() + self.t_base_cam
        normal_base = self.R_base_opt @ R_opt_marker[:, 2]
        side_px = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
        return MarkerMeasurement(
            range=math.hypot(p_base[0], p_base[1]),
            bearing=math.atan2(p_base[1], p_base[0]),
            rel_yaw=math.atan2(normal_base[1], normal_base[0]),
            p_base=p_base, side_px=side_px, reproj_px=float(np.array(errs).flatten()[best]),
            corners=c)
