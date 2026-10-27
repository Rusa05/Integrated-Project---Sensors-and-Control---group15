"""Known map information shared by estimator and controller (the 'map' the robot is given).

The robot knows where the ArUco tags are (a printed, surveyed map), but NOT where the obstacles
are: those it maps with its LiDAR. turtlebot3/scene.xml must match TAGS; part_B_sensing/make_aruco.py
generates the tag textures from this table and checks the scene against it.
"""
from dataclasses import dataclass
import math
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / 'turtlebot3' / 'scene.xml'
ARUCO_DICT = cv2.aruco.DICT_4X4_50
TAG_BORDER_FRAC = 0.2              # white quiet zone on each side, as a fraction of the tag size


@dataclass(frozen=True)
class Tag:
    id: int
    x: float          # m, tag centre in the map frame
    y: float
    z: float
    yaw: float        # rad, heading of the tag's outward normal in the map frame
    size: float       # m, side of the black square


# Wall planes sit 1.5 mm in front of the wall faces (x, y = +-1.96) and the dock backplate (x = 1.91).
_W = 1.9585
TAGS = {t.id: t for t in [
    # charging station: one 10 cm tag seen on the approach, and two 3 cm tags at camera height,
    # 10 cm apart, that stay in view right down to the docked pose
    Tag(0, 1.9085, -1.00, 0.25, math.pi, 0.10),
    Tag(7, 1.9085, -0.95, 0.107, math.pi, 0.03),
    Tag(8, 1.9085, -1.05, 0.107, math.pi, 0.03),
    # 15 cm wall tags for localisation: usable to ~3 m, so one is in range from anywhere in the room
    Tag(1, -0.90, _W, 0.20, -math.pi / 2, 0.15),
    Tag(2, 0.90, _W, 0.20, -math.pi / 2, 0.15),
    Tag(3, -0.90, -_W, 0.20, math.pi / 2, 0.15),
    Tag(4, 0.60, -_W, 0.20, math.pi / 2, 0.15),
    Tag(5, -_W, 0.00, 0.20, 0.0, 0.15),
    Tag(6, _W, 0.90, 0.20, math.pi, 0.15),
]}
DOCK_TAG_IDS = (0, 7, 8)

# The main dock tag, kept under its original names for the scripts that characterise it.
MARKER_ID = 0
MARKER_SIZE = TAGS[0].size
MARKER_XY = (TAGS[0].x, TAGS[0].y)
MARKER_Z = TAGS[0].z
MARKER_YAW = TAGS[0].yaw

# Camera intrinsics (fx, fy, cx, cy) in pixels for 640x480, from the checkerboard calibration
# (part_B_sensing/calibrate_camera.py, 56 views, pinhole chosen on held-out views, 0.057 px).
# The first version assumed cx = W/2 = 320 px; the calibration found 319.37 px (pixel coordinates
# count from the centre of the first pixel), a 0.07 deg bearing bias.
CAMERA_MATRIX = (529.17, 529.17, 319.37, 239.45)

# Docked pose: wheel-axle centre on the pad, facing the dock. The camera (0.086 m ahead of the axle)
# stops 0.20 m from the backplate so both small dock tags stay in view.
DOCK_GOAL = (1.62, -1.0, 0.0)
