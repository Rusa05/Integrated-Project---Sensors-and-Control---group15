"""Known map information shared by estimator and controller (the 'map' the robot is given)."""
import math
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / 'turtlebot3' / 'scene.xml'

# Charging-station marker (see turtlebot3/scene.xml and part_B_sensing/make_aruco.py)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE = 0.10                 # m, black square side
MARKER_XY = (1.9085, -1.0)         # m, marker centre in map frame
MARKER_Z = 0.15                    # m
MARKER_YAW = math.pi               # rad, marker normal points along -X into the room
DOCK_GOAL = (1.78, -1.0, 0.0)      # m, m, rad: wheel-axle centre on the charging pad, facing the marker
