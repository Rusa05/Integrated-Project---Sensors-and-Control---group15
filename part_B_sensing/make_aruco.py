"""Generate the ArUco texture used on the charging station (turtlebot3/assets/aruco_dock.png). Run from anywhere: python part_B_sensing/make_aruco.py.

The marker's black square is MARKER_SIZE metres wide in the scene and is surrounded by a
white quiet zone of BORDER metres, so the textured plane in scene.xml is
MARKER_SIZE + 2*BORDER wide. Keep these numbers in sync with the dock in scene.xml.
"""
from pathlib import Path
import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
MARKER_SIZE = 0.10      # m, black square side length
BORDER = 0.02           # m, white quiet zone on each side
PX_PER_M = 4000         # texture resolution

OUT = Path(__file__).resolve().parent.parent / 'turtlebot3' / 'assets' / 'aruco_dock.png'

def main():
    marker_px = int(MARKER_SIZE * PX_PER_M)
    border_px = int(BORDER * PX_PER_M)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, marker_px)
    image = np.pad(marker, border_px, constant_values=255)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), image)
    print(f'wrote {OUT} ({image.shape[1]}x{image.shape[0]} px, id {MARKER_ID})')

if __name__ == '__main__':
    main()
