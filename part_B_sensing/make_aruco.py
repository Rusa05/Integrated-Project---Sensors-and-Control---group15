"""Part B, step B1: generate the ArUco tag textures and check the scene matches the tag map.

    python part_B_sensing/make_aruco.py

For every tag in src/config.py TAGS this writes turtlebot3/assets/aruco_<id>.png (black square of
the tag's size plus a white quiet zone of TAG_BORDER_FRAC * size on each side), then loads the
scene and checks that each tag's site ('tag_<id>') sits where the map says, facing the right way.
"""
from pathlib import Path
import math
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config      # noqa: E402

PX_PER_M = 4000              # texture resolution (a 10 cm tag is 400 px)
ASSETS = ROOT / 'turtlebot3' / 'assets'


def make_textures():
    dictionary = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
    for tag in config.TAGS.values():
        px = max(int(tag.size * PX_PER_M), 120)
        img = cv2.aruco.generateImageMarker(dictionary, tag.id, px)
        img = np.pad(img, int(round(px * config.TAG_BORDER_FRAC)), constant_values=255)
        out = ASSETS / f'aruco_{tag.id}.png'
        cv2.imwrite(str(out), img)
        print(f'wrote {out.name}: id {tag.id}, {tag.size * 100:.0f} cm, {img.shape[1]} px')


def check_scene():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    worst = 0.0
    for tag in config.TAGS.values():
        site = data.site(f'tag_{tag.id}')
        normal = site.xmat.reshape(3, 3)[:, 2]
        pos_err = np.linalg.norm(site.xpos - [tag.x, tag.y, tag.z])
        yaw_err = abs(math.remainder(math.atan2(normal[1], normal[0]) - tag.yaw, 2 * math.pi))
        plate = model.geom(f'aruco_{tag.id}').size[0] * 2
        size_err = abs(plate - tag.size * (1 + 2 * config.TAG_BORDER_FRAC))
        worst = max(worst, pos_err, yaw_err, size_err)
        if pos_err > 1e-6 or yaw_err > 1e-6 or size_err > 1e-6:
            print(f'MISMATCH tag {tag.id}: position {pos_err:.4f} m, yaw {yaw_err:.4f} rad, plate {size_err:.4f} m')
    print('scene matches the tag map' if worst <= 1e-6 else 'scene does NOT match src/config.py TAGS')


if __name__ == '__main__':
    make_textures()
    check_scene()
