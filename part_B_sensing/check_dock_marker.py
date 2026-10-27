"""Part B, step B2: check that the robot's camera detects the dock tags correctly.

Places the robot at known poses (ground truth is used here only to set up the test and to score
the result), renders front_camera, detects the tags with the robot's own vision code (calibrated
camera, src/vision.py) and compares range and bearing with the true geometry:

  poses 0-2   the 10 cm dock tag (id 0) on the approach, straight on and off-angle
  pose 3      the docked pose: both 3 cm dock tags (ids 7, 8) must be in view

Saves the renders to results/marker_check_*.png. Prints PASS/FAIL.
"""
from pathlib import Path
import math
import sys

import cv2
import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm               # noqa: E402
from src.sensors import Sensors, NoiseConfig      # noqa: E402
from src.vision import MarkerDetector             # noqa: E402
from eval import ground_truth as gt               # noqa: E402

CHECKS = [((1.2, -1.0, 0.0), [0]), ((0.4, -0.7, -0.2), [0]), ((1.3, -1.15, 0.3), [0]),
          (config.DOCK_GOAL, [7, 8])]


def main():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    sens = Sensors(model, data, NoiseConfig.ideal())
    det = MarkerDetector.from_sensors(sens)
    all_ok = True
    for i, (pose, ids) in enumerate(CHECKS):
        gt.set_pose(model, data, *pose)
        grey = sens.camera()
        cv2.imwrite(str(HERE / 'results' / f'marker_check_{i}.png'), grey)
        found = {m.id: m for m in det.measure_all(grey)}
        truth = gt.true_pose(model, data)
        for tid in ids:
            tag = config.TAGS[tid]
            r_t, b_t = rm.h_range_bearing(truth, (tag.x, tag.y))
            if tid not in found:
                print(f'pose {i} {pose}: tag {tid} NOT DETECTED')
                all_ok = False
                continue
            m = found[tid]
            upright = m.corners[0][0] < m.corners[1][0] and m.corners[0][1] < m.corners[3][1]
            ok = abs(m.range - r_t) < 0.03 and abs(rm.wrap(m.bearing - b_t)) < math.radians(2) and upright
            all_ok &= ok
            print(f'pose {i}: tag {tid} ({tag.size * 100:.0f} cm) | range {m.range:.3f} m (true {r_t:.3f}) | '
                  f'bearing {math.degrees(m.bearing):6.2f} deg (true {math.degrees(b_t):6.2f}) | upright={upright}')
    sens.close()
    print('PASS' if all_ok else 'FAIL')


if __name__ == '__main__':
    main()
