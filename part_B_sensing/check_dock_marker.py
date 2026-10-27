"""Test-only check that the charging-station marker is detectable from front_camera.

Places the robot at known poses (ground truth is used here only to set up the test and to
score the result), renders the camera, detects the ArUco marker and compares the
solvePnP range/bearing with the true geometry. Saves the renders to results/marker_check_*.png.
"""
from pathlib import Path
import math
import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = ROOT / 'turtlebot3' / 'scene.xml'
WIDTH, HEIGHT = 640, 480
MARKER_SIZE = 0.10

def intrinsics(model):
    fovy = math.radians(model.cam('front_camera').fovy[0])
    f = HEIGHT / (2 * math.tan(fovy / 2))
    return np.array([[f, 0, WIDTH / 2], [0, f, HEIGHT / 2], [0, 0, 1]])

def place_robot(model, data, x, y, yaw):
    q = model.joint('base_joint').qposadr[0]
    data.qpos[q:q + 3] = [x, y, 0.0]
    data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    K = intrinsics(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    h = MARKER_SIZE / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
    all_ok = True
    for i, (x, y, yaw) in enumerate([(1.2, -1.0, 0.0), (0.4, -0.7, -0.2), (1.5, -1.2, 0.3)]):
        place_robot(model, data, x, y, yaw)
        renderer.update_scene(data, camera='front_camera', scene_option=opt)
        rgb = renderer.render()
        cv2.imwrite(str(HERE / 'results' / f'marker_check_{i}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        cam = data.cam('front_camera')
        rel_world = data.site('marker_centre').xpos - cam.xpos
        true_range = np.linalg.norm(rel_world[:2])
        rel_cam = cam.xmat.reshape(3, 3).T @ rel_world  # MuJoCo camera frame: -Z forward, +X right
        true_bearing = math.atan2(-rel_cam[0], -rel_cam[2])
        if ids is None or 0 not in ids.flatten():
            print(f'pose {i} ({x}, {y}, {yaw}): NOT DETECTED')
            all_ok = False
            continue
        c = corners[list(ids.flatten()).index(0)][0]
        _, rvec, tvec = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        t = tvec.flatten()  # OpenCV optical frame: +Z forward, +X right, +Y down
        rng = math.hypot(t[0], t[2])
        bearing = math.atan2(-t[0], t[2])
        top_left_is_top_left = c[0][0] < c[1][0] and c[0][1] < c[3][1]
        print(f'pose {i}: id 0 detected | range {rng:.3f} m (true {true_range:.3f}) | '
              f'bearing {math.degrees(bearing):6.2f} deg (true {math.degrees(true_bearing):6.2f}) | '
              f'upright={top_left_is_top_left}')
        all_ok &= abs(rng - true_range) < 0.03 and abs(bearing - true_bearing) < math.radians(2) and top_left_is_top_left
    print('PASS' if all_ok else 'FAIL')

if __name__ == '__main__':
    main()
