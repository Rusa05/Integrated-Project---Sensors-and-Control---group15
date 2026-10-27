"""Part B, step B5: checkerboard calibration of the robot camera, as you would on the real robot.

    python part_B_sensing/calibrate_camera.py      (~30 s)

A printed checkerboard (9 x 6 inner corners, 25 mm squares) is shown to a camera identical to the
robot's front_camera (same field of view, 640 x 480, same pixel noise) in N_VIEWS random poses.
Corners are found with OpenCV, then:

  1. naive      cv2.calibrateCamera on every view, default flags
  2. robust     iterative outlier rejection: drop views whose reprojection error is > 3x the median
  3. lens model which distortion terms to fit, chosen on HELD-OUT views (fit on 2/3, test on 1/3),
                so a model that only fits noise loses

The chosen K is compared with the value implied by the field of view and written to
results/camera_calibration.json; src/config.py CAMERA_MATRIX holds the values the robot uses
(copy them there after re-calibrating).
"""
from pathlib import Path
import json
import math
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config                      # noqa: E402
from src.sensors import NoiseConfig         # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
COLS, ROWS, SQUARE = 9, 6, 0.025             # inner corners and square size (m)
N_VIEWS = 60
SEED = 7


def board_texture(path, px_per_sq=60):
    """Checkerboard with (COLS+1) x (ROWS+1) squares and a one-square white margin."""
    nx, ny = COLS + 1, ROWS + 1
    img = np.full(((ny + 2) * px_per_sq, (nx + 2) * px_per_sq), 255, np.uint8)
    for i in range(ny):
        for j in range(nx):
            if (i + j) % 2 == 0:
                img[(i + 1) * px_per_sq:(i + 2) * px_per_sq, (j + 1) * px_per_sq:(j + 2) * px_per_sq] = 0
    cv2.imwrite(str(path), img)
    return (nx + 2) * SQUARE / 2, (ny + 2) * SQUARE / 2       # plane half-sizes


def build_model(tex_path, half_w, half_h):
    robot = mujoco.MjModel.from_xml_path(str(config.SCENE))
    fovy = robot.cam('front_camera').fovy[0]                  # identical optics to the robot camera
    xml = f'''<mujoco>
  <visual><global offwidth="960" offheight="720"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="board" type="2d" file="{tex_path.as_posix()}"/>
    <material name="board" texture="board" texrepeat="1 1" texuniform="false"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{fovy}"/>
    <body name="board" mocap="true" pos="0.6 0 0">
      <geom type="plane" size="{half_w} {half_h} 0.01" material="board" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>'''
    return mujoco.MjModel.from_xml_string(xml), fovy


def render_views(model, rng):
    data = mujoco.MjData(model)
    n = NoiseConfig()
    renderer = mujoco.Renderer(model, n.camera_height, n.camera_width)
    # plane frame: x along board width, y along board height, z = normal; face the camera (normal -x)
    face = Rotation.from_matrix(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]))
    images = []
    while len(images) < N_VIEWS:
        d = rng.uniform(0.35, 0.9)
        pos = [d, rng.uniform(-0.25, 0.25) * d, rng.uniform(-0.18, 0.18) * d]
        tilt = Rotation.from_euler('xyz', rng.uniform([-35, -35, -20], [35, 35, 20]), degrees=True)
        q = (face * tilt).as_quat()                              # x, y, z, w
        data.mocap_pos[0], data.mocap_quat[0] = pos, [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera='cam')
        grey = renderer.render().astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        grey += rng.normal(0, n.pixel_sigma, grey.shape).astype(np.float32)
        images.append(np.clip(grey, 0, 255).astype(np.uint8))
    renderer.close()
    return images


def find_corners(images):
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    obj, img, used = [], [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for k, im in enumerate(images):
        ok, c = cv2.findChessboardCorners(im, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = cv2.cornerSubPix(im, c, (5, 5), (-1, -1), crit)
            obj.append(objp); img.append(c); used.append(k)
    return obj, img, used


LENS_MODELS = {
    'pinhole (no distortion)': cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'k1': cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'k1, k2': cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'k1, k2, p1, p2, k3 (full)': 0,
}


def calibrate(obj, img, size, flags=0):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return rms, K, dist, np.array(per_view)


def held_out_error(obj, img, K, dist):
    """Reprojection RMS on views not used for the fit (pose from solvePnP with the fitted model)."""
    errs = []
    for o, i in zip(obj, img):
        _, r, t = cv2.solvePnP(o, i, K, dist)
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))
    return float(np.sqrt(np.mean(errs)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tex = OUT / '_checkerboard.png'
    half_w, half_h = board_texture(tex)
    model, fovy = build_model(tex, half_w, half_h)
    n = NoiseConfig()
    size = (n.camera_width, n.camera_height)
    f_fov = n.camera_height / (2 * math.tan(math.radians(fovy) / 2))
    K_fov = np.array([[f_fov, 0, (size[0] - 1) / 2], [0, f_fov, (size[1] - 1) / 2], [0, 0, 1]])
    images = render_views(model, rng)
    tex.unlink()
    obj, img, used = find_corners(images)
    print(f'{len(used)}/{N_VIEWS} views with the full board detected')

    # 1. naive
    rms0, K0, dist0, pv0 = calibrate(obj, img, size)
    # 2. robust: iterative outlier rejection
    keep = np.arange(len(obj))
    for _ in range(5):
        rms1, K1, dist1, pv1 = calibrate([obj[k] for k in keep], [img[k] for k in keep], size)
        bad = pv1 > 3 * np.median(pv1)
        if not bad.any():
            break
        keep = keep[~bad]
    # 3. lens model chosen on held-out views
    perm = rng.permutation(keep)
    train, test = perm[: 2 * len(perm) // 3], perm[2 * len(perm) // 3:]
    models = {}
    for name, flags in LENS_MODELS.items():
        rms, K, dist, _ = calibrate([obj[k] for k in train], [img[k] for k in train], size, flags)
        models[name] = {'train_rms_px': float(rms),
                        'test_rms_px': held_out_error([obj[k] for k in test], [img[k] for k in test], K, dist),
                        'K': K.tolist(), 'dist': dist.flatten().tolist()}
    # simplest model whose held-out error is within 2 % of the best (extra terms that only fit noise lose)
    lowest = min(m['test_rms_px'] for m in models.values())
    best = next(name for name in LENS_MODELS if models[name]['test_rms_px'] <= 1.02 * lowest)
    rmsF, KF, distF, _ = calibrate([obj[k] for k in keep], [img[k] for k in keep], size, LENS_MODELS[best])

    def describe(K):
        return {'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]}
    res = {
        'views': {'rendered': N_VIEWS, 'board_found': len(used), 'kept_after_outliers': int(len(keep))},
        'K_from_field_of_view': describe(K_fov),
        'naive': {'rms_px': rms0, **describe(K0), 'dist': dist0.flatten().tolist()},
        'robust': {'rms_px': rms1, **describe(K1)},
        'lens_models_held_out': models, 'chosen_lens_model': best,
        'final': {'rms_px': rmsF, **describe(KF), 'dist': distF.flatten().tolist()},
    }
    (OUT / 'camera_calibration.json').write_text(json.dumps(res, indent=2, default=float))

    print(f'K from field of view : fx=fy={f_fov:.2f}  cx={K_fov[0, 2]:.2f}  cy={K_fov[1, 2]:.2f}')
    print(f'naive  (all views)   : fx={K0[0, 0]:.2f} fy={K0[1, 1]:.2f} cx={K0[0, 2]:.2f} cy={K0[1, 2]:.2f}  rms {rms0:.3f} px')
    print(f'robust ({len(keep)} views)   : fx={K1[0, 0]:.2f} fy={K1[1, 1]:.2f} cx={K1[0, 2]:.2f} cy={K1[1, 2]:.2f}  rms {rms1:.3f} px')
    for name, m in models.items():
        print(f'  lens model {name:28s} train {m["train_rms_px"]:.4f} px  held-out {m["test_rms_px"]:.4f} px')
    print(f'chosen: {best}  ->  fx={KF[0, 0]:.2f} fy={KF[1, 1]:.2f} cx={KF[0, 2]:.2f} cy={KF[1, 2]:.2f}')
    bias = math.degrees(math.atan((KF[0, 2] - 320.0) / KF[0, 0]))
    print(f'principal point vs the W/2 = 320 px we assumed: {KF[0, 2] - 320.0:+.2f} px = {bias:+.3f} deg of bearing')

    # figure: sample views with corners, and the held-out comparison
    fig = plt.figure(figsize=(13, 5.5))
    for k, v in enumerate(used[:6]):
        ax = fig.add_subplot(2, 5, (k // 3) * 5 + k % 3 + 1)
        vis = cv2.cvtColor(images[v], cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, (COLS, ROWS), img[used.index(v)], True)
        ax.imshow(vis[:, :, ::-1]); ax.axis('off'); ax.set_title(f'view {v}', fontsize=8)
    ax = fig.add_subplot(1, 5, (4, 5))
    names = list(models)
    ax.bar(range(len(names)), [models[m]['test_rms_px'] for m in names], color=['C2' if m == best else 'C0' for m in names])
    ax.plot(range(len(names)), [models[m]['train_rms_px'] for m in names], 'ko', label='training error')
    ax.set_xticks(range(len(names)), [m.replace(' (', '\n(') for m in names], fontsize=7)
    ax.set(ylabel='reprojection RMS (px)', title='Lens model chosen on held-out views')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'camera_calibration.png', dpi=120); plt.close(fig)
    print('wrote results/camera_calibration.json and camera_calibration.png')


if __name__ == '__main__':
    main()
