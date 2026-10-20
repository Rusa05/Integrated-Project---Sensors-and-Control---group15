"""Step 3 - Camera intrinsic calibration and ArUco measurement-noise characterisation.

Part 1: a 9x6 checkerboard (40 mm squares) is shown to the robot camera in 30 random poses;
        cv2.calibrateCamera estimates K and distortion. Compared against the analytic pinhole
        model from the MuJoCo fovy.
Part 2: the robot is placed at known poses in front of the dock marker (0.3-3.0 m, several
        bearings and viewing angles). For each pose the image is rendered once and 40 noisy
        copies are processed, giving the bias and standard deviation of
        z = [range, bearing, relative yaw] as a function of range. These fit the EKF's R.

Ground truth is used here exactly as a calibration rig (a tape measure / motion-capture
reference) would be on hardware: offline, to identify sensor parameters. It is never used
online.

Run:  python scripts/03_calibrate_camera.py           (writes results/calibration.json + figures)
      python scripts/03_calibrate_camera.py --refit   (re-fit the saved samples, no rendering)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from tb3nav import world as W
from tb3nav.config import CALIB_FILE, camera_intrinsics, load_config
from tb3nav.frames import wrap
from tb3nav.sensing import MarkerSensor

FIG = Path(__file__).resolve().parent.parent / "results" / "figures"
rng = np.random.default_rng(3)


def update_calib(key, value):
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    c = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
    c[key] = value
    CALIB_FILE.write_text(json.dumps(c, indent=2))


# ------------------------------------------------------------------ Part 1: intrinsics
def checkerboard_png(nx=10, ny=7, px=60):
    img = np.full(((ny + 2) * px, (nx + 2) * px), 255, np.uint8)
    for i in range(ny):
        for j in range(nx):
            if (i + j) % 2 == 0:
                img[(i + 1) * px:(i + 2) * px, (j + 1) * px:(j + 2) * px] = 0
    path = W.GEN_DIR / "checker.png"
    W.GEN_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def calibrate_intrinsics():
    sq, nx, ny = 0.04, 10, 7
    png = checkerboard_png(nx, ny)
    w, h = (nx + 2) * sq, (ny + 2) * sq
    spec = W.WorldSpec(W.default_landmarks(), [], start=(1.0, 2.0, 0.0))
    spec.extra_assets = (f'<texture name="tex_cb" type="2d" file="{png.as_posix()}"/>'
                         f'<material name="mat_cb" texture="tex_cb" texrepeat="1 1" texuniform="false"/>')
    spec.extra_bodies = (f'<body name="board" mocap="true" pos="2 2 0.2">'
                         f'<geom type="plane" size="{w/2} {h/2} 0.01" material="mat_cb" '
                         f'contype="0" conaffinity="0" group="1"/></body>')
    model, data = W.load_model(spec)
    ren = mujoco.Renderer(model, 480, 640)
    cam_pos = None
    obj = np.zeros(((nx - 1) * (ny - 1), 3), np.float32)
    obj[:, :2] = np.mgrid[0:nx - 1, 0:ny - 1].T.reshape(-1, 2) * sq
    objpts, imgpts, shots = [], [], []
    tries = 0
    while len(objpts) < 60 and tries < 400:
        tries += 1
        mujoco.mj_forward(model, data)
        cam_pos = data.cam_xpos[0].copy()
        dist = rng.uniform(0.45, 1.1)
        # cover the whole image, including the corners, so cx, cy and distortion are observable
        lat, vert = rng.uniform(-0.30, 0.30) * dist, rng.uniform(-0.20, 0.20) * dist
        # board faces the camera (normal = -x) with random tilts
        yaw, pitch, roll = np.pi + rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)
        # plane local z (its textured face) must point back at the camera
        n = -np.array([np.cos(yaw - np.pi), np.sin(yaw - np.pi), 0.0])
        n = W_rot(pitch, "y") @ n
        up = W_rot(roll, "x") @ np.array([0, 0, 1.0])
        x = np.cross(up, n); x /= np.linalg.norm(x)
        y = np.cross(n, x)
        Rm = np.stack([x, y, n], 1)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.reshape(-1))
        data.mocap_pos[0] = cam_pos + np.array([dist, -lat, vert])
        data.mocap_quat[0] = q
        mujoco.mj_forward(model, data)
        ren.update_scene(data, camera="cam_front")
        img = ren.render()
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (nx - 1, ny - 1))
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        objpts.append(obj); imgpts.append(corners)
        if len(shots) < 4:
            shots.append(cv2.drawChessboardCorners(img.copy(), (nx - 1, ny - 1), corners, ok))
    ren.close()
    # Outlier rejection: a few views have mis-localised corners (board partly out of frame or
    # at a grazing angle). Calibrate, drop views whose RMS error > 3x the median, repeat.
    K_init = np.array([[640.0, 0, 320], [0, 640.0, 240], [0, 0, 1]])
    n_all = len(objpts)
    for it in range(3):
        _, K_, d_, rvs, tvs = cv2.calibrateCamera(objpts, imgpts, (640, 480), K_init.copy(), None,
                                                  flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        per = np.array([np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K_, d_)[0].reshape(-1, 2)
                                                 - im.reshape(-1, 2)) ** 2, -1)))
                        for o, im, r, t in zip(objpts, imgpts, rvs, tvs)])
        keep = per < 3 * np.median(per)
        if keep.all():
            break
        objpts = [o for o, k in zip(objpts, keep) if k]
        imgpts = [o for o, k in zip(imgpts, keep) if k]
    print(f"outlier rejection kept {len(objpts)}/{n_all} views")
    # Model selection: fit on 2/3 of the views, score reprojection error on the held-out 1/3.
    n_fit = 2 * len(objpts) // 3
    models = {
        "pinhole (no distortion)": cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        | cv2.CALIB_ZERO_TANGENT_DIST,
        "radial k1,k2": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        "full 5-param": 0,
    }
    # K_init: generic guess (f ~ image width, centred principal point) - standard practice,
    # uses nothing about the true camera.
    scores = {}
    for name, flags in models.items():
        _, K_, d_, _, _ = cv2.calibrateCamera(objpts[:n_fit], imgpts[:n_fit], (640, 480),
                                              K_init.copy(), None,
                                              flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
        errs = []
        for o, im in zip(objpts[n_fit:], imgpts[n_fit:]):
            ok, rv, tv = cv2.solvePnP(o, im, K_, d_)
            pr, _ = cv2.projectPoints(o, rv, tv, K_, d_)
            errs.append(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, -1))))
        scores[name] = float(np.mean(errs))
        print(f"  {name:26s} held-out reprojection RMS = {scores[name]:.3f} px")
    best = min(scores, key=scores.get)
    # prefer the simplest model unless a richer one is clearly (>5 %) better
    for name in models:
        if scores[name] <= 1.05 * scores[best]:
            best = name
            break
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, (640, 480), K_init.copy(), None,
                                             flags=models[best] | cv2.CALIB_USE_INTRINSIC_GUESS)
    K0, _ = camera_intrinsics(calib=False)
    print(f"selected model: {best}; views used: {len(objpts)}  RMS reprojection: {rms:.3f} px")
    print("analytic K:\n", K0.round(2), "\ncalibrated K:\n", K.round(2), "\ndist:", dist.ravel().round(4))
    fig, ax = plt.subplots(1, len(shots), figsize=(4 * len(shots), 3))
    for a, s in zip(np.atleast_1d(ax), shots):
        a.imshow(s); a.axis("off")
    fig.suptitle("Simulated checkerboard calibration views (robot camera)")
    fig.tight_layout(); fig.savefig(FIG / "camera_calibration_views.png", dpi=110); plt.close(fig)
    update_calib("camera", {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": rms,
                            "model": best, "heldout_rms_px": scores,
                            "n_views": len(objpts), "K_analytic": K0.tolist()})
    return K, dist


def W_rot(a, ax):
    c, s = np.cos(a), np.sin(a)
    if ax == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ------------------------------------------------------------------ Part 2: marker noise
def characterise_markers(use_calibration=True, n_noise=24):
    cfg = load_config()
    spec = W.WorldSpec(W.default_landmarks(), [])
    ms = MarkerSensor(cfg, spec.landmark_sizes(), use_calibration=use_calibration)
    ms.use_bias = False     # characterise the RAW measurement (bias is what we are fitting)
    dock = spec.dock
    model, data = W.load_model(spec)
    ren = mujoco.Renderer(model, 480, 640)
    rows = []
    for rng_true in np.arange(0.4, 6.01, 0.4):
        for bearing in (-0.3, 0.0, 0.3):            # marker bearing in the robot frame
            for view in (-1.0, -0.5, 0.0, 0.5, 1.0):   # viewing angle off the marker normal
                # robot position on a ray from the marker at angle `view` off its normal
                ang = dock.yaw + view
                pos = np.array([dock.x, dock.y]) + rng_true * np.array([np.cos(ang), np.sin(ang)])
                heading = np.arctan2(dock.y - pos[1], dock.x - pos[0]) - bearing
                data.qpos[:] = 0
                data.qpos[0:2] = pos
                data.qpos[3] = np.cos(heading / 2); data.qpos[6] = np.sin(heading / 2)
                mujoco.mj_forward(model, data)
                # true measurement from base origin (planar)
                d = np.array([dock.x, dock.y]) - pos
                z_true = np.array([np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - heading),
                                   wrap(dock.yaw - heading)])
                ren.update_scene(data, camera="cam_front")
                clean = ren.render().astype(np.float32)
                for _ in range(n_noise):
                    img = np.clip(clean + rng.normal(0, cfg.noise.pixel_noise_std, clean.shape),
                                  0, 255).astype(np.uint8)
                    obs = [o for o in ms.detect(img) if o.id == W.DOCK_ID]
                    if obs:
                        e = obs[0].z - z_true
                        e[1:] = wrap(e[1:])
                        rows.append([rng_true, bearing, view, *e, 1])
                    else:
                        rows.append([rng_true, bearing, view, np.nan, np.nan, np.nan, 0])
    ren.close()
    return np.array(rows)


def fit_noise(rows, tag):
    """Per-range bias/std, a range-bias correction and sigma(r) = a + b r for each channel."""
    ranges = np.unique(rows[:, 0])
    stats = []
    for r in ranges:
        sel = rows[(rows[:, 0] == r) & (rows[:, 6] == 1)]
        det = np.mean(rows[rows[:, 0] == r][:, 6])
        if len(sel) < 5:
            continue
        med = np.median(sel[:, 3:6], 0)
        # Spread: range and bearing use the robust std (1.4826 MAD). Yaw uses a TRIMMED std:
        # only gross IPPE flips (|error| > 0.3 rad, which the EKF gate rejects) are
        # dropped, so near-frontal small-angle ambiguities stay in the noise model.
        mad = 1.4826 * np.median(np.abs(sel[:, 3:6] - med), 0)
        gross = np.abs(sel[:, 5]) > 0.3          # errors vs truth: |e| > 0.3 rad is a flip
        mad[2] = np.std(sel[~gross, 5])
        out_yaw = np.mean(gross)
        stats.append([r, det, *med, *mad, out_yaw])
    stats = np.array(stats)
    r = stats[:, 0]
    # usable range: furthest range with a detection rate >= 40 %.
    # yaw: IPPE has a two-fold ambiguity ("flip"); the EKF resolves gross flips against its
    # prior (sensing.MarkerObs.z_alt), so yaw is used wherever its trimmed std <= 0.05 rad.
    max_range = float(r[stats[:, 1] >= 0.4].max())
    good = (stats[:, 7] <= 0.05) & (r <= max_range)
    yaw_max_range = float(r[good].max())
    keep = r <= max_range
    stats, r = stats[keep], r[keep]
    # Range bias grows with range (corner localisation is pulled inward on small, blurred
    # tags). Fit bias(r) = c1 r + c2 r^2 (zero at r = 0) and subtract it in MarkerSensor.
    A = np.stack([r, r ** 2], 1)
    c_bias = np.linalg.lstsq(A, stats[:, 2], rcond=None)[0]
    resid = {0: stats[:, 2] - A @ c_bias, 1: stats[:, 3], 2: stats[:, 4]}
    fits = {"range_bias_poly": c_bias.tolist(), "yaw_max_range": yaw_max_range,
            "max_range": max_range}
    # Yaw depends strongly on the viewing angle (planar PnP is weakest head-on): fit separate
    # sigma(r) lines for frontal (view = 0) and oblique (|view| >= 0.5) views.
    for key, sel_view in (("r_yaw_frontal", lambda v: v == 0), ("r_yaw_oblique", lambda v: np.abs(v) >= 0.5)):
        pts = []
        for rr in r:
            m = (rows[:, 0] == rr) & (rows[:, 6] == 1) & sel_view(rows[:, 2])
            y = rows[m, 5]
            if len(y) < 10:
                continue
            y = y[np.abs(y) < 0.3]                    # drop gross flips (gated online)
            pts.append([rr, np.sqrt(np.mean(y ** 2))])
        pts = np.array(pts)
        b_, a_ = np.polyfit(pts[:, 0], pts[:, 1], 1)
        fits[key] = [float(max(a_, 0.001)), float(max(b_, 0.0))]
        fits[key + "_points"] = pts.tolist()
    names = ["r_range", "r_bearing", "r_yaw"]
    for k, n in enumerate(names):
        use = np.ones_like(r, bool) if k < 2 else r <= yaw_max_range
        rms = np.sqrt(resid[k][use] ** 2 + stats[use, 5 + k] ** 2)  # total error after correction
        b, a = np.polyfit(r[use], rms, 1)
        fits[n] = [float(max(a, 0.5 * rms.min())), float(max(b, 0.0))]
    print(f"[{tag}] detection rate by range:", dict(zip(r.round(1), stats[:, 1].round(2))))
    print(f"[{tag}] yaw sigma fits: frontal {np.round(fits['r_yaw_frontal'], 4)}, "
          f"oblique {np.round(fits['r_yaw_oblique'], 4)}")
    print(f"[{tag}] range bias fit c1,c2 = {np.round(c_bias, 5)}; usable range {max_range} m;"
          f" yaw used up to {yaw_max_range:.1f} m; yaw flip rate {stats[:, 8].round(3)}")
    for k, n in enumerate(names):
        print(f"[{tag}] {n}: bias {stats[:, 2+k].round(4)}  std {stats[:, 5+k].round(4)}"
              f"  sigma fit a,b={np.round(fits[n], 4)}")
    return stats, fits


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if "--refit" not in sys.argv:
        print("== Part 1: intrinsic calibration ==")
        calibrate_intrinsics()
    if "--intrinsics-only" in sys.argv:
        return
    print("== Part 2: ArUco measurement noise (analytic K vs calibrated K) ==")
    samples = Path(CALIB_FILE).parent / "marker_noise_samples.npz"
    if "--refit" in sys.argv and samples.exists():      # re-fit saved samples, no rendering
        z = np.load(samples); rows0, rows1 = z["analytic"], z["calibrated"]
    else:
        rows0 = characterise_markers(use_calibration=False)
        rows1 = characterise_markers(use_calibration=True)
        np.savez_compressed(samples, analytic=rows0, calibrated=rows1)
    st0, _ = fit_noise(rows0, "analytic K")
    st1, fits = fit_noise(rows1, "calibrated K")
    update_calib("marker_noise", {**fits, "model": "sigma = a + b * range", "ref_size": W.MARKER_SIZE,
                                  "pixel_noise_std": load_config().noise.pixel_noise_std})
    np.savetxt(Path(CALIB_FILE).parent / "marker_noise_stats.csv", st1, delimiter=",",
               header="range,det_rate,median_err_r,median_err_b,median_err_yaw,robust_std_r,robust_std_b,robust_std_yaw,yaw_flip_rate")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    labels = ["range error [m]", "bearing error [rad]", "relative-yaw error [rad]"]
    for k in range(3):
        for st, lab, c in ((st0, "analytic K", "tab:gray"), (st1, "calibrated K", "tab:blue")):
            ax[k].errorbar(st[:, 0], st[:, 2 + k], yerr=st[:, 5 + k], fmt="o-", capsize=3,
                           color=c, label=f"{lab}: median ± σ (robust / trimmed for yaw)")
        a, b = fits[["r_range", "r_bearing", "r_yaw"][k]]
        rr = np.linspace(0.3, fits["max_range"] if k < 2 else fits["yaw_max_range"], 50)
        centre = (fits["range_bias_poly"][0] * rr + fits["range_bias_poly"][1] * rr ** 2
                  if k == 0 else 0 * rr)
        if k == 0:
            ax[k].plot(rr, centre, "r--", lw=1, label="fitted bias (subtracted online)")
        if k < 2:
            ax[k].fill_between(rr, centre - (a + b * rr), centre + (a + b * rr), color="tab:orange",
                               alpha=0.25, label="EKF R model ±σ")
        else:
            for key, c_ in (("r_yaw_frontal", "tab:red"), ("r_yaw_oblique", "tab:green")):
                pts = np.array(fits[key + "_points"]); a_, b_ = fits[key]
                ax[k].plot(pts[:, 0], pts[:, 1], "^", color=c_, ms=4)
                ax[k].plot(rr, a_ + b_ * rr, "-", color=c_, label=f"σ model, {key.split('_')[-1]} view")
        ax[k].axhline(0, color="k", lw=0.5)
        ax[k].set_xlabel("true range [m]"); ax[k].set_ylabel(labels[k])
    ax[0].legend(fontsize=7)
    ax[2].axvline(fits["yaw_max_range"], color="r", ls=":", lw=1)
    ax[2].text(fits["yaw_max_range"], ax[2].get_ylim()[1] * 0.8, " yaw not used beyond", fontsize=7)
    ax[2].set_ylim(-0.5, 0.5)
    fig.suptitle("ArUco measurement error vs range (15 views x 24 noisy frames per range, "
                 "viewing angles up to 57 deg)")
    fig.tight_layout(); fig.savefig(FIG / "marker_noise_vs_range.png", dpi=120); plt.close(fig)
    print("wrote", CALIB_FILE)


if __name__ == "__main__":
    main()
