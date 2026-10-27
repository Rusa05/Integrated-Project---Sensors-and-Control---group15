"""How noisy are the LiDAR and the camera's tag measurements? (Part B)

Puts the robot at lots of distances and angles from the 10 cm dock tag, measures range, bearing
and tag yaw with the camera, and compares them with the truth. Fits the range bias and the noise
curves the EKF uses for R. Also checks the LiDAR noise.
Saves plots and results/summary.json. Takes about 45 s. Press Run.
"""
from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm                                   # noqa: E402
from src.sensors import Sensors, NoiseConfig                          # noqa: E402
from src.vision import MarkerDetector                                 # noqa: E402
from eval import ground_truth as gt                                   # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
STEPS_PER_TICK = round(rm.DT / 0.002)


def load():
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def drive(model, data, v, omega):
    data.ctrl[[model.actuator('wheel_left').id, model.actuator('wheel_right').id]] = rm.body_to_wheels(v, omega)
    mujoco.mj_step(model, data, nstep=STEPS_PER_TICK)


# --------------------------------------------------------------------------
def lidar_noise():
    model, data = load()
    gt.set_pose(model, data, 0.0, 0.0, 0.0)
    ideal = Sensors(model, data, NoiseConfig.ideal()).lidar()
    noisy = Sensors(model, data, NoiseConfig(), seed=2)
    scans = np.array([noisy.lidar() for _ in range(200)])
    ok = np.isfinite(scans) & np.isfinite(ideal)
    err = np.where(ok, scans - ideal, np.nan)
    std = np.nanstd(err, axis=0)
    dropout = float(np.mean(~np.isfinite(scans[:, np.isfinite(ideal)])))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = np.argsort(ideal)
    ax.plot(ideal[order], std[order] * 1000, 'o', ms=3, label='measured std')
    rr = np.linspace(0, 3.5, 100)
    ax.plot(rr, 1000 * np.maximum(0.0075, 0.025 * rr), '--', label='LDS-01 based model')
    ax.set(xlabel='true range (m)', ylabel='std (mm)', title='LiDAR range noise (200 scans at origin)')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'lidar_noise.png', dpi=130); plt.close(fig)
    print(f'\n== LiDAR: dropout {dropout*100:.1f}%, mean |bias| {np.nanmean(np.abs(np.nanmean(err, axis=0)))*1000:.1f} mm ==')
    return {'dropout_rate': dropout}


# --------------------------------------------------------------------------
def marker_noise(frames_per_pose=8):
    """Measure range/bearing/yaw error of the marker measurement over a grid of poses."""
    model, data = load()
    sens = Sensors(model, data, NoiseConfig(), seed=3)
    det = MarkerDetector.from_sensors(sens)
    m_true = None
    rows = []
    obstacles = [((0.9, 0.65), 0.45), ((-0.6, -0.8), 0.4)]   # obstacle + robot footprint
    for d in np.arange(0.3, 2.81, 0.2):
        for view in np.radians([-40, -20, 0, 20, 40]):         # where the robot stands, seen from the marker
            for off in np.radians([-20, 0, 20]):               # where the marker sits in the image
                x = config.MARKER_XY[0] + d * math.cos(config.MARKER_YAW + view)
                y = config.MARKER_XY[1] + d * math.sin(config.MARKER_YAW + view)
                if not (-1.75 < x < 1.75 and -1.75 < y < 1.75):
                    continue
                if any(math.hypot(x - ox, y - oy) < rad for (ox, oy), rad in obstacles):
                    continue
                theta = math.atan2(config.MARKER_XY[1] - y, config.MARKER_XY[0] - x) - off
                gt.set_pose(model, data, x, y, theta)
                m_true = gt.true_marker_xy(data)
                pose = gt.true_pose(model, data)
                r_t, b_t = rm.h_range_bearing(pose, m_true)
                yaw_t = rm.wrap(config.MARKER_YAW - pose[2])
                for _ in range(frames_per_pose):
                    meas = det.measure(sens.camera())
                    if meas is None:
                        rows.append([r_t, b_t, yaw_t, view, np.nan, np.nan, np.nan, np.nan])
                    else:
                        rows.append([r_t, b_t, yaw_t, view, meas.range - r_t, rm.wrap(meas.bearing - b_t),
                                     rm.wrap(meas.rel_yaw - yaw_t), meas.side_px])
    sens.close()
    a = np.array(rows)
    r_t, view, e_r, e_b, e_y = a[:, 0], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    found = np.isfinite(e_r)

    # Range bias calibration: e_r = k * r^2 fits a corner-shrink bias (fewer pixels further away)
    rf, ef = r_t[found], e_r[found]
    k_bias = float(np.sum(ef * rf ** 2) / np.sum(rf ** 4))
    resid = ef - k_bias * rf ** 2

    # bin by true range
    bins = np.arange(0.2, 3.01, 0.2)
    idx = np.digitize(r_t, bins)
    centres, det_rate, r_mean, r_std, b_std, y_std = [], [], [], [], [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() < 5:
            continue
        f = sel & found
        centres.append((bins[k - 1] + bins[k]) / 2)
        det_rate.append(f.sum() / sel.sum())
        r_mean.append(np.nanmean(e_r[f]) if f.any() else np.nan)
        r_std.append(np.nanstd(e_r[f] - k_bias * r_t[f] ** 2) if f.sum() > 2 else np.nan)
        b_std.append(np.nanstd(e_b[f]) if f.sum() > 2 else np.nan)
        y_std.append(np.nanstd(e_y[f]) if f.sum() > 2 else np.nan)
    centres = np.array(centres)

    # Noise model after bias correction: sigma(r) = s0 + s1 r^2 fitted to binned std
    idx_f = idx[found]

    def fit_sigma(err):
        """Power law sigma(r) = a * r^p fitted in log space to the per-bin std; returns [a, p]."""
        stds, rs = [], []
        for k in range(1, len(bins)):
            sel = idx_f == k
            if sel.sum() > 5:
                stds.append(np.std(err[sel]))
                rs.append((bins[k - 1] + bins[k]) / 2)
        p_, log_a = np.polyfit(np.log(rs), np.log(np.maximum(stds, 1e-9)), 1)
        return [float(np.exp(log_a)), float(p_)]

    sig_r = fit_sigma(resid)
    sig_b = fit_sigma(e_b[found])
    sig_y = fit_sigma(e_y[found])
    yaw_outliers = float(np.mean(np.abs(e_y[found]) > math.radians(20)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.scatter(rf, ef * 100, s=4, alpha=0.4, label='samples')
    rr = np.linspace(0.2, 3, 100)
    ax.plot(rr, k_bias * rr ** 2 * 100, 'r', label=f'bias fit {k_bias*100:.2f} r^2 cm')
    ax.set(xlabel='true range (m)', ylabel='range error (cm)', title='Range error (before correction)')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(centres, np.degrees(b_std), 'o-', label='bearing std')
    ax.plot(centres, np.degrees(y_std), 's-', label='marker-yaw std')
    ax.set(xlabel='true range (m)', ylabel='deg', title='Angular noise vs range', yscale='log')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(centres, np.array(r_std) * 100, 'o', label='measured std (after bias corr.)')
    ax.plot(rr, sig_r[0] * rr ** sig_r[1] * 100, '--', label=f'fit {sig_r[0]*100:.2f} r^{sig_r[1]:.1f} cm')
    ax.set(xlabel='true range (m)', ylabel='cm', title='Range noise model for EKF R', yscale='log')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(centres, np.array(det_rate) * 100, 'o-')
    ax.set(xlabel='true range (m)', ylabel='% frames', ylim=(0, 105), title='Detection rate')
    fig.tight_layout(); fig.savefig(OUT / 'marker_noise.png', dpi=130); plt.close(fig)

    print('\n== Marker measurement ==')
    print(f'  {len(a)} frames, detected {found.mean()*100:.1f}%; max detected range {rf.max():.2f} m')
    print(f'  range bias e_r = {k_bias:.4f} r^2  (e.g. {k_bias*1.5**2*100:.1f} cm at 1.5 m)')
    print(f'  sigma_range  = {sig_r[0]*1000:.2f} mm * r^{sig_r[1]:.2f}')
    print(f'  sigma_bearing= {math.degrees(sig_b[0]):.4f} deg * r^{sig_b[1]:.2f}')
    print(f'  sigma_yaw    = {math.degrees(sig_y[0]):.3f} deg * r^{sig_y[1]:.2f};'
          f' |yaw err| > 20 deg in {yaw_outliers*100:.1f}% of frames (IPPE flip)')
    return {'frames': int(len(a)), 'detection_rate': float(found.mean()), 'max_detected_range_m': float(rf.max()),
            'noise_model_form': 'sigma(r) = a * r**p, stored as [a, p], r in m', 'range_bias_k_per_m2': k_bias, 'sigma_range_m': sig_r, 'sigma_bearing_rad': sig_b,
            'sigma_yaw_rad': sig_y, 'yaw_outlier_rate': yaw_outliers,
            'binned': {'range_m': centres.tolist(), 'detection_rate': det_rate,
                       'range_bias_m': r_mean, 'range_std_m': r_std, 'bearing_std_rad': b_std,
                       'yaw_std_rad': y_std}}


def main():
    print('Measuring LiDAR and tag noise (about 45 s)...', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'lidar': lidar_noise(), 'marker': marker_noise()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f'\nWrote figures and summary.json to {OUT}')


if __name__ == '__main__':
    main()
