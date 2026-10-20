"""Step 2 - Odometry calibration: effective wheel radius, turning gain and odometry noise.

  1. Steady-state straight runs at 12 wheel speeds (forward and reverse)
       -> lookup table r_eff(w_mean)   (slip makes the effective radius speed dependent)
  2. In-place turns: 6 steady spins + 24 turns made with the heading controller (as used in
     the mission) -> turning gain g = dtheta / (dphi_r - dphi_l)  (equivalent b = r/g)
     and the instantaneous centre of rotation (ICR) offset x_icr from the true displacement
  3. Residuals of 24 fresh turns -> rotation noise k_rot_th, k_rot_xy
     Random drives split into 0.25 m segments -> odometry noise k_s (sigma ~ sqrt|ds|)
  4. UMBmark-style 1 m square, CW and CCW, with datasheet vs calibrated model -> validation

The reference displacement comes from the simulator's true pose, playing the role of a tape
measure / overhead tracker in an offline calibration session. It is never used online.

Run:  python scripts/02_calibrate_odometry.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tb3nav import world as W
from tb3nav.config import CALIB_FILE, load_config
from tb3nav.frames import wrap
from tb3nav.kinematics import DiffDrive
from tb3nav.sim import Simulator

FIG = Path(__file__).resolve().parent.parent / "results" / "figures"


def open_space_sim(seed, start=(3.0, 2.0, 0.0)):
    cfg = load_config(use_calibration=False)
    spec = W.WorldSpec(W.default_landmarks(), [], start=start)
    return Simulator(spec, cfg, seed=seed, render_camera=False), cfg


def wheel_angles(sim, pkt):
    return pkt.enc_ticks * 2 * np.pi / sim.cfg.robot.ticks_per_rev


def track_heading(sim, T, dt=0.05):
    """Step for T seconds, accumulating the unwrapped true heading change."""
    th_prev, dth = sim.ground_truth_for_evaluation()[2], 0.0
    pkt = None
    for _ in range(int(round(T / dt))):
        pkt = sim.step(dt)
        th = sim.ground_truth_for_evaluation()[2]
        dth += wrap(th - th_prev); th_prev = th
    return pkt, dth


def controller_turns(cfg, n, seed0):
    """In-place turns of 0.3-2 rad with the mission's heading controller. Returns rows
    [dth_true, dphi_r - dphi_l, p0x, p0y, th0, p1x, p1y]."""
    from tb3nav.controller import heading_controller
    dd = DiffDrive(cfg)
    rng = np.random.default_rng(seed0)
    rows = []
    for k in range(n):
        sim, _ = open_space_sim(seed0 + k, start=(3, 2, rng.uniform(-3, 3)))
        pkt = sim.step(0.3)
        a0 = wheel_angles(sim, pkt)
        p0 = sim.ground_truth_for_evaluation()
        odo = p0.copy()                      # the controller only needs a heading estimate
        target = odo[2] + rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)
        prev, dth_true, th_prev = a0, 0.0, p0[2]
        for _ in range(200):
            v, w, err = heading_controller(odo, target)
            sim.set_wheel_speeds(*dd.inverse(v, w))
            pkt = sim.step(0.05)
            a = wheel_angles(sim, pkt)
            odo = dd.integrate(odo, *(a - prev), 0.05); prev = a
            th = sim.ground_truth_for_evaluation()[2]
            dth_true += wrap(th - th_prev); th_prev = th
            if abs(err) < 0.01:
                break
        sim.set_wheel_speeds(0, 0)
        pkt = sim.step(0.3)
        p1 = sim.ground_truth_for_evaluation()
        dth_true += wrap(p1[2] - th_prev)
        dphi = wheel_angles(sim, pkt) - a0
        rows.append([dth_true, dphi[1] - dphi[0], p0[0], p0[1], p0[2], p1[0], p1[1]])
        sim.close()
    return np.array(rows)


def icr_from_turns(turns):
    """ICR c (world) from p1 - c = R(dth)(p0 - c); returned in the body frame at p0."""
    out = []
    for dth, _, x0, y0, th0, x1, y1 in turns:
        if abs(dth) < 0.3:
            continue
        R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
        c = np.linalg.solve(np.eye(2) - R, np.array([x1, y1]) - R @ np.array([x0, y0]))
        d = c - np.array([x0, y0])
        out.append([np.cos(th0) * d[0] + np.sin(th0) * d[1], -np.sin(th0) * d[0] + np.cos(th0) * d[1]])
    return np.array(out)


def calibrate(seed0=0):
    speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    table = []
    seed = seed0
    for w in [-6.0, -4.0, -2.0, -1.0, -0.5] + speeds:
        sim, _ = open_space_sim(seed); seed += 1
        sim.set_wheel_speeds(w, w)
        pkt = sim.step(1.5)                                # let speed and slip settle
        a0, p0 = wheel_angles(sim, pkt).mean(), sim.ground_truth_for_evaluation()
        T = min(5.0, 1.2 / (abs(w) * 0.03))                # <= ~1.2 m of travel
        pkt = sim.step(T)
        a1, p1 = wheel_angles(sim, pkt).mean(), sim.ground_truth_for_evaluation()
        d = np.hypot(*(p1[:2] - p0[:2]))
        table.append([w, d / abs(a1 - a0)])
    table = np.array(table)
    spins = []
    for w in (1.0, 2.0, 3.0):
        for sgn in (1, -1):
            sim, _ = open_space_sim(seed); seed += 1
            sim.set_wheel_speeds(-sgn * w, sgn * w)
            pkt = sim.step(1.0)
            a0 = wheel_angles(sim, pkt)
            pkt, dth = track_heading(sim, 6.0)
            a1 = wheel_angles(sim, pkt)
            dphi = a1 - a0
            spins.append([w * sgn, dth, dphi[1] - dphi[0]])
    spins = np.array(spins)
    g_steady = float(spins[:, 1] @ spins[:, 2] / (spins[:, 2] @ spins[:, 2]))
    turns = controller_turns(load_config(use_calibration=False), 24, seed0=500)
    x = np.concatenate([spins[:, 2], turns[:, 1]]); y = np.concatenate([spins[:, 1], turns[:, 0]])
    g = float(x @ y / (x @ x))                                   # LS through origin, pooled
    icr = icr_from_turns(turns)
    # Turn gain while rolling: steady arcs at several forward speeds, both turn directions
    cfg_t = load_config(use_calibration=False)
    cfg_t.robot.radius_table = (table[:, 0].tolist(), table[:, 1].tolist())
    cfg_t.robot.turn_gain = g
    dd_t = DiffDrive(cfg_t)
    gtab = [[0.0, g]]
    for v in (0.03, 0.06, 0.10, 0.14):
        gs = []
        for w in (0.3, -0.3, 0.6, -0.6):
            sim, _ = open_space_sim(seed); seed += 1
            sim.set_wheel_speeds(*dd_t.inverse(v, w))
            pkt = sim.step(1.0)
            a0 = wheel_angles(sim, pkt)
            pkt, dth = track_heading(sim, 3.0)
            dphi = wheel_angles(sim, pkt) - a0
            gs.append(dth / (dphi[1] - dphi[0]))
        w_mean = float(np.interp(v, dd_t._vv, dd_t._ww))
        gtab.append([w_mean, float(np.mean(gs))])
    return table, g, g_steady, spins, turns, icr, np.array(gtab)


def odo_step(dd, dphi, dt, pose):
    ds, _, dth = dd.forward3(dphi[0], dphi[1], dt)
    return dd.integrate(pose, dphi[0], dphi[1], dt), ds, dth


def rotation_noise(cfg, n=24):
    """Residual heading / position error of the calibrated model over in-place turns."""
    dd = DiffDrive(cfg)
    turns = controller_turns(cfg, n, seed0=900)
    eth, exy, a = [], [], []
    for dth, dphi_d, x0, y0, th0, x1, y1 in turns:
        pred = dd.integrate(np.array([x0, y0, th0]), -dphi_d / 2, dphi_d / 2, 1.0)
        eth.append(wrap(pred[2] - (th0 + dth))); exy.append(np.hypot(pred[0] - x1, pred[1] - y1))
        a.append(abs(dth))
    eth, exy, a = map(np.array, (eth, exy, a))
    return float(np.sqrt(np.mean(eth ** 2 / a))), float(np.sqrt(np.mean(exy ** 2 / a) / 2)), eth, exy, a


def noise_study(cfg, n_runs=8, seg_len=0.25):
    """Random speed drives; compare calibrated odometry with truth over short segments."""
    dd = DiffDrive(cfg)
    errs = []
    for k in range(n_runs):
        sim, _ = open_space_sim(100 + k)
        rng = np.random.default_rng(100 + k)
        pkt = sim.step(0.5)
        prev_a = wheel_angles(sim, pkt)
        prev_p = sim.ground_truth_for_evaluation()
        acc_s, odo = 0.0, np.zeros(3)
        for i in range(400):
            if i % 20 == 0:
                v, w = rng.uniform(0.04, 0.15), rng.uniform(-0.6, 0.6)
                sim.set_wheel_speeds(*dd.inverse(v, w))
            pkt = sim.step(0.05)
            a = wheel_angles(sim, pkt)
            odo, ds, _ = odo_step(dd, a - prev_a, 0.05, odo); prev_a = a
            acc_s += abs(ds)
            if acc_s >= seg_len:
                p = sim.ground_truth_for_evaluation()
                c, s = np.cos(prev_p[2]), np.sin(prev_p[2])
                d = p[:2] - prev_p[:2]
                true_seg = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1],
                                     wrap(p[2] - prev_p[2])])
                e = odo - true_seg; e[2] = wrap(e[2])
                errs.append([acc_s, *e])
                prev_p, acc_s, odo = p, 0.0, np.zeros(3)
    errs = np.array(errs)
    # Along-track: var = (sig_l^2 + sig_r^2)/4 = k^2 |ds| / 2  ->  k = sqrt(2 E[e^2/|ds|])
    k_along = float(np.sqrt(2 * np.mean(errs[:, 1] ** 2 / errs[:, 0])))
    # Heading:     var = (sig_l^2 + sig_r^2)/b^2 = 2 k^2 |ds| / b^2
    b = dd.b_equiv
    k_head = float(np.sqrt(np.mean(errs[:, 3] ** 2 / errs[:, 0]) * b ** 2 / 2))
    return errs, k_along, k_head


def square_test(cfg, direction, side=1.0, seed=0):
    """Drive a square using odometry feedback only; return true end-point error (UMBmark)."""
    dd = DiffDrive(cfg)
    sim, _ = open_space_sim(seed, start=(2.5, 2.0 - direction * 0.5, 0.0))
    pkt = sim.step(0.5)
    prev = wheel_angles(sim, pkt)
    p_start = sim.ground_truth_for_evaluation()
    odo, path, dt = np.zeros(3), [], 0.02
    for leg in range(4):
        target = odo[:2] + side * np.array([np.cos(odo[2]), np.sin(odo[2])])
        while np.hypot(*(target - odo[:2])) > 0.005:
            sim.set_wheel_speeds(*dd.inverse(0.1, 0.0))
            pkt = sim.step(dt)
            a = wheel_angles(sim, pkt); odo, _, _ = odo_step(dd, a - prev, dt, odo); prev = a
            path.append(sim.ground_truth_for_evaluation())
        th_target = odo[2] + direction * np.pi / 2
        while abs(wrap(th_target - odo[2])) > 0.003:
            w = np.clip(2.0 * wrap(th_target - odo[2]), -0.6, 0.6)
            w = np.sign(w) * max(abs(w), 0.08)
            sim.set_wheel_speeds(*dd.inverse(0.0, w))
            pkt = sim.step(dt)
            a = wheel_angles(sim, pkt); odo, _, _ = odo_step(dd, a - prev, dt, odo); prev = a
            path.append(sim.ground_truth_for_evaluation())
    sim.set_wheel_speeds(0, 0); sim.step(0.5)
    p_end = sim.ground_truth_for_evaluation()
    return np.hypot(*(p_end[:2] - p_start[:2])), wrap(p_end[2] - p_start[2]), np.array(path)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cfg0 = load_config(use_calibration=False)
    r0, b0 = cfg0.robot.wheel_radius, cfg0.robot.wheel_base
    table, g, g_steady, spins, turns, icr, gtab = calibrate()
    cfg1 = copy.deepcopy(cfg0)
    cfg1.robot.radius_table = (table[:, 0].tolist(), table[:, 1].tolist())
    cfg1.robot.turn_gain = g
    cfg1.robot.turn_table = (gtab[:, 0].tolist(), gtab[:, 1].tolist())
    x_icr = float(np.median(icr[:, 0]))
    cfg1.robot.icr_x = x_icr
    dd1 = DiffDrive(cfg1)
    cruise = float(dd1.r(5.0))
    print(f"datasheet: r = {r0:.4f} m, b = {b0:.4f} m, g = r/b = {r0/b0:.5f}")
    print("r_eff(w_mean):", dict(zip(table[:, 0].round(1), table[:, 1].round(5))))
    print(f"turn gain: steady spins {g_steady:.5f}, pooled with controller turns g = {g:.5f}  "
          f"(equivalent b at cruise r: {cruise/g:.4f} m, {100*(cruise/g/b0-1):+.1f} % vs datasheet)")
    print("turn gain vs |w_mean| (rolling):", dict(zip(gtab[:, 0].round(2), gtab[:, 1].round(5))))
    print(f"ICR offset (body frame, median of {len(icr)} turns): x = {1000*x_icr:+.1f} mm "
          f"(IQR {1000*np.percentile(icr[:,0],25):+.1f}..{1000*np.percentile(icr[:,0],75):+.1f}),"
          f" y = {1000*np.median(icr[:,1]):+.1f} mm")
    k_rot_th, k_rot_xy, eth, exy, arot = rotation_noise(cfg1)
    cfg_noicr = copy.deepcopy(cfg1); cfg_noicr.robot.icr_x = 0.0
    _, k_xy_noicr, _, exy0, _ = rotation_noise(cfg_noicr)
    print(f"turn residuals (24 fresh turns): k_rot_th = {k_rot_th:.4f} rad^0.5, "
          f"k_rot_xy = {k_rot_xy:.4f} m/rad^0.5 (without ICR model: {k_xy_noicr:.4f}); "
          f"rms position error per turn {100*np.sqrt(np.mean(exy**2)):.2f} cm vs "
          f"{100*np.sqrt(np.mean(exy0**2)):.2f} cm without ICR")
    errs, k_al, k_hd = noise_study(cfg1)
    print(f"odometry noise from {len(errs)} segments: k_s along = {k_al:.4f}, "
          f"from heading = {k_hd:.4f} m^0.5")
    res, paths = {}, {}
    for label, c in (("datasheet", cfg0), ("calibrated", cfg1)):
        for d, name in ((1, "CCW"), (-1, "CW")):
            e_pos, e_th, path = square_test(c, d)
            res[f"{label}_{name}"] = {"pos_err_m": float(e_pos), "heading_err_rad": float(e_th)}
            paths[(label, name)] = path
            print(f"  1 m square {name:3s} with {label:10s} model: end error {100*e_pos:5.1f} cm, "
                  f"{np.degrees(e_th):+6.1f} deg")
    # EKF translation noise: along-track estimate (the heading part of segment errors is now
    # covered by the rotation terms) with a x1.2 margin. Final scaling is checked by the NEES
    # sweep in scripts/06_tradeoffs.py.
    k_s_ekf = 1.2 * k_al
    c = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
    c["odometry"] = {"radius_table": [table[:, 0].tolist(), table[:, 1].tolist()],
                     "turn_gain": g, "turn_gain_steady_spins": g_steady,
                     "turn_table": [gtab[:, 0].tolist(), gtab[:, 1].tolist()],
                     "icr_x": x_icr, "icr_samples": icr.tolist(),
                     "k_rot_th": 1.2 * k_rot_th, "k_rot_xy": 1.2 * k_rot_xy,
                     "k_rot_th_measured": k_rot_th, "k_rot_xy_measured": k_rot_xy,
                     "wheel_radius": cruise, "wheel_base": cruise / g,
                     "k_s": k_s_ekf, "k_s_measured_along": k_al, "k_s_measured_heading": k_hd,
                     "datasheet": {"wheel_radius": r0, "wheel_base": b0},
                     "umbmark_square_1m": res, "spin_runs": spins.tolist()}
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(c, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(table[:, 0], 1000 * table[:, 1], "o-", label="calibrated r_eff")
    ax[0].axhline(1000 * r0, color="k", ls="--", label="datasheet 33 mm")
    ax[0].set_xlabel("mean wheel speed [rad/s] (negative = reverse)")
    ax[0].set_ylabel("effective radius [mm]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Speed-dependent effective wheel radius")
    ax[1].hist(errs[:, 1] / np.sqrt(errs[:, 0]), 25, alpha=0.7, label="along-track")
    ax[1].hist(errs[:, 2] / np.sqrt(errs[:, 0]), 25, alpha=0.7, label="cross-track")
    ax[1].set_xlabel("segment error / sqrt(segment length) [m^0.5]"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Calibrated odometry residuals, {len(errs)} x 0.25 m segments")
    for (label, name), p in paths.items():
        ax[2].plot(p[:, 0], p[:, 1], ls="-" if label == "calibrated" else ":",
                   label=f"{label} {name}")
    ax[2].set_aspect("equal"); ax[2].legend(fontsize=7)
    ax[2].set_title("1 m square driven on odometry (true paths)")
    ax[2].set_xlabel("x [m]"); ax[2].set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(FIG / "odometry_calibration.png", dpi=120); plt.close(fig)
    print("wrote", CALIB_FILE)


if __name__ == "__main__":
    main()
