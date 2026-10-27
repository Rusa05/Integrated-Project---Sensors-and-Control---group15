"""Tests mapping, path planning and the LQR controller (Part D).

The robot starts from 3 different places, maps the room with its LiDAR, plans a path around the
obstacles with A* and follows it with the LQR to a point in front of the dock (no docking here).
Each start is run 3 times, then the LQR is tried with different weights to show the trade-off
between control effort and tracking error.
Saves plots and results/summary.json. Takes about 4 min. Press Run (add --quick for 1 run each).
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm                       # noqa: E402
from src.mapping import ROBOT_RADIUS                       # noqa: E402
from src.mission import Mission, PRE_DOCK                  # noqa: E402
from src.robot import SimRobot                             # noqa: E402
from eval import ground_truth as gt                        # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
P0 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(3) ** 2])

SCENARIOS = {
    'A_cylinder_in_way': (-1.5, -0.8, 0.0),
    'B_box_in_way': (0.5, 1.5, -math.pi / 2),
    'C_far_corner': (-1.5, 1.5, -0.6),
}

TIMEOUT = 120.0


def run(start, seed=0, lqr_r_scale=1.0, goal=PRE_DOCK, verbose=False):
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    gt.set_pose(model, data, *start)
    robot = SimRobot(model, data, seed=seed)
    rng = np.random.default_rng(2000 + seed)
    mission = Mission(np.array(start) + rng.multivariate_normal(np.zeros(3), P0), P0,
                      goal=goal, dock=False, lqr_r_scale=lqr_r_scale)
    log = {k: [] for k in ['t', 'truth', 'est', 'P', 'ref', 'u', 'u_ff', 'e_est', 'state']}
    collisions = set()
    t0 = data.time

    while data.time - t0 < TIMEOUT and not mission.finished:
        dphi, meas, scan = robot.sense()
        v, w = mission.step(robot.time, dphi, meas, scan)          # EKF estimate only
        robot.act(v, w)
        # ---- evaluation only -----------------------------------------------------------
        collisions |= gt.robot_collisions(model, data)
        log['t'].append(data.time); log['truth'].append(gt.true_pose(model, data))
        log['est'].append(mission.ekf.x.copy()); log['P'].append(mission.ekf.P.copy())
        log['ref'].append(mission.x_ref.copy()); log['u'].append((v, w)); log['u_ff'].append(mission.u_ff.copy())
        log['e_est'].append(mission.e.copy()); log['state'].append(mission.state)

    robot.close()
    state, replans, stops, plans = mission.state, mission.replans, mission.safety_stops, mission.plans
    grid = mission.grid
    log = {k: np.array(v) for k, v in log.items()}
    tracking = log['state'] == 'TRACK'
    e_true = np.array([rm.tracking_error(p, r) for p, r in zip(log['truth'], log['ref'])])
    fb = log['u'] - log['u_ff']
    final = log['truth'][-1]
    result = {
        'success': bool(state == 'DONE' and not collisions),
        'reached': bool(state == 'DONE'),
        'time_s': float(log['t'][-1] - t0),
        'collisions': sorted(collisions),
        'replans': replans, 'safety_stops': stops,
        'final_pos_err_true_m': float(np.hypot(*(final[:2] - np.array(goal[:2])))),
        'final_yaw_err_true_deg': float(math.degrees(rm.wrap(final[2] - goal[2]))),
        'final_est_err_m': float(np.hypot(*(log['est'][-1, :2] - final[:2]))),
        'track_rms_cross_m': float(np.sqrt(np.mean(e_true[tracking, 1] ** 2))) if tracking.any() else float('nan'),
        'track_rms_along_m': float(np.sqrt(np.mean(e_true[tracking, 0] ** 2))) if tracking.any() else float('nan'),
        'track_max_pos_m': float(np.max(np.hypot(e_true[tracking, 0], e_true[tracking, 1]))) if tracking.any() else float('nan'),
        'track_rms_yaw_deg': float(np.degrees(np.sqrt(np.mean(e_true[tracking, 2] ** 2)))) if tracking.any() else float('nan'),
        'ctrl_rms_cross_est_m': float(np.sqrt(np.mean(log['e_est'][tracking, 1] ** 2))) if tracking.any() else float('nan'),
        'ctrl_max_cross_est_m': float(np.max(np.abs(log['e_est'][tracking, 1]))) if tracking.any() else float('nan'),
        'effort_rms_v_fb': float(np.sqrt(np.mean(fb[tracking, 0] ** 2))) if tracking.any() else float('nan'),
        'effort_rms_w_fb': float(np.sqrt(np.mean(fb[tracking, 1] ** 2))) if tracking.any() else float('nan'),
        'ekf_rmse_m': float(np.sqrt(np.mean(np.sum((log['est'][:, :2] - log['truth'][:, :2]) ** 2, axis=1)))),
    }
    return result, log, grid, plans, e_true


def plot_run(name, start, log, grid, plans, e_true, path):
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)
    occ = 1 / (1 + np.exp(-grid.logodds))
    ext = [grid.origin[0], grid.origin[0] + grid.n * grid.res, grid.origin[1], grid.origin[1] + grid.n * grid.res]
    ax.imshow(occ.T, origin='lower', extent=ext, cmap='Greys', vmin=0, vmax=1)
    ax.contour(np.linspace(ext[0], ext[1], grid.n), np.linspace(ext[2], ext[3], grid.n), grid.distance().T,
               levels=[ROBOT_RADIUS], colors='C3', linewidths=0.6)
    for i, p in enumerate(plans):
        ax.plot(p.x, p.y, '--', color='C1', lw=1, alpha=0.5 if i < len(plans) - 1 else 1,
                label='planned path' if i == len(plans) - 1 else None)
    ax.plot(log['truth'][:, 0], log['truth'][:, 1], 'k', lw=1.5, label='ground truth')
    ax.plot(log['est'][:, 0], log['est'][:, 1], 'C0:', lw=1.5, label='EKF estimate')
    ax.plot(*start[:2], 'go', label='start'); ax.plot(*PRE_DOCK[:2], 'r*', ms=12, label='pre-dock goal')
    ax.plot(*config.MARKER_XY, 'ks', ms=6)
    ax.set(aspect='equal', xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), title=f'{name}: LiDAR map (grey), inflated edge (red)')
    ax.legend(fontsize=7, loc='upper left')
    t = log['t'] - log['t'][0]
    tr = log['state'] == 'TRACK'
    a1 = fig.add_subplot(2, 2, 2)
    a1.plot(t, np.where(tr, e_true[:, 1] * 100, np.nan), 'C0', label='cross-track: plan vs TRUE pose')
    a1.plot(t, np.where(tr, log['e_est'][:, 1] * 100, np.nan), 'C0--', label='cross-track: plan vs EKF (controller sees)')
    est_err = np.hypot(*(log['est'][:, :2] - log['truth'][:, :2]).T)
    a1.plot(t, est_err * 100, 'C2:', label='EKF position error')
    a1.set(ylabel='cm', title='Control error vs estimation error')
    a1.legend(fontsize=7)
    a2 = fig.add_subplot(2, 2, 4)
    a2.plot(t, log['u'][:, 0], label='v (m/s)')
    a2.plot(t, log['u'][:, 1], label='omega (rad/s)')
    a2.set(xlabel='time (s)', title='Commands')
    a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    print('Testing navigation from 3 starts x 3 seeds plus the LQR study, about 4 min (no window)...', flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = 1 if args.quick else args.seeds
    results = {}
    t0 = time.time()
    for name, start in SCENARIOS.items():
        results[name] = []
        for seed in range(seeds):
            res, log, grid, plans, e_true = run(start, seed)
            results[name].append(res)
            if seed == 0:
                plot_run(name, start, log, grid, plans, e_true, OUT / f'nav_{name}.png')
            print(f'{name} seed {seed}: {"OK " if res["success"] else "FAIL"} {res["time_s"]:5.1f}s, '
                  f'final {res["final_pos_err_true_m"]*100:.1f} cm / {res["final_yaw_err_true_deg"]:.1f} deg, '
                  f'cross-track true RMS {res["track_rms_cross_m"]*100:.1f} cm (max {res["track_max_pos_m"]*100:.1f}), '
                  f'vs EKF {res["ctrl_rms_cross_est_m"]*100:.1f} cm, EKF err {res["ekf_rmse_m"]*100:.1f} cm, '
                  f'replans {res["replans"]}, stops {res["safety_stops"]}, collisions {res["collisions"]}', flush=True)
    out = {'scenarios': results}
    if not args.quick:
        study = {}
        for scale in [0.1, 1.0, 10.0]:
            rs = [run(SCENARIOS['A_cylinder_in_way'], seed, lqr_r_scale=scale)[0] for seed in range(seeds)]
            study[str(scale)] = rs
            m = lambda k: float(np.mean([r[k] for r in rs]))
            print(f'LQR R x{scale:<4}: cross-track RMS {m("track_rms_cross_m")*100:.2f} cm, yaw RMS {m("track_rms_yaw_deg"):.2f} deg, '
                  f'effort v_fb {m("effort_rms_v_fb"):.3f} m/s, w_fb {m("effort_rms_w_fb"):.3f} rad/s, '
                  f'time {m("time_s"):.1f} s, success {sum(r["success"] for r in rs)}/{len(rs)}', flush=True)
        out['lqr_r_scaling'] = study
        sc = [0.1, 1.0, 10.0]
        mean = lambda s, k: np.mean([r[k] for r in study[str(s)]])
        fig, ax = plt.subplots(figsize=(5.5, 4))
        xs = [mean(s, 'effort_rms_w_fb') for s in sc]; ys = [mean(s, 'track_rms_cross_m') * 100 for s in sc]
        ax.plot(xs, ys, 'o-')
        for s, x_, y_ in zip(sc, xs, ys):
            ax.annotate(f'R x{s}', (x_, y_), textcoords='offset points', xytext=(5, 5))
        ax.set(xlabel='feedback effort, RMS omega correction (rad/s)', ylabel='cross-track RMS (cm)',
               title='LQR trade-off: control effort vs tracking error')
        fig.tight_layout(); fig.savefig(OUT / 'lqr_tradeoff.png', dpi=130); plt.close(fig)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2))
    print(f'done in {time.time() - t0:.0f} s; wrote {OUT}')


if __name__ == '__main__':
    main()
