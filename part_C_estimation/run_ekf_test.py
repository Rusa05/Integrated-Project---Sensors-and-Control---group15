"""Phase 3 evaluation: EKF accuracy and consistency on a drive that loses and regains the marker.

The robot follows a fixed open-loop command schedule (no feedback, so no ground truth in the
loop). The EKF runs on noisy encoders (50 Hz) and noisy camera frames (10 Hz). Ground truth,
read only through eval/ground_truth.py, is used to score the estimate.

Monte Carlo: every run draws new sensor noise and a new initial-estimate error from P0, so the
average NEES over runs can be compared with chi-square bounds (Bar-Shalom consistency test).

Run from the project root:
    python part_C_estimation/run_ekf_test.py            # 20 runs + R-scaling study, ~25 min
    python part_C_estimation/run_ekf_test.py --quick    # 4 runs, no study
Writes figures and results/summary.json next to this script.
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
from matplotlib.patches import Ellipse
import mujoco
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm                     # noqa: E402
from src.ekf import EKF                                  # noqa: E402
from src.sensors import Sensors, NoiseConfig             # noqa: E402
from src.vision import MarkerDetector                    # noqa: E402
from eval import ground_truth as gt                      # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
STEPS_PER_TICK = round(rm.DT / 0.002)
START = (0.0, 0.0, 0.0)                                  # where the robot is placed (nominal start)
P0 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(3) ** 2])

# (v m/s, omega rad/s, seconds). Face the marker and approach; turn away and drive off on
# odometry alone (marker lost); turn back and approach to ~0.6 m.
SCHEDULE = [(0.0, -0.5, 1.0),     # face the marker (~29 deg right)
            (0.15, 0.0, 5.0),     # approach to ~1.4 m
            (0.0, 0.6, 5.24),     # turn 180 deg: marker lost
            (0.15, 0.0, 6.0),     # drive away 0.9 m on odometry only
            (0.0, 0.6, 5.24),     # turn 180 deg back towards the marker: reacquire
            (0.15, 0.0, 9.0),     # approach
            (0.12, 0.0, 3.0),     # slow final approach to ~0.6 m
            (0.0, 0.0, 1.0)]


def run_trial(seed, r_scale=1.0, slip_frac=None, min_update_dt=None):
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    gt.set_pose(model, data, *START)
    sens = Sensors(model, data, NoiseConfig(), seed=seed)
    det = MarkerDetector.from_sensors(sens)
    rng = np.random.default_rng(1000 + seed)
    x0 = np.array(START) + rng.multivariate_normal(np.zeros(3), P0)
    kw = {}
    if slip_frac is not None:
        kw['slip_frac'] = slip_frac
    if min_update_dt is not None:
        kw['min_update_dt'] = min_update_dt
    ekf = EKF(x0, P0, r_scale=r_scale, **kw)
    odo = EKF(x0, P0, **kw)                        # same filter, never updated = odometry only
    left, right = model.actuator('wheel_left').id, model.actuator('wheel_right').id
    enc_prev = sens.encoders()

    log = {k: [] for k in ['t', 'truth', 'est', 'P', 'odo', 'odoP', 'seen']}
    upd = {k: [] for k in ['t', 'nis', 'dof', 'used', 'range']}
    for v, w, dur in SCHEDULE:
        for _ in range(int(round(dur / rm.DT))):
            data.ctrl[[left, right]] = rm.body_to_wheels(v, w)
            mujoco.mj_step(model, data, nstep=STEPS_PER_TICK)
            enc = sens.encoders()
            ekf.predict(enc - enc_prev)
            odo.predict(enc - enc_prev)
            enc_prev = enc
            seen = False
            if sens.camera_due():
                for meas in det.measure_all(sens.camera()):
                    res = ekf.update(meas, data.time)
                    if res is not None:
                        seen = True
                        upd['t'].append(data.time); upd['nis'].append(res.nis); upd['dof'].append(res.dof)
                        upd['used'].append(res.used); upd['range'].append(meas.range)
            log['t'].append(data.time); log['truth'].append(gt.true_pose(model, data))
            log['est'].append(ekf.x.copy()); log['P'].append(ekf.P.copy())
            log['odo'].append(odo.x.copy()); log['odoP'].append(odo.P.copy()); log['seen'].append(seen)
    sens.close()
    log = {k: np.array(v) for k, v in log.items()}
    upd = {k: np.array(v) for k, v in upd.items()}
    return log, upd


def errors(est, truth):
    e = est - truth
    e[:, 2] = (e[:, 2] + math.pi) % (2 * math.pi) - math.pi
    return e


def nees(log, key='est', pkey='P'):
    e = errors(log[key], log['truth'])
    return np.einsum('ni,nij,nj->n', e, np.linalg.inv(log[pkey]), e)


def summarise(logs, upds):
    e = np.concatenate([errors(l['est'], l['truth']) for l in logs])
    eo = np.concatenate([errors(l['odo'], l['truth']) for l in logs])
    pos = np.hypot(e[:, 0], e[:, 1]); pos_o = np.hypot(eo[:, 0], eo[:, 1])
    final = np.array([np.hypot(*errors(l['est'], l['truth'])[-1, :2]) for l in logs])
    final_o = np.array([np.hypot(*errors(l['odo'], l['truth'])[-1, :2]) for l in logs])
    sig = np.sqrt(np.concatenate([np.diagonal(l['P'], axis1=1, axis2=2) for l in logs]))
    inside = np.mean(np.all(np.abs(e) <= 3 * sig, axis=1))
    N = len(logs)
    anees = np.mean([nees(l) for l in logs], axis=0)       # average over runs, per time step
    lo, hi = chi2.ppf([0.025, 0.975], 3 * N) / N
    used = np.concatenate([u['used'] for u in upds])
    nis = np.concatenate([u['nis'] for u in upds]); dof = np.concatenate([u['dof'] for u in upds])
    acc = used != 'rejected'
    nis3 = nis[(dof == 3) & acc]
    return {
        'runs': N,
        'pos_rmse_m': float(np.sqrt(np.mean(pos ** 2))), 'yaw_rmse_deg': float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2)))),
        'odo_pos_rmse_m': float(np.sqrt(np.mean(pos_o ** 2))), 'odo_yaw_rmse_deg': float(np.degrees(np.sqrt(np.mean(eo[:, 2] ** 2)))),
        'final_pos_err_mean_m': float(final.mean()), 'odo_final_pos_err_mean_m': float(final_o.mean()),
        'within_3sigma_frac': float(inside),
        'anees_mean': float(anees.mean()), 'anees_bounds_95': [float(lo), float(hi)],
        'anees_frac_in_bounds': float(np.mean((anees >= lo) & (anees <= hi))),
        'updates': int(len(used)), 'pose_updates': int(np.sum(used == 'pose')),
        'range_bearing_updates': int(np.sum(used == 'range_bearing')), 'rejected': int(np.sum(used == 'rejected')),
        'nis3_mean': float(nis3.mean()) if nis3.size else float('nan'),
        'nis3_frac_below_95': float(np.mean(nis3 < chi2.ppf(0.95, 3))) if nis3.size else float('nan'),
    }, anees, (lo, hi)


def plot_run(log, upd, path):
    t, tr, es, P, od = log['t'], log['truth'], log['est'], log['P'], log['odo']
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 2, (1, 3))
    ax.add_patch(plt.Rectangle((0.73, 0.45), 0.34, 0.4, color='orange', alpha=0.5))
    ax.add_patch(plt.Circle((-0.6, -0.8), 0.18, color='seagreen', alpha=0.5))
    ax.plot(*config.MARKER_XY, 'ks', ms=8, label='marker')
    ax.plot(tr[:, 0], tr[:, 1], 'k', lw=2, label='ground truth')
    ax.plot(od[:, 0], od[:, 1], ':', color='grey', label='odometry only')
    ax.plot(es[:, 0], es[:, 1], 'C0--', label='EKF')
    for k in range(0, len(t), 100):
        vals, vecs = np.linalg.eigh(P[k][:2, :2])
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(es[k, :2], *(2 * 3 * np.sqrt(vals[::-1])), angle=ang, fill=False, color='C0', lw=0.8))
    ax.set(aspect='equal', xlim=(-0.9, 2.05), ylim=(-1.4, 0.9), xlabel='x (m)', ylabel='y (m)',
           title='One run: EKF with 3-sigma ellipses (every 2 s)')
    ax.legend(fontsize=8, loc='upper left')
    e = errors(es, tr)
    sig = np.sqrt(np.diagonal(P, axis1=1, axis2=2))
    for i, (lab, scale, unit) in enumerate([('position x', 100, 'cm'), ('heading', 180 / math.pi, 'deg')]):
        a = fig.add_subplot(2, 2, 2 * (i + 1))
        idx = 0 if i == 0 else 2
        a.fill_between(t, -3 * sig[:, idx] * scale, 3 * sig[:, idx] * scale, color='C0', alpha=0.2, label='EKF ±3σ')
        a.plot(t, e[:, idx] * scale, 'C0', label='EKF error')
        a.plot(t, errors(log['odo'], tr)[:, idx] * scale, ':', color='grey', label='odometry error')
        seen = log['seen']
        a.scatter(t[seen], np.full(seen.sum(), a.get_ylim()[0]), s=4, color='green', label='marker update')
        a.set(xlabel='time (s)', ylabel=unit, title=f'{lab} error')
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_consistency(logs, upds, anees, bounds, path):
    t = logs[0]['t']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(t, anees, 'C0', lw=1, label=f'average NEES ({len(logs)} runs)')
    ax.axhline(3, color='k', lw=0.8, label='expected (3)')
    ax.axhspan(*bounds, color='C2', alpha=0.2, label='95 % bounds')
    seen = np.mean([l['seen'] for l in logs], axis=0) > 0.5
    ax.fill_between(t, 0, 0.3, where=seen, color='green', alpha=0.4, label='marker in view')
    ax.set(xlabel='time (s)', ylabel='NEES', yscale='log', title='Estimator consistency (NEES)')
    ax.legend(fontsize=7)
    ax = axes[1]
    nis = np.concatenate([u['nis'][(u['dof'] == 3) & (u['used'] != 'rejected')] for u in upds])
    rng = np.concatenate([u['range'][(u['dof'] == 3) & (u['used'] != 'rejected')] for u in upds])
    ax.scatter(rng, nis, s=4, alpha=0.4, label='NIS of accepted pose updates')
    ax.axhline(3, color='k', lw=0.8, label='expected (3)')
    ax.axhline(chi2.ppf(0.95, 3), color='C3', lw=0.8, ls='--', label='95 % (7.8)')
    ax.set(xlabel='marker range (m)', ylabel='NIS', yscale='log', title='Innovation consistency (NIS)')
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--runs', type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    N = 4 if args.quick else args.runs

    t0 = time.time()
    trials = [run_trial(seed) for seed in range(N)]
    logs, upds = zip(*trials)
    summary, anees, bounds = summarise(logs, upds)
    plot_run(logs[0], upds[0], OUT / 'ekf_run.png')
    plot_consistency(logs, upds, anees, bounds, OUT / 'ekf_consistency.png')
    print(f'{N} runs in {time.time() - t0:.0f} s')
    for k, v in summary.items():
        print(f'  {k:28s} {v}')
    result = {'nominal': summary}

    if not args.quick:
        study = {}
        for scale in [0.1, 1.0, 10.0]:
            n = max(6, N // 2)
            s, _, _ = summarise(*zip(*[run_trial(100 + seed, r_scale=scale) for seed in range(n)]))
            study[str(scale)] = s
            print(f'R x{scale:<4}: pos RMSE {s["pos_rmse_m"]*100:.2f} cm, yaw RMSE {s["yaw_rmse_deg"]:.2f} deg, '
                  f'ANEES {s["anees_mean"]:.2f} (bounds {s["anees_bounds_95"][0]:.2f}-{s["anees_bounds_95"][1]:.2f}), '
                  f'NIS {s["nis3_mean"]:.2f}, rejected {s["rejected"]}/{s["updates"]}')
        result['r_scaling'] = study
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        sc = [0.1, 1.0, 10.0]
        ax[0].plot(sc, [study[str(s)]['pos_rmse_m'] * 100 for s in sc], 'o-')
        ax[0].set(xscale='log', xlabel='R scale', ylabel='position RMSE (cm)', title='Accuracy vs R')
        ax[1].plot(sc, [study[str(s)]['anees_mean'] for s in sc], 'o-', label='ANEES')
        ax[1].plot(sc, [study[str(s)]['nis3_mean'] for s in sc], 's-', label='mean NIS (pose updates)')
        ax[1].axhline(3, color='k', lw=0.8)
        ax[1].set(xscale='log', yscale='log', xlabel='R scale', title='Consistency vs R (3 = ideal)')
        ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / 'r_scaling.png', dpi=130); plt.close(fig)

    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
