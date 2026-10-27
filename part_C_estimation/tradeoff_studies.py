"""Part C: two estimator studies on the full mission (random layouts from Part E).

  Study A  consistency vs responsiveness: scale the process noise (wheel-slip allowance) x0.3 ... x10
           with the robot pushed part-way through. Small Q = confident filter that is slow to accept
           the push; large Q = quick recovery but a pessimistic, noisier estimate in normal driving.
  Study D  which measurements matter: full pose (range, bearing, tag yaw) vs range + bearing only vs
           odometry only (camera ignored), nominal conditions.

    python part_C_estimation/tradeoff_studies.py               (~6 min on 4 cores)
    python part_C_estimation/tradeoff_studies.py --trials 3    (quicker)

Ground truth is used only to score (through the Part E harness).
"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'part_E_integration_and_robustness'))
sys.path.insert(0, str(ROOT))
import run_trials                              # noqa: E402  (Part E harness)
from src.ekf import SLIP_FRAC                   # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
Q_SCALES = [0.3, 1.0, 3.0, 10.0]
MODES = [('pose', 'range + bearing + tag yaw'), ('range_bearing', 'range + bearing only'),
         ('odometry', 'odometry only (no camera)')]


def mean(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    return float(np.mean(v)) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=6)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [('push', i, False, None, {'slip_frac': SLIP_FRAC * q}) for q in Q_SCALES for i in range(args.trials)]
    jobs += [('nominal', i, False, None, {'mode': m}) for m, _ in MODES for i in range(args.trials)]
    t0 = time.time()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        results = pool.map(run_trials.run_trial, jobs)
    for r in results:
        r.pop('traj', None); r.pop('debug', None)

    study_a = {}
    for q in Q_SCALES:
        rs = [r for r in results if r['condition'] == 'push' and r['ekf_options']['slip_frac'] == SLIP_FRAC * q]
        study_a[q] = {'success': sum(r['success'] for r in rs), 'n': len(rs), 'mean_nees': mean(rs, 'mean_nees'),
                      'max_nees': mean(rs, 'max_nees'), 'push_recovery_s': mean(rs, 'push_recovery_s'),
                      'ekf_rmse_cm': 100 * mean(rs, 'ekf_rmse_m'), 'dock_err_cm': 100 * mean(rs, 'dock_pos_err_m')}
    study_d = {}
    for m, label in MODES:
        rs = [r for r in results if r['condition'] == 'nominal' and r['ekf_options']['mode'] == m]
        study_d[m] = {'label': label, 'success': sum(r['success'] for r in rs), 'n': len(rs),
                      'ekf_rmse_cm': 100 * mean(rs, 'ekf_rmse_m'), 'dock_err_cm': 100 * mean(rs, 'dock_pos_err_m'),
                      'dock_err_max_cm': 100 * max(r['dock_pos_err_m'] for r in rs), 'mean_nees': mean(rs, 'mean_nees')}

    print(f'\nStudy A: process-noise scale, pushed robot ({args.trials} random layouts each)')
    print(f'{"Q scale":>8s} {"docked":>7s} {"mean NEES":>10s} {"max NEES":>9s} {"push recovery":>14s} {"EKF RMSE":>9s} {"dock err":>9s}')
    for q, a in study_a.items():
        print(f'{q:>8.1f} {a["success"]:>3d}/{a["n"]:<3d} {a["mean_nees"]:>10.2f} {a["max_nees"]:>9.0f} '
              f'{a["push_recovery_s"]:>12.1f} s {a["ekf_rmse_cm"]:>7.2f} cm {a["dock_err_cm"]:>6.2f} cm')
    print(f'\nStudy D: measurement ablation, nominal ({args.trials} random layouts each)')
    for m, d in study_d.items():
        print(f'{d["label"]:28s} docked {d["success"]}/{d["n"]}  EKF RMSE {d["ekf_rmse_cm"]:6.2f} cm  '
              f'dock err {d["dock_err_cm"]:6.2f} cm (worst {d["dock_err_max_cm"]:.2f})  mean NEES {d["mean_nees"]:.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    qs = list(study_a)
    ax = axes[0]
    ax.plot(qs, [study_a[q]['mean_nees'] for q in qs], 'o-', label='mean NEES (ideal 3)')
    ax.axhline(3, color='k', lw=0.8)
    ax.set(xscale='log', yscale='log', xlabel='process-noise scale (x slip allowance)', title='Study A: consistency')
    ax2 = ax.twinx()
    ax2.plot(qs, [study_a[q]['push_recovery_s'] for q in qs], 's--', color='C3', label='push recovery (s)')
    ax2.set_ylabel('seconds to get back within 3 cm', color='C3')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax = axes[1]
    labels = [study_d[m]['label'].replace(' (', '\n(') for m, _ in MODES]
    ax.bar(range(3), [study_d[m]['dock_err_cm'] for m, _ in MODES], color=['C2', 'C0', 'C3'])
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set(yscale='log', ylabel='mean dock error (cm)', title='Study D: which measurements matter')
    fig.tight_layout(); fig.savefig(OUT / 'tradeoff_studies.png', dpi=130); plt.close(fig)
    (OUT / 'tradeoff_studies.json').write_text(json.dumps({'study_a_q_scale': {str(k): v for k, v in study_a.items()},
                                                            'study_d_ablation': study_d, 'trials': results},
                                                           indent=1, default=float))
    print(f'\ndone in {time.time() - t0:.0f} s; wrote results/tradeoff_studies.*')


if __name__ == '__main__':
    main()
