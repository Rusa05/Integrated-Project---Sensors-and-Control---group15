"""Phase 6: robustness trials. Random starts and obstacle layouts, plus disturbances.

Conditions (each trial also has a random start pose and random obstacle positions):
    nominal     no disturbance
    push        a sideways shove part-way through (8 N for 0.2 s, random direction)
    slip        tyre friction cut from 1.0 to 0.1 (slippery floor)
    occlusion   marker hidden for a random 10 s window (something in front of it)
    latency     every marker measurement arrives 0.2 s after its camera frame
    latency_nocomp  same, with the EKF's latency compensation switched off (for comparison)
    noise_x2    all sensor noise doubled
    combined    push + slip + occlusion + 0.1 s latency

The robot runs entirely on src/ (SimRobot -> Mission). Pushes, friction and obstacle positions
are changes to the simulated world; occlusion and latency are sensor faults inside SimRobot.
Ground truth (eval/ground_truth.py) is used only to score and to diagnose failures.

Run from the project root:
    python part_E_integration_and_robustness/run_trials.py                 # 10 trials per condition, 4 processes
    python part_E_integration_and_robustness/run_trials.py --trials 3      # quicker
    python part_E_integration_and_robustness/run_trials.py --only push     # one condition
"""
from pathlib import Path
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'results' / 'trials'
CONDITIONS = ['nominal', 'push', 'slip', 'occlusion', 'latency', 'latency_nocomp', 'noise_x2', 'combined']
P0 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(3) ** 2])
TIMEOUT = 150.0
DOCK_POS_TOL, DOCK_YAW_TOL = 0.02, math.radians(3)

BOX_HALF = (0.17, 0.20)
CYL_R = 0.18
DOCK_ZONE = lambda x, y: x > 0.9 and abs(y + 1.0) < 0.55       # keep the dock approach clear


# ------------------------------------------------------------------------------------------
def sample_layout(rng):
    """Random obstacle positions and a random start pose, all at least a robot-width apart."""
    def free(x, y, others, r):
        if abs(x) > 1.95 - r or abs(y) > 1.95 - r or DOCK_ZONE(x, y):
            return False
        return all(math.hypot(x - ox, y - oy) > r + orad for ox, oy, orad in others)

    while True:
        placed = []
        bx, by = rng.uniform(-1.5, 1.4), rng.uniform(-1.5, 1.5)
        if not free(bx, by, placed, 0.27):
            continue
        placed.append((bx, by, 0.27))
        cx, cy = rng.uniform(-1.5, 1.4), rng.uniform(-1.5, 1.5)
        if not free(cx, cy, placed, CYL_R + 0.55):          # leave a gap the robot fits through
            continue
        placed.append((cx, cy, CYL_R))
        for _ in range(200):
            sx, sy = rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6)
            if free(sx, sy, [(bx, by, 0.27 + 0.3), (cx, cy, CYL_R + 0.3)], 0.3) and math.hypot(sx - 1.3, sy + 1.0) > 0.8:
                return {'box': (bx, by, rng.uniform(0, math.pi)), 'cyl': (cx, cy),
                        'start': (sx, sy, rng.uniform(-math.pi, math.pi))}


def run_trial(args):
    """args = (condition, trial index[, debug log[, video path]])."""
    cond, idx = args[:2]
    debug = len(args) > 2 and args[2]
    record = args[3] if len(args) > 3 else None
    dbg = []
    sys.path.insert(0, str(ROOT))
    import mujoco
    from src import config, model as rm
    from src.mission import Mission
    from src.planner import PlanningError
    from src.robot import SimRobot
    from src.sensors import NoiseConfig
    from eval import ground_truth as gt

    rng = np.random.default_rng(10_000 + 97 * idx + CONDITIONS.index(cond))
    layout = sample_layout(np.random.default_rng(500 + idx))         # same layout across conditions
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    # --- world changes (not visible to the robot except through its sensors) ---------------
    bx, by, byaw = layout['box']
    model.geom('obstacle_box').pos[:2] = (bx, by)
    model.geom('obstacle_box').quat[:] = (math.cos(byaw / 2), 0, 0, math.sin(byaw / 2))
    model.geom('obstacle_cylinder').pos[:2] = layout['cyl']
    push = slip = occl = None
    latency, noise = 0.0, NoiseConfig()
    if cond in ('push', 'combined'):
        ang = rng.uniform(-math.pi, math.pi)
        push = {'t': rng.uniform(6.0, 20.0), 'dur': 0.2, 'f': 8.0 * np.array([math.cos(ang), math.sin(ang), 0.0])}
    if cond in ('slip', 'combined'):
        slip = 0.1
        for side in ('left', 'right'):
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == model.body(f'wheel_{side}').id:
                    model.geom_friction[g][0] = slip
    if cond in ('occlusion', 'combined'):
        t1 = rng.uniform(5.0, 25.0)
        occl = (t1, t1 + 10.0)
    if cond in ('latency', 'latency_nocomp'):
        latency = 0.2
    if cond == 'combined':
        latency = 0.1
    if cond == 'noise_x2':
        n = NoiseConfig()
        noise = NoiseConfig(encoder_sigma=2 * n.encoder_sigma, lidar_sigma_min=2 * n.lidar_sigma_min,
                            lidar_sigma_frac=2 * n.lidar_sigma_frac, lidar_dropout=2 * n.lidar_dropout,
                            pixel_sigma=2 * n.pixel_sigma)

    data = mujoco.MjData(model)
    gt.set_pose(model, data, *layout['start'])
    robot = SimRobot(model, data, noise=noise, seed=idx,
                     occluded=(lambda t: occl[0] <= t < occl[1]) if occl else None, latency=latency)
    mission = Mission(np.array(layout['start']) + rng.multivariate_normal(np.zeros(3), P0), P0)
    base = model.body('base').id
    collisions = {}
    max_err = max_nees = 0.0
    traj = []
    t0 = data.time
    error = None
    writer = None
    if record:                                                  # same overlays as the phase 5 video
        import cv2
        import importlib.util
        spec = importlib.util.spec_from_file_location('run_mission', Path(__file__).resolve().parent / 'run_mission.py')
        vid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vid)
        renderer = mujoco.Renderer(model, 720, 960)
        writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 720))
        rcam, ropt = vid.overview_camera(), mujoco.MjvOption()
        ropt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        est_trail, true_trail, tick = [], [], 0
    try:
        while data.time - t0 < TIMEOUT and not mission.finished:
            dphi, meas, scan = robot.sense()
            if cond == 'latency_nocomp' and meas is not None:
                meas.t = None                                   # EKF treats the frame as current
            v, w = mission.step(robot.time, dphi, meas, scan)
            data.xfrc_applied[base, :3] = push['f'] if push and push['t'] <= data.time < push['t'] + push['dur'] else 0.0
            robot.act(v, w)
            # ---- evaluation only ------------------------------------------------------------
            truth = gt.true_pose(model, data)
            e = mission.ekf.x - truth
            e[2] = rm.wrap(e[2])
            max_err = max(max_err, math.hypot(e[0], e[1]))
            max_nees = max(max_nees, float(e @ np.linalg.solve(mission.ekf.P, e)))
            for name in gt.robot_collisions(model, data):
                if name not in collisions:
                    collisions[name] = {'t': data.time - t0, 'ekf_err': math.hypot(e[0], e[1]), 'state': mission.state}
            if int(round(data.time / rm.DT)) % 10 == 0:
                traj.append((truth[0], truth[1], mission.ekf.x[0], mission.ekf.x[1]))
            if writer is not None:
                if tick % 5 == 0:
                    est_trail.append(mission.ekf.x[:2].copy()); true_trail.append(truth[:2].copy())
                if tick % 2 == 0:
                    renderer.update_scene(data, camera=rcam, scene_option=ropt)
                    vid.draw_overlays(renderer.scene, mission, est_trail, true_trail)
                    frame = vid.compose_frame(renderer.render(), robot, mission, truth, data.time - t0)
                    tags = [f'condition: {cond}']
                    if push and push['t'] <= data.time < push['t'] + 1.5:
                        tags.append('PUSH!')
                    if occl and occl[0] <= data.time < occl[1]:
                        tags.append('marker occluded')
                        # the occlusion drops detections in software, so show it on the camera inset too
                        H_, W_ = frame.shape[:2]
                        x0_, y0_ = W_ - 10 - W_ // 3, 10
                        inset = frame[y0_:y0_ + H_ // 3, x0_:W_ - 10]
                        inset[:] = (inset * 0.3 + 90).astype(np.uint8)
                        cv2.putText(frame, 'marker detections blocked', (x0_ + 40, y0_ + H_ // 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    if latency:
                        tags.append(f'camera latency {latency:.1f} s')
                    if slip:
                        tags.append(f'tyre friction {slip}')
                    roi = frame[126:160, 0:620]
                    roi[:] = (roi * 0.35).astype(np.uint8)
                    cv2.putText(frame, '   '.join(tags), (8, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 1, cv2.LINE_AA)
                    writer.write(frame)
                tick += 1
            if debug:
                u = mission.last_update
                dbg.append((round(data.time - t0, 2), mission.state, truth.round(4).tolist(), mission.ekf.x.round(4).tolist(),
                            float(3 * math.sqrt(mission.ekf.P[0, 0])), None if u is None else (u.used, round(u.nis, 1)),
                            None if meas is None else round(meas.range, 3)))
    except PlanningError as exc:
        error = f'no path: {exc}'
    except Exception as exc:                                     # report, don't kill the batch
        error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}'
    finally:
        robot.close()
        if writer is not None:
            writer.release()
            renderer.close()

    truth = gt.true_pose(model, data)
    dock = np.array(config.DOCK_GOAL)
    pos_err = float(np.hypot(*(truth[:2] - dock[:2])))
    yaw_err = float(rm.wrap(truth[2] - dock[2]))
    est_err = float(np.hypot(*(mission.ekf.x[:2] - truth[:2])))
    ctrl_err = float(np.hypot(*(mission.ekf.x[:2] - dock[:2])))
    docked = mission.state == 'DOCKED'
    ok = docked and not collisions and pos_err < DOCK_POS_TOL and abs(yaw_err) < DOCK_YAW_TOL and error is None

    # ---- diagnosis: which part of the pipeline caused the failure ---------------------------
    cause = None
    if error:
        cause = 'planning: no path' if error.startswith('no path') else 'software error'
    elif collisions:
        name, c = min(collisions.items(), key=lambda kv: kv[1]['t'])
        cause = (f'collision with {name} in {c["state"]}: ' +
                 ('estimation (EKF off by %.0f cm)' % (100 * c['ekf_err']) if c['ekf_err'] > 0.08 else
                  'planning/control (EKF was accurate)'))
    elif not docked:
        cause = f'timeout in {mission.state}'
    elif not ok:
        est_yaw_err = abs(rm.wrap(mission.ekf.x[2] - truth[2]))
        if est_err > 0.5 * DOCK_POS_TOL:
            cause = 'estimation: EKF %.1f cm off at dock' % (100 * est_err)
        elif abs(yaw_err) >= DOCK_YAW_TOL and est_yaw_err > 0.5 * DOCK_YAW_TOL:
            cause = 'estimation: EKF heading %.1f deg off at dock' % math.degrees(est_yaw_err)
        elif abs(yaw_err) >= DOCK_YAW_TOL:
            cause = 'control/planning: docked %.1f deg crooked (EKF heading was right)' % math.degrees(yaw_err)
        else:
            cause = 'control: estimate %.1f cm from goal' % (100 * ctrl_err)
    return {
        'condition': cond, 'trial': idx, 'success': bool(ok), 'final_state': mission.state, 'cause': cause,
        'time_s': float(data.time - t0), 'dock_pos_err_m': pos_err, 'dock_yaw_err_deg': math.degrees(yaw_err),
        'ekf_err_at_end_m': est_err, 'max_ekf_err_m': float(max_err), 'max_nees': float(max_nees),
        'replans': mission.replans, 'safety_stops': mission.safety_stops, 'p_inflations': mission.ekf.inflations,
        'collisions': sorted(collisions), 'layout': layout, 'push': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                                                     for k, v in push.items()} if push else None,
        'occlusion': occl, 'traj': traj, 'error': error, 'debug': dbg,
    }


# ------------------------------------------------------------------------------------------
def report(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds = [c for c in CONDITIONS if any(r['condition'] == c for r in results)]
    rows = []
    for c in conds:
        rs = [r for r in results if r['condition'] == c]
        okr = [r for r in rs if r['success']]
        rows.append({'condition': c, 'n': len(rs), 'success': len(okr),
                     'dock_err_cm_mean': 100 * float(np.mean([r['dock_pos_err_m'] for r in okr])) if okr else float('nan'),
                     'dock_err_cm_max': 100 * float(np.max([r['dock_pos_err_m'] for r in okr])) if okr else float('nan'),
                     'max_ekf_err_cm_median': 100 * float(np.median([r['max_ekf_err_m'] for r in rs])),
                     'time_s_mean': float(np.mean([r['time_s'] for r in okr])) if okr else float('nan'),
                     'replans_total': int(sum(r['replans'] for r in rs))})
    print(f'\n{"condition":10s} {"success":>8s} {"dock err mean/max (cm)":>24s} {"max EKF err median (cm)":>24s} {"time (s)":>9s} {"replans":>8s}')
    for r in rows:
        print(f'{r["condition"]:10s} {r["success"]:>3d}/{r["n"]:<4d} {r["dock_err_cm_mean"]:>12.2f} / {r["dock_err_cm_max"]:<8.2f} '
              f'{r["max_ekf_err_cm_median"]:>20.1f} {r["time_s_mean"]:>12.1f} {r["replans_total"]:>8d}')
    fails = [r for r in results if not r['success']]
    print(f'\nFailures ({len(fails)}):')
    for r in fails:
        print(f'  {r["condition"]:10s} trial {r["trial"]:2d}: {r["cause"]}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.bar([r['condition'] for r in rows], [100 * r['success'] / r['n'] for r in rows], color='C0')
    ax.set(ylabel='% docked within 2 cm / 3 deg', ylim=(0, 105), title='Success rate by condition')
    ax.tick_params(axis='x', rotation=30)
    ax = axes[1]
    data = [[100 * r['max_ekf_err_m'] for r in results if r['condition'] == c] for c in conds]
    ax.boxplot(data, tick_labels=conds)
    ax.set(ylabel='cm', yscale='log', title='Worst EKF position error during each run')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(OUT / 'success_by_condition.png', dpi=130); plt.close(fig)

    # a grid of example runs, one per condition (first failure if any, else first trial)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    conds_plot = [c for c in conds if c != 'latency_nocomp'] + (['latency_nocomp'] if 'latency_nocomp' in conds else [])
    for k, (ax, c) in enumerate(zip(axes.flat, conds_plot[:8])):
        rs = [r for r in results if r['condition'] == c]
        # show a failure if there is one, else a different random layout in each panel
        r = next((x for x in rs if not x['success']), rs[k % len(rs)])
        L = r['layout']
        bx, by, byaw = L['box']
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * BOX_HALF
        rot = np.array([[math.cos(byaw), -math.sin(byaw)], [math.sin(byaw), math.cos(byaw)]])
        ax.fill(*(corners @ rot.T + [bx, by]).T, color='orange', alpha=0.6)
        ax.add_patch(plt.Circle(L['cyl'], CYL_R, color='seagreen', alpha=0.6))
        ax.add_patch(plt.Rectangle((1.62, -1.18), 0.34, 0.36, color='gold', alpha=0.8))
        tr = np.array(r['traj'])
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], 'k', lw=1.2, label='truth')
            ax.plot(tr[:, 2], tr[:, 3], 'C0:', lw=1.2, label='EKF')
        ax.plot(*L['start'][:2], 'go')
        if r['push']:
            k = min(int(r['push']['t'] / 0.2), len(tr) - 1)
            ax.annotate('', xy=tr[k, :2] + 0.3 * np.array(r['push']['f'][:2]) / 8, xytext=tr[k, :2],
                        arrowprops=dict(color='red', arrowstyle='->'))
        ax.set(aspect='equal', xlim=(-2, 2), ylim=(-2, 2),
               title=f'{c} #{r["trial"]}: {"OK" if r["success"] else "FAIL"}', xticks=[], yticks=[])
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(conds):]:
        ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT / 'example_runs.png', dpi=110); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--only', choices=CONDITIONS)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--report-only', action='store_true', help='redraw plots from results/summary.json')
    ap.add_argument('--record', type=str, help='record ONE trial (--only CONDITION --trial N) to this .mp4')
    ap.add_argument('--trial', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.record:
        r = run_trial((args.only or 'combined', args.trial, False, args.record))
        print(f'{r["condition"]} #{r["trial"]}: {"OK" if r["success"] else "FAIL"}, dock {r["dock_pos_err_m"]*100:.2f} cm, '
              f'max EKF err {r["max_ekf_err_m"]*100:.1f} cm, push {r["push"] and round(r["push"]["t"], 1)} s, '
              f'occlusion {r["occlusion"]}')
        return
    if args.report_only:
        report(json.loads((OUT / 'summary.json').read_text())['trials'])
        return
    conds = [args.only] if args.only else CONDITIONS
    jobs = [(c, i) for c in conds for i in range(args.trials)]
    t0 = time.time()
    results = []
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_trial, jobs):
            results.append(r)
            print(f'[{len(results):3d}/{len(jobs)}] {r["condition"]:10s} #{r["trial"]:2d} '
                  f'{"OK  " if r["success"] else "FAIL"} {r["time_s"]:5.1f}s dock {r["dock_pos_err_m"]*100:6.2f} cm '
                  f'max EKF err {r["max_ekf_err_m"]*100:5.1f} cm  {r["cause"] or ""}', flush=True)
    results.sort(key=lambda r: (CONDITIONS.index(r['condition']), r['trial']))
    rows = report(results)
    (OUT / 'summary.json').write_text(json.dumps({'table': rows, 'trials': results}, indent=1, default=float))
    print(f'done in {time.time() - t0:.0f} s; wrote {OUT}')


if __name__ == '__main__':
    main()
