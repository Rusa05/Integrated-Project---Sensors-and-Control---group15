"""Phase 5: the full Track A mission, start -> navigate around obstacles -> dock on the pad.

Modes (run from the project root; on macOS use mjpython for --viewer):
    python part_E_integration_and_robustness/run_mission.py                   # batch: scenarios x seeds, headless
    python part_E_integration_and_robustness/run_mission.py --viewer          # one run, live MuJoCo viewer
    python part_E_integration_and_robustness/run_mission.py --record run.mp4  # one run, saved as a video
    add --scenario B_box_in_way / --seed 2 to choose the run; --start x y theta for any start pose

Overlays (viewer and video): blue arrow = EKF estimate, blue ellipse = its 3-sigma position
uncertainty, orange dots = current plan, blue/black trails = estimated/true path. The video also
shows the robot's camera with the detected tags outlined, and the estimate-vs-truth error.

The robot runs entirely on src/ (sensors -> EKF -> planner -> LQR). Ground truth is read only in
this file through eval/ground_truth.py, to score the run and draw the true trail.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config, model as rm               # noqa: E402
from src.mission import Mission                    # noqa: E402
from src.robot import SimRobot                     # noqa: E402
from eval import ground_truth as gt                # noqa: E402

OUT = Path(__file__).resolve().parent / 'results' / 'mission'
P0 = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(3) ** 2])
SCENARIOS = {
    'A_cylinder_in_way': (-1.5, -0.8, 0.0),
    'B_box_in_way': (0.5, 1.5, -math.pi / 2),
    'C_far_corner': (-1.5, 1.5, -0.6),
}
TIMEOUT = 150.0
DOCK_POS_TOL, DOCK_YAW_TOL = 0.02, math.radians(3)


# ------------------------------------------------------------------------------------------
# overlays
def _add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _rot_z(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_overlays(scn, mission, est_trail, true_trail):
    x = mission.ekf.x
    # EKF estimate: arrow along the estimated heading (arrow geoms point along their local z)
    z = np.array([math.cos(x[2]), math.sin(x[2]), 0.0])
    xa = np.array([-z[1], z[0], 0.0])
    mat = np.stack([xa, np.cross(z, xa), z], axis=1)
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.012, 0.012, 0.22], [x[0], x[1], 0.30], mat, [0.1, 0.4, 1, 1])
    # 3-sigma position ellipse (drawn at least 1 cm so it stays visible)
    vals, vecs = np.linalg.eigh(mission.ekf.P[:2, :2])
    ang = math.atan2(vecs[1, 1], vecs[0, 1])
    radii = np.maximum(3 * np.sqrt(np.maximum(vals[::-1], 0)), 0.01)
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [radii[0], radii[1], 0.002], [x[0], x[1], 0.012],
              _rot_z(ang), [0.1, 0.4, 1, 0.45])
    # current plan
    if mission.ref is not None:
        for k in range(0, len(mission.ref.x), 8):
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008] * 3, [mission.ref.x[k], mission.ref.y[k], 0.01],
                      np.eye(3), [1, 0.55, 0.1, 0.9])
    for trail, rgba in ((true_trail, [0, 0, 0, 0.9]), (est_trail, [0.1, 0.4, 1, 0.9])):
        for p in trail[::5]:
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005] * 3, [p[0], p[1], 0.006], np.eye(3), rgba)


def compose_frame(scene_rgb, robot, mission, truth, t):
    img = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    # inset: what the robot's camera sees, with the detected tags outlined
    if robot.last_frame is not None:
        cam = cv2.cvtColor(robot.last_frame, cv2.COLOR_GRAY2BGR)
        for m in robot.last_markers:
            cv2.polylines(cam, [m.corners.astype(np.int32)], True, (0, 255, 0), 3)
        cam = cv2.resize(cam, (W // 3, H // 3))
        img[10:10 + cam.shape[0], W - 10 - cam.shape[1]:W - 10] = cam
        cv2.rectangle(img, (W - 10 - cam.shape[1], 10), (W - 10, 10 + cam.shape[0]), (255, 255, 255), 1)
    est, P = mission.ekf.x, mission.ekf.P
    err = np.hypot(*(est[:2] - truth[:2]))
    lines = [f't = {t:5.1f} s   state: {mission.state}',
             f'EKF  x {est[0]:+.3f}  y {est[1]:+.3f}  th {math.degrees(est[2]):+6.1f} deg',
             f'true x {truth[0]:+.3f}  y {truth[1]:+.3f}  th {math.degrees(truth[2]):+6.1f} deg',
             f'error {err * 100:4.1f} cm   3-sigma {300 * math.sqrt(max(P[0, 0], P[1, 1])):4.1f} cm',
             'tags in view: ' + (', '.join(f'{m.id} ({m.range:.2f} m)' for m in robot.last_markers) or 'none')]
    legend = ['blue arrow/ellipse: EKF estimate, 3-sigma', 'blue dots: estimated path   black dots: true path',
              'orange dots: planned path']

    def panel(text_lines, x0, y0, width):
        h = 22 * len(text_lines) + 12
        roi = img[y0:y0 + h, x0:x0 + width]
        roi[:] = (roi * 0.35).astype(np.uint8)              # darken behind the text
        for i, line in enumerate(text_lines):
            cv2.putText(img, line, (x0 + 8, y0 + 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

    panel(lines, 0, 0, 430)
    panel(legend, 0, H - 22 * len(legend) - 12, 430)
    if robot.last_frame is not None:
        cv2.putText(img, 'robot camera', (W - 10 - W // 3 + 6, 10 + H // 3 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
    return img


def overview_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.1, 0.0, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 5.2, 90.0, -62.0
    return cam


# ------------------------------------------------------------------------------------------
def run(start, seed=0, viewer=False, record=None, realtime=True):
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    data = mujoco.MjData(model)
    gt.set_pose(model, data, *start)
    robot = SimRobot(model, data, seed=seed)
    rng = np.random.default_rng(3000 + seed)
    mission = Mission(np.array(start) + rng.multivariate_normal(np.zeros(3), P0), P0, dock=True)
    collisions, est_trail, true_trail = set(), [], []
    marker_last_seen_range, t0 = math.inf, data.time
    v_handle = writer = renderer = None
    if viewer:
        v_handle = mujoco.viewer.launch_passive(model, data)
        cam = overview_camera()
        v_handle.cam.lookat[:], v_handle.cam.distance = cam.lookat, cam.distance
        v_handle.cam.azimuth, v_handle.cam.elevation = cam.azimuth, cam.elevation
    if record:
        renderer = mujoco.Renderer(model, 720, 960)
        writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 720))
        rcam, ropt = overview_camera(), mujoco.MjvOption()
        ropt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    tick, hold_after = 0, 2.0
    t_done = None
    try:
        while data.time - t0 < TIMEOUT:
            started = time.perf_counter()
            dphi, meas, scan = robot.sense()
            v, w = mission.step(robot.time, dphi, meas, scan)       # EKF estimate only
            dock_seen = [m.range for m in meas if m.id in config.DOCK_TAG_IDS]
            if dock_seen:
                marker_last_seen_range = min(dock_seen)
            robot.act(v, w)
            # ---- evaluation / display only ------------------------------------------------
            truth = gt.true_pose(model, data)
            collisions |= gt.robot_collisions(model, data)
            if tick % 5 == 0:
                est_trail.append(mission.ekf.x[:2].copy()); true_trail.append(truth[:2].copy())
            if v_handle is not None:
                if not v_handle.is_running():
                    break
                with v_handle.lock():
                    v_handle.user_scn.ngeom = 0
                    draw_overlays(v_handle.user_scn, mission, est_trail, true_trail)
                v_handle.sync()
                if realtime:
                    time.sleep(max(0.0, rm.DT - (time.perf_counter() - started)))
            if writer is not None and tick % 2 == 0:
                renderer.update_scene(data, camera=rcam, scene_option=ropt)
                draw_overlays(renderer.scene, mission, est_trail, true_trail)
                writer.write(compose_frame(renderer.render(), robot, mission, truth, data.time - t0))
            tick += 1
            if mission.finished:
                t_done = t_done or data.time
                if data.time - t_done >= (hold_after if (viewer or record) else 0):
                    break
    finally:
        robot.close()
        if writer is not None:
            writer.release()
            renderer.close()
        if v_handle is not None:
            v_handle.close()

    truth = gt.true_pose(model, data)
    dock = np.array(config.DOCK_GOAL)
    pos_err = float(np.hypot(*(truth[:2] - dock[:2])))
    yaw_err = float(math.degrees(rm.wrap(truth[2] - dock[2])))
    return {
        'docked': mission.state == 'DOCKED',
        'success': bool(mission.state == 'DOCKED' and not collisions and pos_err < DOCK_POS_TOL
                        and abs(math.radians(yaw_err)) < DOCK_YAW_TOL),
        'time_s': float((t_done or data.time) - t0),
        'dock_pos_err_m': pos_err, 'dock_lateral_err_m': float(truth[1] - dock[1]),
        'dock_depth_err_m': float(truth[0] - dock[0]), 'dock_yaw_err_deg': yaw_err,
        'ekf_err_at_dock_m': float(np.hypot(*(mission.ekf.x[:2] - truth[:2]))),
        'ekf_3sigma_at_dock_m': float(3 * math.sqrt(max(mission.ekf.P[0, 0], mission.ekf.P[1, 1]))),
        'marker_last_seen_range_m': float(marker_last_seen_range),
        'collisions': sorted(collisions), 'replans': mission.replans,
        'lidar_dock_stop': mission.dock_stop_triggered, 'final_state': mission.state,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--record', type=str)
    ap.add_argument('--scenario', default='A_cylinder_in_way', choices=list(SCENARIOS))
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'THETA'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, default=3, help='batch mode: seeds per scenario')
    args = ap.parse_args()
    start = tuple(args.start) if args.start else SCENARIOS[args.scenario]

    if args.viewer or args.record:
        res = run(start, args.seed, viewer=args.viewer, record=args.record)
        print(json.dumps(res, indent=2))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    results, t0 = {}, time.time()
    for name, st in SCENARIOS.items():
        results[name] = []
        for seed in range(args.seeds):
            r = run(st, seed)
            results[name].append(r)
            print(f'{name} seed {seed}: {"OK " if r["success"] else "FAIL"} {r["final_state"]:7s} {r["time_s"]:5.1f}s | '
                  f'dock err {r["dock_pos_err_m"]*100:.2f} cm (lateral {r["dock_lateral_err_m"]*100:+.2f}, depth {r["dock_depth_err_m"]*100:+.2f}), '
                  f'yaw {r["dock_yaw_err_deg"]:+.2f} deg | EKF err {r["ekf_err_at_dock_m"]*100:.2f} cm (3s {r["ekf_3sigma_at_dock_m"]*100:.2f}) | '
                  f'marker last seen {r["marker_last_seen_range_m"]:.2f} m | collisions {r["collisions"]}', flush=True)
    allr = [r for rs in results.values() for r in rs]
    summary = {'runs': len(allr), 'success': sum(r['success'] for r in allr),
               'mean_dock_pos_err_m': float(np.mean([r['dock_pos_err_m'] for r in allr])),
               'max_dock_pos_err_m': float(np.max([r['dock_pos_err_m'] for r in allr])),
               'mean_abs_yaw_err_deg': float(np.mean([abs(r['dock_yaw_err_deg']) for r in allr])),
               'mean_time_s': float(np.mean([r['time_s'] for r in allr]))}
    print(json.dumps(summary, indent=2))
    (OUT / 'summary.json').write_text(json.dumps({'summary': summary, 'runs': results}, indent=2))
    print(f'done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
