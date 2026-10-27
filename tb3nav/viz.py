"""Live viewer overlays and an offscreen MP4 recorder for the demo video.

Overlays (drawn into the MuJoCo scene, never fed back to the robot):
  green arrow + green ellipse : EKF estimate and its 3-sigma position ellipse
  blue dots                   : current reference trajectory
  yellow dots                 : LiDAR returns placed with the ESTIMATED pose
The robot model itself is drawn at its TRUE pose, so estimate vs truth is visible directly.
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from . import world as W
from .frames import rot2

GREEN = np.array([0.1, 0.9, 0.2, 0.9], np.float32)
BLUE = np.array([0.2, 0.4, 1.0, 0.8], np.float32)
YELLOW = np.array([1.0, 0.85, 0.1, 0.9], np.float32)


def _add(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).reshape(-1), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_overlays(scn, robot, pkt=None, show_lidar=True):
    ekf = robot.ekf
    if ekf.initialised:
        s, P = ekf.s, ekf.P
        # 3-sigma ellipse as a flat ellipsoid
        vals, vecs = np.linalg.eigh(P[:2, :2])
        R = np.eye(3); R[:2, :2] = vecs
        size = [3 * np.sqrt(max(vals[0], 1e-8)) + 0.01, 3 * np.sqrt(max(vals[1], 1e-8)) + 0.01, 0.002]
        _add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, size, [s[0], s[1], 0.25], R,
             [0.1, 0.9, 0.2, 0.35])
        # heading arrow, with a +-3 sigma heading fan
        for dth, rgba in ((0.0, GREEN), (3 * np.sqrt(P[2, 2]), [0.1, 0.9, 0.2, 0.35]),
                          (-3 * np.sqrt(P[2, 2]), [0.1, 0.9, 0.2, 0.35])):
            th = s[2] + dth
            d = np.array([np.cos(th), np.sin(th), 0.0])
            z = d; x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
            _add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.012, 0.012, 0.25], [s[0], s[1], 0.25],
                 np.stack([x, y, z], 1), rgba)
    if robot.traj is not None:
        for p in robot.traj.xy[::8]:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012] * 3, [p[0], p[1], 0.01], np.eye(3), BLUE)
    if show_lidar and pkt is not None and pkt.scan is not None and ekf.initialised:
        ok = np.isfinite(pkt.scan)
        r, a = pkt.scan[ok][::3], pkt.scan_angles[ok][::3]
        pts = np.stack([r * np.cos(a) - 0.049, r * np.sin(a)], 1) @ rot2(ekf.s[2]).T + ekf.s[:2]
        for p in pts:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.015] * 3, [p[0], p[1], 0.17], np.eye(3), YELLOW)


def viewer_callback_factory(viewer, realtime=True):
    import time
    state = {"wall": None, "sim": None}

    def cb(sim, robot, pkt):
        viewer.user_scn.ngeom = 0
        draw_overlays(viewer.user_scn, robot, pkt)
        viewer.sync()
        if realtime:
            if state["wall"] is None:
                state["wall"], state["sim"] = time.time(), sim.data.time
            ahead = (sim.data.time - state["sim"]) - (time.time() - state["wall"])
            if ahead > 0:
                time.sleep(ahead)
        if not viewer.is_running():
            raise KeyboardInterrupt
    return cb


class Recorder:
    """Writes a composite MP4: tracking overview | robot camera, with a text panel."""

    def __init__(self, path, every=2, fps=10, size=(960, 540)):
        self.every, self.k = every, 0
        self.W, self.H = size
        self.ren = None
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.2, -55, 90
        self.cam.lookat[:] = [W.ROOM_X / 2, W.ROOM_Y / 2, 0]
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                      (self.W, self.H))
        self.last_cam = np.zeros((self.H // 2, self.W // 3, 3), np.uint8)
        self.last_scan_pkt = None   # LiDAR runs at 5 Hz: keep the latest scan for display
        self.n_seen = 0             # camera runs at 10 Hz: keep the latest detection count
        self.title = ""

    def __call__(self, sim, robot, pkt):
        if self.ren is None:
            self.ren = mujoco.Renderer(sim.model, self.H, int(self.W * 2 / 3))
        self.k += 1
        if pkt.scan is not None:
            self.last_scan_pkt = pkt
        if pkt.image is not None:
            self.n_seen = len(robot.last_obs)
            img = pkt.image.copy()
            for o in robot.last_obs:
                cv2.polylines(img, [o.corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(img, f"id{o.id} r={o.z[0]:.2f}", tuple(o.corners[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.last_cam = cv2.resize(img, (self.W // 3, self.H // 2))
        if self.k % self.every:
            return
        truth = sim.ground_truth_for_evaluation()        # display only
        # follow the robot loosely
        self.cam.lookat[:2] = 0.8 * self.cam.lookat[:2] + 0.2 * truth[:2]
        self.ren.update_scene(sim.data, camera=self.cam)
        draw_overlays(self.ren.scene, robot, self.last_scan_pkt)
        left = self.ren.render()
        panel = np.full((self.H // 2, self.W // 3, 3), 25, np.uint8)
        est, P = robot.ekf.s, robot.ekf.P
        err = np.hypot(*(est[:2] - truth[:2])) if robot.ekf.initialised else np.nan
        lines = [self.title, f"t = {pkt.t:6.1f} s   state: {robot.state}",
                 f"true  x={truth[0]:5.2f} y={truth[1]:5.2f} th={np.degrees(truth[2]):6.1f}",
                 f"est   x={est[0]:5.2f} y={est[1]:5.2f} th={np.degrees(est[2]):6.1f}",
                 f"3sig  xy={3*np.sqrt(max(P[0,0],P[1,1])):.3f} m  th={np.degrees(3*np.sqrt(P[2,2])):.1f} deg",
                 f"|est-true| = {100*err:5.1f} cm", f"markers seen: {self.n_seen}",
                 "green: estimate+3sig  blue: ref", "yellow: LiDAR @ est. pose"]
        for i, l in enumerate(lines):
            cv2.putText(panel, l, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        right = np.vstack([self.last_cam, panel])
        frame = np.hstack([left, right])
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()
        if self.ren is not None:
            self.ren.close()
