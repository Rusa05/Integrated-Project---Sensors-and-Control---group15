"""Mission state machine: navigate to the pre-dock pose, then dock on the charging pad.

    ALIGN ──► TRACK ──► FINAL_ALIGN ──► DOCK ──► DOCKED
      ▲         │  (path blocked / obstacle ahead: replan)
      └─────────┘

ALIGN        turn on the spot to the path heading (the tracker is not controllable at v = 0)
TRACK        LQR on the A* reference, LiDAR safety stop and replanning
FINAL_ALIGN  turn to face the marker at the pre-dock pose (0.6 m out, marker well in view)
DOCK         slow straight-line reference onto the pad, tracked by the same LQR. The marker
             keeps correcting the EKF until it leaves the camera's view (~0.4 m out); the last
             ~0.3 m runs on odometry, which drifts only ~0.35 % of distance (phase 2).
             A LiDAR check stops the robot if the backplate is closer than expected.
DOCKED / DONE

Inputs are only sensor data (encoder increments, marker measurements, LiDAR scans); the pose
used everywhere is the EKF estimate.
"""
import math
import numpy as np

from src import config, model as rm
from src.controller import LQRTracker, saturate, turn_to
from src.ekf import EKF
from src.mapping import OccupancyGrid, ROBOT_RADIUS
from src.planner import plan, time_parametrise, PlanningError
from src.sensors import LIDAR_ANGLES

PRE_DOCK = (1.30, -1.0, 0.0)          # 0.6 m in front of the marker, facing it
REPLAN_CHECK = 1.0                    # s
SAFETY_STOP = 0.18                    # m, front LiDAR range that triggers a stop while travelling
GOAL_TOL = 0.03                       # m, pre-dock arrival
ALIGN_TOL = math.radians(10)
HEADING_TOL = math.radians(0.5)
DOCK_V = 0.06                         # m/s, docking speed
DOCK_A = 0.05                         # m/s^2
DOCK_TOL = 0.01                       # m
# At the dock goal the LiDAR centre (0.035 m behind the axle, x = 1.745) is 0.165 m from the
# backplate face (x = 1.91). A front range below DOCK_STOP means we are about to touch it.
DOCK_STOP = 0.13


def _front_range(scan, half_width=math.radians(35)):
    ahead = np.abs((LIDAR_ANGLES + math.pi) % (2 * math.pi) - math.pi) < half_width
    vals = scan[ahead]
    return float(np.min(vals)) if np.isfinite(vals).any() else math.inf


class Mission:
    def __init__(self, x0, P0, goal=PRE_DOCK, dock=True, lqr_r_scale=1.0):
        self.ekf = EKF(x0, P0)
        self.grid = OccupancyGrid()
        self.lqr = LQRTracker(r_scale=lqr_r_scale)
        self.goal = np.array(goal, dtype=float)
        self.dock = dock
        self.state = 'START'
        self.ref, self.plans = None, []
        self.t_ref = 0.0
        self.replans = self.safety_stops = 0
        self.dock_stop_triggered = False
        self._next_check = 0.0
        # for logging / display
        self.x_ref, self.u_ff, self.e = np.zeros(3), np.zeros(2), np.zeros(3)
        self.last_update = None

    @property
    def finished(self):
        return self.state in ('DONE', 'DOCKED')

    # ------------------------------------------------------------------------------
    def _plan(self, t):
        self.ref, _ = plan(self.grid, self.ekf.x[:2], self.goal[:2])
        self.plans.append(self.ref)
        self.t_ref, self.state = 0.0, 'ALIGN'
        self._next_check = t + REPLAN_CHECK

    def _start_dock(self):
        # Cubic Hermite curve from the current estimate (current heading) to the dock goal (dock
        # heading), so the robot finishes lined up with the dock even if it reached the pre-dock
        # point a few cm off-axis. A straight line would dock at that angle (phase 6 failures).
        dock = np.array(config.DOCK_GOAL)
        p0, p1 = self.ekf.x[:2], dock[:2]
        L = np.hypot(*(p1 - p0))
        t0 = L * np.array([math.cos(self.ekf.x[2]), math.sin(self.ekf.x[2])])
        t1 = L * np.array([math.cos(dock[2]), math.sin(dock[2])])
        s_ = np.linspace(0, 1, 60)[:, None]
        h00, h10, h01, h11 = 2 * s_**3 - 3 * s_**2 + 1, s_**3 - 2 * s_**2 + s_, -2 * s_**3 + 3 * s_**2, s_**3 - s_**2
        pts = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
        self.ref = time_parametrise(pts, v_max=DOCK_V, a_max=DOCK_A)
        self.plans.append(self.ref)
        self.t_ref, self.state = 0.0, 'DOCK'

    def _track(self):
        x_ref, u_ref = self.ref.at(self.t_ref)
        (v, w), e = self.lqr.control(self.ekf.x, x_ref, u_ref)
        self.x_ref, self.e = x_ref, e
        self.u_ff = np.array([u_ref[0] * math.cos(e[2]), u_ref[1]])
        if e[0] < 0.10:                      # reference governor: wait if the robot falls behind
            self.t_ref += rm.DT
        return v, w

    # ------------------------------------------------------------------------------
    def step(self, t, dphi, meas=None, scan=None):
        """One 20 ms tick. Returns (v, omega)."""
        ekf = self.ekf
        ekf.predict(dphi, t)
        self.last_update = ekf.update(meas, t, t_meas=meas.t) if meas is not None else None
        front = None
        if scan is not None:
            self.grid.update(ekf.x, scan)
            front = _front_range(scan)

        if self.state == 'START':
            self._plan(t)

        # replan if the remaining path became blocked, or something is right in front
        if self.state == 'TRACK':
            blocked = False
            if t >= self._next_check:
                self._next_check = t + REPLAN_CHECK
                k0 = np.searchsorted(self.ref.t, self.t_ref)
                upcoming = np.stack([self.ref.x[k0:], self.ref.y[k0:]], axis=1)[::5]
                blocked = any(self.grid.clearance_at(p) < ROBOT_RADIUS * 0.9 for p in upcoming)
            if front is not None and front < SAFETY_STOP:
                self.safety_stops += 1
                blocked = True
            if blocked:
                try:
                    self._plan(t)
                    self.replans += 1
                except PlanningError:
                    pass                      # keep the old plan if no new one exists

        v = w = 0.0
        self.u_ff, self.e = np.zeros(2), np.zeros(3)
        if self.state == 'ALIGN':
            self.x_ref = self.ref.at(0.0)[0]
            v, w, err = turn_to(ekf.x[2], self.ref.theta[0])
            if abs(err) < ALIGN_TOL:
                self.state = 'TRACK'
                v, w = self._track()
        elif self.state == 'TRACK':
            v, w = self._track()
            at_goal = np.hypot(*(ekf.x[:2] - self.goal[:2])) < GOAL_TOL
            if (self.t_ref >= self.ref.duration and at_goal) or self.t_ref >= self.ref.duration + 5.0:
                self.state = 'FINAL_ALIGN'
        elif self.state == 'DOCK':
            v, w = self._track()
            if front is not None and front < DOCK_STOP:
                self.dock_stop_triggered = True
                self.state, v, w = 'DOCKED', 0.0, 0.0
            dock_err = np.hypot(*(ekf.x[:2] - np.array(config.DOCK_GOAL[:2])))
            if (self.t_ref >= self.ref.duration and dock_err < DOCK_TOL) or self.t_ref >= self.ref.duration + 5.0:
                self.state, v, w = 'DOCKED', 0.0, 0.0

        if self.state == 'FINAL_ALIGN':
            v, w, err = turn_to(ekf.x[2], self.goal[2])
            if abs(err) < HEADING_TOL:
                v, w = 0.0, 0.0
                if self.dock:
                    self._start_dock()
                else:
                    self.state = 'DONE'
        return saturate(v, w)
