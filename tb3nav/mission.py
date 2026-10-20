"""Mission state machine and the closed-loop runner.

    LOCALISE -> PLAN -> ALIGN -> TRACK -> DOCK_ALIGN -> DOCK -> DONE
                  ^                |
                  +---- replan ----+   (path blocked by new LiDAR obstacles, safety stop,
                                         or tracking error too large)

Data flow each 50 ms control step (see docs/architecture.md):
    sim.step() -> SensorPacket -> Odometry -> EKF.predict
                               -> MarkerSensor -> EKF.update
                               -> OccupancyGrid.integrate (with EKF pose)
    EKF state -> planner / LQRTracker -> (v, w) -> DiffDrive.inverse -> sim.set_wheel_speeds

`Robot` (everything robot-side) never touches the simulator. `run_mission` owns the
simulator and is the ONLY place that reads ground truth, and only to log it for evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import world as W
from .config import Config, load_config
from .controller import LQRTracker, heading_controller
from .ekf import EKF
from .frames import wrap
from .kinematics import DiffDrive
from .planner import OccupancyGrid, Trajectory, plan_path
from .sensing import MarkerSensor, Odometry, front_clearance
from .sim import Disturbances, Simulator


@dataclass
class RobotOptions:
    ekf_r_scale: float = 1.0
    ekf_q_scale: float = 1.0
    lqr_r_scale: float = 1.0
    use_yaw: bool | None = None
    odometry_only: bool = False       # baseline: no camera updates after initialisation


class Robot:
    """Everything that would run on the TurtleBot's computer. Inputs: SensorPackets only."""

    def __init__(self, cfg: Config, landmark_map: dict, dock_pose: np.ndarray,
                 opts: RobotOptions | None = None, sizes: dict | None = None):
        self.cfg, self.opts = cfg, opts or RobotOptions()
        self.odo = Odometry(cfg)
        self.markers = MarkerSensor(cfg, sizes)
        self.ekf = EKF(cfg, landmark_map, sizes, self.opts.ekf_r_scale, self.opts.ekf_q_scale,
                       self.opts.use_yaw)
        self.dd = DiffDrive(cfg)
        self.lqr = LQRTracker(cfg, self.opts.lqr_r_scale)
        self.grid = OccupancyGrid(cfg)
        self.dock_pose = np.asarray(dock_pose, float)
        n = np.array([np.cos(dock_pose[2]), np.sin(dock_pose[2])])
        pd = cfg.planner.predock_dist - W.DOCK_STANDOFF
        self.predock = np.array([*(dock_pose[:2] - pd * n), dock_pose[2]])
        self.state = "LOCALISE"
        self.traj: Trajectory | None = None
        self.t_ref = 0.0
        self.last_plan_t = -1e9
        self.spin_t = 0.0
        self.last_obs = []
        self.events = []
        self.dt = cfg.timing.control_dt
        self.ref = None
        self.e = np.zeros(3)
        self.blocked_t = 0.0
        self.n_relocalise = 0
        self.no_path_t = 0.0

    # ------------------------------------------------------------------ main step
    def step(self, pkt):
        cfg = self.cfg
        dphi = self.odo.update(pkt.enc_ticks)
        self.ekf.predict(dphi, self.dt)
        obs = []
        if pkt.image is not None:
            obs = self.markers.detect(pkt.image)
            if not self.ekf.initialised:
                if self.ekf.try_initialise(obs, pkt.t):
                    self.events.append((pkt.t, f"EKF initialised ({self.ekf.init_method})"))
            elif not (self.opts.odometry_only and self.state != "LOCALISE"):
                self.ekf.update(obs, pkt.t)
                # Divergence detector: several consecutive gated updates mean the prior is
                # wrong (e.g. the robot was pushed or slid): re-localise and rebuild the map.
                if self.ekf.consec_rejects >= 6:
                    self.events.append((pkt.t, "estimate inconsistent (6 gated updates): "
                                               "re-localising, map reset"))
                    self.ekf.reset()
                    self.grid = OccupancyGrid(self.cfg)
                    self.n_relocalise += 1
                    self.state = "LOCALISE"
                    self.traj = None
        self.last_obs = obs
        if pkt.scan is not None and self.ekf.initialised:
            self.grid.integrate(self.ekf.s, pkt.scan, pkt.scan_angles)
        clear = front_clearance(pkt.scan, pkt.scan_angles) if pkt.scan is not None else None
        if clear is not None:
            self._clear = clear
        v, w = self._decide(pkt.t)
        return self.dd.inverse(v, w), (v, w)

    # ------------------------------------------------------------------ behaviour
    def _set_state(self, s, t, why=""):
        if s != self.state:
            self.events.append((t, f"{self.state} -> {s} {why}".strip()))
            self.state = s

    def _decide(self, t):
        cfg, ekf = self.cfg, self.ekf
        s = ekf.s
        self.ref = None
        if self.state == "LOCALISE":
            # spin slowly until initialised, heading std < 3 deg and position std < 3 cm
            self.spin_t += self.dt
            if ekf.initialised and np.sqrt(ekf.P[2, 2]) < 0.05 and ekf.n_updates >= 3 \
                    and np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]) < 0.03 and t - ekf.t_init > 0.3:
                self._set_state("PLAN", t, "(localised)")
                return 0.0, 0.0
            return 0.0, 0.35

        if self.state == "PLAN":
            return self._plan(t)

        if self.state == "ALIGN":
            v, w, err = heading_controller(s, self.traj.start_heading)
            if abs(err) < 0.06:
                self._set_state("TRACK", t)
                self.t_ref = 0.0
            return v, w

        if self.state == "TRACK":
            # replan if the path ahead is no longer free in the (updated) map
            if t - self.last_plan_t > cfg.planner.replan_period:
                i0 = np.searchsorted(self.traj.t, self.t_ref)
                if not self.grid.path_free(self.traj.xy[i0:i0 + 150], margin=-0.05):
                    self._set_state("PLAN", t, "(path blocked in map)")
                    return 0.0, 0.0
                self.last_plan_t = t
            if getattr(self, "_clear", np.inf) < cfg.planner.safety_stop:
                self.blocked_t += self.dt
                if self.blocked_t > 0.5:
                    self.blocked_t = 0.0
                    self._set_state("PLAN", t, "(LiDAR safety stop)")
                return 0.0, 0.0
            self.blocked_t = 0.0
            return self._track(t, next_state="DOCK_ALIGN")

        if self.state == "DOCK_ALIGN":
            d = self.dock_pose[:2] - s[:2]
            v, w, err = heading_controller(s, np.arctan2(d[1], d[0]))
            if abs(err) < 0.03:
                self.traj = Trajectory([s[:2], self.dock_pose[:2]], cfg, v_max=cfg.planner.dock_speed,
                                       smooth_iters=0)
                self.t_ref = 0.0
                self._set_state("DOCK", t)
            return v, w

        if self.state == "DOCK":
            # stop when the estimate says the axle has reached the dock pose
            d = self.dock_pose[:2] - s[:2]
            along = d @ np.array([np.cos(self.dock_pose[2]), np.sin(self.dock_pose[2])])
            if along < 0.003 or self.t_ref > self.traj.T + 5.0:
                self._set_state("DONE", t)
                return 0.0, 0.0
            return self._track(t, next_state=None)

        return 0.0, 0.0  # DONE

    def _plan(self, t):
        s = self.ekf.s
        path = plan_path(self.grid, s[:2], self.predock[:2])
        self.last_plan_t = t
        if path is None:
            if self.no_path_t == 0.0:
                self.events.append((t, "planner: no path, rotating to scan"))
            self.no_path_t += self.dt
            if self.no_path_t > 6.0:             # persistent: the map is probably corrupted
                self.events.append((t, "planner: no path for 6 s, map reset"))
                self.grid = OccupancyGrid(self.cfg)
                self.no_path_t = 0.0
            return 0.0, 0.3
        self.no_path_t = 0.0
        self.traj = Trajectory(path, self.cfg, grid=self.grid, end_heading=self.predock[2])
        self.path = path
        self.events.append((t, f"planned {self.traj.length:.2f} m, {len(path)} waypoints"))
        if abs(wrap(self.traj.start_heading - s[2])) > 0.35:
            self._set_state("ALIGN", t)
        else:
            self._set_state("TRACK", t)
            self.t_ref = 0.0
        return 0.0, 0.0

    def _track(self, t, next_state):
        ref = self.traj.ref(self.t_ref)
        v, w, e, u = self.lqr.command(self.ekf.s, ref)
        self.ref, self.e = ref, e
        # Time-scaling: if the robot falls behind (e_x > 5 cm) slow the reference down so a
        # stop (safety, slip) does not leave the reference running away.
        lag = np.clip(1.0 - (e[0] - 0.05) / 0.15, 0.0, 1.0)
        self.t_ref += self.dt * lag
        if np.hypot(e[0], e[1]) > 0.35 and self.state == "TRACK":
            self._set_state("PLAN", t, "(tracking error > 0.35 m)")
            return 0.0, 0.0
        if next_state and self.t_ref >= self.traj.T and np.hypot(e[0], e[1]) < 0.08:
            self._set_state(next_state, t)
            return 0.0, 0.0
        return v, w


# ============================================================================ runner
@dataclass
class RunResult:
    log: dict
    events: list
    success: bool
    final_err: np.ndarray
    collisions: int
    t_end: float
    ekf_log: list = field(default_factory=list)
    spec: W.WorldSpec | None = None
    grid: np.ndarray | None = None
    wall_time: float = 0.0


def run_mission(spec: W.WorldSpec, dist: Disturbances | None = None, seed: int = 0,
                cfg: Config | None = None, opts: RobotOptions | None = None,
                t_max: float = 240.0, viewer=None, frame_cb=None, fast_camera=True):
    cfg = cfg or load_config()
    sim = Simulator(spec, cfg, dist, seed=seed, fast_camera=fast_camera)
    robot = Robot(cfg, spec.landmark_map(), spec.dock_pose(), opts, spec.landmark_sizes())
    L = {k: [] for k in ("t", "est", "P", "true", "ref", "cmd", "state", "e", "n_obs")}
    pkt = sim.read_sensors()
    t0 = time.time()
    while sim.data.time < t_max:
        (wl, wr), (v, w) = robot.step(pkt)
        sim.set_wheel_speeds(wl, wr)
        # ---------------- logging (ground truth read here ONLY, for evaluation) ----------
        L["t"].append(pkt.t); L["est"].append(robot.ekf.s.copy()); L["P"].append(robot.ekf.P.copy())
        L["true"].append(sim.ground_truth_for_evaluation())
        L["ref"].append(robot.ref if robot.ref is not None else np.full(5, np.nan))
        L["cmd"].append([v, w]); L["state"].append(robot.state); L["e"].append(robot.e.copy())
        L["n_obs"].append(len(robot.last_obs))
        if viewer is not None:
            viewer(sim, robot, pkt)
        if frame_cb is not None:
            frame_cb(sim, robot, pkt)
        if robot.state == "DONE":
            sim.set_wheel_speeds(0, 0)
            sim.step(0.5)
            break
        pkt = sim.step(cfg.timing.control_dt)
    true_end = sim.ground_truth_for_evaluation()
    dp = spec.dock_pose()
    err = np.array([*(true_end[:2] - dp[:2]), wrap(true_end[2] - dp[2])])
    # docking error in the dock frame: along (approach direction) and lateral
    c, s_ = np.cos(dp[2]), np.sin(dp[2])
    err_dock = np.array([c * err[0] + s_ * err[1], -s_ * err[0] + c * err[1], err[2]])
    success = (robot.state == "DONE" and np.hypot(*err_dock[:2]) < 0.03
               and abs(err_dock[2]) < np.radians(5) and sim.collisions == 0)
    log = {k: (np.array(v) if k != "state" else v) for k, v in L.items()}
    res = RunResult(log, robot.events, bool(success), err_dock, sim.collisions, sim.data.time,
                    robot.ekf.log, spec, robot.grid.logodds.copy(), time.time() - t0)
    res.robot = robot
    res.dist = sim.dist
    sim.close()
    return res
