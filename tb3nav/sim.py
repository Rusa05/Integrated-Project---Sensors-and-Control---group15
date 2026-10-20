"""MuJoCo simulator wrapper: the "real world" side of the pipeline.

Everything that knows the true state lives in this file. The robot-side code (sensing.py,
ekf.py, controller.py, planner.py, mission.py) only ever receives a `SensorPacket`, which
contains encoder ticks, a LiDAR scan and a camera image - never a pose.

`Simulator.ground_truth_for_evaluation()` is the single accessor for the true pose. It is
called only by the logger in mission.run_mission(), to score the run afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import world as W
from .config import Config
from .frames import P_B_L, quat_to_yaw


@dataclass
class Disturbances:
    """Everything that can be perturbed in a trial. Defaults = nominal conditions."""
    enc_scale: tuple = (1.0, 1.0)      # true/assumed wheel radius ratio seen by the encoders
    push_time: float | None = None     # s, start of an external push on the base
    push_force: tuple = (0.0, 0.0)     # N, world-frame force during the push
    push_duration: float = 0.3         # s
    camera_dropout: float = 0.0        # probability a camera frame is lost
    pixel_noise_scale: float = 1.0     # multiplies Noise.pixel_noise_std
    lidar_noise_scale: float = 1.0     # multiplies the LiDAR range noise


@dataclass
class SensorPacket:
    t: float
    enc_ticks: np.ndarray                     # int (left, right), always present
    scan: np.ndarray | None = None            # (N,) ranges [m], inf = no return
    scan_angles: np.ndarray | None = None     # (N,) beam angles in the lidar frame [rad]
    image: np.ndarray | None = None           # (H, W, 3) uint8 RGB


class Simulator:
    def __init__(self, spec: W.WorldSpec, cfg: Config, dist: Disturbances | None = None,
                 seed: int = 0, render_camera: bool = True, fast_camera: bool = True):
        self.spec, self.cfg = spec, cfg
        self.fast_camera = fast_camera
        self.dist = dist or Disturbances()
        self.rng = np.random.default_rng(seed)
        self.model, self.data = W.load_model(spec)
        self.model.opt.timestep = cfg.timing.physics_dt
        m = self.model
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                    for n in ("wheel_left", "wheel_right")]
        self.jnt_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in ("wheel_left", "wheel_right")]
        self.dof_adr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                        for n in ("wheel_left", "wheel_right")]
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_bodies = {self.base_id,
                             mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_left"),
                             mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_right")}
        self.obstacle_geoms = {i for i in range(m.ngeom)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                               .startswith(("wall", "obs", "dock"))}
        self.wheel_cmd = np.zeros(2)
        self.integ = np.zeros(2)
        self.enc_slip = np.zeros(2)
        self.prev_wheel_angle = self._wheel_angle().copy()
        self.collisions = 0
        self._in_collision = False
        self.next_cam = 0.0
        self.next_lidar = 0.0
        self.renderer = mujoco.Renderer(m, 480, 640) if render_camera else None
        n = cfg.noise.lidar_beams
        self.scan_angles = np.arange(n) * 2 * np.pi / n
        self.lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")

    # ------------------------------------------------------------------ actuation
    def set_wheel_speeds(self, w_left, w_right):
        """Command wheel angular velocities [rad/s] (what a Dynamixel in velocity mode takes)."""
        lim = self.cfg.robot.max_wheel_speed
        self.wheel_cmd = np.clip([w_left, w_right], -lim, lim)

    def _firmware(self):
        """500 Hz motor firmware: velocity actuator + integral term removes friction offset."""
        d = self.data
        w = np.array([d.qvel[a] for a in self.dof_adr])
        self.integ += self.cfg.robot.firmware_ki * (self.wheel_cmd - w) * self.model.opt.timestep
        self.integ = np.clip(self.integ, -2.0, 2.0)
        for i, a in enumerate(self.act):
            d.ctrl[a] = self.wheel_cmd[i] + self.integ[i]

    # ------------------------------------------------------------------ stepping
    def step(self, duration: float):
        """Advance the world by `duration` seconds and return the SensorPacket at the end."""
        d, m = self.data, self.model
        n = int(round(duration / m.opt.timestep))
        for _ in range(n):
            self._firmware()
            self._apply_push()
            mujoco.mj_step(m, d)
            self._integrate_encoder_slip()
        self._count_collisions()
        return self.read_sensors()

    def _apply_push(self):
        dd, d = self.dist, self.data
        d.xfrc_applied[self.base_id, :] = 0
        if dd.push_time is not None and dd.push_time <= d.time < dd.push_time + dd.push_duration:
            d.xfrc_applied[self.base_id, 0:2] = dd.push_force

    def _wheel_angle(self):
        return np.array([self.data.qpos[a] for a in self.jnt_adr])

    def _integrate_encoder_slip(self):
        ang = self._wheel_angle()
        dphi = ang - self.prev_wheel_angle
        self.prev_wheel_angle = ang
        ds = np.abs(dphi) * self.cfg.robot.wheel_radius
        self.enc_slip += self.rng.normal(0, 1, 2) * self.cfg.noise.enc_slip_std_per_m * np.sqrt(ds)

    def _count_collisions(self):
        d, m = self.data, self.model
        hit = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ((b1 in self.robot_bodies and c.geom2 in self.obstacle_geoms) or
                    (b2 in self.robot_bodies and c.geom1 in self.obstacle_geoms)):
                hit = True
        if hit and not self._in_collision:
            self.collisions += 1
        self._in_collision = hit

    # ------------------------------------------------------------------ sensors
    def read_sensors(self) -> SensorPacket:
        d, cfg = self.data, self.cfg
        t = d.time
        # Encoders: true angle x scale error + slip random walk, quantised to ticks.
        ang = self._wheel_angle() * np.array(self.dist.enc_scale) + self.enc_slip
        ticks = np.floor(ang / (2 * np.pi) * cfg.robot.ticks_per_rev).astype(int)
        pkt = SensorPacket(t=t, enc_ticks=ticks)
        if t + 1e-9 >= self.next_lidar:
            self.next_lidar += cfg.timing.lidar_period
            pkt.scan, pkt.scan_angles = self._lidar(), self.scan_angles
        if self.renderer is not None and t + 1e-9 >= self.next_cam:
            self.next_cam += cfg.timing.camera_period
            if self.rng.random() >= self.dist.camera_dropout:
                pkt.image = self._camera()   # None = frame with no possible tag
        return pkt

    def _lidar(self):
        m, d, nz = self.model, self.data, self.cfg.noise
        origin = d.site_xpos[self.lidar_site].copy()
        R = d.site_xmat[self.lidar_site].reshape(3, 3)
        dirs_local = np.stack([np.cos(self.scan_angles), np.sin(self.scan_angles),
                               np.zeros_like(self.scan_angles)], 1)
        vec = (dirs_local @ R.T).reshape(-1)
        n = len(self.scan_angles)
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        # geomgroup: only group 0 (walls, obstacles, dock); markers are group 1.
        mujoco.mj_multiRay(m, d, origin, vec, np.array([1, 0, 0, 0, 0, 0], np.uint8), 1,
                           self.base_id, geomid, dist, n, nz.lidar_max)
        r = np.where(dist < 0, np.inf, dist)
        fin = np.isfinite(r)
        sigma = np.where(r[fin] < 0.5, nz.lidar_std_near, nz.lidar_rel_std_far * r[fin])
        r[fin] += self.rng.normal(0, 1, fin.sum()) * sigma * self.dist.lidar_noise_scale
        r[(r < nz.lidar_min) | (r > nz.lidar_max)] = np.inf
        r[self.rng.random(n) < nz.lidar_dropout] = np.inf
        return r

    def _any_marker_in_view(self):
        """Speed-up only: True if some marker centre is inside the camera frustum (+margin)
        and faces the camera. If not, the rendered image could contain no tag, so rendering
        is skipped. Uses the true pose, but only to decide whether to *render*."""
        cid = 0
        pos = self.data.cam_xpos[cid]
        R = self.data.cam_xmat[cid].reshape(3, 3)
        for lm in self.spec.landmarks:
            p = np.array([lm.x, lm.y, lm.z]) - pos
            pc = R.T @ p                       # MuJoCo camera frame, looks along -z
            depth = -pc[2]
            if depth < 0.05 or depth > 40 * lm.size:
                continue
            if abs(pc[0] / depth) > 0.75 or abs(pc[1] / depth) > 0.55:   # FOV + margin
                continue
            n = np.array([np.cos(lm.yaw), np.sin(lm.yaw), 0.0])
            if n @ p < 0:                      # marker faces the camera
                return True
        return False

    def _camera(self):
        if self.fast_camera and not self._any_marker_in_view():
            return None
        self.renderer.update_scene(self.data, camera="cam_front")
        img = self.renderer.render().astype(np.float32)
        s = self.cfg.noise.pixel_noise_std * self.dist.pixel_noise_scale
        img += self.rng.normal(0, s, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ evaluation only
    def ground_truth_for_evaluation(self):
        """TRUE base pose (x, y, yaw) in W.  EVALUATION / LOGGING ONLY - never fed back."""
        d = self.data
        return np.array([d.qpos[0], d.qpos[1], quat_to_yaw(d.qpos[3:7])])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
