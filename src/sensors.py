"""Adds realistic noise to MuJoCo's perfect sensors (Part B).

  encoders: rounded to 4096 ticks per turn, like the real motors
  LiDAR:    noise and missing beams based on the LDS-01 datasheet
  camera:   10 Hz images with pixel noise
Also has Odometry, which adds up the wheel movement into a position.
"""
from dataclasses import dataclass
import math
import mujoco
import numpy as np

from src import config, model as robot_model

N_BEAMS = 72
LIDAR_ANGLES = np.arange(N_BEAMS) * 2 * math.pi / N_BEAMS   # 0 = forward, positive = left
TICKS_PER_REV = 4096


@dataclass
class NoiseConfig:
    encoder_sigma: float = 0.002        # rad, additive angle noise per reading
    lidar_sigma_min: float = 0.0075     # m
    lidar_sigma_frac: float = 0.025     # fraction of range
    lidar_min_range: float = 0.12       # m
    lidar_max_range: float = 3.5        # m
    lidar_dropout: float = 0.01         # probability a beam returns nothing
    pixel_sigma: float = 3.0            # 8-bit grey levels
    camera_width: int = 640
    camera_height: int = 480
    camera_rate: float = 10.0           # Hz

    @classmethod
    def ideal(cls):
        return cls(encoder_sigma=0.0, lidar_sigma_min=0.0, lidar_sigma_frac=0.0,
                   lidar_dropout=0.0, pixel_sigma=0.0)


class Sensors:
    """Wraps a MuJoCo model/data pair and returns noisy encoder, LiDAR and camera readings."""

    def __init__(self, model, data, noise=None, seed=0):
        self.model, self.data = model, data
        self.noise = noise or NoiseConfig()
        self.rng = np.random.default_rng(seed)
        self._adr = {name: model.sensor(name).adr[0] for name in
                     ['encoder_left', 'encoder_right'] + [f'lidar_{i:03d}' for i in range(N_BEAMS)]}
        self._renderer = None
        self._opt = mujoco.MjvOption()
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self._last_frame_time = -np.inf

    # -- encoders ---------------------------------------------------------
    def encoders(self):
        """Wheel angles (rad), quantised to encoder ticks, [left, right]."""
        raw = np.array([self.data.sensordata[self._adr['encoder_left']],
                        self.data.sensordata[self._adr['encoder_right']]])
        raw = raw + self.rng.normal(0, self.noise.encoder_sigma, 2) if self.noise.encoder_sigma else raw
        tick = 2 * math.pi / TICKS_PER_REV
        return np.round(raw / tick) * tick

    # -- LiDAR ------------------------------------------------------------
    def lidar(self):
        """72 ranges (m) at LIDAR_ANGLES; invalid returns are np.inf."""
        n = self.noise
        r = np.array([self.data.sensordata[self._adr[f'lidar_{i:03d}']] for i in range(N_BEAMS)])
        valid = r >= 0                               # MuJoCo returns -1 for no hit
        sigma = np.maximum(n.lidar_sigma_min, n.lidar_sigma_frac * np.where(valid, r, 0))
        r = r + self.rng.normal(0, 1, N_BEAMS) * sigma
        valid &= (r >= n.lidar_min_range) & (r <= n.lidar_max_range)
        valid &= self.rng.random(N_BEAMS) >= n.lidar_dropout
        return np.where(valid, r, np.inf)

    # -- camera -----------------------------------------------------------
    def camera_due(self):
        return self.data.time - self._last_frame_time >= 1 / self.noise.camera_rate - 1e-9

    def camera(self):
        """Noisy 8-bit greyscale image from front_camera."""
        n = self.noise
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, n.camera_height, n.camera_width)
        self._renderer.update_scene(self.data, camera='front_camera', scene_option=self._opt)
        rgb = self._renderer.render()
        grey = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if n.pixel_sigma:
            grey += self.rng.normal(0, n.pixel_sigma, grey.shape).astype(np.float32)
        self._last_frame_time = self.data.time
        return np.clip(grey, 0, 255).astype(np.uint8)

    def camera_intrinsics(self):
        """Calibrated pinhole intrinsics (checkerboard calibration, part_B_sensing/calibrate_camera.py).
        The calibration chose a pinhole model with no distortion on held-out views."""
        fx, fy, cx, cy = config.CAMERA_MATRIX
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def camera_intrinsics_from_fov(self):
        """What the field of view alone implies (pixel centres at (W-1)/2, (H-1)/2), for comparison."""
        n = self.noise
        fovy = math.radians(self.model.cam('front_camera').fovy[0])
        f = n.camera_height / (2 * math.tan(fovy / 2))
        return np.array([[f, 0, (n.camera_width - 1) / 2], [0, f, (n.camera_height - 1) / 2], [0, 0, 1]])

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class Odometry:
    """Integrates encoder increments into a dead-reckoned pose (prediction input for the EKF)."""

    def __init__(self, first_reading, x0=(0.0, 0.0, 0.0)):
        self.prev = np.array(first_reading, dtype=float)
        self.pose = np.array(x0, dtype=float)

    def update(self, reading, dt):
        """Returns (v, omega) over the last dt and advances the dead-reckoned pose."""
        d = np.array(reading) - self.prev
        self.prev = np.array(reading, dtype=float)
        v, omega = robot_model.wheels_to_body(d[0] / dt, d[1] / dt)
        self.pose = robot_model.f_discrete(self.pose, (v, omega), dt)
        return v, omega
