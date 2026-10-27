"""The robot's hardware interface in simulation: sensors at their real rates in, wheel speeds out.

Everything here is something the real TurtleBot3 would provide (encoders every tick, LiDAR at
5 Hz, camera frames at 10 Hz with on-board marker detection). It never reads the base pose.
"""
import mujoco

from src import model as rm
from src.sensors import Sensors, NoiseConfig
from src.vision import MarkerDetector

LIDAR_PERIOD = 0.2          # s (the LDS-01 spins at 5 Hz)
PHYSICS_DT = 0.002


class SimRobot:
    """faults (optional, for robustness trials):
        occluded(t) -> bool   marker detection fails while True (something covers the marker)
        latency     seconds   each marker measurement is delivered this long after its frame
    """
    def __init__(self, model, data, noise=None, seed=0, occluded=None, latency=0.0):
        self.model, self.data = model, data
        self.sensors = Sensors(model, data, noise or NoiseConfig(), seed=seed)
        self.detector = MarkerDetector.from_sensors(self.sensors)
        self._act = [model.actuator('wheel_left').id, model.actuator('wheel_right').id]
        self._enc_prev = self.sensors.encoders()
        self._next_lidar = data.time
        self.last_frame = None          # most recent camera image (for the video overlay)
        self.last_marker = None
        self.occluded = occluded
        self.latency = latency
        self._pending = []              # (release time, measurement) for camera latency

    @property
    def time(self):
        return self.data.time

    def sense(self):
        """Returns (encoder increments [rad], marker measurement or None, LiDAR scan or None)."""
        enc = self.sensors.encoders()
        dphi, self._enc_prev = enc - self._enc_prev, enc
        meas = None
        if self.sensors.camera_due():
            self.last_frame = self.sensors.camera()
            m = self.detector.measure(self.last_frame)
            if m is not None:
                m.t = self.data.time                        # camera timestamp
            if m is not None and self.occluded is not None and self.occluded(self.data.time):
                m = None
            self.last_marker = m
            if m is not None:
                self._pending.append((self.data.time + self.latency, m))
        if self._pending and self._pending[0][0] <= self.data.time + 1e-9:
            meas = self._pending.pop(0)[1]
        scan = None
        if self.data.time >= self._next_lidar - 1e-9:
            scan = self.sensors.lidar()
            self._next_lidar += LIDAR_PERIOD
        return dphi, meas, scan

    def act(self, v, omega):
        self.data.ctrl[self._act] = rm.body_to_wheels(v, omega)
        mujoco.mj_step(self.model, self.data, nstep=round(rm.DT / PHYSICS_DT))

    def close(self):
        self.sensors.close()
