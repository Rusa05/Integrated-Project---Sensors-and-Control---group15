"""The robot's "hardware" in the simulation: reads the sensors and drives the wheels.

Encoders every 20 ms, LiDAR every 0.2 s, camera every 0.1 s. It never looks at the true position.
It can also fake camera problems (blinded camera, delay) for the robustness tests.
"""
import mujoco

from src import model as rm
from src.sensors import Sensors, NoiseConfig
from src.vision import MarkerDetector

LIDAR_PERIOD = 0.2          # s (the LDS-01 spins at 5 Hz)
PHYSICS_DT = 0.002


class SimRobot:
    """faults (optional, for robustness trials):
        occluded(t) -> bool   no tag is detected while True (camera blocked or blinded)
        latency     seconds   each frame's tag measurements are delivered this long after the frame
    """
    def __init__(self, model, data, noise=None, seed=0, occluded=None, latency=0.0):
        self.model, self.data = model, data
        self.sensors = Sensors(model, data, noise or NoiseConfig(), seed=seed)
        self.detector = MarkerDetector.from_sensors(self.sensors)
        self._act = [model.actuator('wheel_left').id, model.actuator('wheel_right').id]
        self._enc_prev = self.sensors.encoders()
        self._next_lidar = data.time
        self.last_frame = None          # most recent camera image (for the video overlay)
        self.last_markers = []          # tags detected in that image
        self.occluded = occluded
        self.latency = latency
        self._pending = []              # (release time, measurement) for camera latency

    @property
    def time(self):
        return self.data.time

    def sense(self):
        """Returns (encoder increments [rad], list of tag measurements delivered this tick,
        LiDAR scan or None)."""
        enc = self.sensors.encoders()
        dphi, self._enc_prev = enc - self._enc_prev, enc
        meas = []
        if self.sensors.camera_due():
            self.last_frame = self.sensors.camera()
            ms = self.detector.measure_all(self.last_frame)
            for m in ms:
                m.t = self.data.time                        # camera timestamp
            if self.occluded is not None and self.occluded(self.data.time):
                ms = []
            self.last_markers = ms
            if ms:
                self._pending.append((self.data.time + self.latency, ms))
        while self._pending and self._pending[0][0] <= self.data.time + 1e-9:
            meas += self._pending.pop(0)[1]
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
