"""The EKF that estimates where the robot is: x, y and heading (Part C).

  predict: moves the estimate forward using the wheel encoders (every 20 ms)
  update:  corrects it using each ArUco tag the camera sees (range, bearing and tag yaw)
It also handles camera delay, rejects bad readings, and recovers if it keeps rejecting them.
How the noise values below were chosen is explained in part_C_estimation/README.md.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np

from src import config, model as rm

# --- process noise (wheel-level) ------------------------------------------------------
SLIP_FRAC = 0.01            # 1 % of wheel travel, 1 sigma
SIGMA_PHI_MIN = 1e-5        # rad per tick

# --- measurement noise: sigma(r) = a * r^p (Part B fits for the 10 cm tag) with floors -----
REF_SIZE = 0.10                            # m, the tag the noise model was measured on
RANGE_BIAS_K = 0.0067                      # m per m^2: corrected range = r - K r^2 (10 cm tag)
SIG_RANGE = (0.00107, 3.33, 0.02)          # a [m], p, floor [m]
SIG_BEARING = (0.000117, 1.08, math.radians(0.5))
SIG_YAW = (0.0110, 2.68, math.radians(2.0))
# With several tags in view, each tag's viewpoint-dependent error (a few mm and tenths of a degree,
# repeated frame after frame) leaves one direction poorly constrained: a sideways offset paired with
# a heading error, which the clustered dock tags see as nearly the same bearings. Fusing every tag
# at the noise level above made the filter about 3x over-confident (mean NEES 6.4 over 6 random
# missions). Scaling R by 3 gave mean NEES 3.0 (ideal) with unchanged accuracy (1.47 vs 1.44 cm), so
# R is scaled by R_INFLATION; r_scale (the tuning study's x0.1 ... x10) multiplies on top of it.
R_INFLATION = 3.0
MIN_UPDATE_DT = 0.2         # s; fuse at most 5 camera frames per second
# A tag's error depends on the viewpoint, so frames from (almost) the same viewpoint repeat the same
# error and carry no new information. With several tags in view, fusing them every 0.2 s while the
# robot stood still or turned on the spot made the filter over-confident in heading (NEES 14-24 in
# FINAL_ALIGN). So a new frame is fused only once the robot has moved MIN_MOVE or turned MIN_TURN
# since the last fused frame, or MAX_WAIT has passed.
MIN_MOVE = 0.02             # m
MIN_TURN = math.radians(2.0)
MAX_WAIT = 2.0              # s
HISTORY = 1.0               # s of predict steps kept for latency compensation
# Readings further than this (for the 10 cm tag; scaled by size for others) are not fused. Beyond
# it the yaw is too noisy to use, and range+bearing to one tag cannot fix the heading (rank 2):
# while the heading is uncertain, a far update slides the estimate around the tag the wrong way
# (Part E: three accepted updates at 2.6-2.8 m took the error from 5 to 19 cm).
MAX_FUSE_RANGE = 2.0
# The tag yaw degrades faster than range and bearing: at 2x the reference range it is 12-16 deg RMS
# (IPPE flips plus noise) against 0.8-2 deg at up to 1.75x (measured on 3, 10 and 15 cm tags,
# part_B_sensing/sensor_noise.py). Beyond 1.75x only range and bearing are fused.
YAW_MAX_RANGE = 1.75
# Small tags give a biased yaw: the 3 cm dock tags read +-0.4 deg docked and up to +-5 deg at 0.6 m,
# with only 0.1-0.3 deg scatter (a consistent, viewpoint-dependent error the filter cannot average
# out). Their range and bearing are excellent, and two of them 10 cm apart already make the pose
# observable (rank 3), so tags smaller than this only contribute range and bearing.
YAW_MIN_SIZE = 0.05         # m

# Gate lock-out recovery: if many readings in a row fail the gate, the filter is more likely wrong
# than the camera. After REJECT_LIMIT consecutive rejections, each further rejection inflates P.
REJECT_LIMIT = 5
P_INFLATE = np.diag([0.03 ** 2, 0.03 ** 2, math.radians(2) ** 2])

# --- innovation gates (chi-square, 99 %) ------------------------------------------------
GATE_3DOF = 11.345
GATE_2DOF = 9.210


def _sigma(params, r, k):
    """Noise of a tag k times the reference size at range r (pinhole similarity)."""
    a, p, floor = params
    return max(k * a * (r / k) ** p, floor)


@dataclass
class UpdateResult:
    used: str                 # 'pose', 'range_bearing', or 'rejected'
    nis: float                # NIS of the innovation that was tested last
    dof: int
    innovation: np.ndarray = field(repr=False)
    tag: int = None
    flipped: bool = False     # True if the other IPPE solution was the consistent one


class EKF:
    """mode: 'pose' (range, bearing and tag yaw; default), 'range_bearing' (never use the yaw)
    or 'odometry' (ignore the camera): the last two exist for the measurement ablation."""

    def __init__(self, x0, P0, slip_frac=SLIP_FRAC, r_scale=1.0, min_update_dt=MIN_UPDATE_DT,
                 tags=config.TAGS, mode='pose'):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.slip_frac = slip_frac
        self.r_scale = r_scale          # multiplies R (used for the tuning study)
        self.min_update_dt = min_update_dt   # s; skip frames closer together than this
        self._last_update_t = -math.inf
        self.tags = tags
        self.mode = mode
        self.rejections = 0             # consecutive gate rejections (lock-out detection)
        self.inflations = 0
        self.flips = 0                  # IPPE ambiguities resolved in favour of the other solution
        self._moved = self._turned = math.inf    # motion since the last fused frame (from odometry)
        # (time after the step, wheel increments, state and covariance before the step)
        self.history = deque(maxlen=int(round(HISTORY / rm.DT)))

    # ---------------------------------------------------------------------------------
    def predict(self, dphi, t=None):
        """dphi = [left, right] wheel angle increments (rad) since the last call; t = time after
        the step (pass it to enable latency compensation in update())."""
        if t is not None:
            self.history.append((t, np.array(dphi, float), self.x.copy(), self.P.copy()))
        return self._predict(dphi)

    def _predict(self, dphi):
        dl, dr = dphi
        r, b = rm.WHEEL_RADIUS, rm.WHEEL_SEPARATION
        ds = r * (dl + dr) / 2
        dth = r * (dr - dl) / b
        self._moved += abs(ds)
        self._turned += abs(dth)
        th_m = self.x[2] + dth / 2
        c, s = math.cos(th_m), math.sin(th_m)

        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1]])
        G_motion = np.array([[c, -ds / 2 * s], [s, ds / 2 * c], [0, 1]])   # d x / d(ds, dth)
        G_w = G_motion @ rm.wheel_increment_jacobian()                     # d x / d(dphi_L, dphi_R)
        Sw = np.diag([(self.slip_frac * dl) ** 2 + SIGMA_PHI_MIN ** 2,
                      (self.slip_frac * dr) ** 2 + SIGMA_PHI_MIN ** 2])
        Q = G_w @ Sw @ G_w.T

        self.x = np.array([self.x[0] + ds * c, self.x[1] + ds * s, rm.wrap(self.x[2] + dth)])
        self.P = F @ self.P @ F.T + Q
        return Q

    # ---------------------------------------------------------------------------------
    def measurement_noise(self, rng, size=REF_SIZE):
        k = size / REF_SIZE
        return R_INFLATION * self.r_scale * np.diag([_sigma(SIG_RANGE, rng, k) ** 2,
                                       _sigma(SIG_BEARING, rng, k) ** 2,
                                       _sigma(SIG_YAW, rng, k) ** 2])

    @staticmethod
    def corrected(meas, solution=None):
        """[range, bearing, rel_yaw] with the range bias removed. solution = meas.alt for the other
        IPPE solution; the bias scales as K r^2 / k for a tag k times the reference size."""
        rng, brg, yaw = solution if solution is not None else (meas.range, meas.bearing, meas.rel_yaw)
        k = meas.size / REF_SIZE
        return np.array([rng - RANGE_BIAS_K * rng ** 2 / k, brg, yaw])

    def _innovation(self, z, h, H, R):
        nu = z - h
        nu[1:] = [rm.wrap(a) for a in nu[1:]]
        S = H @ self.P @ H.T + R
        return nu, S, float(nu @ np.linalg.solve(S, nu))

    def _apply(self, nu, S, H, R):
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = rm.wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form: stays symmetric PSD

    def update(self, meas, t=None, t_meas=None):
        """Fuse one tag measurement. Tries the full pose; falls back to range+bearing if the yaw
        is out of range or fails the gate.

        t       current time; t_meas  time the camera frame was captured. If the frame is older
        than the current state (camera latency), the filter rewinds to t_meas, applies the update
        there and replays the stored wheel increments up to now, so a late measurement is compared
        with where the robot *was*, not where it is now.
        Several tags from the same frame (same t_meas) are all fused; frames closer together than
        min_update_dt are skipped. Returns None if skipped, out of range or older than the history."""
        stamp = t_meas if t_meas is not None else t
        new_frame = stamp is None or stamp - self._last_update_t > 1e-9
        if new_frame and stamp is not None:
            dt = stamp - self._last_update_t
            if dt < self.min_update_dt - 1e-9:
                return None
            if self._moved < MIN_MOVE and self._turned < MIN_TURN and dt < MAX_WAIT:
                return None                                  # same viewpoint: no new information
        replay = []
        if t_meas is not None and self.history and t_meas < self.history[-1][0] - 1e-9:
            if t_meas < self.history[0][0] - rm.DT - 1e-9:
                return None                                  # older than the stored history
            j = next(k for k, h in enumerate(self.history) if h[0] > t_meas + 1e-9)
            replay = list(self.history)[j:]
            for _ in range(len(replay)):
                self.history.pop()
            self.x, self.P = replay[0][2].copy(), replay[0][3].copy()
        result = self._update_now(meas)
        if stamp is not None and result is not None:
            if new_frame:
                self._moved = self._turned = 0.0
            self._last_update_t = stamp
        for t_k, dphi_k, _, _ in replay:                      # roll forward to the present
            self.history.append((t_k, dphi_k, self.x.copy(), self.P.copy()))
            self._predict(dphi_k)
        return result

    def _update_now(self, meas):
        if self.mode == 'odometry' or meas.id not in self.tags:
            return None
        k = meas.size / REF_SIZE
        if meas.range > k * MAX_FUSE_RANGE:
            return None
        result = self._fuse(meas, k)
        if result.used == 'rejected':
            self.rejections += 1
            if self.rejections >= REJECT_LIMIT:
                self.P = self.P + P_INFLATE
                self.inflations += 1
        else:
            self.rejections = 0
        return result

    def _fuse(self, meas, k):
        tag = self.tags[meas.id]
        m = (tag.x, tag.y)
        R = self.measurement_noise(meas.range, meas.size)
        if self.mode == 'pose' and meas.size >= YAW_MIN_SIZE and meas.range <= k * YAW_MAX_RANGE:
            h = rm.h_range_bearing_yaw(self.x, m, tag.yaw)
            H = rm.H_range_bearing_yaw(self.x, m)
            # IPPE ambiguity: keep whichever of the two solutions agrees with the prior
            cands = [(self.corrected(meas), False)]
            if meas.alt is not None:
                cands.append((self.corrected(meas, meas.alt), True))
            (nu, S, nis), flip = min(((self._innovation(z, h, H, R), f) for z, f in cands),
                                     key=lambda c: c[0][2])
            if nis <= GATE_3DOF:
                self._apply(nu, S, H, R)
                self.flips += flip
                return UpdateResult('pose', nis, 3, nu, meas.id, flip)
        z = self.corrected(meas)[:2]
        h = rm.h_range_bearing(self.x, m)
        H = rm.H_range_bearing(self.x, m)
        nu, S, nis = self._innovation(z, h, H, R[:2, :2])
        if nis > GATE_2DOF:
            return UpdateResult('rejected', nis, 2, nu, meas.id)
        self._apply(nu, S, H, R[:2, :2])
        return UpdateResult('range_bearing', nis, 2, nu, meas.id)

    # ---------------------------------------------------------------------------------
    @staticmethod
    def pose_from_marker(meas, tags=config.TAGS):
        """Invert one measurement into a robot pose (for initialising without a prior)."""
        tag = tags[meas.id]
        rng, brg, rel_yaw = EKF.corrected(meas)
        th = rm.wrap(tag.yaw - rel_yaw)
        return np.array([tag.x - rng * math.cos(th + brg), tag.y - rng * math.sin(th + brg), th])
