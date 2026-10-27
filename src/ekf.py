"""Extended Kalman filter for the robot pose [x, y, theta] (Part C).

Predict  every control tick (50 Hz) from wheel-encoder increments.
Update   whenever the camera sees the charging-station marker (up to 10 Hz), with
         z = [range, bearing, marker yaw relative to the robot] (a relative pose measurement).

Why a pose measurement: with range + bearing to a single marker the pose is not observable
(rank 2, see part_A_modelling); the marker's yaw supplies the missing direction.

Noise values and where they come from
-------------------------------------
Process noise Q is built from the wheels, not tuned on x/y/theta directly. Each wheel's
angle increment over one tick has variance
    sigma_phi^2 = (SLIP_FRAC * |dphi|)^2 + SIGMA_PHI_MIN^2
  * SLIP_FRAC    unmodelled slip/scrub as a fraction of wheel travel: the error that really
                 accumulates. Phase 2 measured 0.35 % position drift over 7.3 m.
  * SIGMA_PHI_MIN a tiny floor so Q is never singular at standstill.
Encoder quantisation (4096 ticks/rev) and per-reading jitter are deliberately NOT in Q: they are
errors on the *absolute* wheel angle, so successive increments cancel (the sum telescopes) and
they do not grow with distance. Putting them in Q at 50 Hz made the filter far too pessimistic
when driving blind (NEES well below its bounds); see part_C_estimation/README.md.
Q = G_w diag(sigma_phi_L^2, sigma_phi_R^2) G_w^T with G_w = d f / d(dphi_L, dphi_R).

Measurement noise R uses the power laws fitted in part_B_sensing/results/summary.json,
sigma(r) = a * r^p, with floors (2 cm range, 0.5 deg bearing, 2 deg yaw). The fits see only
frame-to-frame pixel noise, but the camera's real error is mostly a small systematic offset that
depends on the viewpoint and repeats in consecutive frames (~0.5 cm in range at 1 m). Fusing it as
if it were independent made the filter over-confident (NEES up to ~18). Two measures, chosen with
the NEES/NIS study in part_C_estimation:
  * floors on R sized to that systematic error, and
  * at most 5 updates per second (MIN_UPDATE_DT), so near-identical frames are not
    counted as independent evidence.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np

from src import config, model as rm

# --- process noise (wheel-level) ------------------------------------------------------
SLIP_FRAC = 0.01            # 1 % of wheel travel, 1 sigma
SIGMA_PHI_MIN = 1e-5        # rad per tick

# --- measurement noise: sigma(r) = a * r^p (phase 2 fits) with floors -----------------
RANGE_BIAS_K = 0.0067                      # m per m^2: corrected range = r - K r^2
SIG_RANGE = (0.00107, 3.33, 0.02)          # a [m], p, floor [m]
SIG_BEARING = (0.000117, 1.08, math.radians(0.5))
SIG_YAW = (0.0110, 2.68, math.radians(2.0))
MIN_UPDATE_DT = 0.2         # s; fuse at most 5 camera frames per second
HISTORY = 1.0               # s of predict steps kept for latency compensation
YAW_MAX_RANGE = 2.0         # m; beyond this the marker yaw is too noisy/ambiguous to use
# Marker readings further than this are not fused at all. Without the yaw, range+bearing to one
# marker cannot fix the heading (rank 2), and while the heading is uncertain a far update slides
# the estimate around the marker the wrong way (phase 6: three accepted updates at 2.6-2.8 m took
# the error from 5 to 19 cm). So beyond the yaw range the camera is ignored.
MAX_FUSE_RANGE = YAW_MAX_RANGE

# Gate lock-out recovery: if many readings in a row fail the gate, the filter is more likely wrong
# than the camera. After REJECT_LIMIT consecutive rejections, each further rejection inflates P.
REJECT_LIMIT = 5                                            # 1 s at 5 Hz
P_INFLATE = np.diag([0.03 ** 2, 0.03 ** 2, math.radians(2) ** 2])

# --- innovation gates (chi-square, 99 %) ------------------------------------------------
GATE_3DOF = 11.345
GATE_2DOF = 9.210


def _sigma(params, r):
    a, p, floor = params
    return max(a * r ** p, floor)


@dataclass
class UpdateResult:
    used: str                 # 'pose', 'range_bearing', or 'rejected'
    nis: float                # NIS of the innovation that was tested last
    dof: int
    innovation: np.ndarray = field(repr=False)


class EKF:
    def __init__(self, x0, P0, slip_frac=SLIP_FRAC, r_scale=1.0, min_update_dt=MIN_UPDATE_DT,
                 marker_xy=config.MARKER_XY, marker_yaw=config.MARKER_YAW):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.slip_frac = slip_frac
        self.r_scale = r_scale          # multiplies R (used for the tuning study)
        self.min_update_dt = min_update_dt   # s; skip frames closer together than this
        self._last_update_t = -math.inf
        self.m = np.array(marker_xy, dtype=float)
        self.m_yaw = marker_yaw
        self.rejections = 0             # consecutive gate rejections (lock-out detection)
        self.inflations = 0
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
    def measurement_noise(self, rng):
        return self.r_scale * np.diag([_sigma(SIG_RANGE, rng) ** 2,
                                       _sigma(SIG_BEARING, rng) ** 2,
                                       _sigma(SIG_YAW, rng) ** 2])

    @staticmethod
    def corrected(meas):
        """[range, bearing, rel_yaw] from a vision.MarkerMeasurement, with the range bias removed."""
        return np.array([meas.range - RANGE_BIAS_K * meas.range ** 2, meas.bearing, meas.rel_yaw])

    def _try(self, z, h, H, R, gate):
        nu = z - h
        nu[1:] = [rm.wrap(a) for a in nu[1:]]
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > gate:
            return False, nis, nu
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = rm.wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form: stays symmetric PSD
        return True, nis, nu

    def update(self, meas, t=None, t_meas=None):
        """Fuse one marker measurement. Tries the full pose; falls back to range+bearing if the
        yaw is out of range or fails the gate (e.g. an IPPE pose flip).

        t       current time; t_meas  time the camera frame was captured. If the frame is older
        than the current state (camera latency), the filter rewinds to t_meas, applies the update
        there and replays the stored wheel increments up to now, so a late measurement is compared
        with where the robot *was*, not where it is now.
        Returns None if the frame was skipped by min_update_dt or is older than the history."""
        stamp = t_meas if t_meas is not None else t
        if stamp is not None:
            if stamp - self._last_update_t < self.min_update_dt - 1e-9:
                return None
        replay = []
        if t_meas is not None and self.history and t_meas < self.history[-1][0] - 1e-9:
            if t_meas < self.history[0][0] - rm.DT - 1e-9:
                return None                                  # older than the stored history
            j = next(k for k, h in enumerate(self.history) if h[0] > t_meas + 1e-9)
            replay = list(self.history)[j:]
            for _ in range(len(replay)):
                self.history.pop()
            self.x, self.P = replay[0][2].copy(), replay[0][3].copy()
        if stamp is not None:
            self._last_update_t = stamp
        result = self._update_now(meas)
        for t_k, dphi_k, _, _ in replay:                      # roll forward to the present
            self.history.append((t_k, dphi_k, self.x.copy(), self.P.copy()))
            self._predict(dphi_k)
        return result

    def _update_now(self, meas):
        if meas.range > MAX_FUSE_RANGE:
            return None
        result = self._fuse(meas)
        if result.used == 'rejected':
            self.rejections += 1
            if self.rejections >= REJECT_LIMIT:
                self.P = self.P + P_INFLATE
                self.inflations += 1
        else:
            self.rejections = 0
        return result

    def _fuse(self, meas):
        z = self.corrected(meas)
        R = self.measurement_noise(z[0])
        if z[0] <= YAW_MAX_RANGE:
            h = rm.h_range_bearing_yaw(self.x, self.m, self.m_yaw)
            H = rm.H_range_bearing_yaw(self.x, self.m)
            ok, nis, nu = self._try(z, h, H, R, GATE_3DOF)
            if ok:
                return UpdateResult('pose', nis, 3, nu)
        h = rm.h_range_bearing(self.x, self.m)
        H = rm.H_range_bearing(self.x, self.m)
        ok, nis, nu = self._try(z[:2], h, H, R[:2, :2], GATE_2DOF)
        return UpdateResult('range_bearing' if ok else 'rejected', nis, 2, nu)

    # ---------------------------------------------------------------------------------
    @staticmethod
    def pose_from_marker(meas, marker_xy=config.MARKER_XY, marker_yaw=config.MARKER_YAW):
        """Invert one measurement into a robot pose (for initialising without a prior)."""
        rng, brg, rel_yaw = EKF.corrected(meas)
        th = rm.wrap(marker_yaw - rel_yaw)
        return np.array([marker_xy[0] - rng * math.cos(th + brg),
                         marker_xy[1] - rng * math.sin(th + brg), th])
