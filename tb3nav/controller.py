"""LQR trajectory tracker on the linearised tracking-error model (see model.py).

At every control step:
  1. e = [e_x, e_y, e_th] from the ESTIMATED pose and the reference (x_r, y_r, th_r, v_r, w_r)
  2. (A, B) = error_model(v_r, w_r), discretised at the control period (ZOH)
  3. K = dlqr(Ad, Bd, Q, R), gain-scheduled on (v_r, w_r) (cached on a grid)
  4. u_fb = -K e;   v = v_r cos e_th - u1,   w = w_r - u2
  5. saturate v, w (scaling both to keep curvature), convert to wheel speeds.

Q and R from Bryson's rule (config.LQRParams). The controller never sees the true pose: its
only pose input is the EKF state passed in by mission.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_discrete_are

from .config import Config
from .frames import rot2, wrap
from .model import discretise, error_model


@lru_cache(maxsize=4096)
def _dlqr_cached(v_r, w_r, dt, q, r):
    A, B = error_model(v_r, w_r)
    Ad, Bd = discretise(A, B, dt)
    Q, R = np.diag(q), np.diag(r)
    # At v_r = w_r = 0 the error model loses controllability of e_y (Part A), so the caller
    # clamps v_r >= 1 cm/s and pure rotations use heading_controller() instead.
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K


class LQRTracker:
    def __init__(self, cfg: Config, r_scale: float = 1.0):
        L = cfg.lqr
        self.dt = cfg.timing.control_dt
        self.q = tuple(1.0 / np.array(L.e_max) ** 2)
        self.r = tuple(r_scale / np.array(L.u_max) ** 2)
        self.v_max, self.w_max = L.v_max * 1.3, L.w_max

    def gain(self, v_r, w_r):
        # Quantise the schedule so the cache stays small (1 cm/s and 0.02 rad/s steps).
        v_q = round(max(v_r, 0.01) / 0.01) * 0.01
        w_q = round(w_r / 0.02) * 0.02
        return _dlqr_cached(v_q, w_q, self.dt, self.q, self.r)

    @staticmethod
    def error(s_est, ref):
        d = rot2(s_est[2]).T @ (ref[:2] - s_est[:2])
        return np.array([d[0], d[1], wrap(ref[2] - s_est[2])])

    def command(self, s_est, ref):
        """ref = [x_r, y_r, th_r, v_r, w_r] -> (v, w, e, u_fb)."""
        e = self.error(s_est, ref)
        v_r, w_r = ref[3], ref[4]
        K = self.gain(v_r, w_r)
        u = -K @ e
        v = v_r * np.cos(e[2]) - u[0]
        w = w_r - u[1]
        v = np.clip(v, -0.05, self.v_max)       # little reversing allowed
        w = np.clip(w, -self.w_max, self.w_max)
        return v, w, e, u


def heading_controller(s_est, th_target, kp=1.5, w_max=0.6, w_min=0.08):
    """In-place rotation (used where v_r = 0 makes the LQR error model uncontrollable)."""
    err = wrap(th_target - s_est[2])
    w = np.clip(kp * err, -w_max, w_max)
    if abs(w) < w_min and abs(err) > 0.01:
        w = np.sign(w) * w_min
    return 0.0, w, err
