"""Differential-drive kinematics with the calibrated (speed-dependent) odometry model.

Calibration (scripts/02_calibrate_odometry.py) showed three platform effects that the
textbook model (r = 33 mm, b = 287 mm, rotation about the axle midpoint) misses:

  1. the effective rolling radius depends on wheel speed and direction (longitudinal slip
     against the dragging rear casters grows with forward speed)  -> lookup table r_eff(w)
  2. the turning gain is ~20 % lower than r/b (tyre scrub)        -> calibrated g
  3. the instantaneous centre of rotation (ICR) is NOT the axle midpoint: on the spot the
     robot turns about a point x_icr ahead/behind the axle, so base_link moves sideways
     by -x_icr * dtheta                                            -> calibrated x_icr

    ds     = r_eff(w_mean) * (dphi_l + dphi_r) / 2       (forward, body x)
    dlat   = -x_icr * dtheta                              (sideways, body y)
    dtheta = g * (dphi_r - dphi_l)

With the datasheet values: r_eff = 0.033 everywhere, g = 0.033 / 0.288, x_icr = 0.
"""
from __future__ import annotations

import numpy as np

from .config import Config


class DiffDrive:
    def __init__(self, cfg: Config):
        rb = cfg.robot
        if rb.radius_table is not None:
            self.w_tab = np.asarray(rb.radius_table[0], float)
            self.r_tab = np.asarray(rb.radius_table[1], float)
        else:
            self.w_tab = np.array([-10.0, 10.0])
            self.r_tab = np.array([rb.wheel_radius, rb.wheel_radius])
        self.g = rb.turn_gain if rb.turn_gain is not None else rb.wheel_radius / rb.wheel_base
        self.x_icr = rb.icr_x
        self.b_equiv = rb.wheel_radius / self.g     # equivalent track width, for reporting
        self.max_w = rb.max_wheel_speed
        # monotonic map w_mean -> v for the inverse
        ww = np.linspace(-self.max_w, self.max_w, 401)
        self._ww, self._vv = ww, ww * self.r(ww)

    def r(self, w_mean):
        return np.interp(w_mean, self.w_tab, self.r_tab)

    def forward(self, dphi_l, dphi_r, dt):
        """Wheel rotations over dt -> (ds, dtheta) of the base (ds along body x)."""
        w_mean = (dphi_l + dphi_r) / (2 * dt) if dt > 0 else 0.0
        ds = self.r(w_mean) * (dphi_l + dphi_r) / 2
        return ds, self.g * (dphi_r - dphi_l)

    def forward3(self, dphi_l, dphi_r, dt):
        """(ds, dlat, dtheta): includes the sideways motion from the ICR offset."""
        ds, dth = self.forward(dphi_l, dphi_r, dt)
        return ds, -self.x_icr * dth, dth

    def integrate(self, pose, dphi_l, dphi_r, dt):
        """Midpoint integration of one odometry step (used by calibration and the EKF)."""
        ds, dlat, dth = self.forward3(dphi_l, dphi_r, dt)
        th = pose[2] + dth / 2
        c, s = np.cos(th), np.sin(th)
        return np.array([pose[0] + ds * c - dlat * s, pose[1] + ds * s + dlat * c, pose[2] + dth])

    def inverse(self, v, w):
        """Desired (v [m/s], w [rad/s]) -> wheel speed commands (left, right) [rad/s]."""
        w_mean = np.interp(v, self._vv, self._ww)
        half = w / (2 * self.g)
        wl, wr = w_mean - half, w_mean + half
        # scale both down together if a wheel would saturate (keeps the path curvature)
        s = max(1.0, abs(wl) / self.max_w, abs(wr) / self.max_w)
        return wl / s, wr / s

    def wheel_sigma(self, dphi, k_s, floor):
        """Std of one wheel's arc-length increment: k_s sqrt|ds| + quantisation floor."""
        return k_s * np.sqrt(np.abs(dphi) * self.r_tab.mean()) + floor
