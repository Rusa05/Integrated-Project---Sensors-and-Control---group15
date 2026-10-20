"""State-space model of the TurtleBot3 Waffle Pi and its structural analysis (Part A).

Full model (5 states), continuous time
--------------------------------------
    state  X = [x, y, theta, w_l, w_r]        (pose in W, wheel angular speeds)
    input  u = [w_l_cmd, w_r_cmd]             (wheel speed set-points to the motor firmware)

    x_dot     = r/2 (w_l + w_r) cos(theta)
    y_dot     = r/2 (w_l + w_r) sin(theta)
    theta_dot = r/b (w_r - w_l)
    w_i_dot   = (w_i_cmd - w_i) / tau          (closed motor loop, first order, tau identified)

Because tau (~10-20 ms) is far faster than the 20 Hz outer loop and the ~0.5 Hz tracking
bandwidth, the controller and estimator use the reduced kinematic (unicycle) model
    s = [x, y, theta],  u = [v, w],   s_dot = [v cos th, v sin th, w]
with v = r (w_l + w_r)/2 and w = r (w_r - w_l)/b.

Tracking-error model (used by the LQR)
--------------------------------------
Error in the ROBOT frame relative to a reference (x_r, y_r, th_r, v_r, w_r):
    e = [e_x, e_y, e_th],  [e_x, e_y] = R(th)^T ([x_r, y_r] - [x, y]),  e_th = th_r - th
    e_x_dot  =  w e_y - v + v_r cos e_th
    e_y_dot  = -w e_x + v_r sin e_th
    e_th_dot =  w_r - w
With the feedback inputs u1 = v_r cos e_th - v, u2 = w_r - w, linearised about e = 0:
    A(v_r, w_r) = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],   B = [[1, 0], [0, 0], [0, 1]]

Measurement model (used by the EKF), landmark i at (l_x, l_y) with normal yaw l_psi
-----------------------------------------------------------------------------------
    h(s) = [ sqrt(dx^2 + dy^2),  atan2(dy, dx) - theta,  l_psi - theta ],  dx = l_x - x, dy = l_y - y
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def unicycle_jacobians(s, u):
    """Continuous-time Jacobians of the unicycle about (s, u)."""
    th = s[2]
    v = u[0]
    A = np.array([[0, 0, -v * np.sin(th)], [0, 0, v * np.cos(th)], [0, 0, 0.0]])
    B = np.array([[np.cos(th), 0], [np.sin(th), 0], [0, 1.0]])
    return A, B


def full_model_linear(s, wl, wr, r, b, tau):
    """5-state model Jacobians about (pose s, wheel speeds wl, wr)."""
    th = s[2]
    A = np.zeros((5, 5))
    A[0, 2] = -r / 2 * (wl + wr) * np.sin(th)
    A[1, 2] = r / 2 * (wl + wr) * np.cos(th)
    A[0, 3] = A[0, 4] = r / 2 * np.cos(th)
    A[1, 3] = A[1, 4] = r / 2 * np.sin(th)
    A[2, 3], A[2, 4] = -r / b, r / b
    A[3, 3] = A[4, 4] = -1 / tau
    B = np.zeros((5, 2))
    B[3, 0] = B[4, 1] = 1 / tau
    return A, B


def error_model(v_r, w_r):
    A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0.0]])
    B = np.array([[1, 0], [0, 0], [0, 1.0]])
    return A, B


def discretise(A, B, dt):
    """Zero-order-hold discretisation via the matrix exponential."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def obsv(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def meas_jacobian(s, lm, use_yaw=True):
    dx, dy = lm[0] - s[0], lm[1] - s[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    H = np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0], [0.0, 0.0, -1.0]])
    return H if use_yaw else H[:2]


def local_observability_gramian(traj, landmarks_seen, dt, use_yaw):
    """Discrete observability Gramian of the linearised (time-varying) system along a path.

    traj: Nx5 [x, y, th, v, w]; landmarks_seen: list (per step) of landmark arrays (k x 3).
    Measurement contributions are whitened by unit noise; the Gramian rank / conditioning shows
    which directions are observable from that sensing history.
    """
    n = 3
    Phi = np.eye(n)
    G = np.zeros((n, n))
    for k in range(len(traj)):
        s, u = traj[k, :3], traj[k, 3:5]
        for lm in landmarks_seen[k]:
            H = meas_jacobian(s, lm, use_yaw)
            G += Phi.T @ H.T @ H @ Phi
        A, _ = unicycle_jacobians(s, u)
        Ad = np.eye(n) + A * dt
        Phi = Ad @ Phi
    return G
