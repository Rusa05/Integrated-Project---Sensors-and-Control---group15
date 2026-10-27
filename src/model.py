"""The robot's maths model (Part A).

Unicycle model: x' = v cos(theta), y' = v sin(theta), theta' = omega. Everything else is built on
it: wheel speeds <-> body speeds, the 50 Hz discrete model and its Jacobians (for the EKF), the
tracking-error model (for the LQR), the tag measurement model, and the rank checks.
Assumes a flat floor, no sideways slip, and ignores motor lag (all checked in part_A_modelling).
"""
import math
import numpy as np

WHEEL_RADIUS = 0.033      # m
WHEEL_SEPARATION = 0.288  # m, wheel centre to wheel centre (official Waffle Pi: wheels at y = +-0.144)
WHEEL_SPEED_LIMIT = 7.88  # rad/s, actuator ctrlrange (official Waffle Pi; 0.26 m/s top speed)
DT = 0.02                 # s, control / odometry period (50 Hz)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


# --- wheel <-> body velocity -------------------------------------------------

def wheels_to_body(w_left, w_right):
    """Wheel angular speeds (rad/s) -> (v, omega)."""
    v = WHEEL_RADIUS * (w_right + w_left) / 2
    omega = WHEEL_RADIUS * (w_right - w_left) / WHEEL_SEPARATION
    return v, omega


def body_to_wheels(v, omega):
    """(v, omega) -> wheel angular speeds (rad/s), clipped to the actuator limit."""
    left = (v - omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    right = (v + omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
    return np.clip([left, right], -WHEEL_SPEED_LIMIT, WHEEL_SPEED_LIMIT)


# --- continuous and discrete motion model ----------------------------------

def f_continuous(x, u):
    """x_dot = f(x, u)."""
    return np.array([u[0] * math.cos(x[2]), u[0] * math.sin(x[2]), u[1]])


def f_discrete(x, u, dt=DT):
    """Midpoint (second-order) discretisation, exact for constant v when omega = 0."""
    v, w = u
    th = x[2] + w * dt / 2
    return np.array([x[0] + v * dt * math.cos(th),
                     x[1] + v * dt * math.sin(th),
                     wrap(x[2] + w * dt)])


def jacobian_state(x, u, dt=DT):
    """F = d f_discrete / d x."""
    v, w = u
    th = x[2] + w * dt / 2
    return np.array([[1, 0, -v * dt * math.sin(th)],
                     [0, 1,  v * dt * math.cos(th)],
                     [0, 0, 1]])


def jacobian_input(x, u, dt=DT):
    """G = d f_discrete / d u."""
    v, w = u
    th = x[2] + w * dt / 2
    return np.array([[dt * math.cos(th), -v * dt * dt / 2 * math.sin(th)],
                     [dt * math.sin(th),  v * dt * dt / 2 * math.cos(th)],
                     [0, dt]])


def wheel_increment_jacobian():
    """d(v*dt, omega*dt) / d(delta_phi_left, delta_phi_right): maps encoder angle noise to motion noise."""
    r, b = WHEEL_RADIUS, WHEEL_SEPARATION
    return np.array([[r / 2, r / 2], [-r / b, r / b]])


# --- linearised tracking-error model (used by the LQR tracker) --------------

def tracking_error(x, x_ref):
    """Error of the reference relative to the robot, expressed in the robot frame."""
    c, s = math.cos(x[2]), math.sin(x[2])
    dx, dy = x_ref[0] - x[0], x_ref[1] - x[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(x_ref[2] - x[2])])


def error_dynamics(v_ref, omega_ref):
    """Continuous linearisation e_dot = A e + B u_fb about a reference (v_ref, omega_ref).

    With u = u_ref + u_fb, the robot-frame error obeys (Kanayama et al. 1990, linearised)
        e1_dot =  omega_ref e2 - v_fb
        e2_dot = -omega_ref e1 + v_ref e3
        e3_dot = -omega_fb
    so the feedback input enters with B = -[[1,0],[0,0],[0,1]].
    """
    A = np.array([[0, omega_ref, 0], [-omega_ref, 0, v_ref], [0, 0, 0]], dtype=float)
    B = -np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
    return A, B


def discretise(A, B, dt=DT):
    """Zero-order-hold discretisation via the matrix exponential of the augmented system."""
    from scipy.linalg import expm
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


# --- measurement models (marker at known map position) ----------------------

def h_range_bearing(x, marker_xy):
    dx, dy = marker_xy[0] - x[0], marker_xy[1] - x[1]
    return np.array([math.hypot(dx, dy), wrap(math.atan2(dy, dx) - x[2])])


def H_range_bearing(x, marker_xy):
    dx, dy = marker_xy[0] - x[0], marker_xy[1] - x[1]
    q = dx * dx + dy * dy
    r = math.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])


def h_range_bearing_yaw(x, marker_xy, marker_yaw):
    """Range, bearing and marker heading relative to the robot (full relative pose)."""
    rb = h_range_bearing(x, marker_xy)
    return np.array([rb[0], rb[1], wrap(marker_yaw - x[2])])


def H_range_bearing_yaw(x, marker_xy):
    return np.vstack([H_range_bearing(x, marker_xy), [0, 0, -1]])


# --- structural analysis ----------------------------------------------------

def controllability_rank(A, B):
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return np.linalg.matrix_rank(C, tol=1e-9), C


def observability_rank(F_seq, H_seq):
    """Rank of the stacked observability matrix of a (possibly time-varying) discrete system.

    F_seq[k] maps x_k -> x_{k+1}; H_seq[k] is the measurement Jacobian at step k (may be empty).
    """
    n = F_seq[0].shape[0]
    rows, Phi = [], np.eye(n)
    for F, H in zip(F_seq, H_seq):
        if H is not None and H.size:
            rows.append(H @ Phi)
        Phi = F @ Phi
    if not rows:
        return 0, np.zeros((0, n))
    O = np.vstack(rows)
    return np.linalg.matrix_rank(O, tol=1e-9), O
