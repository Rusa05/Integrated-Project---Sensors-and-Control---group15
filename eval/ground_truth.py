"""The ONLY place that reads the simulator's true robot pose.

Used for setting up test poses and scoring results. Nothing under src/ may import this.
"""
import math
import mujoco
import numpy as np


def true_pose(model, data):
    """(x, y, theta) of base_link in the world."""
    q = model.joint('base_joint').qposadr[0]
    w, x, y, z = data.qpos[q + 3:q + 7]
    return np.array([data.qpos[q], data.qpos[q + 1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


def set_pose(model, data, x, y, theta, z=0.0):
    q = model.joint('base_joint').qposadr[0]
    data.qpos[q:q + 3] = [x, y, z]
    data.qpos[q + 3:q + 7] = [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]
    v = model.joint('base_joint').dofadr[0]
    data.qvel[v:v + 6] = 0
    mujoco.mj_forward(model, data)


def true_marker_xy(data):
    return data.site('marker_centre').xpos[:2].copy()


ROBOT_BODIES = ('base', 'wheel_left', 'wheel_right')


def robot_collisions(model, data):
    """Names of non-floor geoms the robot is touching (evaluation only)."""
    robot = {model.body(b).id for b in ROBOT_BODIES}
    floor = model.geom('floor').id
    hits = set()
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 in robot) == (b2 in robot) or floor in (c.geom1, c.geom2):
            continue
        other = c.geom2 if b1 in robot else c.geom1
        hits.add(model.geom(other).name or model.body(model.geom_bodyid[other]).name)
    return hits
