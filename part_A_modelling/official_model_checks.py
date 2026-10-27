"""Why turtlebot3_waffle_pi_track_a.xml differs from the official ROBOTIS Waffle Pi model.

Starts from the official dynamics and applies the three Track A fixes one at a time.
For each step the robot drives straight (4 rad/s) and turns on the spot (-3/+3 rad/s) for 2 s,
and we compare true motion with what the wheel encoders say. Ground truth is used only to
score. Run from the project root:
    python part_A_modelling/official_model_checks.py
"""
from pathlib import Path
import math
import os
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T3 = ROOT / 'turtlebot3'
TRACK_A = (T3 / 'turtlebot3_waffle_pi_track_a.xml').read_text()
TMP_ROBOT, TMP_SCENE = T3 / '_check_robot.xml', T3 / '_check_scene.xml'
R, B = 0.033, 0.288


def variant(caster_priority, chassis_box, kv):
    s = TRACK_A
    if not caster_priority:
        s = s.replace(' priority="1"/>', '/>')
    if not chassis_box:
        s = s.replace('      <geom pos="-0.064 0 0.01" mesh="waffle_pi_base" class="collision"/> -->',
                      '      -->\n      <geom pos="-0.064 0 0.01" mesh="waffle_pi_base" class="collision"/>')
        s = s.replace('      <geom name="chassis_collision"', '      <!-- <geom name="chassis_collision"')
        s = s.replace('group="3" density="0"/>', 'group="3" density="0"/> -->', 1)
    s = s.replace('velocity kv="1.0"', f'velocity kv="{kv}"')
    TMP_ROBOT.write_text(s)
    TMP_SCENE.write_text((T3 / 'scene.xml').read_text().replace('turtlebot3_waffle_pi_track_a.xml', TMP_ROBOT.name))
    return mujoco.MjModel.from_xml_path(str(TMP_SCENE))


def yaw(q):
    return math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))


def drive(m, d, cmd, steps=1000):
    mujoco.mj_resetData(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    e0 = np.array([d.sensor('encoder_left').data[0], d.sensor('encoder_right').data[0]])
    q0 = d.qpos[:7].copy()
    for _ in range(steps):
        d.ctrl[:] = cmd
        mujoco.mj_step(m, d)
    e = np.array([d.sensor('encoder_left').data[0], d.sensor('encoder_right').data[0]]) - e0
    return (R * e.sum() / 2, np.hypot(*(d.qpos[:2] - q0[:2])),
            R * (e[1] - e[0]) / B, yaw(d.qpos) - yaw(q0), d.sensor('speed_left').data[0])


def chassis_load(m, d):
    mujoco.mj_resetData(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    floor, base, f, load = m.geom('floor').id, m.body('base').id, np.zeros(6), 0.0
    for i, c in enumerate(d.contact[:d.ncon]):
        g = c.geom2 if c.geom1 == floor else c.geom1
        if m.geom_bodyid[g] == base and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
            mujoco.mj_contactForce(m, d, i, f)
            load += f[0]
    return load / (m.body_subtreemass[base] * 9.81)


def main():
    steps = [('official dynamics', False, False, 0.1),
             ('+ chassis box off the floor', False, True, 0.1),
             ('+ casters frictionless (priority)', True, True, 0.1),
             ('+ wheel servo kv 1.0  = Track A', True, True, 1.0)]
    print(f'{"":36s} {"straight true/odo":>18s} {"turn true/odo":>14s} {"wheel speed/cmd":>16s} {"chassis load":>13s}')
    try:
        for name, cp, box, kv in steps:
            m = variant(cp, box, kv)
            d = mujoco.MjData(m)
            od, td, _, _, w = drive(m, d, (4, 4))
            _, _, oy, ty, _ = drive(m, d, (-3, 3))
            print(f'{name:36s} {td / od:18.3f} {ty / oy:14.3f} {w / 4:16.0%} {chassis_load(m, d):13.0%}')
    finally:
        for p in (TMP_ROBOT, TMP_SCENE):
            if p.exists():
                os.remove(p)


if __name__ == '__main__':
    main()
