"""Drive the TurtleBot yourself with the keyboard.

Press Run, click on the MuJoCo window, then use:
  W / S  forward / back      A / D  turn left / right      Space  stop
  R  reset                   C  robot camera view          L  show LiDAR rays
Prints the nearest LiDAR distance and the wheel speeds every 2 s.
"""
from pathlib import Path
import math
import threading
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / 'turtlebot3' / 'scene.xml'
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.288

def wheel_commands(v, omega):
    return np.clip([(v - omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS,
                    (v + omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS], -7.88, 7.88)

def read_sensors(model, data):
    """Encoder radians, rad/s and planar rangefinder metres; no base truth pose."""
    def value(name):
        sensor = model.sensor(name)
        return data.sensordata[sensor.adr[0]:sensor.adr[0] + sensor.dim[0]].copy()
    ranges = np.array([value(f'lidar_{i:03d}')[0] for i in range(72)])
    return {'wheel_position': np.array([value('encoder_left')[0], value('encoder_right')[0]]),
            'wheel_velocity': np.array([value('speed_left')[0], value('speed_right')[0]]),
            'lidar': np.where((ranges < 0) | (ranges > 3.5), 3.5, ranges),
            'lidar_angles': np.arange(72) * 2 * math.pi / 72}

def run():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    commands = {'v': 0., 'omega': 0., 'reset': False, 'camera': False, 'rays': False}
    lock = threading.Lock()
    def key_callback(key):
        # Commands latch: press SPACE to stop. No key-release callback is available.
        with lock:
            if key == ord('W'): commands.update(v=0.15, omega=0.)
            elif key == ord('S'): commands.update(v=-0.12, omega=0.)
            elif key == ord('A'): commands.update(v=0., omega=0.8)
            elif key == ord('D'): commands.update(v=0., omega=-0.8)
            elif key == 32: commands.update(v=0., omega=0.)
            elif key == ord('R'): commands.update(v=0., omega=0., reset=True)
            elif key == ord('C'): commands['camera'] = not commands['camera']
            elif key == ord('L'): commands['rays'] = not commands['rays']
    print('TurtleBot3 Waffle Pi loaded. Click the simulation window first.')
    print('W forward | S reverse | A/D turn | SPACE stop | R reset | C camera | L LiDAR rays')
    print('Commands continue until another key is pressed. This is manual control.')
    left = model.actuator('wheel_left').id
    right = model.actuator('wheel_right').id
    next_report = 0.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.2, 0., 0.05]
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -40
        while viewer.is_running():
            started = time.perf_counter()
            with lock:
                v, omega, reset, camera, rays = (commands[k] for k in ('v', 'omega', 'reset', 'camera', 'rays'))
                commands['reset'] = False
            if reset:
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                next_report = 0.
            data.ctrl[[left, right]] = wheel_commands(v, omega)
            # Eight physics steps per UI update: 500 Hz physics, ~62.5 Hz display.
            mujoco.mj_step(model, data, nstep=8)
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = rays and not camera
                viewer.cam.type = (mujoco.mjtCamera.mjCAMERA_FIXED if camera
                                   else mujoco.mjtCamera.mjCAMERA_FREE)
                viewer.cam.fixedcamid = model.camera('front_camera').id if camera else -1
            viewer.sync()
            if data.time >= next_report:
                sensors = read_sensors(model, data)
                print(f't={data.time:5.1f}s | nearest LiDAR={min(sensors["lidar"]):.2f}m | '
                      f'wheel speeds={np.round(sensors["wheel_velocity"], 2)} rad/s', flush=True)
                next_report = data.time + 2.
            time.sleep(max(0., model.opt.timestep * 8 - (time.perf_counter() - started)))

if __name__ == '__main__':
    run()
