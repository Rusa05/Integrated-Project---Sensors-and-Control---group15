"""Track A milestone 1: an open-loop TurtleBot3 movement check.

Adapted from the simulation loop in the supplied move_ur3e.py tutorial.
Robot model: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie

Originally simulate_turtlebot.py in the group15 GitHub repo; now loads the Track A room
(turtlebot3/scene.xml, official Waffle Pi model plus Track A sensors).
Windows/Linux:  python part_A_modelling/open_loop_test.py
macOS:          mjpython part_A_modelling/open_loop_test.py

This is a model/actuator check, not the assignment's feedback controller.
It has no EKF, camera processing, LiDAR processing or autonomous navigation.
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer


def movement_command(simulation_time):
    """Return left/right wheel speed targets in rad/s and the phase name."""
    if simulation_time < 1.0:
        return 0.0, 0.0, "Settling on the floor"
    elif simulation_time < 5.0:
        return 5.0, 5.0, "Driving forward"
    elif simulation_time < 7.0:
        return -6.0, 6.0, "Turning left"
    elif simulation_time < 10.0:
        return 5.0, 5.0, "Driving forward again"
    else:
        return 0.0, 0.0, "Stopped - close the viewer when finished"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true",
                        help="Run 12 simulation seconds without a viewer.")
    args = parser.parse_args()

    # Resolve from this file, so spaces and the terminal's folder do not matter.
    model_path = Path(__file__).resolve().parent.parent / "turtlebot3" / "scene.xml"
    if not model_path.is_file():
        raise FileNotFoundError("Scene not found: " + str(model_path))

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    left_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
    right_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")
    if left_id < 0 or right_id < 0:
        raise ValueError("Expected wheel_left and wheel_right actuators.")

    print("Loaded:", model_path)
    print("Sequence: settle, forward, left turn, forward, stop.")
    print("This test commands wheel speeds; it does not use the robot's true pose.")

    def step():
        left_speed, right_speed, phase = movement_command(data.time)
        # These actuators take velocity targets directly, unlike the UR3 example.
        # Friction and actuator dynamics mean actual wheel speeds can differ.
        data.ctrl[left_id] = left_speed
        data.ctrl[right_id] = right_speed
        mujoco.mj_step(model, data)
        return phase

    if args.headless:
        while data.time < 12.0:
            step()
        print("Finished 12 simulation seconds without a viewer.")
    else:
        previous_phase = ""
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # A fixed overview avoids using true robot position to steer the view.
            viewer.cam.lookat[:] = [0.25, 0.0, 0.1]
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35

            while viewer.is_running():
                step_start = time.perf_counter()
                phase = step()
                if phase != previous_phase:
                    print(phase)
                    previous_phase = phase
                viewer.sync()

                # Keep the simulation close to real time, as in the tutor's demo.
                remaining_time = model.opt.timestep - (time.perf_counter() - step_start)
                if remaining_time > 0:
                    time.sleep(remaining_time)


if __name__ == "__main__":
    main()
