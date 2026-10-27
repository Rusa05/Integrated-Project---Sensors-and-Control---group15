# Track A — TurtleBot3 Navigation and Visual Docking

UTS Sensors and Control Integrated Project.

## Project goal
Navigate a room with obstacles using LiDAR and visually dock at an
ArUco-marked charging station. The final controller will use state
estimates from an EKF combining wheel odometry and camera measurements.

## Current progress
Initial MuJoCo setup and a forward-turn-stop movement test.
Camera processing, LiDAR processing, EKF estimation and autonomous
navigation are still to be implemented.

## Files
- simulate_turtlebot.py: initial movement test.
- robotis_mujoco_menagerie/robotis_tb3/: robot model and assets.
- requirements.txt: Python dependencies.

## Setup on macOS
From the project folder, run these commands in order:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python .venv/bin/mjpython simulate_turtlebot.py

## Sources
Robot model:
https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie

Simulation-loop structure adapted from the supplied UR3 tutorial.
The model licence is retained in the robotis_tb3 folder.

## Team and contributions
Add the group number, team members and each member's contributions here.

## Generative AI declaration
ChatGPT assisted with setup troubleshooting, project planning and
adapting the initial movement-test script. Update this declaration as
the project develops.