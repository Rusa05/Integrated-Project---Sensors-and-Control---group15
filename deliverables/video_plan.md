# 5-minute demo video: plan and script

**Hard limit: 5:00.** Markers need not watch past it. Aim for 4:45.

The marking guide says the video must show:

1. the system completing the task;
2. at least one varied or disturbed trial;
3. the estimate and its uncertainty alongside the true state.

Both recorded clips below show all three on screen: the blue EKF arrow and 3σ ellipse, the true robot and its black trail, and a panel with EKF pose, true pose, error and 3σ.

## Clips to record (already generated, re-record any time)

| Clip | File | Length | Re-record with |
|---|---|---|---|
| A. Full mission, nominal | `part_E_integration_and_robustness/results/mission/mission_A.mp4` | 35 s | `python part_E_integration_and_robustness/run_mission.py --record mission_A.mp4 --scenario A_cylinder_in_way` |
| B. Disturbed run: push + slip + occlusion + latency | `part_E_integration_and_robustness/results/trials/combined_trial7.mp4` | 33 s | `python part_E_integration_and_robustness/run_trials.py --record combined_trial7.mp4 --only combined --trial 7` |
| C. Optional: a second nominal layout | n/a | ~35 s | `python part_E_integration_and_robustness/run_mission.py --record mission_B.mp4 --scenario B_box_in_way` |

Stills to show (all in the repo):

- `part_B_sensing/results/frame_diagram.png`
- `part_B_sensing/results/marker_noise.png`
- `part_C_estimation/results/ekf_run.png`
- `part_C_estimation/results/ekf_consistency.png`
- `part_D_control/results/nav_B_box_in_way.png`
- `part_E_integration_and_robustness/results/trials/success_by_condition.png`
- The "Failures diagnosed and fixed" table from the project page

## Timeline

| Time | Shows | Say (key points only; keep it conversational) | Speaker |
|---|---|---|---|
| 0:00–0:15 | Title card: group, members, Track A; then the first frame of clip A | "Track A: a TurtleBot3 crosses an unknown room and docks on an ArUco-marked charger, using only its own sensors." | *name* |
| 0:15–0:45 | Clip A at 2× speed, first half | Point out the blue arrow and ellipse (EKF estimate, 3σ) against the real robot. The ellipse grows while the marker is out of range, then snaps small when the marker is seen (top-right inset, green outline). | *name* |
| 0:45–1:15 | Model slide: unicycle equations, controllability/observability table | Kinematic model is justified (14 ms motor lag against 20 ms control). Range + bearing to one marker is rank 2, so we also measure the marker's yaw. Mention the 3 fixes to the official robot model in one sentence. | *name* |
| 1:15–1:50 | `frame_diagram.png`, then `marker_noise.png` | Walk from pixels to EKF measurement: solvePnP → optical frame → base_link → range, bearing, yaw. Noise grows with r³; range bias 1.5 cm at 1.5 m is calibrated out. | *name* |
| 1:50–2:30 | `ekf_run.png`, `ekf_consistency.png` | EKF 2.2 cm vs odometry 8.1 cm; truth inside 3σ 100 %; NEES 2.3 in bounds. Where Q and R came from (measured), and what R ×0.1 does (NEES 8, over-confident). | *name* |
| 2:30–3:00 | `nav_B_box_in_way.png` | LiDAR map built at the EKF pose, A* with 0.25 m inflation, LQR with Bryson weights. Controller error is 1.6 cm against the estimate; the rest is estimation error. | *name* |
| 3:00–3:30 | Clip A, docking part at 1× | Final align, then curved approach; marker lost at 0.3 m, last 17 cm on odometry; docked 0.6 cm / 0.15°. | *name* |
| 3:30–4:15 | Clip B at 1.5× | Banner shows the disturbances. At 8 s the robot is shoved: true and estimated paths split, ellipse balloons to about 22 cm. Marker reacquired → corrected. Marker blocked during docking (inset greyed). Still docks at 0.64 cm. | *name* |
| 4:15–4:40 | `success_by_condition.png`, then the failures table | 80/80 across 8 conditions. Four failures found on the way, each traced: latency (rewind-and-replay), far updates (2 m limit), gate lock-out (P inflation), crooked docking (Hermite curve). | *name* |
| 4:40–4:55 | Limitations slide | Simulation-only camera, one marker, static obstacles, 10 trials per condition. | *name* |

## Recording tips

- Make the slides in Google Slides or PowerPoint with the stills above. Put clips A and B on slides as embedded videos, or cut them in a video editor (Clipchamp on Windows, iMovie on Mac).
- Speed clips up in the editor rather than re-recording; the overlays stay readable at 2×.
- Record narration separately per section; it's much easier to keep to time.
- Check the final export is ≤ 5:00, then upload it unlisted (YouTube or OneDrive) and put the link on the project page.
- Every speaker should be able to answer questions on the part they narrate *and* one part they didn't lead (viva rule).
