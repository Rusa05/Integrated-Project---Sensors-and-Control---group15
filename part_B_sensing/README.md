# Part B: Sensing, calibration and frames

> **Marking guide, Part B (20 marks):** "calibration, noise handling and frame management".
> Brief: "visually dock to the ArUco-marked charging station … range and bearing, or pose, measurements from the camera".

This part covers the charging-station marker, the sensor noise models, the camera calibration, the frame chain from pixels to the robot frame, and how accurate the marker measurement actually is. The measured accuracy sets the EKF's R in [Part C](../part_C_estimation/).

## Steps, in order

| Step | Script | What it does | Time |
|---|---|---|---|
| B1 | `make_aruco.py` | Generates the marker texture (`turtlebot3/assets/aruco_dock.png`) | 1 s |
| B2 | `check_dock_marker.py` | Camera detects the marker from 3 test poses; range/bearing checked against truth. Prints PASS/FAIL | 5 s |
| B3 | `frame_diagram.py` | Draws the frame diagram from the actual offsets in the code | 3 s |
| B4 | `sensor_noise.py` | LiDAR noise check; marker accuracy over 1224 frames; range-bias calibration; noise fits for R | 45 s |

Run from the project root, for example `python part_B_sensing/sensor_noise.py`.

Code: [`src/sensors.py`](../src/sensors.py) (noise), [`src/vision.py`](../src/vision.py) (marker measurement), [`src/config.py`](../src/config.py) (the known map).

---

## B1–B2. The charging station and marker

Added to [`turtlebot3/scene.xml`](../turtlebot3/scene.xml), against the east wall at y = −1.0 m, facing into the room (−x):

- **ArUco DICT_4X4_50, ID 0.** A 0.10 m black square with a 0.02 m white border, centre (1.9085, −1.0, 0.15) m, just above the camera's 0.107 m height.
- **Dock goal:** wheel-axle centre at (1.78, −1.0), facing +x. There the robot's front stops 4 cm short of the backplate.

| Test pose | Range error | Bearing error |
|---|---|---|
| 0.62 m, straight on | +0.2 cm | 0.07° |
| 1.45 m, off-angle | +1.5 cm | 0.05° |
| 0.37 m, 11° to the side | +0.1 cm | 0.04° |

- The texture is checked upright and not mirrored; a mirrored marker doesn't decode.
- Sub-pixel corner refinement roughly halved the range error at distance.
- Renders from the robot camera are saved as `results/marker_check_0..2.png`.

## B3. Frames

![Frame diagram](results/frame_diagram.png)

| Frame | Definition | Transform to its parent |
|---|---|---|
| map {M} | x east, y north, z up; room spans ±2 m | n/a |
| base_link {B} | x forward, y left, z up; on the floor under the wheel axle | T_MB = **EKF state** (x, y, θ) |
| LiDAR {L} | Scan centre in the LDS-01 turret | (−0.035, 0, 0.120) m in B |
| camera {C} | OpenCV optical frame: x right, y down, z forward | (0.086, 0, 0.107) m in B, looking along +x of B |
| MuJoCo camera | x right, y up, z backward | R_cam,opt = diag(1, −1, −1), i.e. 180° about x |
| marker {K} | x right, y up, z out of the face (solvePnP) | T_MK known: (1.9085, −1.0, 0.15) m, normal along −x of M |

**From pixels to what the controller acts on:**

1. 4 marker corners (sub-pixel).
2. IPPE solvePnP gives the marker pose in {C}. Both planar solutions are computed; the lower reprojection error wins.
3. Transform into {B}: p_B = R_B,opt t_opt + t_B,cam, and the marker normal n_B = R_B,opt z_K.
4. Measurement: range = |p_B,xy|, bearing = atan2(p_B,y, p_B,x), marker yaw = atan2(n_B,y, n_B,x).
5. The EKF compares this with h(x), using the known T_MK.
6. The EKF pose places LiDAR beams on the map, then the planner and LQR use it.

## Noise models and calibration

MuJoCo's sensors are perfect, so realistic noise is added ([`src/sensors.py`](../src/sensors.py)):

| Sensor | Model | Source |
|---|---|---|
| Encoders | 4096 ticks/rev quantisation + 0.002 rad jitter | Dynamixel XL430 |
| LiDAR | σ = max(7.5 mm, 2.5 % of range); 0.12–3.5 m; 1 % dropout | LDS-01 datasheet (±15 mm / ±5 % read as 2σ) |
| Camera | 640×480 at 10 Hz; Pi Camera v2 field of view (48.8° vertical); pixel noise σ = 3 grey levels | Raspberry Pi Camera v2 |

| Calibration | Value | How |
|---|---|---|
| Camera intrinsics | f = H / (2 tan(fovy/2)), principal point at image centre | From the field of view; verified by B2 |
| Camera extrinsics | (0.086, 0, 0.107) m, rotation from the mount | Read from the robot model ([`src/vision.py`](../src/vision.py)) |
| **Range bias** | **+0.0067·r² m**, subtracted (1.5 cm at 1.5 m) | Least-squares fit over 1224 frames (B4). Cause: detected corners sit slightly inside the true marker edge, and that matters more the fewer pixels the marker covers |

## B4. Measured accuracy (`results/marker_noise.png`, `results/lidar_noise.png`)

Robot placed at 0.3–2.7 m, viewing angles ±40°, marker at ±20° in the image, 8 frames each.

| Quantity | Measured | What we do about it |
|---|---|---|
| LiDAR | Injected noise matches the model; 1.1 % dropout | Log-odds map tolerates single bad beams ([Part D](../part_D_control/)) |
| Detection | 98–100 % from 0.5 to 2.5 m; 85 % at 2.7 m; **7 % at 0.3 m** (marker leaves the 62° field of view) | The last ~0.17 m of docking is blind, on odometry |
| Range noise (after bias correction) | σ = 1.1 mm · r^3.3 | Range-dependent R |
| Bearing noise | σ = 0.007° · r^1.1 | Unrealistically small (no lens distortion or blur in simulation), so R has a 0.5° floor |
| Marker-yaw noise | σ = 0.63° · r^2.7; 1.8 % of frames flip by more than 20° (IPPE planar ambiguity) | Gate outliers; fuse only within 2 m |

![Marker accuracy](results/marker_noise.png)
