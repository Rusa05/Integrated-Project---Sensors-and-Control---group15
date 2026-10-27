# Part B: Sensing, calibration and frames

> **Marking guide, Part B (20 marks):** "calibration, noise handling and frame management".
> Brief: "visually dock to the ArUco-marked charging station … range and bearing, or pose, measurements from the camera".

This part covers the ArUco tag map, the camera calibration, the sensor noise models, the frame chain from pixels to the robot frame, and how accurate each tag measurement really is. The measured accuracy sets the EKF's R in [Part C](../part_C_estimation/).

## Steps, in order

| Step | Script | What it does | Time |
|---|---|---|---|
| B1 | `make_aruco.py` | Generates every tag texture from the tag map in `src/config.py`, then checks the scene matches the map | 2 s |
| B2 | `check_dock_marker.py` | Camera detects the dock tags from 4 test poses, including the docked pose; checked against truth. Prints PASS/FAIL | 5 s |
| B3 | `frame_diagram.py` | Draws the frame diagram from the actual offsets in the code | 3 s |
| B4 | `sensor_noise.py` | LiDAR noise check; 10 cm tag accuracy over 1224 frames; range-bias calibration; noise fits for R | 45 s |
| B5 | `calibrate_camera.py` | Checkerboard calibration of the camera; lens model chosen on held-out views | 30 s |

Run from the project root, for example `python part_B_sensing/calibrate_camera.py`.

Code: [`src/config.py`](../src/config.py) (tag map, calibrated camera), [`src/sensors.py`](../src/sensors.py) (noise), [`src/vision.py`](../src/vision.py) (tag measurement).

---

## B1. The tag map

Nine ArUco tags (DICT_4X4_50) in known positions. The robot is given the tag map, like a surveyed set of printed tags, but **not** the obstacles, which it maps with its LiDAR.

| Tags | Size | Where | Why |
|---|---|---|---|
| 1–6 | 15 cm | Two on the north wall, two on the south, one west, one east, centre 0.20 m high | Usable to about 3 m (see "size scaling" below), so some tag is in range from almost anywhere in the 4 × 4 m room |
| 0 | 10 cm | Dock backplate, 0.25 m high | Seen on the approach to the dock |
| 7, 8 | 3 cm | Dock backplate, at camera height (0.107 m), 10 cm apart | Stay in view right down to the docked pose, so the last part of docking is no longer blind. Two points 10 cm apart also make the pose observable from range and bearing alone ([Part A](../part_A_modelling/)) |

- **Docked pose** (wheel-axle centre): (1.62, −1.0), facing +x. The camera stops 0.20 m from the backplate, so both small tags fit in its view.
- **Where the design came from:** the multi-tag layout and the two small dock tags are adapted from Claudia's branch.

## B2. Detection check

| Test pose | Tag | Range error | Bearing error |
|---|---|---|---|
| 0.71 m, straight on | 0 (10 cm) | +0.2 cm | 0.01° |
| 1.54 m, off-angle | 0 (10 cm) | +1.2 cm | 0.02° |
| 0.63 m, 3° to the side | 0 (10 cm) | +0.1 cm | 0.01° |
| Docked pose | 7 and 8 (3 cm) | +0.1 cm | 0.02° |

The tag textures are checked upright and not mirrored; a mirrored tag doesn't decode. Renders are in `results/marker_check_0..3.png`.

## B3. Frames

![Frame diagram](results/frame_diagram.png)

| Frame | Definition | Transform to its parent |
|---|---|---|
| map {M} | x east, y north, z up; room spans ±2 m | n/a |
| base_link {B} | x forward, y left, z up; on the floor under the wheel axle | T_MB = **EKF state** (x, y, θ) |
| LiDAR {L} | Scan centre in the LDS-01 turret | (−0.035, 0, 0.120) m in B |
| camera {C} | OpenCV optical frame: x right, y down, z forward | (0.086, 0, 0.107) m in B, looking along +x of B |
| MuJoCo camera | x right, y up, z backward | R_cam,opt = diag(1, −1, −1), 180° about x |
| tag {K_i} | x right, y up, z out of the face (solvePnP) | T_MK_i known, from the tag map in `src/config.py` |

**From pixels to what the controller acts on:**

1. Find the 4 corners of every known tag (sub-pixel).
2. IPPE solvePnP gives the tag pose in {C}. **Both** planar solutions are kept.
3. Transform into {B}: p_B = R_B,opt t_opt + t_B,cam, and the tag normal n_B = R_B,opt z_K.
4. Per tag: range = |p_B,xy|, bearing = atan2(p_B,y, p_B,x), relative yaw = atan2(n_B,y, n_B,x).
5. The EKF compares each with h(x) using that tag's known pose, and keeps whichever of the two solutions agrees with its estimate.
6. The EKF pose places the LiDAR beams on the map; the planner and LQR use it.

## B5. Camera calibration (`results/camera_calibration.png`)

The camera is calibrated with a printed checkerboard, the same way you would on the real robot:

- 9 × 6 inner corners, 25 mm squares.
- Rendered at 60 random distances and tilts by a camera identical to the robot's (same field of view, 640 × 480, same pixel noise).

| | fx | fy | cx | cy | Reprojection error |
|---|---|---|---|---|---|
| Implied by the field of view | 529.08 | 529.08 | 319.50 | 239.50 | n/a |
| What the first version used | 529.08 | 529.08 | **320.00** | **240.00** | n/a |
| **Checkerboard calibration (used)** | **529.17** | **529.17** | **319.37** | **239.45** | **0.06 px** |

- **Views used:** 56 of 60 showed the whole board. Outlier rejection (any view above 3× the median error) removed none.
- **Lens model chosen on held-out views:** fit on 2/3 of the views, test on the other 1/3.
  - Every model scored 0.057 px on the held-out views: pinhole, k1, k1 + k2, and the full 5-term model.
  - The simplest one wins, because extra distortion terms here only fit noise. That's what you'd expect for a simulated camera, and the test proves it rather than assuming it.
- **A real error found:**
  - The first version put the image centre at W/2 = 320 px. Pixel coordinates count from the *centre* of the first pixel, so the true centre is about 319.4 px.
  - That 0.63 px offset is a **0.07° bearing bias**. The robot now uses the calibrated matrix (`src/config.py` `CAMERA_MATRIX`).

## Noise models

MuJoCo's sensors are perfect, so realistic noise is added ([`src/sensors.py`](../src/sensors.py)):

| Sensor | Model | Source |
|---|---|---|
| Encoders | 4096 ticks/rev quantisation + 0.002 rad jitter | Dynamixel XL430 |
| LiDAR | σ = max(7.5 mm, 2.5 % of range); 0.12–3.5 m; 1 % dropout | LDS-01 datasheet (±15 mm / ±5 % read as 2σ) |
| Camera | 640×480 at 10 Hz; Pi Camera v2 field of view (48.8° vertical); pixel noise σ = 3 grey levels | Raspberry Pi Camera v2 |

## B4. Measured tag accuracy (`results/marker_noise.png`, `results/lidar_noise.png`)

The 10 cm tag was measured from 0.3–2.7 m, at viewing angles of ±40°, with the tag at ±20° in the image (1224 frames). This sets the reference noise model:

| Quantity | Fit used by the EKF | What we do about it |
|---|---|---|
| **Range bias** (calibration) | **+0.0067·r² m**, subtracted (1.5 cm at 1.5 m) | Detected corners sit slightly inside the true edge; it matters more the fewer pixels the tag covers |
| Range noise | σ = 1.1 mm · r^3.3 | Range-dependent R |
| Bearing noise | σ = 0.007° · r^1.1 | Unrealistically small (no lens distortion or blur in simulation), so R has a 0.5° floor |
| Tag-yaw noise | σ = 0.63° · r^2.7 | Yaw only used where it is reliable (below) |
| LiDAR | Injected noise matches the model; 1.1 % dropout | Log-odds map tolerates single bad beams ([Part D](../part_D_control/)) |

**Re-run with the final setup:** the dock tag raised to 0.25 m, and the calibrated camera.

- The fit moves slightly: range bias 0.0079·r², range σ 1.1 mm·r^3.6. That's a 3 mm difference at 1.5 m, well inside R (floor 2 cm, × 3; see Part C). The EKF keeps the original fit.
- The raised tag is no longer seen from closer than about 0.6 m. That's expected, and the small dock tags cover that range.

### Other tag sizes: pinhole similarity

A tag k times bigger, seen from k times further away, covers exactly the same pixels. So one noise model serves all sizes:

- noise: σ_s(r) = k · σ_ref(r / k);
- range bias: K·r² / k;
- usable range: k times longer.

**Checked by measuring the 3 cm and 15 cm tags directly** against the R the EKF would assign:

| Tag | Range | Range error mean / RMS | Yaw RMS | R says |
|---|---|---|---|---|
| 15 cm | 1.5 m | −0.2 / 0.4 cm | 0.9° | range σ 2 cm, yaw 2° |
| 15 cm | 2.5 m | −0.6 / 0.9 cm | 0.8° | range 2 cm, yaw 3.7° |
| 15 cm | 3.0 m (2× ref) | +1.8 / 2.9 cm | **16.4°** | yaw 6.1° |
| 10 cm | 2.0 m (2× ref) | +0.6 / 1.8 cm | **11.9°** | yaw 4.0° |
| 3 cm | 0.3–0.6 m | −0.1 / 0.1–0.3 cm | 0.6–2.0° | range 2 cm, yaw 2° |

The model holds, with two exceptions that change how the EKF uses tags:

1. **Tag yaw falls apart near 2× the reference range** (12–16° RMS, from pose flips plus noise). The EKF uses yaw only out to **1.75×** the reference range: 2.6 m for 15 cm tags, 1.75 m for the 10 cm tag.
2. **Small tags give a biased yaw.** The 3 cm dock tags read their yaw consistently wrong: about ±0.4° when docked and up to ±5° at 0.6 m, with only 0.1–0.3° scatter. A small square seen off-centre gets its tilt misjudged, and a consistent error like that can't be averaged out. Their range (1–6 mm bias) and bearing (0.01–0.03°) are excellent, so **tags under 5 cm contribute range and bearing only**.

### IPPE ambiguity

A square seen nearly head-on has two pose solutions that fit the image almost equally well; they differ mainly in the tag's apparent tilt. Both are kept, and the EKF uses the one that agrees with its estimate. The first version simply took the better-fitting one.
