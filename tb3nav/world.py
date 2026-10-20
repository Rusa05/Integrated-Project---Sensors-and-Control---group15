"""Procedural MuJoCo world: a walled room, obstacles, ArUco landmarks and a docking station.

The world is generated (rather than hand-written) so that Monte Carlo trials can vary the
obstacle layout while the landmark map stays fixed and known to the robot.

World frame W: origin at the south-west inner corner of the room, x east, y north, z up.
Every marker is a textured plane whose local +z (its outward normal) points into the room
and whose local +y is world up, so OpenCV sees an upright, un-mirrored ArUco tag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
GEN_DIR = MODEL_DIR / "generated"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE = 0.15          # wall tags: side of the black ArUco square [m]; sized so that
                            # some tag is within usable range (~5 m) everywhere in the room
MARKER_HEIGHT = 0.20        # wall-tag centre height [m] (camera is at 0.107 m)
DOCK_MARKER_HEIGHT = 0.25   # dock tag sits higher, leaving room for the small tag below it
# Two small tags on the dock, 10 cm apart, at camera height. They stay in view to the docked
# pose and, being two distinct points, make the pose fully observable on the final approach
# (one point landmark leaves rotation about it unobservable: scripts/01_model_analysis.py).
DOCK_SMALL_IDS = (10, 11)
DOCK_SMALL_OFFSET = 0.05    # lateral offset of each small tag from the dock centre [m]
DOCK_SMALL_SIZE = 0.04
DOCK_SMALL_HEIGHT = 0.107
ROOM_X, ROOM_Y = 6.0, 4.0   # inner room size [m]
WALL_T, WALL_H = 0.05, 0.40

DOCK_ID = 0
# Robot base pose when docked: axle 0.22 m in front of the dock face (front bumper ~5 cm
# from the charging block), facing the dock.
DOCK_STANDOFF = 0.22


@dataclass
class Landmark:
    id: int
    x: float
    y: float
    yaw: float  # direction of the marker's outward normal in W [rad]
    size: float = MARKER_SIZE
    z: float = MARKER_HEIGHT

    @property
    def normal(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])


@dataclass
class Obstacle:
    kind: str          # "box" or "cylinder"
    x: float
    y: float
    size: tuple        # box: (half_x, half_y); cylinder: (radius,)
    yaw: float = 0.0


@dataclass
class WorldSpec:
    landmarks: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    start: tuple = (0.6, 0.6, 0.0)  # x, y, yaw of the robot base in W
    floor_friction: float = 1.0   # floor/tyre friction; odometry was calibrated at 1.0
    extra_assets: str = ""   # raw MJCF, used by the calibration scenes
    extra_bodies: str = ""

    @property
    def dock(self) -> Landmark:
        return next(l for l in self.landmarks if l.id == DOCK_ID)


    def dock_pose(self):
        """Target base pose (x, y, yaw) for a completed dock."""
        d = self.dock
        p = np.array([d.x, d.y]) + DOCK_STANDOFF * d.normal
        return np.array([p[0], p[1], np.arctan2(-d.normal[1], -d.normal[0])])

    def landmark_map(self):
        """id -> (x, y, yaw). This is the only prior map the robot is given."""
        return {l.id: np.array([l.x, l.y, l.yaw]) for l in self.landmarks}

    def landmark_sizes(self):
        """id -> tag size [m]; part of the known map (printed tag sizes)."""
        return {l.id: l.size for l in self.landmarks}


def default_landmarks():
    e = 0.004  # markers sit 4 mm proud of the wall face
    L = []
    L.append(Landmark(DOCK_ID, ROOM_X - e, 3.0, np.pi, z=DOCK_MARKER_HEIGHT))   # dock, east wall
    for i, side in zip(DOCK_SMALL_IDS, (-1, 1)):
        L.append(Landmark(i, ROOM_X - e, 3.0 + side * DOCK_SMALL_OFFSET, np.pi,
                          DOCK_SMALL_SIZE, DOCK_SMALL_HEIGHT))
    L += [Landmark(1, e, 1.0, 0.0), Landmark(2, e, 3.0, 0.0)]     # west wall
    L += [Landmark(3, 1.5, e, np.pi / 2), Landmark(4, 3.0, e, np.pi / 2),
          Landmark(5, 4.5, e, np.pi / 2)]                         # south wall
    L += [Landmark(6, 1.5, ROOM_Y - e, -np.pi / 2), Landmark(7, 3.0, ROOM_Y - e, -np.pi / 2),
          Landmark(8, 4.5, ROOM_Y - e, -np.pi / 2)]               # north wall
    L += [Landmark(9, ROOM_X - e, 1.0, np.pi)]                    # east wall
    return L


def default_obstacles():
    return [
        Obstacle("box", 2.0, 1.6, (0.20, 0.45)),
        Obstacle("cylinder", 3.5, 2.6, (0.25,)),
        Obstacle("box", 4.4, 1.3, (0.30, 0.20), 0.4),
        Obstacle("box", 1.4, 3.1, (0.25, 0.15)),
    ]


def nominal_world() -> WorldSpec:
    return WorldSpec(default_landmarks(), default_obstacles())


def randomised_world(rng: np.random.Generator, jitter=0.25, start_jitter=0.2) -> WorldSpec:
    """Same landmarks, obstacles moved by up to `jitter` m, start pose perturbed."""
    w = nominal_world()
    for o in w.obstacles:
        o.x += rng.uniform(-jitter, jitter)
        o.y += rng.uniform(-jitter, jitter)
        o.yaw += rng.uniform(-0.5, 0.5)
    sx, sy, syaw = w.start
    w.start = (sx + rng.uniform(-start_jitter, start_jitter),
               sy + rng.uniform(-start_jitter, start_jitter),
               syaw + rng.uniform(-0.6, 0.6))
    return w


# ----------------------------------------------------------------------------- textures
def marker_png(marker_id: int) -> Path:
    """Write an ArUco tag with a 1-cell white quiet zone; return its path."""
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GEN_DIR / f"aruco_{marker_id}.png"
    if not path.exists():
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        cells = 6                          # 4x4 bits + 1-cell black border each side
        px = 40
        tag = cv2.aruco.generateImageMarker(d, marker_id, cells * px)
        img = cv2.copyMakeBorder(tag, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return path


def _xyaxes_for_normal(yaw):
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, n)  # so that x × up = n  (plane +z is the outward normal)
    return f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} 0 0 1"


def build_xml(spec: WorldSpec) -> str:
    """Return a complete MJCF string for the scene described by `spec`."""
    assets, bodies = [], []
    for lm in spec.landmarks:
        plate = lm.size * 8 / 6  # tag + quiet zone
        png = marker_png(lm.id).as_posix()
        assets.append(f'<texture name="tex_aruco{lm.id}" type="2d" file="{png}"/>')
        assets.append(f'<material name="mat_aruco{lm.id}" texture="tex_aruco{lm.id}" '
                      f'texrepeat="1 1" texuniform="false" specular="0" shininess="0"/>')
        bodies.append(
            f'<geom name="aruco{lm.id}" type="plane" size="{plate/2} {plate/2} 0.01" '
            f'pos="{lm.x} {lm.y} {lm.z}" xyaxes="{_xyaxes_for_normal(lm.yaw)}" '
            f'material="mat_aruco{lm.id}" contype="0" conaffinity="0" group="1"/>')

    d = spec.dock
    dn = d.normal
    # Charging station: a block on the wall below/around the marker (visual + collision).
    bodies.append(f'<geom name="dock_base" type="box" size="0.03 0.12 0.03" '
                  f'pos="{d.x + 0.03*dn[0]} {d.y + 0.03*dn[1]} 0.03" euler="0 0 {d.yaw}" '
                  f'rgba="0.9 0.6 0.1 1"/>')

    hx, hy = ROOM_X / 2, ROOM_Y / 2
    t = WALL_T / 2
    walls = [(hx, -t, hx + WALL_T, t), (hx, ROOM_Y + t, hx + WALL_T, t),
             (-t, hy, t, hy + WALL_T), (ROOM_X + t, hy, t, hy + WALL_T)]
    for i, (x, y, sx, sy) in enumerate(walls):
        bodies.append(f'<geom name="wall{i}" type="box" size="{sx} {sy} {WALL_H/2}" '
                      f'pos="{x} {y} {WALL_H/2}" rgba="0.85 0.85 0.8 1"/>')

    for i, o in enumerate(spec.obstacles):
        if o.kind == "box":
            bodies.append(f'<geom name="obs{i}" type="box" size="{o.size[0]} {o.size[1]} 0.20" '
                          f'pos="{o.x} {o.y} 0.20" euler="0 0 {o.yaw}" rgba="0.6 0.3 0.3 1"/>')
        else:
            bodies.append(f'<geom name="obs{i}" type="cylinder" size="{o.size[0]} 0.20" '
                          f'pos="{o.x} {o.y} 0.20" rgba="0.3 0.4 0.6 1"/>')

    sx, sy, syaw = spec.start
    robot = (MODEL_DIR / "turtlebot3_waffle_pi_sensors.xml").as_posix()
    meshdir = (ROOT / "robotis_mujoco_menagerie" / "robotis_tb3" / "assets").as_posix()
    return f"""<mujoco model="track_a_room">
  <compiler meshdir="{meshdir}"/>
  <include file="{robot}"/>
  <option timestep="0.002"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.55 0.55 0.55"
      rgb2="0.5 0.5 0.5" markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4"/>
    {' '.join(assets)}
    {spec.extra_assets}
  </asset>
  <worldbody>
    <light pos="{hx} {hy} 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" size="{hx+1} {hy+1} 0.05" pos="{hx} {hy} 0" type="plane" material="groundplane"
      friction="{spec.floor_friction} 0.005 0.0001"/>
    {chr(10).join('    ' + b for b in bodies)}
    {spec.extra_bodies}
  </worldbody>
  <keyframe>
    <key name="start" qpos="{sx} {sy} 0 {np.cos(syaw/2)} 0 0 {np.sin(syaw/2)} 0 0"/>
  </keyframe>
</mujoco>
"""


def load_model(spec: WorldSpec):
    """Compile the scene and return (MjModel, MjData) reset to the start keyframe."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(build_xml(spec))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data
