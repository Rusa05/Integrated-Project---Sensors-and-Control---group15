"""Draws the frame diagram for the README (Part B).

Shows the map, robot, LiDAR, camera and tag frames, and how a camera measurement turns into
something the EKF can use. Saves results/frame_diagram.png. Press Run.
"""
from pathlib import Path
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config                       # noqa: E402
from src.mapping import LIDAR_OFFSET          # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'
FOOT = (-0.197, 0.077, 0.153)                 # footprint about the wheel axle: x min, x max, half width


def axes(ax, origin, theta, length, label, color, lw=2.0):
    o = np.asarray(origin, float)
    for vec, name in ((np.array([math.cos(theta), math.sin(theta)]), 'x'),
                      (np.array([-math.sin(theta), math.cos(theta)]), 'y')):
        ax.add_patch(FancyArrowPatch(o, o + length * vec, arrowstyle='-|>', mutation_scale=12, color=color, lw=lw))
        ax.text(*(o + 1.12 * length * vec), name, color=color, fontsize=8, ha='center', va='center')
    ax.text(o[0], o[1] - 0.06, label, color=color, fontsize=9, ha='center', va='top', fontweight='bold')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(config.SCENE))
    cam = model.camera('front_camera')
    cam_xyz = cam.pos.copy()
    lidar_z = model.site('ray_000').pos[2]

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1])

    # ---- 1. top view: map, base_link, LiDAR, camera, marker ---------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, lw=1.5, color='grey'))
    ax.add_patch(Rectangle((1.62, -1.18), 0.34, 0.36, color='gold', alpha=0.5))
    ax.add_patch(Rectangle((1.91, -1.2), 0.04, 0.4, color='dimgrey'))
    axes(ax, (0, 0), 0.0, 0.35, 'map  {M}', 'black')
    ax.text(0.05, 0.42, 'world / map frame\nx east, y north, z up', fontsize=8, color='black')

    x, y, th = 0.75, -0.45, math.radians(-25)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    fp = np.array([[FOOT[0], -FOOT[2]], [FOOT[1], -FOOT[2]], [FOOT[1], FOOT[2]], [FOOT[0], FOOT[2]]])
    ax.add_patch(Polygon(fp @ R.T + [x, y], closed=True, color='#3a3a3a', alpha=0.25))
    axes(ax, (x, y), th, 0.28, 'base_link {B}', 'C0')
    lid = R @ np.array(LIDAR_OFFSET) + [x, y]
    ax.plot(*lid, 'o', color='C2', ms=6)
    ax.annotate(f'LiDAR {{L}}\n({LIDAR_OFFSET[0]:+.3f}, 0, {lidar_z:.3f}) in B', lid, xytext=(-0.5, -0.72),
                fontsize=8, color='C2', arrowprops=dict(arrowstyle='->', color='C2'))
    camp = R @ cam_xyz[:2] + [x, y]
    ax.plot(*camp, 's', color='C3', ms=6)
    ax.add_patch(FancyArrowPatch(camp, camp + 0.3 * np.array([math.cos(th), math.sin(th)]), arrowstyle='-|>',
                                 mutation_scale=12, color='C3', lw=2, ls='--'))
    ax.annotate(f'camera {{C}}\n({cam_xyz[0]:+.3f}, 0, {cam_xyz[2]:.3f}) in B\noptical axis = +x of B', camp,
                xytext=(camp[0] + 0.15, camp[1] + 0.45), fontsize=8, color='C3', arrowprops=dict(arrowstyle='->', color='C3'))
    # field of view wedge
    hfov = 2 * math.atan(4 / 3 * math.tan(math.radians(model.cam('front_camera').fovy[0]) / 2))
    for s in (-1, 1):
        a = th + s * hfov / 2
        ax.plot([camp[0], camp[0] + 1.5 * math.cos(a)], [camp[1], camp[1] + 1.5 * math.sin(a)], ':', color='C3', lw=0.8)

    mx, my = config.MARKER_XY
    ax.add_patch(FancyArrowPatch((mx, my), (mx - 0.25, my), arrowstyle='-|>', mutation_scale=12, color='C1', lw=2))
    ax.text(mx - 0.27, my + 0.05, 'z_K (normal)', color='C1', fontsize=8, ha='right')
    ax.annotate(f'marker {{K}}\n({mx:.3f}, {my:.2f}, {config.MARKER_Z:.2f}) in M\nnormal along -x of M', (mx, my), xytext=(1.3, 0.45),
                fontsize=8, color='C1', arrowprops=dict(arrowstyle='->', color='C1'))
    dg = config.DOCK_GOAL
    ax.plot(dg[0], dg[1], 'x', color='darkgoldenrod', ms=9, mew=2)
    ax.annotate(f'dock goal (wheel axle)\n({dg[0]:.2f}, {dg[1]:.1f}, 0°)', (dg[0], dg[1]), xytext=(0.95, -1.38), fontsize=8, color='darkgoldenrod', arrowprops=dict(arrowstyle='->', color='darkgoldenrod'))
    # measurement arrow base -> marker
    ax.add_patch(FancyArrowPatch((x, y), (mx, my), arrowstyle='-|>', mutation_scale=10, color='C1', lw=1, ls='-.'))
    ax.text(1.12, -0.28, 'measurement\nz = [range, bearing,\n       marker yaw] in B', fontsize=8, color='C1')
    ax.text(-0.5, -1.3, 'T_MB = EKF state (x, y, θ): the only robot\npose the controller and the map use', fontsize=8, color='C0')
    ax.set(aspect='equal', xlim=(-0.6, 2.15), ylim=(-1.45, 0.95), title='Frames, top view (example pose)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    # ---- 2. side view: camera frame conventions --------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.add_patch(Rectangle((FOOT[0], 0.01), FOOT[1] - FOOT[0], 0.11, color='#3a3a3a', alpha=0.25))
    ax2.add_patch(plt.Circle((0, 0.033), 0.033, color='black', alpha=0.4))
    ax2.plot([-0.25, 0.3], [0, 0], color='grey')
    def arrow(o, d, c, txt, off=(0, 0)):
        o = np.asarray(o); d = np.asarray(d)
        ax2.add_patch(FancyArrowPatch(o, o + d, arrowstyle='-|>', mutation_scale=10, color=c, lw=1.6))
        ax2.text(*(o + 1.25 * d + off), txt, color=c, fontsize=8, ha='center', va='center')
    arrow((0, 0), (0.07, 0), 'C0', 'x_B')
    arrow((0, 0), (0, 0.07), 'C0', 'z_B')
    c = np.array([cam_xyz[0], cam_xyz[2]])
    arrow(c, (0.06, 0), 'C3', 'z_opt\n(forward)', (0.012, 0))
    arrow(c, (0, -0.05), 'C3', 'y_opt (down)', (0.035, -0.005))
    arrow(c + [0, 0.004], (-0.05, 0), 'purple', 'z_cam (MuJoCo)', (-0.02, 0.022))
    arrow(c, (0, 0.05), 'purple', 'y_cam (up)', (0.03, 0.005))
    ax2.text(-0.24, 0.215, 'OpenCV optical {C}: x right, y down, z forward\n'
                           'MuJoCo camera: x right, y up, z backward\n'
                           'R_cam_opt = diag(1, -1, -1)   (180° about x)', fontsize=8)
    ax2.set(aspect='equal', xlim=(-0.25, 0.3), ylim=(-0.02, 0.29), title='Camera frames, side view', xticks=[], yticks=[])

    # ---- 3. measurement chain ----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    steps = [
        ('image', '4 marker corners (px), sub-pixel refined'),
        ('solvePnP (IPPE)', 'T_C K: marker pose in optical frame'),
        ('extrinsics', 'p_B = R_B,opt · t_opt + t_B,cam\nn_B = R_B,opt · z_K'),
        ('measurement', 'range = |p_B,xy| − bias(r)\nbearing = atan2(p_B,y, p_B,x)\nmarker yaw = atan2(n_B,y, n_B,x)'),
        ('EKF', 'h(x) uses T_MK (known map) and T_MB (state)\nfuses only within 2 m; rewinds for latency'),
        ('LiDAR', 'beam i at angle 5i° in L → map via T_MB\n→ occupancy grid → A* → LQR'),
    ]
    for i, (h, t) in enumerate(steps):
        yy = 0.95 - i * 0.165
        ax3.text(0.0, yy, h, fontsize=9, fontweight='bold', va='top')
        ax3.text(0.3, yy, t, fontsize=8, va='top')
        if i < len(steps) - 2:
            ax3.annotate('', xy=(0.12, yy - 0.12), xytext=(0.12, yy - 0.06), arrowprops=dict(arrowstyle='->'))
    ax3.set_title('From sensor to the quantity the controller acts on', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / 'frame_diagram.png', dpi=130)
    print('wrote', OUT / 'frame_diagram.png')


if __name__ == '__main__':
    main()
