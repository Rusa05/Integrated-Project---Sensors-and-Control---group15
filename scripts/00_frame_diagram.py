"""Step 0 - Draw the frame diagram (results/figures/frames.png) from the actual extrinsics."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, Rectangle

from tb3nav import world as W
from tb3nav.frames import P_B_C, P_B_L

FIG = ROOT / "results" / "figures"


def axis(ax, o, d, c, lab, L=0.06):
    ax.add_patch(FancyArrow(o[0], o[1], L * d[0], L * d[1], width=0.002, head_width=0.012,
                            length_includes_head=True, color=c))
    ax.text(o[0] + 1.15 * L * d[0], o[1] + 1.15 * L * d[1], lab, color=c, fontsize=9,
            ha="center", va="center")


def main():
    icr = 0.0
    cf = ROOT / "results" / "calibration.json"
    if cf.exists():
        icr = json.loads(cf.read_text()).get("odometry", {}).get("icr_x", 0.0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    # chassis (Waffle Pi footprint ~ 281 x 306 mm, centred 64 mm behind the axle)
    a.add_patch(Rectangle((-0.064 - 0.1405, -0.153), 0.281, 0.306, fill=False, lw=1.5))
    for y in (0.144, -0.144):
        a.add_patch(Rectangle((-0.033, y - 0.009), 0.066, 0.018, color="0.3"))
    for y in (0.064, -0.064):
        a.add_patch(Circle((-0.177, y), 0.006, color="0.5"))
    a.text(-0.177, 0.085, "casters", fontsize=8, ha="center")
    axis(a, (0, 0), (1, 0), "r", "x_B"); axis(a, (0, 0), (0, 1), "g", "y_B")
    a.plot(0, 0, "ko"); a.text(0.01, -0.025, "B (base_link)\naxle midpoint", fontsize=8)
    c = P_B_C[:2]
    axis(a, c, (1, 0), "b", "z_C", 0.05); axis(a, c, (0, -1), "m", "x_C", 0.05)
    a.text(c[0] + 0.005, c[1] + 0.03, "C camera\n(y_C down)", fontsize=8, color="b")
    l = P_B_L[:2]
    a.add_patch(Circle(l, 0.035, fill=False, color="tab:orange", lw=1.5))
    a.text(l[0] - 0.02, l[1] + 0.045, "L LiDAR", fontsize=8, color="tab:orange")
    a.plot(icr, 0, "x", color="tab:purple", ms=10, mew=2)
    a.annotate(f"ICR (calibrated)\nx = {1000*icr:.1f} mm", (icr, 0), (-0.12, -0.11), fontsize=8,
               color="tab:purple", arrowprops=dict(arrowstyle="->", color="tab:purple", lw=0.8))
    a.set_xlim(-0.25, 0.2); a.set_ylim(-0.2, 0.2); a.set_aspect("equal")
    a.set_title("Top view: robot frames (extrinsics from the MJCF)")
    a.set_xlabel("x_B [m]"); a.set_ylabel("y_B [m]"); a.grid(alpha=0.3)
    txt = (f"B->C: t = ({P_B_C[0]:.4f}, {P_B_C[1]:.1f}, {P_B_C[2]:.3f}) m, R = [[0,0,1],[-1,0,0],[0,-1,0]]\n"
           f"B->L: t = ({P_B_L[0]:.3f}, {P_B_L[1]:.1f}, {P_B_L[2]:.3f}) m, R = I")
    a.text(-0.245, -0.195, txt, fontsize=7.5, family="monospace", va="bottom")

    b = ax[1]
    b.add_patch(Rectangle((W.ROOM_X - 0.6, 0), 0.6, 0.4, fill=False, ls=":"))
    b.add_patch(Rectangle((W.ROOM_X, 0), 0.05, 0.4, color="0.8"))
    for z, s, lab, col, dz in ((W.DOCK_MARKER_HEIGHT, 0.15, "dock tag 0 (15 cm)", "k", 0.05),
                               (W.DOCK_SMALL_HEIGHT, 0.04, "dock tags 10/11 (4 cm, y = ±5 cm)",
                                "tab:orange", 0.045)):
        b.add_patch(Rectangle((W.ROOM_X - 0.006, z - s / 2), 0.006, s, color=col))
        b.annotate(lab + f", centre h = {z:.3f} m", (W.ROOM_X - 0.006, z), (W.ROOM_X - 0.58, z + dz),
                   fontsize=8, color=col, arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
    b.add_patch(Rectangle((W.ROOM_X - 0.06, 0), 0.06, 0.06, color="tab:orange", alpha=0.4))
    b.text(W.ROOM_X - 0.03, 0.03, "dock", fontsize=7, ha="center", va="center")
    b.axhline(W.MARKER_HEIGHT, color="0.6", ls="--", lw=0.8)
    b.text(W.ROOM_X - 0.58, W.MARKER_HEIGHT + 0.006, f"wall tags 1-9 (15 cm) centre h = {W.MARKER_HEIGHT} m",
           fontsize=7.5, color="0.4")
    dp = W.ROOM_X - W.DOCK_STANDOFF
    b.add_patch(Rectangle((dp - 0.064 - 0.1405, 0.01), 0.281, 0.1, fill=False))
    b.add_patch(Circle((dp, 0.033), 0.033, fill=False))
    b.plot(dp + P_B_C[0], P_B_C[2], "bs"); b.text(dp + P_B_C[0], P_B_C[2] + 0.015, "C", color="b")
    b.plot(dp + P_B_L[0], P_B_L[2], "o", color="tab:orange")
    b.text(dp + P_B_L[0] - 0.02, P_B_L[2] + 0.015, "L", color="tab:orange")
    b.plot(dp, 0, "ko"); b.text(dp, -0.025, "B on floor", fontsize=8, ha="center")
    b.annotate("", (W.ROOM_X, -0.045), (dp, -0.045), arrowprops=dict(arrowstyle="<->"))
    b.text((W.ROOM_X + dp) / 2, -0.065, f"docked: {W.DOCK_STANDOFF} m", ha="center", fontsize=8)
    for k in (-1, 1):
        b.plot([dp + P_B_C[0], W.ROOM_X], [P_B_C[2], P_B_C[2] + k * np.tan(np.radians(24.4)) *
                                           (W.ROOM_X - dp - P_B_C[0])], "b:", lw=0.8)
    b.set_xlim(W.ROOM_X - 0.6, W.ROOM_X + 0.08); b.set_ylim(-0.08, 0.42); b.set_aspect("equal")
    b.set_title("Side view at the docked pose (camera vertical FOV dotted)")
    b.set_xlabel("x_W [m]"); b.set_ylabel("z [m]")
    fig.text(0.5, 0.01, "World W: origin at the room's SW inner corner, x east, y north, z up. "
             "Marker frame M: origin at tag centre, x along the wall, y up, z = outward normal.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "frames.png", dpi=130)
    print("wrote", FIG / "frames.png")


if __name__ == "__main__":
    main()
