"""WHERE in the bin to drop the next piece: MOSS's arm camera looks in, a
rule picks the clearest spot.

The split is deliberate (decision 2026-09-26, "option A"): perception and the
choice are SCRIPTED, and the stow policy learns to deliver to whatever spot it
is handed in `moss.OBS_DROP`. The alternative — learning the choice — needs a
summary of the bin's contents in the observation, which is a new contract and
a retrain of every leg.

What is modelled and what is not:

  * The look is taken from `BIN_LOOK_POSE`, not from where a delivery leaves
    the arm. No pose near that one sees the bin floor with every joint
    0.10 rad clear of its stop (the side mount looks ALONG the jaws), so a
    brain that uses it moves the whole arm after a delivery — shoulder and
    elbow by 0.2-0.5 rad and the wrist ROLL by 2.8 rad — looks, then folds.
    MEASURED: the straight joint move from the post-delivery pose (-1.577,
    -1.033, 0.017, 1.56, -0.43) touches nothing at 101 kinematic steps; a
    real servo path is not checked. No brain here takes the look yet —
    only `MossStowEnv(drop_target=True)`, kinematically.
  * Each item is detected only inside the arm camera's real field of view and
    range, only if the line of sight clears the bin's walls, with position
    noise and dropout.
  * NOT modelled: occlusion between items, and pixels. A detector reports
    positions; this is the position it would report, not an image.

`choose_drop_point` is pure and frame-agnostic so the brain can call it on a
real detector's output unchanged.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import moss

#: The arm pose the bin is looked at from. RE-SEARCHED 2026-09-27 for the
#: side mount (`moss.ARM_CAMERA_POS`): the first look pose, (-1.577, -1.033,
#: 0.017, 0.37, -0.43), was found for a lens that sat past the jaw tips and
#: looked back up the approach, and from the real mount it sees 0% of the bin.
#: The side lens looks ALONG the jaws, so the look now points them into the
#: bin from above and rolls the camera round to it. Search: 60,000 poses near
#: the post-delivery pose, 0.10 rad clear of every joint limit, arm touching
#: nothing, lines of sight clearing the rim. None sees the WHOLE floor (that
#: needs wrist_flex on its stop); this one sees 94% of a 6 x 6 floor grid and
#: keeps >= 90% of it with no contact in 189 of 200 poses perturbed by up to
#: 0.05 rad on every joint (worst 86%).
BIN_LOOK_POSE = (-1.776, -1.195, 0.559, 1.507, 2.338)
#: sd of a detected item's position, m. Of the order of the arm camera's own
#: range noise at 0.3 m.
DETECT_POS_SD = 0.010
#: chance a visible item is missed entirely
DETECT_DROPOUT = 0.10
#: the chooser's grid, m
GRID_M = 0.005
#: kept between the dropped object's edge and a wall, m
WALL_CLEARANCE_M = 0.004
#: 1 m of distance from the bin centre costs this much clearance, so equal
#: spots resolve to the one nearest the centre — an empty bin gets its centre
CENTRE_PULL = 0.05
#: scale of the published offset: +-1 at 5 cm from the bin centre
DROP_OBS_SCALE_M = 0.05

#: WHERE THE JAWS CAN GET TO. `data/moss_bin_reach.npz`: inverse kinematics
#: over the bin interior on a 1 cm grid, tool point 2 cm below the rim, four
#: restarts per cell (2026-09-26). 147 of 252 cells reachable — the +y side
#: only. The arm moves in one vertical plane swung by shoulder_pan, and pan
#: stops at -1.92 rad before that plane reaches y < -0.02 m; 3 cm and 8 cm
#: ABOVE the rim give the same boundary (150/252). A spot chosen past it was
#: 45 mm out of reach and the released object landed ON THE RIM 11 times in 25.
REACH_FILE = Path(__file__).parent / "data" / "moss_bin_reach.npz"

BIN_CENTRE = ((moss.BIN_INTERIOR_X[0] + moss.BIN_INTERIOR_X[1]) / 2,
              (moss.BIN_INTERIOR_Y[0] + moss.BIN_INTERIOR_Y[1]) / 2)


def in_view(cam_pos, cam_mat, point) -> bool:
    """Is `point` (world) inside the ARM camera's field and range?

    Same convention as `MossPickEnv._sense_arm`: the camera body looks along
    its local +x.
    """
    v = np.asarray(cam_mat, float).reshape(3, 3).T @ (
        np.asarray(point, float) - np.asarray(cam_pos, float))
    rng = float(np.linalg.norm(v))
    if rng > moss.ARM_CAMERA_MAX_RANGE_M or v[0] <= 1e-6:
        return False
    az = abs(math.degrees(math.atan2(v[1], v[0])))
    el = abs(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))))
    return (az <= moss.ARM_CAMERA_HFOV_DEG / 2
            and el <= moss.ARM_CAMERA_VFOV_DEG / 2)


def footprint_radius(prop, xmat) -> float:
    """The radius of a circle that covers the item's FLOOR footprint, from
    its shape and orientation — what a detector's bounding box gives. Using
    the bare radius treated a tall can lying on its side (up to 18 cm long)
    as a 5 cm disc."""
    R = np.asarray(xmat, float).reshape(3, 3)
    if prop.shape == "sphere":
        return float(prop.radius)
    if prop.shape == "cylinder":
        axial = prop.half_height * float(np.hypot(R[0, 2], R[1, 2]))
        return float(axial + prop.radius)
    s = np.asarray(prop.size[:3], float)
    corners = [R[:2] @ (s * np.array(sg)) for sg in
               ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))]
    return float(max(np.hypot(*c) for c in corners))


def clears_walls(cam_pos, point, base_xy_yaw) -> bool:
    """Does the line of sight from the camera to `point` (world) enter the
    bin through its MOUTH — i.e. cross the rim height inside the interior?
    A camera below the rim sees nothing inside."""
    cp = np.asarray(cam_pos, float)
    w = np.asarray(point, float)
    if cp[2] <= moss.BIN_RIM_Z or w[2] >= moss.BIN_RIM_Z:
        return cp[2] > moss.BIN_RIM_Z
    t = (moss.BIN_RIM_Z - cp[2]) / (w[2] - cp[2])
    rx, ry = cp[0] + t * (w[0] - cp[0]), cp[1] + t * (w[1] - cp[1])
    x0, y0, yaw = base_xy_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    bx = (rx - x0) * c + (ry - y0) * s
    by = -(rx - x0) * s + (ry - y0) * c
    return bool(moss.BIN_INTERIOR_X[0] < bx < moss.BIN_INTERIOR_X[1]
                and moss.BIN_INTERIOR_Y[0] < by < moss.BIN_INTERIOR_Y[1])


@lru_cache(maxsize=1)
def _reach():
    if not REACH_FILE.is_file():
        return None
    z = np.load(REACH_FILE)
    ok = np.asarray(z["reach"], bool)
    # ONE CELL OF MARGIN: a cell whose neighbours are all reachable. An aim
    # on the boundary row sits at the edge of what the solver could reach.
    er = ok.copy()
    er[1:, :] &= ok[:-1, :]
    er[:-1, :] &= ok[1:, :]
    er[:, 1:] &= ok[:, :-1]
    er[:, :-1] &= ok[:, 1:]
    return z["xs"], z["ys"], er


def reachable(x: float, y: float) -> bool:
    """Can the jaws get over (x, y) (base frame)? Nearest cell of the map;
    True everywhere when no map is on disk."""
    r = _reach()
    if r is None:
        return True
    xs, ys, ok = r
    i = int(np.argmin(np.abs(xs - x)))
    k = int(np.argmin(np.abs(ys - y)))
    return bool(ok[i, k])


def choose_drop_point(detections, radius: float) -> tuple[float, float]:
    """The spot, in the BASE frame, with the most room for an object of
    `radius` — away from every detected item and from the walls.

    `detections` is an iterable of (x, y, r) in the base frame. Clearance is
    edge to edge; ties (and an empty bin) go to the centre.
    """
    lo_x, hi_x = moss.BIN_INTERIOR_X
    lo_y, hi_y = moss.BIN_INTERIOR_Y
    m = radius + WALL_CLEARANCE_M
    xs = np.arange(lo_x + m, hi_x - m + 1e-9, GRID_M)
    ys = np.arange(lo_y + m, hi_y - m + 1e-9, GRID_M)
    if xs.size == 0 or ys.size == 0:
        return BIN_CENTRE
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    clear = np.minimum.reduce([gx - lo_x, hi_x - gx, gy - lo_y, hi_y - gy]) - m
    for x, y, r in detections:
        clear = np.minimum(clear, np.hypot(gx - x, gy - y) - r - radius)
    score = clear - CENTRE_PULL * np.hypot(gx - BIN_CENTRE[0],
                                           gy - BIN_CENTRE[1])
    r = _reach()
    if r is not None:            # only where the arm can put it
        rx, ry, ok = r
        ii = np.abs(gx[..., None] - rx).argmin(-1)
        kk = np.abs(gy[..., None] - ry).argmin(-1)
        score = np.where(ok[ii, kk], score, -np.inf)
        if not np.isfinite(score).any():
            return BIN_CENTRE
    i = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(gx[i]), float(gy[i])


def drop_obs(drop_xy) -> tuple[float, float]:
    """The two floats `moss.OBS_DROP` carries: offset from the bin centre."""
    return ((drop_xy[0] - BIN_CENTRE[0]) / DROP_OBS_SCALE_M,
            (drop_xy[1] - BIN_CENTRE[1]) / DROP_OBS_SCALE_M)
