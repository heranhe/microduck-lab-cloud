"""PLAN the retract instead of guessing it.

Ten hand-built paths failed (reward tuning, more steps, direct fold, two
waypoint searches, task-space lift, three-phase, alternate target, six joint
orders). The definitive diagnostic: commanding straight to tuck and HOLDING
600 steps leaves the arm jammed on `bin_x1` with shoulder_pan 0.65 rad short.
Both endpoints are collision-free on their own; the bin wall sits between
them, so the arm has to go around and the route is not a straight line in any
space I picked.

This is the textbook case for a planner. A 5-DoF arm with two known-good
endpoints needs nothing heavyweight: sample intermediate configurations, keep
the ones that connect to BOTH ends by a collision-free straight segment in
joint space, and take the cheapest. That is a one-waypoint PRM, and it checks
collisions with kinematics only (mj_forward), so it costs microseconds per
sample rather than a physics rollout.
"""
import numpy as np
import mujoco

from microduck_local.robots import moss
from microduck_local.robots.moss_env import MossStowEnv

ARM = moss.ARM_JOINTS[:5]


def collides(m, d, adrs, q, arm_geoms) -> bool:
    """Is the arm touching anything at this configuration? Kinematics only."""
    for a, v in zip(adrs, q):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    for c in range(d.ncon):
        con = d.contact[c]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
        if ({n1, n2} & arm_geoms) and con.dist < -1e-4:
            # ignore the object it just let go of
            if "can_geom" in (n1, n2):
                continue
            return True
    return False


def seg_free(m, d, adrs, a, b, arm_geoms, n=24) -> bool:
    for f in np.linspace(0.0, 1.0, n):
        if collides(m, d, adrs, a + f * (b - a), arm_geoms):
            return False
    return True


def plan(m, d, adrs, start, goal, arm_geoms, lo, hi, tries=4000, rng=None):
    """One intermediate configuration that connects start to goal."""
    rng = rng or np.random.default_rng(0)
    if seg_free(m, d, adrs, start, goal, arm_geoms):
        return []                                  # straight line is fine
    best = None
    for _ in range(tries):
        q = rng.uniform(lo, hi)
        if collides(m, d, adrs, q, arm_geoms):
            continue
        if not seg_free(m, d, adrs, start, q, arm_geoms, 16):
            continue
        if not seg_free(m, d, adrs, q, goal, arm_geoms, 16):
            continue
        cost = float(np.abs(q - start).sum() + np.abs(goal - q).sum())
        if best is None or cost < best[0]:
            best = (cost, q)
    return [best[1]] if best else None


#: **THE ONE WAYPOINT, found by the planner and then FIXED.**
#:
#: The planner solves each episode, but samples a different route every time —
#: so as a TEACHER it demonstrates conflicting actions for near-identical
#: observations, and a clone of it averages them into mush (250 episodes,
#: action MSE 0.022, still 0/24 folded home). Behaviour cloning needs a
#: DETERMINISTIC teacher.
#:
#: So the planner was run once against 14 real post-delivery configurations and
#: asked for one waypoint connecting ALL of them to the tuck pose by
#: collision-free straight segments. This is it: 14/14 connected, and flown in
#: physics it folds the arm home on 14/24 seeds at a 0.154 rad residual,
#: bringing the forward reach to 114 mm — the tuck pose's own value — against
#: 178 mm untouched. Every hand-built route scored 0-4/24.
UNIVERSAL_WAYPOINT = np.array([0.5259, -0.8036, -1.5515, -1.6033, 1.7983])
