"""The deployment contract, mirrored from microduck_rl.

Single source of truth for everything that must match the runtime and the
official training stack. Values are copied from:
  - microduck_rl/scripts/infer_policy.py  (DEFAULT_POSE, obs order, 50 Hz timing)
  - microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py
    (action scale 1.0, command ranges, obs noise magnitudes)

Obs layout (61D, order is the hot-swap contract — never reorder):
  [ base_ang_vel(3), projected_gravity(3), joint_pos_rel(14), joint_vel(14),
    last_action(14), twist_cmd(3), head_pose_cmd(4), body_pose_cmd(6) ]
Action (14D): target = DEFAULT_POSE + action  (scale 1.0, radians).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# microduck_rl checkout providing the MJCF models. Sibling of this project by
# default; override with MICRODUCK_RL_DIR for a non-standard layout.
MICRODUCK_RL_DIR = Path(
    os.environ.get(
        "MICRODUCK_RL_DIR",
        Path(__file__).resolve().parents[3] / "microduck_rl",
    )
)
SCENE_WALK_XML = MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
# Full-collision scene (head/trunk/hips can rest on the floor) — required by
# inverted/ground tricks (headstand); the walk scene strips those contacts.
SCENE_ALL_XML = MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck/scene.xml"

# Joint order (14 servos) — identical to model order in robot_walk.xml and to
# infer_policy.py's DEFAULT_POSE: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
NUM_JOINTS = 14
LEG_JOINT_IDS = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
HEAD_JOINT_IDS = np.array([5, 6, 7, 8])

# STAND2 pose (matches HOME_FRAME in microduck_constants.py and the STAND keyframe).
DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,        # left leg
    0.3491, 0.3491, 0.0, 0.0,                      # neck/head
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,          # right leg
], dtype=np.float32)

# Timing — infer_policy.py sets model.opt.timestep = 0.005, decimation 4 → 50 Hz.
PHYSICS_DT = 0.005
DECIMATION = 4
CTRL_DT = PHYSICS_DT * DECIMATION  # 0.02 s

# Obs block sizes, in contract order.
OBS_DIM = 61
CMD_DIM = 13  # twist(3) + head_pose(4) + body_pose(6)

# Command ranges (velocity env cfg — fixed, no widening curriculum).
LIN_VEL_X_RANGE = (-0.4, 0.4)
LIN_VEL_Y_RANGE = (-0.3, 0.3)
ANG_VEL_Z_RANGE = (-1.0, 1.0)
# Keep-alive ranges for the unused command slots (velocity env initial ranges):
# neurons for these inputs must stay alive for later curricula / other tasks.
HEAD_CMD_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
BODY_CMD_RANGES = tuple(((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3)

# Actor observation noise (uniform, from the velocity env cfg).
NOISE_GYRO = 0.03
NOISE_GRAVITY = 0.01
NOISE_JOINT_POS = 0.001
NOISE_JOINT_VEL = 0.25


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` by the inverse of quaternion [w, x, y, z] — verbatim from infer_policy.py.

    The crosses are unrolled by hand: `np.cross` on single 3-vectors spends
    ~15 us in moveaxis/broadcast plumbing and this runs 4x per control step
    (obs + rewards). Each component below is the exact multiply-subtract
    np.cross performs, so the result is bit-identical (held to the bit by
    test_bam_perf_parity.py).
    """
    w = quat_wxyz[0]
    x, y, z = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    v0, v1, v2 = vec[0], vec[1], vec[2]
    t0 = (y * v2 - z * v1) * 2   # t = cross(xyz, vec) * 2
    t1 = (z * v0 - x * v2) * 2
    t2 = (x * v1 - y * v0) * 2
    return np.array((v0 - w * t0 + (y * t2 - z * t1),
                     v1 - w * t1 + (z * t0 - x * t2),
                     v2 - w * t2 + (x * t1 - y * t0)), dtype=np.float32)


def scene_walk_ball_xml() -> Path:
    """The walk scene with upstream's 70 mm / 15 g kick ball (roadmap item 7,
    4c revisit): `scene_walk.xml` rewritten beside SYMLINKS to every file of
    the robot directory - MuJoCo resolves meshes relative to the main file -
    with `ball.xml` included last and every keyframe padded by the ball's
    seven qpos, since a free joint changes nq and upstream's own
    scene_ball.xml drops its keyframes for exactly that reason. Generated on
    demand under microduck_local/.cache; the pinned upstream checkout is
    never written to. The walk env indexes joints by address, so nothing
    else in it changes: measured, the shipped walker walks this scene
    without a fall."""
    import re
    src, d = SCENE_WALK_XML, SCENE_WALK_XML.parent
    out = Path(__file__).resolve().parents[2] / ".cache" / "scene"
    out.mkdir(parents=True, exist_ok=True)
    for entry in d.iterdir():
        link = out / entry.name
        target = entry.resolve()
        if link.is_symlink():
            if os.readlink(link) == str(target):
                continue
            link.unlink()
        elif link.exists():
            link.unlink()
        try:
            link.symlink_to(target)
        except FileExistsError:                       # a sibling vec-env worker got there first
            pass
    xml = src.read_text().replace("</mujoco>", '    <include file="ball.xml"/>\n</mujoco>')
    xml = re.sub(r'qpos="([^"]+)"', lambda m: f'qpos="{" ".join(m.group(1).split())} 0.3 0 0.035 1 0 0 0"', xml)
    p = out / "scene_walk_ball.xml"
    if not p.exists() or p.read_text() != xml:
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")        # atomic: a worker never reads a half-written scene
        tmp.write_text(xml)
        os.replace(tmp, p)
    return p
