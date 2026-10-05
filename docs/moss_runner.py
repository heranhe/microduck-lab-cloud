"""Reference implementation of the MOSS policy loop — `moss-arm-32-v1`.

    python moss_runner.py --selfcheck            # arithmetic, no robot needed
    python moss_runner.py --selfcheck --runs DIR # ...and the real ONNX graphs

This is the file `docs/moss-policy-schema.md` refers to. It carries the whole
deploy-side contract in one place: how to build the 32 observation floats, how
to apply the 8 action floats, and which of the three policies drives at any
moment. It depends on `numpy` and `onnxruntime` and NOTHING from the training
repo, so it can be copied onto the Jetson as-is.

WHAT IT DOES NOT DO, because those are yours: there is no detector here (feed
`Tracker.update` a bearing and a range from whatever you run), and no servo
bus (take `Command.joint_targets` and `Command.track_speeds` and write them).
Everything between those two is here and is checked by `--selfcheck`.

Three things in here are the ones that bite, all of them measured rather than
assumed:

* **Clip the graph output to +/-1 before scaling.** The exported pickup graph
  reaches 3.129 on a real observation. It is a Gaussian mean, not a tanh.
* **Actions are INCREMENTS.** `cmd += a * scale`, clamped to the joint range.
  A nudge can say "hold" by being zero; an absolute target cannot.
* **`target_seen` must be honest.** During the stow it is legitimately 0 on
  92% of ticks — the can is inside the jaws, occluded, and far nearer than the
  D455f's 0.52 m depth minimum. That leg runs on proprioception, and handing
  it a stale belief dressed as a fresh one is the mismatch that breaks it.

The observation normalizer is baked INTO each graph. Do not scale inputs.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

CONTRACT_ID = "moss-arm-32-v1"
CONTROL_HZ = 25.0
CONTROL_DT = 1.0 / CONTROL_HZ

# ---------------------------------------------------------------- the joints
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
               "wrist_roll", "finger_left", "finger_right")
ARM_JOINTS = JOINT_NAMES[:5]
GRIPPER_JOINT = "finger_left"          # ONE servo drives both jaws
#: Subtracted from measured positions to make obs slots 0-6.
DEFAULT_POSE = np.array([1.34, -0.65, 0.40, 1.25, -0.05, 0.037, 0.037],
                        dtype=np.float32)
JOINT_RANGE = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "finger_left": (0.0, 0.041),
    "finger_right": (0.0, 0.041),
}

# ------------------------------------------------------------ the layout v1
OBS_DIM = 32
NUM_ACTIONS = 8
OBS_JOINT_POS = slice(0, 7)
OBS_JOINT_VEL = slice(7, 14)
OBS_LAST_ACTION = slice(14, 22)
OBS_BASE_TWIST = slice(22, 24)
OBS_TARGET_BASE = slice(24, 27)
OBS_TARGET_SEEN = slice(27, 28)
OBS_RESERVED = slice(28, 32)
ACT_ARM = slice(0, 5)
ACT_GRIPPER = slice(5, 6)
ACT_BASE = slice(6, 8)

# ------------------------------------------------------------- the scalings
ARM_DELTA_RAD = 0.03        # per action unit, per tick
GRIP_DELTA_M = 0.004
VX_SCALE_MPS = 0.20
WZ_SCALE_RPS = 1.0

# --------------------------------------------------------------- the chassis
TRACK_CENTRES_M = 0.266     # V0.4 CAD belt centres — NOT the 0.244 proxies
MAX_TRACK_SPEED_MPS = 0.6   # per track, his firmware envelope
CAMERA_POS_X = 0.156        # camera front face ahead of the rover origin
CAN_HALF_HEIGHT_M = 0.0575
FIX_STALE_S = 0.6           # past this, slots 24-27 go to zero

# ----------------------------------------------------------------- the legs
GRASP_JAW_CTRL_M = 0.027    # the gentlest jaw command that still holds
MISSION_OPEN_M = 0.037
DEPLOY_STANDOFF_M = 0.55    # approach hands over inside this
GRASP_STANDOFF_M = 0.26
#: The box the PICKUP policy is competent in, base frame (x_lo, x_hi, |y|max).
#: The mission must deliver the can into it before handing over: measured in
#: simulation, only 28% of handovers started inside it and the leg is 12/12
#: when its env starts it inside.
PICK_HANDOVER_BOX = (0.36, 0.55, 0.12)
GRASP_POSE = (1.3534, 0.3752, 0.7008, 0.5032, -0.05)
TUCK_POSE = (-0.9129, 0.6491, 0.5893, 1.6044, -0.7775)
LIFT_POSE = (1.3534, -0.8146, 0.3999, 1.4345, -0.05)
#: The scripted delivery: up, round, then down. The last pose is the one the
#: arm SETTLES at resting against the bin's front wall, not one commanded
#: through it — commanding the latter held five servos in a stall at 33 N for
#: the whole release. See the schema's note before changing it.
STOW_HIGH = (1.3534, -1.35, 0.10, 1.4345, -0.05)
STOW_TURNED = (-1.776, -1.35, 0.10, 1.4345, -0.05)
STOW_INSIDE = (-1.7706, -0.8998, 1.2069, -0.717, 1.154)


def twist_to_tracks(vx: float, wz: float,
                    width: float = TRACK_CENTRES_M,
                    limit: float = MAX_TRACK_SPEED_MPS) -> tuple[float, float]:
    """(vx, wz) -> (left, right) track speeds, m/s, scaled not clipped.

    An unreachable twist SLOWS BOTH TRACKS by a common factor rather than
    saturating one of them, which would flatten the turn into a straight line
    at exactly the moment the robot most needs to turn.
    """
    half = 0.5 * width * wz
    left, right = vx - half, vx + half
    peak = max(abs(left), abs(right))
    if peak > limit:
        left *= limit / peak
        right *= limit / peak
    return float(left), float(right)


@dataclass
class Tracker:
    """The can's position in the BASE frame, held between detector frames.

    Feed it `update(bearing, range, t)` from your detector. Between frames it
    re-expresses the last fix with odometry; it does NOT extrapolate the can's
    own motion, because a velocity differenced from two 10 Hz fixes carries
    ~0.2 m/s of noise and predicting measured WORSE (22.2 mm against 16.5).
    """
    xy: tuple[float, float] | None = None
    t: float = -1e9

    def update(self, bearing_rad: float, range_m: float, t: float) -> None:
        self.xy = (CAMERA_POS_X + range_m * math.cos(bearing_rad),
                   range_m * math.sin(bearing_rad))
        self.t = float(t)

    def carry(self, d_forward: float, d_yaw: float) -> None:
        """Move the held fix with the robot: call once per tick with the
        odometry increment since the last call."""
        if self.xy is None:
            return
        x, y = self.xy
        x -= d_forward
        c, s = math.cos(-d_yaw), math.sin(-d_yaw)
        self.xy = (x * c - y * s, x * s + y * c)

    def fresh(self, t: float) -> bool:
        return self.xy is not None and (t - self.t) < FIX_STALE_S

    def slots(self, t: float) -> tuple[float, float, float, float]:
        """(x, y, z, seen) for obs 24-27. Stale reads as four zeros."""
        if not self.fresh(t):
            return 0.0, 0.0, 0.0, 0.0
        return self.xy[0], self.xy[1], CAN_HALF_HEIGHT_M, 1.0


@dataclass
class Senses:
    """What the robot reports, in SI, at one tick."""
    t: float
    joint_pos: np.ndarray            # 7, rad x5 then m x2, MEASURED
    joint_vel: np.ndarray            # 7, rad/s x5 then m/s x2
    base_twist: tuple[float, float]  # (vx m/s, wz rad/s), MEASURED
    holding: bool = False            # both pads on the same object


@dataclass
class Command:
    joint_targets: dict = field(default_factory=dict)
    track_speeds: tuple[float, float] = (0.0, 0.0)
    note: str = ""


def build_obs(s: Senses, tracker: Tracker, last_action: np.ndarray) -> np.ndarray:
    """The 32 floats, in contract order. Nothing here is normalized: the
    normalizer is inside the graph."""
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[OBS_JOINT_POS] = np.asarray(s.joint_pos, np.float32) - DEFAULT_POSE
    o[OBS_JOINT_VEL] = np.asarray(s.joint_vel, np.float32)
    o[OBS_LAST_ACTION] = np.asarray(last_action, np.float32)
    o[OBS_BASE_TWIST] = np.asarray(s.base_twist, np.float32)
    x, y, z, seen = tracker.slots(s.t)
    o[OBS_TARGET_BASE] = (x, y, z)
    o[OBS_TARGET_SEEN] = seen
    o[OBS_RESERVED] = 0.0
    return o


def apply_action(action, cmd: dict) -> tuple[dict, tuple[float, float]]:
    """8 floats -> updated joint targets and (left, right) track speeds.

    `cmd` is the running command dict and is updated IN PLACE, because the
    arm slots are increments on it rather than targets.
    """
    a = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
    if a.size != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} actions, got {a.size}")
    # `float(a[i])` before the arithmetic, deliberately. NumPy's weak
    # promotion makes `python_float + np.float32` a float32, so doing this in
    # place would round every joint target to float32 and land it ~6e-8 PAST
    # the limit it was just clamped to. Caught by this file's own selfcheck.
    for i, j in enumerate(ARM_JOINTS):
        lo, hi = JOINT_RANGE[j]
        cmd[j] = min(max(cmd.get(j, 0.0) + float(a[i]) * ARM_DELTA_RAD, lo), hi)
    lo, hi = JOINT_RANGE[GRIPPER_JOINT]
    cmd[GRIPPER_JOINT] = min(max(
        cmd.get(GRIPPER_JOINT, MISSION_OPEN_M)
        + float(a[ACT_GRIPPER][0]) * GRIP_DELTA_M, lo), hi)
    vx = float(a[ACT_BASE][0]) * VX_SCALE_MPS
    wz = float(a[ACT_BASE][1]) * WZ_SCALE_RPS
    return cmd, twist_to_tracks(vx, wz)


class Policy:
    """One exported leg. The graph takes obs[1,32] and returns actions[1,8]."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path,
                                         providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        shape = list(self.sess.get_inputs()[0].shape)
        if shape[-1] != OBS_DIM:
            raise ValueError(f"{path}: input is {shape}, not [...,{OBS_DIM}] "
                             f"— is this a {CONTRACT_ID} export?")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.name: obs.reshape(1, -1).astype(np.float32)})
        return np.asarray(out[0][0], np.float32)


def in_handover_box(xy) -> bool:
    """Is the can where the pickup policy is competent? See the constant."""
    if xy is None:
        return False
    lo, hi, ymax = PICK_HANDOVER_BOX
    return lo <= xy[0] <= hi and abs(xy[1]) <= ymax


# ------------------------------------------------------------------ selfcheck

def selfcheck(runs: str | None = None) -> int:
    ok = 0

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok += 0 if cond else 1

    print(f"{CONTRACT_ID} @ {CONTROL_HZ:g} Hz")
    s = Senses(t=0.0, joint_pos=DEFAULT_POSE.copy(),
               joint_vel=np.zeros(7, np.float32), base_twist=(0.0, 0.0))
    tr = Tracker()
    o = build_obs(s, tr, np.zeros(NUM_ACTIONS, np.float32))
    check("obs is 32 float32", o.shape == (OBS_DIM,) and o.dtype == np.float32)
    check("at DEFAULT_POSE the joint slots are zero",
          np.allclose(o[OBS_JOINT_POS], 0.0))
    check("no fix -> target slots AND seen are zero",
          np.allclose(o[24:28], 0.0))
    check("reserved slots are zero", np.allclose(o[OBS_RESERVED], 0.0))

    tr.update(bearing_rad=0.0, range_m=0.30, t=0.0)
    o = build_obs(s, tr, np.zeros(NUM_ACTIONS, np.float32))
    check("a fresh fix is x = 0.156 + range, seen = 1",
          abs(o[24] - (CAMERA_POS_X + 0.30)) < 1e-6 and o[27] == 1.0)
    s_late = Senses(t=FIX_STALE_S + 0.01, joint_pos=DEFAULT_POSE.copy(),
                    joint_vel=np.zeros(7, np.float32), base_twist=(0.0, 0.0))
    o = build_obs(s_late, tr, np.zeros(NUM_ACTIONS, np.float32))
    check(f"a fix older than {FIX_STALE_S}s reads as zeros",
          np.allclose(o[24:28], 0.0))

    cmd = {j: 0.0 for j in ARM_JOINTS} | {GRIPPER_JOINT: MISSION_OPEN_M}
    cmd, tracks = apply_action(np.full(NUM_ACTIONS, 5.0), dict(cmd))
    check("an out-of-range action is CLIPPED to +1 before scaling",
          abs(cmd["shoulder_pan"] - ARM_DELTA_RAD) < 1e-6)
    check("track speeds respect the per-track limit",
          max(abs(t) for t in tracks) <= MAX_TRACK_SPEED_MPS + 1e-9)
    cmd2, _ = apply_action(np.zeros(NUM_ACTIONS), dict(cmd))
    check("a zero action HOLDS the command", cmd2 == cmd)
    cmd3 = {j: JOINT_RANGE[j][1] for j in ARM_JOINTS} | {GRIPPER_JOINT: 0.041}
    cmd3, _ = apply_action(np.ones(NUM_ACTIONS), cmd3)
    check("increments clamp to the joint range",
          all(cmd3[j] <= JOINT_RANGE[j][1] + 1e-9 for j in ARM_JOINTS))

    l, r = twist_to_tracks(0.0, 1.0)
    check("a spin gives equal and opposite tracks", abs(l + r) < 1e-9 and r > 0)
    l, r = twist_to_tracks(10.0, 10.0)
    check("an unreachable twist scales BOTH tracks, keeping the ratio",
          abs(max(abs(l), abs(r)) - MAX_TRACK_SPEED_MPS) < 1e-9)
    check("the handover box accepts its own middle", in_handover_box((0.45, 0.0)))
    check("the handover box rejects a can too far", not in_handover_box((0.9, 0.0)))

    if runs:
        import os
        for leg, run in (("approach", "moss-approach-v1"),
                         ("pick", "moss-pick-v1"), ("stow", "moss-stow-v1")):
            path = os.path.join(runs, run, "policy.onnx")
            if not os.path.isfile(path):
                print(f"  SKIP  {leg}: no {path}")
                continue
            p = Policy(path)
            a = p(build_obs(s, tr, np.zeros(NUM_ACTIONS, np.float32)))
            check(f"{leg}: graph returns {NUM_ACTIONS} finite floats",
                  a.shape == (NUM_ACTIONS,) and np.all(np.isfinite(a)))
            c, t2 = apply_action(a, {j: 0.0 for j in ARM_JOINTS}
                                 | {GRIPPER_JOINT: MISSION_OPEN_M})
            check(f"{leg}: raw |a|max {np.abs(a).max():.3f} -> tracks within "
                  f"the envelope", max(abs(x) for x in t2) <= MAX_TRACK_SPEED_MPS + 1e-9)
    print("OK" if ok == 0 else f"{ok} FAILURE(S)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--runs", help="directory holding moss-*-v1/policy.onnx")
    args = ap.parse_args()
    raise SystemExit(selfcheck(args.runs) if args.selfcheck else
                     ap.print_help() or 0)
