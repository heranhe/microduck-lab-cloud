"""`probe_head_pitch`: how far down can this walker actually look, and what
does that buy the detector?

    uv run python scripts/probe_head_pitch.py --part cam    # command -> camera angle
    uv run python scripts/probe_head_pitch.py --part blind  # ...and what it sees
    uv run python scripts/probe_head_pitch.py               # both

Two questions, both of which the chase brain answers today with a constant
nobody measured:

1. `ChaseParams.head_down` clamps the head-pitch COMMAND at 0.6. The MJCF
   lets `neck_pitch` reach +1.047 and `head_pitch` +1.571, and the command
   is not a joint angle - the walker policy maps it - so the clamp could be
   the joint, the policy, or a number somebody picked. `camera()` sweeps the
   command and reads the achieved joints, the camera axis, whether the duck
   stays up and what the pose costs in forward speed. Same shape as
   `walker_facts.camera_pitch()`, which measures exactly two points of it.

2. The brain also never commands `neck_pitch` AT ALL - `Chase.step` emits
   `(0.0, gaze, 0.0, 0.0)` and slot 0 is the neck. So the second sweep is
   over both slots, together and apart.

3. `blind()` then asks what a pose is worth: with the head held at it, the
   closest floor ball the REAL `Detector` on a REAL composed `World` still
   reports. The brain's `refresh_min` = 0.35 exists because "the level camera
   loses a floor ball inside ~0.3 m"; this measures that number per pose.

Ranges are reported twice, because the brain and the world do not use the
same one: `ground` is the 2-D trunk-to-ball distance (the physical thing),
`range_est` is what the detector reports and what `refresh_min`,
`lineup_range` and `head_range` are compared against - a 3-D SLANT range from
the camera SITE, which at these distances is several cm larger (AGENTS.md,
verification discipline #8).
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.world import World
from microduck_local.world.scenario import Ball, Duck, Scenario, Wall

WALKER = POLICIES_DIR / "alpha_walking.onnx"
BALL_R = 0.035


def _flat(with_ball: bool = False, yaw: float = 0.0):
    """An empty 13 x 13 m floor, one duck at the origin. Big enough that a
    6 s walk never reaches a wall (walker_facts learned this the hard way:
    a 3 x 2.5 m room measures the boxes, not the gait)."""
    h = 6.0
    cs = [(-h, -h), (h, -h), (h, h), (-h, h)]
    sc = Scenario(
        name="headprobe", seed=0, floor=(13.0, 13.0),
        walls=[Wall(cs[i], cs[(i + 1) % 4], 0.3, 0.02) for i in range(4)],
        boxes=[], balls=[Ball((3.0, 3.0), BALL_R)] if with_ball else [],
        ducks=[Duck("d0", (0.0, 0.0, float(yaw)), detector="datasheet", tof=None)],
    )
    w = World(sc, infer_for={"d0": onnx_infer(WALKER)}, seed=0)
    return w, w.ducks["d0"]


def _cam(w) -> int:
    return mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "d0/head_camera")


def _depression(w, cam: int) -> float:
    """Depression of the camera's optical axis below horizontal, rad — the
    identical definition `DetectionFrame.cam_pitch` is stamped with."""
    R = w.data.site_xmat[cam].reshape(3, 3)
    return float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))


# ---------------------------------------------------------------- part 1 ----

def _hold(w, d, cam, neck: float, head: float, vx: float,
          secs: float = 4.0, warm: float = 2.0) -> dict:
    """Command (neck, head) for `secs`; average everything over the tail."""
    n = int(secs / C.CTRL_DT)
    n0 = int(warm / C.CTRL_DT)
    deps, zs, njs, hjs, spd, up = [], [], [], [], [], []
    jq = d.adr.joint_qpos
    i_neck = list(C.JOINT_NAMES).index("neck_pitch")
    i_head = list(C.JOINT_NAMES).index("head_pitch")
    for i in range(n):
        d.set_cmd(w.data, (vx, 0.0, 0.0), (neck, head, 0.0, 0.0))
        w.step()
        if i >= n0:
            deps.append(_depression(w, cam))
            zs.append(float(w.data.site_xpos[cam][2]))
            njs.append(float(w.data.qpos[jq[i_neck]]))
            hjs.append(float(w.data.qpos[jq[i_head]]))
            spd.append(d.heading_speed(w.data))
            up.append(0.0 if d.fallen(w.data) else 1.0)
    return {"dep": float(np.mean(deps)), "dep_sd": float(np.std(deps)),
            "cam_z": float(np.mean(zs)),
            "neck_q": float(np.mean(njs)), "head_q": float(np.mean(hjs)),
            "speed": float(np.mean(spd)), "upright": float(np.mean(up))}


def camera(args) -> None:
    lo, hi = C.MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck/robot_walk.xml", None
    print(f"  (joint ranges from {lo.name}; DEFAULT_POSE neck_pitch = head_pitch = "
          f"{C.DEFAULT_POSE[5]:.4f} rad)")
    del hi
    cmds = [round(v, 2) for v in np.arange(0.0, args.cmd_max + 1e-9, args.cmd_step)]
    # SIGN, measured: +head_pitch looks DOWN, +neck_pitch looks UP. "neck"
    # therefore sweeps NEGATIVE, and "both" is (-c, +c) — the two slots
    # pulling the same way.
    for vx, tag in ((0.0, "standing"), (0.3, "walking vx=0.30")):
        for mode in ("head", "neck (negative = down)", "both (neck -c, head +c)"):
            print(f"  {tag}, sweeping {mode}:")
            base = None
            for c in cmds:
                neck = -c if mode.startswith(("neck", "both")) else 0.0
                head = c if mode.startswith(("head", "both")) else 0.0
                w, d = _flat()
                cam = _cam(w)
                r = _hold(w, d, cam, neck, head, vx, secs=args.secs)
                if base is None:
                    base = r
                cost = ("" if vx == 0 else
                        f" · speed {r['speed']:+.3f} m/s ({100 * (r['speed'] / base['speed'] - 1):+.0f}%)")
                print(f"    cmd {c:.2f} (neck {neck:+.2f}, head {head:+.2f}) -> camera {r['dep']:.3f} ± {r['dep_sd']:.3f} rad "
                      f"({math.degrees(r['dep']):5.1f}°), z {r['cam_z']:.3f} · joints neck "
                      f"{r['neck_q']:+.3f} head {r['head_q']:+.3f}"
                      f"{cost} · upright {r['upright']:.2f}")


# ---------------------------------------------------------------- part 2 ----

# (neck, head) commands worth asking about, from the sweep above: today's
# clamp (0, 0.6), the head slot on its own to its saturation (0, 1.25), the
# neck slot on its own, and the two together — which is where the depression
# actually peaks (standing 1.47 rad at (-1.0, +1.0), walking 1.40 at
# (-0.75, +0.75)) before the head joint hits +1.571 and folding the neck
# further starts bringing the axis back UP.
POSES = [(0.0, 0.0), (0.0, 0.6), (0.0, 0.9), (0.0, 1.25),
         (-1.0, 0.0), (-0.5, 0.5), (-0.75, 0.75), (-1.0, 1.0)]


def _place_ball(w, d, ground: float) -> tuple[float, float]:
    """Put the ball `ground` metres ahead of the trunk along the heading,
    resting on the floor. Returns (ground, range_est-truth) — the SLANT range
    from the camera site, which is what the detector reports."""
    jb = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "ball0")
    q = int(w.model.jnt_qposadr[w.model.body_jntadr[jb]])
    pos = d.trunk_pos(w.data)
    yaw = d.yaw(w.data)
    x, y = pos[0] + ground * math.cos(yaw), pos[1] + ground * math.sin(yaw)
    w.data.qpos[q:q + 3] = [x, y, BALL_R]
    w.data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
    w.data.qvel[int(w.model.jnt_dofadr[w.model.body_jntadr[jb]]):][:6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    cam = w.data.site_xpos[_cam(w)]
    return ground, float(np.linalg.norm(np.array([x, y, BALL_R]) - cam))


def _seen(w, d, reps: int) -> float:
    """Fraction of `reps` fresh captures that report the ball."""
    hits = 0
    for _ in range(reps):
        f = d.detector.capture(w.data, w.t)
        hits += any(x.cls == "ball" and x.name == "ball0" for x in f.detections)
    return hits / reps


def blind(args) -> None:
    grounds = [round(v, 3) for v in np.arange(0.04, args.far + 1e-9, 0.01)]
    print(f"  ground = 2-D trunk->ball; slant = the detector's own range_est "
          f"(what `refresh_min` compares to). {args.reps} captures a point, "
          f"datasheet noise.")
    print("  STANDING (the settle, and the `look` after a kick):")
    for neck, head in POSES:
        w, d = _flat(with_ball=True)
        cam = _cam(w)
        for _ in range(int(2.5 / C.CTRL_DT)):
            d.set_cmd(w.data, (0.0, 0.0, 0.0), (neck, head, 0.0, 0.0))
            w.step()
        dep, camz = _depression(w, cam), float(w.data.site_xpos[cam][2])
        fell = d.fallen(w.data)
        ps, slants = [], []
        for g in grounds:
            _, s = _place_ball(w, d, g)
            slants.append(s)
            ps.append(_seen(w, d, args.reps))
        ok = [i for i, p in enumerate(ps) if p >= 0.5]
        near = grounds[ok[0]] if ok else float("nan")
        near_s = slants[ok[0]] if ok else float("nan")
        far_ok = grounds[ok[-1]] if ok else float("nan")
        print(f"    neck {neck:.1f} head {head:.1f} · camera {dep:.3f} rad "
              f"({math.degrees(dep):5.1f}°), z {camz:.3f}{' FELL' if fell else ''} "
              f"-> nearest ball seen: ground {near:.2f} m (slant {near_s:.2f}), "
              f"visible out to {far_ok:.2f} m")
    print("  WALKING vx=0.30 (the chase and the walk-in; the gait holds the head higher):")
    for neck, head in POSES:
        w, d = _flat(with_ball=True)
        cam = _cam(w)
        for _ in range(int(2.0 / C.CTRL_DT)):
            d.set_cmd(w.data, (0.3, 0.0, 0.0), (neck, head, 0.0, 0.0))
            w.step()
        hits = {g: [0, 0] for g in grounds}
        deps = []
        jb = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "ball0")
        qb = int(w.model.jnt_qposadr[w.model.body_jntadr[jb]])
        for i in range(int(args.walk_s / C.CTRL_DT)):
            g = grounds[i % len(grounds)]
            keep = w.data.qpos[qb:qb + 7].copy()
            _place_ball(w, d, g)
            hits[g][0] += _seen(w, d, 1)
            hits[g][1] += 1
            deps.append(_depression(w, cam))
            w.data.qpos[qb:qb + 7] = keep      # the ball never really moved
            mujoco.mj_forward(w.model, w.data)
            d.set_cmd(w.data, (0.3, 0.0, 0.0), (neck, head, 0.0, 0.0))
            w.step()
        ps = [hits[g][0] / max(hits[g][1], 1) for g in grounds]
        ok = [i for i, p in enumerate(ps) if p >= 0.5]
        near = grounds[ok[0]] if ok else float("nan")
        print(f"    neck {neck:.1f} head {head:.1f} · camera {np.mean(deps):.3f} "
              f"± {np.std(deps):.3f} rad -> nearest ball seen: ground {near:.2f} m "
              f"({sum(h[1] for h in hits.values())} captures)")


# ---------------------------------------------------------------- part 3 ----

# Same depression, two ways of asking for it. The single sweep above says the
# two slots have different gains AND different costs, so a gaze that is worth
# 1.0 rad of depression can be bought at several prices; this prices them
# against each other with repeats, because one 4 s run of a gait is not a
# measurement (walker_facts: 4 headings, steady rate over seconds 2-6).
SPLITS = [(0.0, 0.0),
          (0.0, 0.4), (0.0, 0.6), (0.0, 0.8), (0.0, 1.0), (0.0, 1.25),
          (-0.2, 0.4), (-0.3, 0.6), (-0.4, 0.8), (-0.5, 1.0),
          (-0.4, 0.4), (-0.5, 0.5), (-0.6, 0.6), (-0.75, 0.75), (-1.0, 1.0),
          (-0.6, 0.0), (-1.0, 0.0), (-1.5, 0.0)]


def split(args) -> None:
    yaws = (0.0, 1.6, 3.1, 4.7)
    print(f"  walking vx=0.30, steady over seconds 2-{args.split_s:g}, {len(yaws)} headings "
          f"(the same repeat scheme walker_facts uses):")
    base = None
    for neck, head in SPLITS:
        deps, spds = [], []
        for y in yaws:
            w, d = _flat(yaw=y)
            cam = _cam(w)
            r = _hold(w, d, cam, neck, head, 0.3, secs=args.split_s, warm=2.0)
            deps.append(r["dep"])
            spds.append(r["speed"])
        dep, spd = float(np.mean(deps)), float(np.mean(spds))
        if base is None:
            base = spd
        print(f"    neck {neck:+.2f} head {head:+.2f} -> camera {dep:.3f} rad "
              f"({math.degrees(dep):5.1f}°) · speed {spd:.4f} ± {np.std(spds):.4f} m/s "
              f"({100 * (spd / base - 1):+5.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both", choices=("cam", "blind", "split", "both"))
    ap.add_argument("--split-s", type=float, default=6.0)
    ap.add_argument("--cmd-max", type=float, default=2.0)
    ap.add_argument("--cmd-step", type=float, default=0.2)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--far", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--walk-s", type=float, default=12.0)
    args = ap.parse_args()
    print(f"walker: {WALKER}")
    if args.part in ("cam", "both"):
        print("command -> camera (what the walker does with a head-pitch command):")
        camera(args)
    if args.part in ("blind", "both"):
        print("blind radius (the real Detector on a real World):")
        blind(args)
    if args.part in ("split", "both"):
        print("what a given depression COSTS, head slot against neck slot:")
        split(args)


if __name__ == "__main__":
    main()
