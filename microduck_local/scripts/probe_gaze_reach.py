"""How far down can a duck look, what does it then SEE, and what does it cost?
(roadmap Track 4 item 12k)

    cd microduck_local
    uv run python scripts/probe_gaze_reach.py

The question this answers came off the /sim page: a duck looks down for the
ball and still does not scan the ground by its own feet — is there room in the
joints, and does using it need a walker retrained on wider head commands?

`C.HEAD_CMD_RANGES` is ±0.05 rad on the head and neck slots, so EVERY gaze the
chase brain sends is already extrapolation. This drives the shipped walker at
each pose while walking and reports what it actually delivers:

  * **depression** — the camera site's own pitch, read off the model, so it
    includes whatever the gait does to the head;
  * **the floor window** — where the vertical field of view (`fov_v_deg`, 48°
    and FIXED) meets z = 0, near and far edge, measured from the trunk origin.
    This is the number that matters: a deeper gaze SLIDES that window down, it
    does not widen it, so what a duck gains at its feet it loses in the mid
    range where the ball is while it walks its line-up;
  * **forward speed and falls** — the cost, and whether the walker copes at
    all with a command 20× outside its training range.

Measured 2026-09-08: it copes (0 falls in 28 trials, depression monotonic in
the command), and the split across both slots is CHEAPER than the head slot
alone for more depression. But the window slide is why `gaze_neck` still ships
off — see item 12k for the in-play numbers.
"""

from __future__ import annotations

import argparse
import math

import mujoco

from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.sensors.detector import DetectorSpec
from microduck_local.world import World, make_pitch

# (neck command, head command, label) — offsets on C.DEFAULT_POSE, the same
# units `Chase` emits in `head_pose_cmd`. A DOWNWARD gaze is +head and −neck.
POSES = (
    (0.0, 0.0, "level (what the walker trained on)"),
    (0.0, 0.30, "head +0.30"),
    (0.0, 0.60, "head +0.60  (the shipped gaze cap)"),
    (-0.30, 0.60, "neck -0.30 / head +0.60  (gaze_neck 0.5)"),
    (-0.60, 0.60, "neck -0.60 / head +0.60  (gaze_neck 1.0)"),
    (-0.60, 0.90, "neck -0.60 / head +0.90"),
    (-1.00, 1.00, "neck -1.00 / head +1.00  (past any knob)"),
)


def run(neck: float, head: float, vx: float, seconds: float, seeds: int) -> dict:
    """One pose, held while walking, over `seeds` worlds."""
    half_v = math.radians(DetectorSpec().fov_v_deg) / 2
    deps, speeds, nears, fars, falls = [], [], [], [], 0
    for sd in range(seeds):
        sc = make_pitch(per_side=1)
        infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
        w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=sd)
        d = w.ducks["d0"]
        cam = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, "d0/head_camera")
        start, t0 = None, None
        while w.t < seconds:
            d.set_cmd(w.data, (vx, 0.0, 0.0), (neck, head, 0.0, 0.0))
            w.step()
            if w.t < 2.0:                     # let the gait and the head settle first
                continue
            if start is None:
                start, t0 = w.odom(d)[:2], w.t
            R = w.data.site_xmat[cam].reshape(3, 3)
            dep = -math.asin(max(-1.0, min(1.0, R[2, 0])))
            deps.append(dep)
            z = float(w.data.site_xpos[cam][2])
            ahead = float(w.data.site_xpos[cam][0]) - float(w.data.qpos[d.adr.root_qpos])
            nears.append((z / math.tan(dep + half_v) if dep + half_v < math.pi / 2 else 0.0) + ahead)
            fars.append((z / math.tan(dep - half_v) + ahead) if dep - half_v > 0.02 else math.inf)
        if d.fallen(w.data):
            falls += 1
        elif start is not None:
            speeds.append(math.dist(w.odom(d)[:2], start) / max(w.t - t0, 1e-9))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")      # noqa: E731
    return {"dep": mean(deps), "near": mean(nears), "speed": mean(speeds), "falls": falls,
            "far": math.inf if any(math.isinf(f) for f in fars) else mean(fars)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--vx", type=float, default=0.3, help="forward command held through the sweep")
    args = ap.parse_args()
    print(f"the shipped walker, walking at vx={args.vx:g}, {args.seeds} seeds x {args.seconds:g} s a pose")
    print(f"vertical field of view {DetectorSpec().fov_v_deg:.0f} deg (FIXED: a deeper gaze slides this window, "
          "it does not widen it)")
    print(f"\n  {'commanded pose':<40}{'depression':>11}{'floor window (from the root)':>30}{'speed':>9}{'vs level':>10}{'falls':>7}")
    base = None
    for neck, head, label in POSES:
        r = run(neck, head, args.vx, args.seconds, args.seeds)
        base = r["speed"] if base is None else base
        far = "the horizon" if math.isinf(r["far"]) else f"{r['far']:.2f} m"
        window = f"{r['near']:.2f} .. {far}"
        rel = (r["speed"] / base - 1) * 100 if base else float("nan")
        print(f"  {label:<40}{math.degrees(r['dep']):>8.0f} deg{window:>30}{r['speed']:>9.3f}{rel:>+9.1f}%"
              f"{r['falls']:>4}/{args.seeds}")


if __name__ == "__main__":
    main()
