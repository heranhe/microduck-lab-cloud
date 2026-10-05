"""`probe_handover`: the state the brain HANDS the kick, against the state the
kick was TRAINED from (roadmap 12aw).

    cd microduck_local
    uv run python scripts/kick_gym.py --episodes 60 --seeds 16 --jobs 6 --out gym.jsonl
    uv run python scripts/probe_handover.py gym.jsonl

`lastmetre` (and the wide kick before it) spawns from windows somebody drew:
a uniform box for the ball, a uniform window per gaze axis, and — because the
spawn is the walk env's standing reset — a body and a set of joints that are
NOT MOVING. Play hands the skill something else. This prints the two side by
side and scores the only thing that decides whether re-spawning from play can
help: **the share of real handovers that the training spawn's support
contains.** An axis at 100% is an axis where a replay spawn is a no-op by
construction and its null would be expected, not informative (AGENTS.md:
"check a knob's reachable set first").

Two kinds of axis, kept apart because they are not the same claim:

  * **windowed** — the spawn draws them uniformly from a range, so coverage is
    a real fraction and the row prints it.
  * **pinned** — the spawn sets them to ZERO (`_reset_standing` zeroes qvel;
    the tip stages draw head yaw from (0, 0)), so the support is a single
    point and no tolerance makes a fraction honest. The row prints what play
    does instead, and the spawn column says `0`.

Ball offsets are read in the KICKING FOOT's sign convention, the one
`_lm_reset` spawns in (`sgn * uniform(*KICK_BOX_SIDE)`), not the duck's left;
gaze axes are OFFSETS off `DEFAULT_POSE`, which is what the windows are.
Push rows are dropped: a push is not a swing the kick skill was handed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_local import contract as C  # noqa: E402
from microduck_local.behaviors.kick import KICK_BOX_AHEAD, KICK_BOX_SIDE  # noqa: E402
from microduck_local.behaviors.lastmetre import (  # noqa: E402
    LM_GAZE_STAGE1,
    LM_GAZE_TIP_HEAD,
    LM_GAZE_TIP_NECK,
    LM_GAZE_YAW,
)

# The walk env's own spawn noise on every joint (`walk_env.reset`, +-0.03 rad).
# It is what a "(0, 0)" gaze window actually spawns, so the yaw axis is scored
# against it rather than against a bare point — the kinder reading of the two.
POSE_NOISE = 0.03


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _win(spec) -> tuple[float, float]:
    """A window that may be a ("lo,hi", "lo,hi") string pair or a tuple."""
    if isinstance(spec, str):
        lo, hi = spec.split(",")
        return float(lo), float(hi)
    return float(spec[0]), float(spec[1])


def axes(rows, gaze: str):
    """(label, play values, spawn window or None) per axis."""
    neck_w = _win(LM_GAZE_STAGE1[0] if gaze == "drill" else LM_GAZE_TIP_NECK)
    head_w = _win(LM_GAZE_STAGE1[1] if gaze == "drill" else LM_GAZE_TIP_HEAD)
    yaw_w = _win(LM_GAZE_STAGE1[2] if gaze == "drill" else LM_GAZE_YAW)
    yaw_w = (yaw_w[0] - POSE_NOISE, yaw_w[1] + POSE_NOISE)
    i_neck, i_head, i_yaw = (C.JOINT_NAMES.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw"))

    def sgn(r):
        return -1.0 if r["foot"] == "kick_right" else 1.0

    out = [
        ("ball ahead (m)", [r["ahead"] for r in rows], _win(KICK_BOX_AHEAD)),
        ("ball to the foot's side (m)", [sgn(r) * r["side"] for r in rows], _win(KICK_BOX_SIDE)),
        ("neck_pitch off HOME (rad)", [r["joints"][i_neck] - C.DEFAULT_POSE[i_neck] for r in rows], neck_w),
        ("head_pitch off HOME (rad)", [r["joints"][i_head] - C.DEFAULT_POSE[i_head] for r in rows], head_w),
        ("head_yaw toward the foot (rad)",
         [sgn(r) * (r["joints"][i_yaw] - C.DEFAULT_POSE[i_yaw]) for r in rows], yaw_w),
        ("|body linear vel| (m/s)", [math.hypot(r["body_vx"], r["body_vy"]) for r in rows], None),
        ("|body yaw rate| (rad/s)", [abs(r["body_wz"]) for r in rows], None),
        ("max |joint vel| (rad/s)", [max(abs(x) for x in r["joint_vel"]) for r in rows], None),
        ("|ball vel| (m/s)", [math.hypot(r["ball_vx"], r["ball_vy"]) for r in rows], None),
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="gym.jsonl files from kick_gym.py --out")
    ap.add_argument("--gaze", choices=("tip", "drill"), default="tip",
                    help="which lastmetre gaze window to score against (default: the tip, "
                         "which is what the finished policy trains under)")
    a = ap.parse_args()

    rows = []
    for p in a.paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("swing") and r.get("touch") == "kick" and "joints" in r:
                rows.append(r)
    if not rows:
        sys.exit("no swing rows with a `joints` column — re-run kick_gym.py after the "
                 "handover columns landed")

    print(f"\n{len(rows)} swings from {len(a.paths)} file(s); scored against lastmetre's "
          f"{a.gaze} spawn\n")
    print(f"{'axis':32} {'play: p10':>9} {'median':>8} {'p90':>8}   {'spawn draws from':>18}  "
          f"{'covered':>8}")
    print("-" * 92)
    windowed = []
    for label, vals, win in axes(rows, a.gaze):
        if win is None:
            span, cov = "0 (pinned)", ""
        else:
            lo, hi = win
            inside = sum(1 for v in vals if lo <= v <= hi)
            windowed.append([lo <= v <= hi for v in vals])
            span, cov = f"{lo:+.2f}..{hi:+.2f}", f"{inside / len(vals):7.0%}"
        print(f"{label:32} {q(vals, .10):9.3f} {st.median(vals):8.3f} {q(vals, .90):8.3f}   "
              f"{span:>18}  {cov:>8}")

    both = sum(1 for i in range(len(rows)) if all(col[i] for col in windowed))
    print("-" * 92)
    print(f"inside EVERY windowed axis at once: {both}/{len(rows)} ({both / len(rows):.0%})")
    print("the four pinned axes are zero in the spawn and non-zero in play on "
          f"{sum(1 for r in rows if max(abs(x) for x in r['joint_vel']) > 0.5) / len(rows):.0%} "
          "of swings (max |joint vel| > 0.5 rad/s)\n")


if __name__ == "__main__":
    main()
