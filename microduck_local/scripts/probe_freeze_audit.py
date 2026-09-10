"""Every way a duck can stop moving, and which watchdog catches it — the audit.

The corner trap (roadmap 12aa) was one freeze with no watchdog. This asks
whether it was the ONLY one, by counting the freezes the brain actually
produces and sorting them by which escape can reach them. Two exist:

    A  `stuck_s`      states avoid / blocked / yield, no translation (< 0.05 m)
                      AND no rotation (< 0.3 rad) over `stuck_s`
    B  `support_unstick_s`   states support / wait, under `support_unstick_move`
                      of travel across `support_unstick_s`, and NOT on its post

So A cannot see a duck that is TURNING (the corner trap turned), and B cannot
see one away from the boards. A freeze is counted here as `FREEZE_M` of net
displacement or less over `FREEZE_S`, whatever the duck's heading did — which
is the quantity that matters, because a duck spinning on the spot has stopped
playing exactly as much as one standing still.

Each freeze is attributed to the FIRST watchdog whose gate it satisfies, and
`uncovered` is the answer: freezes neither can reach. The states are reported
because `kick` / `look` / `settle` / `push` stand still BY DESIGN and are
bounded by their own clocks — a freeze in those is not a bug and is counted
apart.

    uv run python scripts/probe_freeze_audit.py --seeds 8 --seconds 180
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains, throw_in_brains
from microduck_local.world import World, make_pitch

# 15 s in which the trunk moved under 15 cm. A SHORTER bar (6 s / 0.10 m) is
# not a freeze — it catches ordinary transient blocking, and a first run of
# this probe at that bar reported 10 'freezes' a minute, most of them a duck
# in `retreat` or `lineup` that was shoving past something and got there.
# A duck that has stopped PLAYING is the thing to count.
FREEZE_S, FREEZE_M = 15.0, 0.15
BOUNDED = ("kick", "look", "settle", "push")   # states that stand by design, on their own clocks
A_STATES = ("avoid", "blocked", "yield")
B_STATES = ("support", "wait")


def run(seed: float, seconds: float, per_side: int) -> dict:
    sc = make_pitch(per_side=per_side, formation=True)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    getup = onnx_infer(POLICIES_DIR / "alpha_stand.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed,
              ball_out_s=5.0, getup_s=5.0, getup_infer=getup)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    hist: dict = {d: [] for d in brains}
    open_f: dict = {d: None for d in brains}
    t: dict = defaultdict(float)
    goal_seq, out_seq = 0, w.ball_out_seq
    while w.t < seconds:
        for did, b in brains.items():
            d = w.ducks[did]
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            it = b.step(s)
            w.apply_intent(d, it)
            if d.skill is None:
                d.set_cmd(w.data, it.twist, it.head)
            pos = d.trunk_pos(w.data)
            x, y = float(pos[0]), float(pos[1])
            h = hist[did]
            h.append((w.t, x, y, b.state, d.yaw(w.data)))
            while h and w.t - h[0][0] > FREEZE_S:
                h.pop(0)
            t["ticks"] += 1
            if len(h) < 2 or w.t - h[0][0] < FREEZE_S - 0.05:
                continue
            moved = max(math.hypot(px - x, py - y) for _, px, py, _, _ in h)
            if moved > FREEZE_M:
                open_f[did] = None
                continue
            near = min(hx - abs(x), hy - abs(y))
            st = b.state
            if open_f[did] is None:                 # a NEW freeze: classify it once
                # by the state held for MOST of the window, not the state at the
                # instant it was noticed — a freeze that ends in `retreat` for one
                # tick was not a retreat freeze.
                counts: dict = {}
                for _, _, _, stt, _ in h:
                    counts[stt] = counts.get(stt, 0) + 1
                st = max(counts, key=counts.get)
                open_f[did] = st
                # A SUPPORTER STANDING AT ITS POST IS CORRECT PLAY, not a
                # freeze: `_support` servos to the post and stops within 0.12 m
                # of it, then faces the ball. Counting those as stuck would
                # report the brain working as a bug. Three cases, and only the
                # last two are pathological:
                #   at post   `post` set and the duck is within the servo's stop
                #   stranded  `post` set and the duck is NOT there — it wants to
                #             move and something is stopping it
                #   blind     `post` is None — nobody knows where the ball is,
                #             which is `_support`'s no-ball branch: the corner
                #             trap's own state, here in open play
                if st in B_STATES:
                    post = b.post
                    if post is None:
                        t["sup/blind"] += 1
                    elif math.hypot(post[0] - x, post[1] - y) <= 0.12:
                        t["sup/at_post"] += 1
                        t["legit"] += 1
                        continue        # never reaches the coverage buckets below
                    else:
                        t["sup/stranded"] += 1
                t["freezes"] += 1
                t[f"state/{st}"] += 1
                # A's OTHER gate, and the one the corner trap failed: it wants a
                # still HEAD as well as a still body, so a duck spinning on the
                # spot is invisible to it however long it stands there.
                yaw_still = all(abs(_wrap(yaw - h[-1][4])) < 0.3 for *_, yaw in h)
                if st in BOUNDED:
                    t["bounded"] += 1
                elif st in A_STATES and yaw_still:
                    t["A"] += 1
                elif st in B_STATES:      # the motion gate: this probe's own freeze test
                    t["B"] += 1
                else:
                    t["uncovered"] += 1
                    t[f"uncovered/{st}"] += 1
                    t[f"uncovered_near/{'board' if near < 0.30 else 'open'}"] += 1
                    if st in A_STATES and not yaw_still:
                        t["uncovered_spinning"] += 1
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
        if w.ball_out_seq != out_seq:
            out_seq = w.ball_out_seq
            throw_in_brains(brains, teams)
    t["duck_seconds"] = len(brains) * w.t
    return t


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=2)
    a = ap.parse_args()
    tot: dict = defaultdict(float)
    for s in range(a.seed0, a.seed0 + a.seeds):
        for k, v in run(s, a.seconds, a.per_side).items():
            tot[k] += v
        print(f"  seed {s} done ({tot.get('freezes', 0):.0f} freezes)", flush=True)
    # `freezes` is incremented AFTER the at-post `continue`, so it ALREADY
    # excludes the legitimate holds — subtracting them again here double-counted
    # and made the buckets sum past 100%.
    n = tot.get("freezes", 0)
    ds = tot.get("duck_seconds", 1)
    print(f"\n{ds:.0f} duck-seconds; {n + tot.get('legit', 0):.0f} stands of "
          f"{FREEZE_S:.0f} s with {FREEZE_M} m or less of travel, of which "
          f"{tot.get('legit', 0):.0f} are a supporter AT its post (correct play).")
    print(f"{n:.0f} REAL freezes remain — {n / ds * 3600:.1f} an hour of duck time:")
    if not n:
        print("!! none at all — that is a broken measurement, not a clean brain")
        return
    if n <= 0:
        print("  none: every stand found was a duck legitimately holding a post.")
        return
    for k, label in (("bounded", "on a clock of their own (kick/look/settle/push)"),
                     ("A", "reachable by `stuck_s`"),
                     ("B", "reachable by `support_unstick_s`"),
                     ("uncovered", "NEITHER — no watchdog can reach these")):
        print(f"  {label:<48}{tot.get(k, 0):>6.0f}  ({100 * tot.get(k, 0) / n:.0f}%)")
    sp = tot.get("uncovered_spinning", 0)
    if sp:
        print(f"\n  of the uncovered, {sp:.0f} are in one of `stuck_s`'s OWN states and escape it"
              "\n  only by TURNING — the corner trap's exact signature, in another place.")
    unc = [(k[10:], v) for k, v in tot.items() if k.startswith("uncovered/")]
    if unc:
        print("\n  the uncovered ones, by state: "
              + ", ".join(f"{k} {v:.0f}" for k, v in sorted(unc, key=lambda kv: -kv[1])))
        print(f"  …and by place: at a board {tot.get('uncovered_near/board', 0):.0f}, "
              f"in open play {tot.get('uncovered_near/open', 0):.0f}")
    print(f"\n  supporters: at post {tot.get('sup/at_post', 0):.0f} (legit), "
          f"stranded off it {tot.get('sup/stranded', 0):.0f}, "
          f"blind — no ball belief at all {tot.get('sup/blind', 0):.0f}")
    st = [(k[6:], v) for k, v in tot.items() if k.startswith("state/")]
    print("\n  ALL freezes by state: "
          + ", ".join(f"{k} {v:.0f}" for k, v in sorted(st, key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
