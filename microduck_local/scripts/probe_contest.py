"""How often can the duel rule act AT ALL? — the population behind three nulls.

`lineup_keepout`, `opp_keepout` and `contest_margin` all measured null, and
this repo wrote that up as "the duel does not matter". microduck-62's camera
work (roadmap 12n) suggests a different reading: those rules only fire when
two ducks contest a ball at least one of them can SEE, and on the sim's camera
nobody on a team sees the ball 61.3% of the time — on the crop the robot
actually runs, 85.2%. A rule that rarely gets the chance to act measures null
for a reason that has nothing to do with whether it works.

That is playbook rule 5 — measure the opportunity rate BEFORE spending a
battery — which was never applied to the duel. This applies it:

    uv run python scripts/probe_contest.py --seeds 4 --per-side 3

It runs the contest arm with its gate set (`use_color=1`, without which the
rule is inert — see `brain/knob_gates.py`) and counts, per duck-tick, how far
down the chain of preconditions the duck actually gets. `Chase.contesting` is
read off the brain itself, so this is the rule firing, not a reconstruction.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# The arm must be set BEFORE any brain is built: `brain_kwargs` reads
# `ChaseParams.from_env()` at construction (playbook rule 0).
os.environ.setdefault("MICRODUCK_CHASE", "use_color=1,contest_margin=0.15")

from microduck_local.brain import REGISTRY, Senses  # noqa: E402
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.brain.team import brain_kwargs, kickoff_brains  # noqa: E402
from microduck_local.world import World, make_pitch  # noqa: E402


def run(seed: int, seconds: float, per_side: int) -> list[dict]:
    sc = make_pitch(per_side=per_side, formation=True)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=5.0)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    team_of = {d.id: d.team for d in sc.ducks}
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    rows: list[dict] = []
    goal_seq = 0
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

            ball = next((t for t in b.tracker.tracks
                         if t.cls == "ball" and t.age(w.t) <= b.p.lost_s), None)
            ducks = [t for t in b.tracker.tracks
                     if t.cls == "duck" and t.age(w.t) <= b.p.lost_s]
            opp = [t for t in ducks if team_of.get(t.name) not in (None, team_of[did])]
            near = min((t.range for t in opp), default=float("inf"))
            rows.append({
                "sees_ball": ball is not None,
                "sees_opp": bool(opp),
                "both": ball is not None and bool(opp),
                "opp_close": ball is not None and near <= b.p.duck_touch,
                "contesting": bool(getattr(b, "contesting", False)),
            })
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
    return rows


def report(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print("no ticks")
        return
    print(f"\n{n} duck-ticks\n")
    print(f"{'the chain of preconditions':<44}{'ticks':>9}{'share':>9}")
    for key, label in (("sees_ball", "the duck can see the ball"),
                       ("sees_opp", "…and/or a live opponent track"),
                       ("both", "sees the ball AND an opponent"),
                       ("opp_close", "…and that opponent is within duck_touch"),
                       ("contesting", "THE RULE FIRES (Chase.contesting)")):
        c = sum(r[key] for r in rows)
        print(f"{label:<44}{c:>9}{100 * c / n:>8.2f}%")
    fires = sum(r["contesting"] for r in rows)
    if not fires:
        print("\n!! THE RULE NEVER FIRED. That is a BROKEN MEASUREMENT, not a result of"
              "\n   0%. Before believing it, check the thing being measured could have"
              "\n   moved at all: is the arm's gate set (`use_color=1` — see"
              "\n   brain/knob_gates.py), is MICRODUCK_CHASE set BEFORE the brain imports"
              "\n   (brain_kwargs reads ChaseParams.from_env() at construction), and are"
              "\n   the ducks it needs actually on the pitch (--per-side >= 2)?"
              "\n   A zero from a probe that could not have seen a non-zero reads exactly"
              "\n   like a profound finding, which is how it gets published.")
        return
    print(f"\nThe contest rule acts on {100 * fires / n:.2f}% of duck-ticks."
          "\nA whole-match average cannot move on a population this size whatever the"
          "\nrule does when it fires — which is what a null on ballAdvance was always"
          "\ngoing to say. Read it against the effect the battery could resolve at all"
          "\n(scripts/audit_power.py: kicks 28% of baseline, advance 19%).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--per-side", type=int, default=3)
    a = ap.parse_args()
    rows: list[dict] = []
    for s in range(a.seeds):
        rows += run(s, a.seconds, a.per_side)
        print(f"  seed {s} done ({len(rows)} ticks)")
    report(rows)


if __name__ == "__main__":
    main()
