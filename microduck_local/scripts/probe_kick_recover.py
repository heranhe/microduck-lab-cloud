"""After a WHIFF, does the duck ever look at the ball again? (roadmap 12i, re-opened)

    cd microduck_local
    uv run python scripts/probe_kick_recover.py --episodes 40 --seeds 4 --jobs 4
    uv run python scripts/probe_kick_recover.py --arm "shipped=" --arm "dip=look_sweep=0"

Item 12i shipped a post-kick look that sweeps the head across the predicted
exit line at `look_sweep_range` 1.0 m — a CONNECTED kick's range. It was
measured on connected kicks only ("connected kicks seen again within 2 s
37 -> 46%"). A WHIFFED ball never left: it is 0.1-0.3 m from the trunk, which
is what the older `look_range` 0.3 m was for, and `look_sweep > 0` overrides
that range unconditionally. If the look then sees nothing, `hunt` walks 3 s
along the line the ball never took.

This measures the half that was never measured. Every episode is kick_gym's:
one duck, one ball, the real chase brain, one swing. After the swing the world
keeps running for `--recover` seconds and the probe records, per kick:

  * `whiff`      — the ball moved < 10 cm in CARRY_S (kick_gym's rule)
  * `reacq`      — seconds from the swing to the next FRESH ball track
                   (>= 0.5 s after, so the pre-swing track does not count it)
  * `d0`/`d_end` — trunk-to-ball distance at the swing and at the end of the
                   window: does the duck walk AWAY from a ball it missed?
  * `states`     — seconds spent in each brain state during the window
  * `rekick`     — seconds to the next swing, if there was one
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kick_gym import EPISODE_S, WHIFF_M, _drive, _place, gym_scenario  # noqa: E402

from microduck_local import contract as C  # noqa: E402
from microduck_local.brain import REGISTRY  # noqa: E402
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: E402
from microduck_local.world.arena import World  # noqa: E402
from microduck_local.world.metrics import CARRY_S  # noqa: E402


def _wrapdeg(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


REACQ_MIN_S = 0.5      # probe_kick_line's rule: a sighting sooner is the pre-swing track
FRESH_S = 0.2          # a track this old or younger is "seeing it now"


def run(seed: int, episodes: int, spread: float, recover: float, knobs: str = "") -> list[dict]:
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    sc = gym_scenario()
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, teams)) for x in sc.ducks}
    brain = brains["d0"]
    d = w.ducks["d0"]
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        q, v = _place(w, rng, spread)
        for b in brains.values():
            b.reset()
        t0 = w.t
        swing = None
        prev_skill = None
        while w.t - t0 < EPISODE_S:
            _drive(w, brains)
            w.step()
            if d.skill is not None and prev_skill is None and str(d.skill).startswith("kick"):
                p = d.trunk_pos(w.data)
                bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
                yaw0 = d.yaw(w.data)
                swing = {"ep": ep, "seed": seed, "foot": str(d.skill), "arm": knobs,
                         "ball0": (bx, by),
                         "d0": round(math.dist((float(p[0]), float(p[1])), (bx, by)), 4),
                         # the same frame the look frames are measured in, at the
                         # swing: a sanity anchor, because the kick only fires
                         # with the ball AHEAD (`kick_ahead` 0.08 m).
                         "bear0": round(math.degrees(math.atan2(
                             -(bx - float(p[0])) * math.sin(yaw0) + (by - float(p[1])) * math.cos(yaw0),
                             (bx - float(p[0])) * math.cos(yaw0) + (by - float(p[1])) * math.sin(yaw0))), 1),
                         "_yaw0": yaw0, "_xy0": (float(p[0]), float(p[1]))}
                break
            prev_skill = d.skill
        if swing is None:
            continue
        # --- the window this probe exists for ---
        ts = w.t
        states: dict[str, float] = {}
        reacq = None
        rekick = None
        prev = d.skill
        ball_at_carry = None
        d_min = swing["d0"]
        # What the CAMERA did during the look, per detector frame: the
        # depression the head actually reached (`DetectionFrame.cam_pitch`, the
        # achieved angle, not the command) and whether the ball was in it. A
        # gaze that is commanded and a gaze that arrives inside `look_s` are
        # not the same thing, and only the second one can confirm a whiff.
        look_frames = look_ball = 0
        look_pitch = 0.0
        seen_frame_t = None
        look_rng: list[float] = []      # where the ball WAS on each of those frames,
        look_bear: list[float] = []     # in the duck's own yaw frame: range and |bearing|
        while w.t - ts < recover:
            _drive(w, brains)
            w.step()
            dt = C.CTRL_DT
            states[brain.state] = states.get(brain.state, 0.0) + dt
            el = w.t - ts
            if ball_at_carry is None and el >= CARRY_S:
                ball_at_carry = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
            if reacq is None and el >= REACQ_MIN_S:
                trk = brain.tracker.best(brain.p.target_cls, w.t, min_hits=1)
                if trk is not None and trk.age(w.t) <= FRESH_S:
                    reacq = round(el, 2)
            if rekick is None and d.skill is not None and prev is None and str(d.skill).startswith("kick"):
                rekick = round(el, 2)
            prev = d.skill
            fr = d.detector.last
            if fr is not None and fr.t != seen_frame_t:
                seen_frame_t = fr.t
                if brain.state == "look":
                    if look_frames == 0:
                        # WHAT THE SWING ITSELF DID TO THE BODY, read at the
                        # first frame of the look: the kick skill owns the legs
                        # for ~0.5 s and the brain is standing still by now, so
                        # any turn or step here is the KICK's, not the brain's.
                        tp0 = d.trunk_pos(w.data)
                        swing["turn_deg"] = round(math.degrees(_wrapdeg(
                            float(d.yaw(w.data)) - swing["_yaw0"])), 1)
                        swing["step_m"] = round(math.dist(
                            (float(tp0[0]), float(tp0[1])), swing["_xy0"]), 3)
                    look_frames += 1
                    look_pitch = max(look_pitch, float(fr.cam_pitch))
                    if any(x.cls == brain.p.target_cls for x in fr.detections):
                        look_ball += 1
                    tp = d.trunk_pos(w.data)
                    yaw = d.yaw(w.data)
                    ddx = float(w.data.qpos[q]) - float(tp[0])
                    ddy = float(w.data.qpos[q + 1]) - float(tp[1])
                    look_rng.append(math.hypot(ddx, ddy))
                    look_bear.append(abs(math.degrees(math.atan2(
                        -ddx * math.sin(yaw) + ddy * math.cos(yaw),
                        ddx * math.cos(yaw) + ddy * math.sin(yaw)))))
            p = d.trunk_pos(w.data)
            bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
            d_min = min(d_min, math.dist((float(p[0]), float(p[1])), (bx, by)))
        p = d.trunk_pos(w.data)
        bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        if ball_at_carry is None:
            ball_at_carry = (bx, by)
        travel = math.dist(swing["ball0"], ball_at_carry)
        swing.pop("ball0")
        swing.pop("_yaw0")
        swing.pop("_xy0")
        swing.update(travel=round(travel, 4), whiff=travel < WHIFF_M, reacq=reacq, rekick=rekick,
                     d_end=round(math.dist((float(p[0]), float(p[1])), (bx, by)), 4),
                     d_min=round(d_min, 4),
                     look_frames=look_frames, look_ball=look_ball,
                     look_pitch=round(math.degrees(look_pitch), 1),
                     look_rng=None if not look_rng else round(st.median(look_rng), 3),
                     look_bear=None if not look_bear else round(st.median(look_bear), 1),
                     look_behind=None if not look_bear else round(
                         sum(x > 58.0 for x in look_bear) / len(look_bear), 2),
                     states={k: round(x, 2) for k, x in sorted(states.items())})
        rows.append(swing)
    return rows


def _run(a):
    return run(*a)


def _col(rs, k):
    return [r[k] for r in rs if r.get(k) is not None]


def report(rows: list[dict], recover: float) -> None:
    if not rows:
        print("no swings")
        return
    wh = [r for r in rows if r["whiff"]]
    co = [r for r in rows if not r["whiff"]]
    print(f"\n{len(rows)} swings — {len(wh)} whiffed ({len(wh) / len(rows):.0%}), {len(co)} connected")
    print(f"\n{'':38}{'WHIFFED':>12}{'CONNECTED':>12}")

    def line(label, f):
        a = f(wh) if wh else "—"
        b = f(co) if co else "—"
        print(f"  {label:36}{a:>12}{b:>12}")

    def pct_seen(rs, lim):
        n = sum(1 for r in rs if r["reacq"] is not None and r["reacq"] <= lim)
        return f"{n / len(rs):.0%}" if rs else "—"

    line("ball seen again within 2 s", lambda rs: pct_seen(rs, 2.0))
    line(f"ball seen again within {recover:g} s", lambda rs: pct_seen(rs, recover))
    line("median seconds to that sighting",
         lambda rs: f"{st.median(_col(rs, 'reacq')):.2f}" if _col(rs, "reacq") else "—")
    line("duck→ball at the swing (m)", lambda rs: f"{st.mean(_col(rs, 'd0')):.2f}")
    line("…its |bearing| off the nose then (deg)",
         lambda rs: f"{st.median(abs(x) for x in _col(rs, 'bear0')):.0f}")
    line(f"duck→ball at +{recover:g} s (m)", lambda rs: f"{st.mean(_col(rs, 'd_end')):.2f}")
    line("closest it got afterwards (m)", lambda rs: f"{st.mean(_col(rs, 'd_min')):.2f}")
    line("ENDED FURTHER AWAY than at the swing",
         lambda rs: f"{sum(1 for r in rs if r['d_end'] > r['d0']) / len(rs):.0%}")
    line("swung again inside the window",
         lambda rs: f"{sum(1 for r in rs if r['rekick'] is not None) / len(rs):.0%}")
    line("the swing turned the body by (deg)",
         lambda rs: f"{st.median(abs(x) for x in _col(rs, 'turn_deg')):.0f}" if _col(rs, "turn_deg") else "—")
    line("…and carried it forward by (m)",
         lambda rs: f"{st.median(_col(rs, 'step_m')):.2f}" if _col(rs, "step_m") else "—")
    line("camera frames during the `look`", lambda rs: f"{st.mean(_col(rs, 'look_frames')):.1f}")
    line("…deepest depression it reached (deg)", lambda rs: f"{st.mean(_col(rs, 'look_pitch')):.0f}")
    line("…of those, frames with the ball in them",
         lambda rs: f"{sum(r['look_ball'] for r in rs) / max(sum(r['look_frames'] for r in rs), 1):.0%}")
    line("…the ball's range on those frames (m)", lambda rs: f"{st.median(_col(rs, 'look_rng')):.2f}"
         if _col(rs, "look_rng") else "—")
    line("…its |bearing| off the nose (deg)", lambda rs: f"{st.median(_col(rs, 'look_bear')):.0f}"
         if _col(rs, "look_bear") else "—")
    line("…share of look frames with it OUTSIDE ±58°",
         lambda rs: f"{st.mean(_col(rs, 'look_behind')):.0%}" if _col(rs, "look_behind") else "—")
    keys = sorted({k for r in rows for k in r["states"]})
    print(f"\n  seconds of the {recover:g} s window in each state")
    for k in keys:
        line(f"    {k}", lambda rs, k=k: f"{st.mean([r['states'].get(k, 0.0) for r in rs]):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40, help="episodes PER seed")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--recover", type=float, default=6.0, help="seconds watched after the swing")
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=KNOBS")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in (a.arm or ["shipped="])]
    all_rows: list[dict] = []
    for label, knobs in arms:
        jobs = [(a.seed0 + i, a.episodes, a.spread, a.recover, knobs) for i in range(a.seeds)]
        if a.jobs > 1:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                rows = [r for part in ex.map(_run, jobs) for r in part]
        else:
            rows = [r for j in jobs for r in _run(j)]
        for r in rows:
            r["label"] = label
        print(f"\n===== {label or 'shipped'}  ({knobs or 'no knobs'}) =====")
        report(rows, a.recover)
        all_rows += rows
    if a.out:
        with open(a.out, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(all_rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
