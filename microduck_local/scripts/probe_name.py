"""Does the SOCCER brain depend on `Detection.name` carrying the sim id?

The roadmap (item 5) says dropping it "makes every duck track worse and
moves every soccer number", because "the Tracker associates detections to
tracks by name". Reading `Tracker._associate`, it matches on class, a
bearing gate and a range gate — the name is only voted onto the track and
never gated on, and `Chase` reads `color` but never `name`. That is a claim
about the code, so this measures it instead: run the SAME seeds with every
detection's name blanked, and diff the track payloads and the play ledger.

No src edit — the name is stripped between the detector and the brain.
"""
from __future__ import annotations

import argparse

import numpy as np

from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.brain.team import brain_kwargs, kickoff_brains
from microduck_local.sensors.detector import DetectionFrame
from microduck_local.world import World, make_pitch


def blank(det):
    """The same frame with every detection's sim id removed."""
    if det is None:
        return None
    out = []
    for d in det.detections:
        out.append(type(d)(**{**d.__dict__, "name": ""}))
    fr = DetectionFrame(det.t, out)
    for k, v in det.__dict__.items():
        if k not in ("t", "detections"):
            setattr(fr, k, v)
    return fr


def run(seed: float, seconds: float, per_side: int, strip: bool) -> dict:
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    goal_seq = 0
    ntracks, named, kicks = [], 0, 0
    sig = []
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            if strip:
                det = blank(det)
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            b = brains[d.id]
            it = b.step(s)
            if it.skill in ("kick_left", "kick_right"):
                kicks += 1
            w.apply_intent(d, it)
            if d.skill is None:
                d.set_cmd(w.data, it.twist, it.head)
        w.step()
        if w.tick % 25 == 0:
            for b in brains.values():
                trs = [t for t in b.tracker.tracks if t.cls == "duck"]
                ntracks.append(len(trs))
                named += sum(1 for t in trs if t.name)
                sig.append(round(sum(t.bearing for t in trs), 4))
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)          # with the world: the kickoff rule the benchmark plays under
    m = w.metrics.row() if hasattr(w, "metrics") else {}
    return {"duckTracks": float(np.mean(ntracks)), "namedTracks": named,
            "kicks": kicks, "sig": sig,
            "falls": sum(d.falls for d in w.ducks.values()),
            "possession": float(np.mean(list(m.get("possession", {0: 0}).values()))) if m else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--per-side", type=int, default=2)
    args = ap.parse_args()
    ident = 0
    print(f"{'seed':<6}{'duck tracks kept':>18}{'named':>16}{'kicks':>14}{'falls':>12}")
    for s in range(args.seeds):
        a = run(s, args.seconds, args.per_side, False)
        b = run(s, args.seconds, args.per_side, True)
        ident += (a["sig"] == b["sig"])
        print(f"{s:<6}{a['duckTracks']:>8.2f} -> {b['duckTracks']:<8.2f}"
              f"{a['namedTracks']:>8} -> {b['namedTracks']:<7}"
              f"{a['kicks']:>7} -> {b['kicks']:<6}"
              f"{a['falls']:>6} -> {b['falls']:<5}")
    print(f"\nRuns bit-for-bit identical on {ident} of {args.seeds} seeds "
          f"(track-bearing signature over the whole run).")
    print("If that is ALL of them, the soccer brain does not read the sim id and")
    print("dropping it is a detector/tidy change, not a soccer one.")


if __name__ == "__main__":
    main()
