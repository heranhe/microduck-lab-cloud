"""THE KICK CHOICE DATASET, and the model fitted on it (roadmap E.2).

E.2 asks for the first RL-shaped DECISION in this brain, bounded and
measurable. The decision is A.3's: at every line-up `Chase._plan` lays a fan
of candidate kick lines (both feet, inside the aim window, plus the push when
it is on) and `brain/kickselect.py` ranks them by rolling each one out ~30
times under the measured kick model. That ranking is a model of the BALL. It
does not see the body that has to walk round to the spot, and it is
calibrated on a kick that connects — while half the gym's swings do not.

This script is the instrument and the fit:

    collect   drive the SAME single-duck kick gym `scripts/kick_gym.py` runs,
              record at every line-up the whole safe candidate list (line,
              foot, and the roll-out's own p_goal / p_own / value / ...) plus
              the context (the ball in the pitch and in the body frame, the
              distance to the boards, the body's pose, the opponents), pick
              ONE of them, and record what the swing then did — travel,
              advance over the carry window, the exit line, whiff, fall.
    fit       ridge (and an MLP) from those rows to the realised advance and
              the backward line, written as a weights file that
              `ChaseParams.kick_select_learned` loads.
    report    the columns the comparison is read on, including the backward
              LINE share (the ball's WORLD direction 0.5 s after the touch),
              which `compare_gym.py` does not print.

EXPLORATION IS THE POINT OF `--explore`. A regressor fitted only on the
candidates the shipped selector picked would be extrapolating on every
candidate it did not — which is every candidate the learned scorer would
ever prefer. So a share of episodes pick a RANDOM safe candidate instead,
held fixed for the whole line-up (a choice that jitters tick to tick is a
spot the duck can never reach, and would measure the jitter). The row says
which mode produced it.

    uv run python scripts/kick_choice_data.py collect --seeds 12 --episodes 40 \
        --jobs 3 --explore 0.6 --out runs/kickchoice/data-b0.jsonl
    uv run python scripts/kick_choice_data.py fit runs/kickchoice/data-b0.jsonl \
        --out runs/kickchoice/model.json
    uv run python scripts/kick_choice_data.py report runs/kickchoice/gym-shipped.jsonl \
        runs/kickchoice/gym-learned.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import kick_gym as G  # noqa: E402


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Recorder:
    """Installed on the brain as `_kick_choice`, so it sees exactly what the
    shipped selector saw — the SAFE candidate list, after the own-goal veto —
    and returns the pick the episode is exploring. Nothing else about the tick
    changes: `kickselect.select` has already done its roll-outs by the time
    this is called."""

    def __init__(self) -> None:
        self.explore: tuple[float, str] | None = None   # (quantile, preferred foot) for this episode
        self.last: dict | None = None                   # the latched line-up, replaced every tick

    def bind(self, odom, los: float, goal):
        def choose(ball, pitch, safe):
            from microduck_local.brain.kickselect import _best  # noqa: PLC0415
            roll = _best(safe, safe[0].n, pitch)
            if self.explore is None:
                pick = roll
            else:
                q, foot = self.explore
                pool = [v for v in safe if v.foot == foot] or list(safe)
                pool = sorted(pool, key=lambda v: (v.heading, v.foot))
                pick = pool[min(len(pool) - 1, int(q * len(pool)))]
            self.last = {
                "cands": [[round(v.heading, 5), v.foot, round(v.p_goal, 4), round(v.p_own, 4),
                           round(v.p_block, 4), round(v.p_pass, 4), round(v.value, 5)] for v in safe],
                "pick": next(i for i, v in enumerate(safe) if v is pick),
                "roll_pick": next(i for i, v in enumerate(safe) if v is roll),
                "los": round(float(los), 5), "odom": [round(float(c), 4) for c in odom],
                "ball": [round(float(ball[0]), 4), round(float(ball[1]), 4)],
                "goal": [round(float(goal[0]), 4), round(float(goal[1]), 4)],
                "pitch": [pitch.half_x, pitch.half_y, pitch.goal_w, pitch.attack_sign],
                "explored": self.explore is not None,
            }
            return pick
        return choose


def collect(seed: int, episodes: int, spread: float, explore: float, knobs: str = "") -> list[dict]:
    """One seed of line-ups, in `kick_gym`'s own gym and with its own
    placement, timing and outcome definitions (imported, never re-typed, so
    the dataset and the A/B it is judged by score the same event)."""
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    from microduck_local.brain import REGISTRY
    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
    from microduck_local.world.arena import World
    sc = G.gym_scenario()
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    bk = __import__("microduck_local.brain.team", fromlist=["brain_kwargs"]).brain_kwargs
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **bk(x, w, teams)) for x in sc.ducks}
    brain = brains["d0"]
    d = w.ducks["d0"]
    rec = Recorder()
    brain._kick_choice = rec
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        q, v = G._place(w, rng, spread)
        for b in brains.values():
            b.reset()
        brain._kick_choice = rec                      # reset() does not clear it, but say so out loud
        # `--explore 0` must draw NOTHING, so the episode stream is bit-for-bit
        # `kick_gym`'s own and the positive control below is a real control:
        # with no exploration this script's pick IS `kickselect._best`, which
        # is what `select` returns on the gym's shipped knobs (no push, so no
        # `shoot` branch), and the two must produce identical episodes.
        rec.explore = None
        if explore > 0.0 and rng.random() < explore:
            rec.explore = (float(rng.random()), "kick_left" if rng.random() < 0.5 else "kick_right")
        rec.last = None
        t0 = w.t
        latched = None
        prev_skill = None
        pushes0 = brain.pushes
        swing = None
        while w.t - t0 < G.EPISODE_S:
            G._drive(w, brains)
            if rec.last is not None:
                latched = rec.last                    # the last line-up before the skill takes the body
            w.step()
            pushed = brain.pushes > pushes0
            if pushed or (d.skill is not None and prev_skill is None and str(d.skill).startswith("kick")):
                yaw = d.yaw(w.data)
                pos = d.trunk_pos(w.data)
                bx, by = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
                dx, dy = bx - float(pos[0]), by - float(pos[1])
                swing = {"foot": "push" if pushed else str(d.skill), "swing_yaw": round(float(yaw), 4),
                         "ball0": (bx, by), "falls_before": int(d.falls),
                         "ahead": round(dx * math.cos(yaw) + dy * math.sin(yaw), 4),
                         "side": round(-dx * math.sin(yaw) + dy * math.cos(yaw), 4),
                         "t": round(w.t - t0, 2)}
                prev_skill = d.skill
                break
            prev_skill = d.skill
        if swing is None or latched is None:
            rows.append({"seed": seed, "ep": ep, "swing": False,
                         **({} if latched is None else latched)})
            continue
        ts = w.t
        b_exit = None
        while w.t - ts < G.SETTLE_S:
            G._drive(w, brains)
            w.step()
            if b_exit is None and w.t - ts >= G.EXIT_S:
                b_exit = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
        bx1, by1 = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
        travel = math.dist(swing["ball0"], (bx1, by1))
        advance = bx1 - swing["ball0"][0]             # +x is the attacked mouth in this gym
        e_play = None if b_exit is None else G.exit_angle(swing["ball0"], b_exit, swing["swing_yaw"])
        rows.append({"seed": seed, "ep": ep, "swing": True, **latched, **swing,
                     "travel": round(travel, 4), "advance": round(advance, 4),
                     "whiff": travel < G.WHIFF_M, "back": bool(advance < 0.0),
                     "fell": int(d.falls) > swing["falls_before"],
                     "exit_play": None if e_play is None else round(e_play, 4),
                     "exit_world": None if e_play is None else round(_wrap(e_play + swing["swing_yaw"]), 4),
                     "ball1": [round(bx1, 4), round(by1, 4)]})
        rows[-1].pop("ball0", None)
    return rows


def _collect(a):
    return collect(*a)


# --- the fit -----------------------------------------------------------------

def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """The PICKED candidate's features and what the swing then did. One row
    per swing: the label belongs to the candidate that was executed, and
    (because `--explore` picked it at random) it is an unbiased sample of
    that candidate's outcome rather than of the selector's taste."""
    from microduck_local.brain.kickchoice import features
    from microduck_local.brain.kickselect import Pitch, Verdict
    X, Y, keep = [], [], []
    for r in rows:
        if not r.get("swing") or not r.get("cands"):
            continue
        c = r["cands"][int(r["pick"])]
        v = Verdict(float(c[0]), str(c[1]), float(c[2]), float(c[3]), float(c[6]), 30,
                    float(c[5]), float(c[4]))
        pitch = Pitch(*[float(x) for x in r["pitch"]])
        X.append(features(v, r["ball"], pitch, float(r["los"]), r["goal"], r["odom"]))
        back_line = (r.get("exit_world") is not None
                     and abs(_wrap(float(r["exit_world"]) - (0.0 if pitch.attack_sign >= 0 else math.pi)))
                     > math.pi / 2)
        Y.append([float(r["advance"]), float(back_line)])
        keep.append(r)
    return np.array(X, float), np.array(Y, float), keep


def r2(y, yh) -> float:
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ss = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss <= 0 else 1.0 - float(((y - yh) ** 2).sum()) / ss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--seeds", type=int, default=12)
    c.add_argument("--seed0", type=int, default=0)
    c.add_argument("--episodes", type=int, default=40)
    c.add_argument("--spread", type=float, default=0.8)
    c.add_argument("--explore", type=float, default=0.6,
                   help="share of episodes that pick a RANDOM safe candidate instead of the roll-out's")
    c.add_argument("--arm", default="", help="a MICRODUCK_CHASE string applied in the worker")
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--out", required=True)

    f = sub.add_parser("fit")
    f.add_argument("data", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--alpha", type=float, default=1.0)
    f.add_argument("--kind", default="auto", choices=("auto", "ridge", "mlp"))
    f.add_argument("--lam-back", type=float, default=None,
                   help="the backward-line penalty; default picks it on a held-out split of THIS data")
    f.add_argument("--holdout", type=float, default=0.25)

    rp = sub.add_parser("report")
    rp.add_argument("files", nargs="+", metavar="LABEL=PATH_OR_PATH")

    a = ap.parse_args()

    if a.cmd == "collect":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        args = [(s, a.episodes, a.spread, a.explore, a.arm) for s in range(a.seed0, a.seed0 + a.seeds)]
        rows: list[dict] = []
        if a.jobs > 1 and len(args) > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                for r in ex.map(_collect, args):
                    rows += r
        else:
            for x in args:
                rows += collect(*x)
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        sw = [r for r in rows if r.get("swing")]
        cands = [len(r["cands"]) for r in rows if r.get("cands")]
        print(f"{len(rows)} line-ups, {len(sw)} swings, "
              f"{sum(r.get('explored', False) for r in sw)} of them exploring; "
              f"median candidate set {int(np.median(cands)) if cands else 0}; "
              f"whiff {100 * sum(r['whiff'] for r in sw) / max(len(sw), 1):.0f}%; -> {a.out}")
        return

    if a.cmd == "fit":
        from microduck_local.brain.kickchoice import FEATURES, ChoiceModel, fit_mlp, fit_ridge
        rows = [json.loads(ln) for p in a.data for ln in open(p) if ln.strip()]
        X, Y, keep = design(rows)
        print(f"{len(rows)} rows, {len(X)} labelled swings, {X.shape[1]} features")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        nte = int(a.holdout * len(X))
        te, tr = idx[:nte], idx[nte:]
        fits = {}
        mean, scale, P = fit_ridge(X[tr], Y[tr], a.alpha)
        fits["ridge"] = ChoiceModel("ridge", mean, scale, P)
        mean, scale, P = fit_mlp(X[tr], Y[tr])
        fits["mlp"] = ChoiceModel("mlp", mean, scale, P)
        best, best_r2 = None, -1e9
        for kind, m in fits.items():
            h = m.heads(X[te])
            ra, rb = r2(Y[te, 0], h[:, 0]), r2(Y[te, 1], h[:, 1])
            print(f"  {kind:<6} held-out R2  advance {ra:+.3f}   backward-line {rb:+.3f}")
            if (a.kind in ("auto", kind)) and ra > best_r2:
                best, best_r2 = kind, ra
        m = fits[best]
        lam = a.lam_back
        if lam is None:
            # The penalty is chosen on the same held-out split, by the only
            # thing it is for: the realised advance of the candidate the
            # score would have picked among that line-up's whole safe set is
            # not observable, so rank the HELD-OUT swings by score and take
            # the lam whose top half has the best mean advance and no more
            # backward lines. A crude 1-D sweep, and it is written into the
            # weights file so the arm is one decision.
            grid, bestv = [0.0, 0.1, 0.2, 0.5, 1.0], None
            for g in grid:
                m.lam_back = g
                s = m.score(X[te])
                top = np.argsort(-s)[: max(1, len(s) // 2)]
                v = (float(Y[te][top, 0].mean()), -float(Y[te][top, 1].mean()))
                if bestv is None or v > bestv[0]:
                    bestv = (v, g)
                print(f"  lam_back {g:<4} top-half advance {v[0]:+.3f} m, backward-line {-v[1]:.2f}")
            lam = bestv[1]
        m.lam_back = float(lam)
        m.meta = {"rows": len(X), "kind": best, "lam_back": float(lam), "alpha": a.alpha,
                  "features": list(FEATURES), "data": [str(p) for p in a.data],
                  "seeds": sorted({int(r["seed"]) for r in keep})}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        m.save(a.out)
        print(f"chose {best}, lam_back {lam} -> {a.out}")
        return

    # report
    arms = {}
    for spec in a.files:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(label).stem, label
        arms[label] = [json.loads(ln) for ln in open(path) if ln.strip()]
    print(f"\n{'arm':<16}{'swings':>8}{'whiff':>8}{'conn':>7}{'advance':>10}{'back':>7}"
          f"{'backline':>10}{'fell':>7}{'carry':>8}")
    for lab, rows in arms.items():
        sw = [r for r in rows if r.get("swing")]
        if not sw:
            continue
        conn = [r for r in sw if not r.get("whiff")]
        bl = [r for r in sw if r.get("exit_play") is not None and r.get("swing_yaw") is not None]
        nb = sum(abs(_wrap(float(r["exit_play"]) + float(r["swing_yaw"]))) > math.pi / 2 for r in bl)
        print(f"{lab:<16}{len(sw):>8}{100 * sum(bool(r['whiff']) for r in sw) / len(sw):>7.0f}%"
              f"{len(conn):>7}{np.mean([r['advance'] for r in sw]):>+10.3f}"
              f"{100 * sum(bool(r.get('back')) for r in sw) / len(sw):>6.0f}%"
              f"{(100 * nb / len(bl) if bl else float('nan')):>9.1f}%"
              f"{100 * sum(bool(r.get('fell')) for r in sw) / len(sw):>6.1f}%"
              f"{np.mean([r['travel'] for r in conn]) if conn else float('nan'):>8.2f}")
    labels = list(arms)
    if len(labels) < 2:
        return
    base = labels[0]

    def col(rows, key):
        return [float(r[key]) for r in rows if r.get("swing") and r.get(key) is not None]

    def prop(rows, pred):
        rs = [r for r in rows if r.get("swing")]
        return sum(bool(pred(r)) for r in rs), len(rs)

    def backline(rows):
        rs = [r for r in rows if r.get("swing") and r.get("exit_play") is not None]
        return sum(abs(_wrap(float(r["exit_play"]) + float(r["swing_yaw"]))) > math.pi / 2 for r in rs), len(rs)

    print(f"\n{'arm':<16}{'metric':<12}{'base':>9}{'arm':>9}{'shift':>9}{'±MDE':>8}{'p':>8}  verdict")
    for lab in labels[1:]:
        for name, fn in (("whiff", lambda rs: prop(rs, lambda r: r.get("whiff"))),
                         ("back", lambda rs: prop(rs, lambda r: r.get("back"))),
                         ("backline", backline),
                         ("fell", lambda rs: prop(rs, lambda r: r.get("fell")))):
            xb, nbse = fn(arms[base])
            xa, na = fn(arms[lab])
            if not (nbse and na):
                continue
            d_, p_, mde = G.two_proportions(xb, nbse, xa, na)
            print(f"{lab:<16}{name:<12}{100 * xb / nbse:>8.1f}%{100 * xa / na:>8.1f}%{100 * d_:>+8.1f}%"
                  f"{100 * mde:>7.1f}%{p_:>8.3f}  {G.verdict_prop(p_, mde)}")
        for name in ("advance", "travel"):
            b_, a_ = col(arms[base], name), col(arms[lab], name)
            if not (b_ and a_):
                continue
            se = math.sqrt(np.var(b_, ddof=1) / len(b_) + np.var(a_, ddof=1) / len(a_))
            d_ = float(np.mean(a_) - np.mean(b_))
            p_ = math.erfc(abs(d_ / se) / math.sqrt(2)) if se > 0 else 1.0
            print(f"{lab:<16}{name:<12}{np.mean(b_):>+9.3f}{np.mean(a_):>+9.3f}{d_:>+9.3f}"
                  f"{1.96 * se:>8.3f}{p_:>8.3f}  {'effect' if p_ < 0.05 else 'null/NO RESULT'}")
    print("\nFALLS ARE THE VETO (`fell` = the duck went down inside the carry window)."
          "\n`backline` is the ball's WORLD direction 0.5 s after the touch pointing away from"
          "\nthe attacked mouth — quote it beside `back`, never `back` alone (12at).")


if __name__ == "__main__":
    main()
