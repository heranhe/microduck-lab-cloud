"""Is the pitch INSIDE the box a gym-fitted scorer was fitted in? (roadmap E.2)

    cd microduck_local
    uv run python scripts/probe_model_box.py --model runs/kickchoice/model-b0.json \
        --data runs/kickchoice/data-b0.jsonl --seeds 3 --seconds 120

A regressor is only as good as the region it was fitted on. E.2 fitted the
learned kick ranking (`brain/kickchoice.py`) on `scripts/kick_gym.py`
line-ups, measured a +0.28 m-a-swing win that reproduced on three independent
gym blocks INCLUDING an opposed one — and then lost on a 48-seed pitch ledger
with carry per kick dead flat. The reason was not the ranking. It was the
INPUT: 47% of the candidates the chooser scores on a 2v2 pitch are outside
the training range of `range` (body-to-ball at the line-up) and 31% outside
it on `ball_board`, because the gym's walk-in placement (`_place`, a 0.45-1.4
m draw) and its 3.0 x 2.5 m boards do not cover what a pitch presents.

So this is [[check-a-knobs-reachable-set-first]] applied to a MODEL'S INPUT
BOX instead of to a knob, and it is the check to run BEFORE a gym-fitted
scorer goes on a ledger, not after: three seeds x 120 s against an hour a
block. It spies on EVERY candidate the chooser scores — not just the one it
picks — because every one of them is a row the model has to rank.

  * PASS/FAIL is one line, on a threshold (default: more than 10% of pitch
    rows outside the training range of ANY feature is a FAIL, "do not put
    this scorer on a ledger"). The exit status follows it, so a battery
    script can gate on it.
  * A feature whose TRAINING column is constant is flagged `dead`: the fit
    never saw it vary, so it carries no information whatever the pitch does
    with it (E.2: `p_block` is live only in the opposed gym and identically
    zero on a default pitch, and `p_pass` is dead in both arenas).
  * The spy DRIVES the picks, so the line-ups it measures are the ones this
    model itself produces — the deployment condition, not the shipped arm's.

The training box comes from `scripts/kick_choice_data.py`'s own `design()`,
so the matrix here is byte-for-byte the matrix the model was fitted on; a
feature that moves in `kickchoice.FEATURES` fails the model load rather than
silently shifting a column.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# More than this share of pitch rows outside the training range of any one
# feature and the scorer is extrapolating where it matters. 10% is a judgement
# call and is a flag, not a law — E.2's two failures were 47% and 31%.
OUTSIDE_PCT = 10.0


# --- the training box ---------------------------------------------------------

def training_box(paths: list[str]) -> np.ndarray:
    """The design matrix the model was fitted on, from the dataset itself."""
    import kick_choice_data as K  # noqa: PLC0415  (a sibling script, not a module)

    rows = [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines() if ln.strip()]
    X, _Y, _keep = K.design(rows)
    if not len(X):
        raise SystemExit(f"{paths}: no labelled swings in the dataset — nothing to take a box from")
    return np.asarray(X, float)


# --- the pitch's candidates ---------------------------------------------------

def _spy_class():
    """Built lazily so this module imports without a World (the tests do)."""
    from microduck_local.brain.kickchoice import Chooser, features  # noqa: PLC0415

    class Spy(Chooser):
        """The chooser, plus a record of every candidate it was asked to
        score. It still PICKS — the line-ups downstream are this model's own,
        which is the condition a ledger would run it under."""

        def __init__(self, model):
            super().__init__(model)
            self.rows: list[list[float]] = []

        def bind(self, odom, los, goal):
            inner = super().bind(odom, los, goal)

            def choose(ball, pitch, safe):
                for v in safe:
                    self.rows.append(features(v, ball, pitch, los, goal, odom))
                return inner(ball, pitch, safe)
            return choose

    return Spy


def seed_features(seed: int, model_path: str, seconds: float, per_side: int,
                  ball_out_s: float) -> np.ndarray:
    """One seed of the ledger's own pitch, returning every candidate row the
    chooser scored. The loop is `eval_pitch.run_one`'s, down to the seed's
    ball jitter — a probe that measures a different pitch from the battery is
    measuring the wrong box."""
    from microduck_local.brain import REGISTRY, Senses  # noqa: PLC0415
    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: PLC0415
    from microduck_local.brain.kickchoice import ChoiceModel  # noqa: PLC0415
    from microduck_local.brain.team import (  # noqa: PLC0415
        brain_kwargs,
        kickoff_brains,
        throw_in_brains,
    )
    from microduck_local.world import World, make_pitch  # noqa: PLC0415

    model = ChoiceModel.load(model_path)
    Spy = _spy_class()
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed, ball_out_s=ball_out_s)
    rng = np.random.default_rng(seed)
    q = int(w.model.jnt_qposadr[w._ball_joint])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    spies = {}
    for k, b in brains.items():
        spies[k] = Spy(model)
        b._kick_choice = spies[k]          # the documented probe hook (controllers.py)
    goal_seq, out_seq = w.goal_seq, w.ball_out_seq
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if w.goal_seq != goal_seq:
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams, w)
        if w.ball_out_seq != out_seq:
            out_seq = w.ball_out_seq
            throw_in_brains(brains, teams)
    rows = [r for sp in spies.values() for r in sp.rows]
    return np.array(rows, float) if rows else np.zeros((0, 0))


def _seed_features(a: tuple) -> np.ndarray:
    return seed_features(*a)


def pitch_box(model_path: str, seeds: int, seed0: int, seconds: float, per_side: int,
              ball_out_s: float, jobs: int) -> np.ndarray:
    args = [(seed0 + i, model_path, seconds, per_side, ball_out_s) for i in range(seeds)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(_seed_features, args))
    else:
        parts = [_seed_features(a) for a in args]
    parts = [p for p in parts if p.size]
    if not parts:
        raise SystemExit("the chooser was never asked to score anything — a DEAD path, not a pass "
                         "(AGENTS.md: a knob that changes nothing is broken, not null)")
    return np.vstack(parts)


# --- the reading --------------------------------------------------------------

def box_report(train: np.ndarray, pitch: np.ndarray, names: tuple[str, ...],
               threshold: float = OUTSIDE_PCT) -> tuple[list[str], bool]:
    """The per-feature table and the verdict. Pure: the tests drive it with a
    stubbed feature stream, because the live one costs three pitch seeds."""
    if train.shape[1] != len(names) or pitch.shape[1] != len(names):
        raise SystemExit(f"feature count mismatch: training {train.shape[1]}, pitch {pitch.shape[1]}, "
                         f"names {len(names)} — the model and the code disagree about the columns")
    out = [f"pitch candidate rows {len(pitch)}   training rows {len(train)}", "",
           f"{'feature':<14}{'train min':>10}{'train max':>10}{'train mean':>11}"
           f"{'pitch mean':>11}{'outside box':>12}"]
    worst, worst_f = 0.0, ""
    for i, f in enumerate(names):
        lo, hi = float(train[:, i].min()), float(train[:, i].max())
        pct = 100.0 * float(((pitch[:, i] < lo) | (pitch[:, i] > hi)).mean())
        note = "  <<<" if pct > threshold else ""
        if hi - lo < 1e-12:
            # The fit never saw this column vary: it carries no information,
            # whatever the pitch does with it.
            note += "  dead in training"
        if pct > worst:
            worst, worst_f = pct, f
        out.append(f"{f:<14}{lo:>10.3f}{hi:>10.3f}{train[:, i].mean():>11.3f}"
                   f"{pitch[:, i].mean():>11.3f}{pct:>11.1f}%{note}")
    ok = worst <= threshold
    out.append("")
    if ok:
        out.append(f"PASS: no feature has more than {threshold:.0f}% of pitch rows outside its "
                   f"training range (worst: {worst_f} at {worst:.1f}%).")
    else:
        out.append(f"FAIL: {worst_f} has {worst:.1f}% of pitch rows outside its training range "
                   f"(threshold {threshold:.0f}%). Do not put this scorer on a ledger — fit it on "
                   f"the line-ups it will be asked to rank, or widen the collector's placement.")
    return out, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the weights file (brain/kickchoice.py's format)")
    ap.add_argument("--data", nargs="+", required=True, help="the dataset it was fitted on (.jsonl)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--per-side", type=int, default=2)
    ap.add_argument("--ball-out-s", type=float, default=5.0,
                    help="the ledger's referee (eval-pitch --ball-out-s); 0 for none")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=OUTSIDE_PCT,
                    help="%% of pitch rows outside a feature's training range that FAILS "
                         "(default: %(default)s)")
    a = ap.parse_args()

    from microduck_local.brain.kickchoice import FEATURES  # noqa: PLC0415

    train = training_box(a.data)
    pitch = pitch_box(a.model, a.seeds, a.seed0, a.seconds, a.per_side, a.ball_out_s, a.jobs)
    lines, ok = box_report(train, pitch, FEATURES, a.threshold)
    print(f"model {a.model}\ndata  {' '.join(a.data)}\n"
          f"pitch {a.seeds} seed(s) from {a.seed0} x {a.seconds:.0f} s, {a.per_side}v{a.per_side}, "
          f"ball-out {a.ball_out_s:g} s\n")
    print("\n".join(lines))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
