"""The model-input-box probe (roadmap E.2's acceptance test, made committed).

E.2's learned kick ranking won in the gym on three independent blocks and
lost on a 48-seed pitch ledger, and the reason was the INPUT and not the
ranking: 47% of the candidates the chooser scores on a pitch are outside the
training range of `range`, 31% outside it on `ball_board`. The probe that
found that lived in a scratch directory; `scripts/probe_model_box.py` is it,
and this is what the instrument itself has to get right.

The live half costs three pitch seeds x 120 s, so what is tested here is the
READING — `box_report`, a pure function — against a stubbed feature stream,
plus the dataset path (`training_box`) on a tiny synthetic dataset in the
collector's own row format:

  * a training set that COVERS the pitch's line-ups PASSES, and the same
    pitch rows against a gym-shaped box (the real one: `range` 0.141-0.447 m)
    FAIL, naming `range`. Both halves matter — a checker that only ever says
    FAIL is not a check.
  * a column the fit never saw vary is called `dead`, because a constant
    feature carries no information however the pitch moves it (E.2:
    `p_block` is live only in the opposed gym, `p_pass` in neither arena).
  * the feature NAMES come from `kickchoice.FEATURES`, so a feature that
    moves fails loudly rather than shifting a column silently.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import probe_model_box as P  # noqa: E402

from microduck_local.brain.kickchoice import FEATURES, features  # noqa: E402
from microduck_local.brain.kickselect import Pitch, Verdict  # noqa: E402

PITCH = (1.5, 1.25, 0.6, 1.0)          # half_x, half_y, goal_w, attack_sign


def _cand(rng, rge: float, by: float = 0.0) -> list[float]:
    """One candidate's feature row at a chosen body-to-ball `range`. The
    features are the real ones — this is a stubbed feature STREAM, not a
    stubbed feature function."""
    ball = (float(rng.uniform(-1.0, 1.0)), by)
    v = Verdict(float(rng.uniform(-math.pi, math.pi)), "kick_left", 0.1, 0.01, 0.5, 30, 0.0, 0.0)
    me = (ball[0] - rge, ball[1])
    return features(v, ball, Pitch(*PITCH), 0.0, (1.5, 0.0), me)


def _stream(n: int, lo: float, hi: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([_cand(rng, float(rng.uniform(lo, hi))) for _ in range(n)], float)


def test_a_box_that_covers_the_pitch_passes_and_a_gym_shaped_one_fails_on_range():
    """The two verdicts, on the same pitch rows. The gym box is E.2's measured
    one (`range` 0.141-0.447 m from `kick_gym`'s 0.45-1.4 m walk-in draw)."""
    pitch = _stream(600, 0.10, 1.40, seed=1)
    wide = _stream(600, 0.05, 1.60, seed=2)              # fitted on pitch-like line-ups
    gym = _stream(600, 0.141, 0.447, seed=3)             # fitted in the gym

    lines, ok = P.box_report(wide, pitch, FEATURES)
    assert ok and "PASS" in lines[-1]

    lines, ok = P.box_report(gym, pitch, FEATURES)
    assert not ok
    assert "FAIL" in lines[-1] and "range" in lines[-1]
    assert "do not put this scorer on a ledger" in lines[-1].lower()
    # …and the table says how far outside, on the row for that feature.
    row = next(ln for ln in lines if ln.startswith("range"))
    outside = float(row.split()[-2].rstrip("%")) if row.endswith("<<<") else float(row.split()[-1].rstrip("%"))
    assert outside > 40.0                               # E.2 measured 47.4% on the real pitch


def test_a_constant_training_column_is_called_dead():
    """`p_pass` is identically 0 in every training row of the unopposed
    dataset and identically 0 on a default pitch too. That is not a pass, it
    is a column the fit could never have learned anything from."""
    pitch = _stream(200, 0.2, 0.4, seed=4)
    train = _stream(200, 0.2, 0.4, seed=5)
    lines, ok = P.box_report(train, pitch, FEATURES)
    assert ok
    dead = [ln.split()[0] for ln in lines if "dead in training" in ln]
    assert "p_pass" in dead and "p_block" in dead       # both are 0 in this stream


def test_the_threshold_is_the_knob_and_it_moves_the_verdict():
    pitch = _stream(400, 0.10, 1.40, seed=6)
    gym = _stream(400, 0.141, 0.447, seed=7)
    assert P.box_report(gym, pitch, FEATURES, threshold=10.0)[1] is False
    assert P.box_report(gym, pitch, FEATURES, threshold=99.0)[1] is True
    assert P.OUTSIDE_PCT == 10.0


def test_a_column_count_that_disagrees_is_fatal_not_silent():
    """The failure this would otherwise become is the worst kind: a model
    scored on a column it was not fitted on, with no error."""
    pitch = _stream(20, 0.2, 0.4, seed=8)
    with pytest.raises(SystemExit):
        P.box_report(pitch[:, :-1], pitch, FEATURES)


# --- the dataset half ---------------------------------------------------------

def _train_row(rge: float, bx: float = 0.2) -> dict:
    """One row in `scripts/kick_choice_data.py`'s collected format: the
    candidate fan, which one was swung, and what the swing then did."""
    return {"swing": True, "pick": 0,
            "cands": [[0.1, "kick_left", 0.2, 0.01, 0.0, 0.0, 0.5]],
            "pitch": list(PITCH), "ball": [bx, 0.0], "los": 0.0,
            "odom": [bx - rge, 0.0, 0.0], "goal": [1.5, 0.0],
            "advance": 0.4, "exit_world": 0.0}


def test_the_training_box_is_read_through_the_collectors_own_design(tmp_path):
    """Not re-typed here: the matrix has to be the one the model was fitted
    on, so it comes from `kick_choice_data.design` itself."""
    f = tmp_path / "data.jsonl"
    ranges = [0.15, 0.30, 0.45]
    f.write_text("".join(json.dumps(_train_row(r)) + "\n" for r in ranges)
                 + json.dumps({"swing": False, "cands": []}) + "\n")   # a line-up with no swing
    X = P.training_box([str(f)])
    assert X.shape == (len(ranges), len(FEATURES))                     # the unswung row is dropped
    i = FEATURES.index("range")
    assert X[:, i] == pytest.approx(ranges, abs=1e-6)


def test_an_empty_dataset_is_refused_rather_than_boxed(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text(json.dumps({"swing": False, "cands": []}) + "\n")
    with pytest.raises(SystemExit):
        P.training_box([str(f)])
