"""Locks for the LEARNED kick-choice scorer (brain/kickchoice.py, roadmap
E.2): the knob ships off and the shipped ranking is bit-for-bit untouched
when it is, a chooser replaces the RANKING and nothing else (the fan, the
roll-outs and the own-goal veto are the selector's), the weights file round
trips, and the dataset's candidate columns land in the Verdict fields they
name — a column swap there would fit a model on `p_block` and call it
`p_goal`, and every number downstream would still look plausible.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kick_choice_data import design  # noqa: E402

from microduck_local.brain.controllers import Chase, ChaseParams  # noqa: E402
from microduck_local.brain.kickchoice import (  # noqa: E402
    FEATURES,
    ChoiceModel,
    Chooser,
    features,
    fit_ridge,
)
from microduck_local.brain.kickselect import KickModel, Pitch, Verdict, select  # noqa: E402
from microduck_local.brain.runtime import Senses  # noqa: E402

PITCH = Pitch(half_x=1.5, half_y=1.25, goal_w=0.7, attack_sign=1.0)
MODEL = KickModel()


def _fan(los: float = 0.0, k: int = 4) -> list[tuple[float, str]]:
    return [(los + i * 0.35, foot) for i in range(-k, k + 1) for foot in ("kick_left", "kick_right")]


def test_the_knob_ships_off_and_off_is_the_shipped_ranking():
    """`kick_select_learned` defaults to "" — read off a CONSTRUCTED brain,
    not off a fresh ChaseParams — and with it off `select` is never handed a
    chooser, so the A.3 path runs as it did."""
    b = Chase(ChaseParams(kick_select=True), goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    assert b.p.kick_select_learned == ""
    assert b._kick_choice is None
    b._senses = Senses(t=10.0)
    import microduck_local.brain.kickselect as ks
    seen, real = [], ks.select

    def spy(*a, **k):
        seen.append(k.get("chooser"))
        return real(*a, **k)
    ks.select = spy
    try:
        b._select_kick_line((-0.1, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0)
    finally:
        ks.select = real
    assert seen == [None] and b._kick_choice is None


def test_passing_no_chooser_changes_nothing_in_select():
    """The inertness lock, on the function rather than on the brain: the same
    seeded generator, the same fan, with and without the (None) argument —
    identical verdicts, over a sweep of balls including ones in front of our
    own mouth where the veto bites."""
    for bx in (-1.4, -0.8, 0.0, 0.6, 1.3):
        for by in (-1.1, -0.3, 0.4, 1.2):
            a = select((bx, by), _fan(), MODEL, PITCH, np.random.default_rng(7), n=30, t_own=0.10)
            c = select((bx, by), _fan(), MODEL, PITCH, np.random.default_rng(7), n=30, t_own=0.10,
                       chooser=None)
            assert a == c


def test_a_chooser_sees_only_candidates_that_passed_the_own_goal_veto():
    """The learned scorer inherits A.3's safety rule: `select` filters on
    `t_own` BEFORE it ranks, so no weights file can choose a line the roll-out
    says goes in our own net. Checked in front of our own mouth, where the
    fan genuinely contains such lines."""
    handed: list[list[Verdict]] = []

    def chooser(ball, pitch, safe):
        handed.append(list(safe))
        return safe[-1]
    v = select((-1.2, 0.0), _fan(los=math.pi), MODEL, PITCH, np.random.default_rng(3), n=30, t_own=0.10,
               chooser=chooser)
    assert handed, "the chooser was never called"
    assert v is handed[0][-1]                                   # the chooser's pick is the verdict
    assert all(x.p_own <= 0.10 for x in handed[0])
    # ...and the fan really did contain a line the veto had to remove, so the
    # assertion above is not vacuous.
    everything = [select((-1.2, 0.0), [c], MODEL, PITCH, np.random.default_rng(3), n=30, t_own=1.0)
                  for c in _fan(los=math.pi)]
    assert any(x is not None and x.p_own > 0.10 for x in everything)


def test_features_are_named_finite_and_read_the_verdict():
    v = Verdict(0.3, "kick_left", 0.4, 0.02, 1.1, 30, 0.1, 0.05)
    f = features(v, (0.5, -0.2), PITCH, 0.1, (1.5, 0.0), (0.0, 0.0))
    assert len(f) == len(FEATURES) and all(math.isfinite(x) for x in f)
    idx = {n: i for i, n in enumerate(FEATURES)}
    assert f[idx["p_goal"]] == 0.4 and f[idx["p_own"]] == 0.02 and f[idx["value"]] == 1.1
    assert f[idx["is_left"]] == 1.0 and f[idx["is_push"]] == 0.0
    assert abs(f[idx["turn"]] - 0.2) < 1e-9 and abs(f[idx["abs_turn"]] - 0.2) < 1e-9
    # A push gets its own slope columns, and a kick's are zero.
    p = features(Verdict(0.3, "push", 0.0, 0.0, 1.1, 30), (0.5, -0.2), PITCH, 0.1, (1.5, 0.0), (0.0, 0.0))
    assert p[idx["is_push"]] == 1.0 and p[idx["value_push"]] == 1.1 and f[idx["value_push"]] == 0.0


def test_the_weights_file_round_trips_and_refuses_a_stale_feature_list(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES)))
    truth = rng.normal(size=len(FEATURES))
    Y = np.stack([X @ truth, (X[:, 0] > 0).astype(float)], 1)
    mean, scale, P = fit_ridge(X, Y, alpha=1e-6)
    m = ChoiceModel("ridge", mean, scale, P, lam_back=0.25)
    # It fits a linear target it can represent.
    assert float(np.corrcoef(m.heads(X)[:, 0], Y[:, 0])[0, 1]) > 0.99
    path = tmp_path / "w.json"
    m.save(path)
    back = ChoiceModel.load(path)
    assert np.allclose(back.score(X), m.score(X)) and back.lam_back == 0.25
    d = json.loads(path.read_text())
    d["feat"] = list(FEATURES)[:-1]
    try:
        ChoiceModel.from_json(d)
    except ValueError:
        pass
    else:                                                       # a silent column shift is the failure mode
        raise AssertionError("a stale feature list must refuse to load")


def test_the_score_is_advance_minus_the_backward_line_penalty():
    """Two heads, one number: the penalty is IN the weights file, so an arm is
    one decision and cannot be half-applied."""
    n = len(FEATURES)
    W = np.zeros((n, 2))
    W[0, 0] = 1.0                                               # head 0 reads the first feature
    W[1, 1] = 1.0                                               # head 1 the second
    m = ChoiceModel("ridge", np.zeros(n), np.ones(n), {"W": W, "b": np.zeros(2)}, lam_back=2.0)
    X = np.zeros((2, n))
    X[0, 0], X[0, 1] = 1.0, 0.0
    X[1, 0], X[1, 1] = 1.0, 0.5
    assert np.allclose(m.score(X), [1.0, 0.0])


def test_the_knob_builds_the_chooser_and_it_picks_the_argmax(tmp_path):
    """End to end from the knob: a weights file whose score IS `abs_turn`
    makes the selector take the widest line in the fan it was handed — so the
    ranking really is the model's, and the fan really is still the brain's."""
    idx = {n: i for i, n in enumerate(FEATURES)}
    n = len(FEATURES)
    W = np.zeros((n, 2))
    W[idx["abs_turn"], 0] = 1.0
    ChoiceModel("ridge", np.zeros(n), np.ones(n), {"W": W, "b": np.zeros(2)}).save(tmp_path / "w.json")
    b = Chase(ChaseParams(kick_select=True, kick_select_learned=str(tmp_path / "w.json")),
              goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    b._senses = Senses(t=10.0)
    handed: list[list[Verdict]] = []
    import microduck_local.brain.kickselect as ks
    real = ks.select

    def spy(*a, **k):
        ch = k["chooser"]

        def wrapped(ball, pitch, safe):
            handed.append(list(safe))
            return ch(ball, pitch, safe)
        return real(*a, **{**k, "chooser": wrapped})
    ks.select = spy
    try:
        got = b._select_kick_line((-0.1, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0)
    finally:
        ks.select = real
    assert isinstance(b._kick_choice, Chooser) and b._kick_choice.calls == 1
    safe = handed[0]
    want = max(safe, key=lambda v: abs(v.heading - 0.0))        # `los` is 0.0 in this call
    assert got == (want.heading, want.foot)


def test_the_knob_is_settable_from_the_battery_string(tmp_path):
    p = ChaseParams.from_env(f"kick_select_learned={tmp_path / 'w.json'}")
    assert p.kick_select_learned == str(tmp_path / "w.json")
    assert "kick_select_learned" in ChaseParams.env_names(f"kick_select_learned={tmp_path / 'w.json'}")


def test_the_dataset_columns_land_in_the_verdict_fields_they_name():
    """`design()` rebuilds the picked candidate's Verdict from the row's
    seven-column candidate list. A swap there (p_block read as p_goal, say)
    is invisible in every number downstream, so it is checked against
    features whose values are all distinct."""
    row = {"swing": True, "pick": 1,
           "cands": [[0.0, "kick_left", 0.9, 0.0, 0.0, 0.0, 0.0],
                     [0.25, "kick_right", 0.11, 0.02, 0.33, 0.44, 1.25]],
           "pitch": [1.5, 1.25, 0.7, 1.0], "ball": [0.5, -0.2], "los": 0.05,
           "goal": [1.5, 0.0], "odom": [0.0, 0.0, 0.0], "advance": 0.7,
           "exit_world": 0.1, "swing_yaw": 0.0}
    X, Y, keep = design([row, {"swing": False}])
    assert X.shape == (1, len(FEATURES)) and keep == [row]
    idx = {n: i for i, n in enumerate(FEATURES)}
    assert X[0, idx["p_goal"]] == 0.11 and X[0, idx["p_own"]] == 0.02
    assert X[0, idx["p_block"]] == 0.33 and X[0, idx["p_pass"]] == 0.44 and X[0, idx["value"]] == 1.25
    assert X[0, idx["is_left"]] == 0.0 and abs(X[0, idx["turn"]] - 0.20) < 1e-9
    assert Y[0, 0] == 0.7 and Y[0, 1] == 0.0                    # 0.1 rad off +x is not a backward line
    row2 = {**row, "exit_world": 3.0}                           # ...and 3.0 rad is
    assert design([row2])[1][0, 1] == 1.0


# --- THE PITCH COLLECTOR (E.2 follow-up (1)) ---------------------------------
# The first cut was fitted in `kick_gym` and lost on the ledger because 47% of
# the pitch's line-ups sit outside the gym's `range` box. `--pitch` collects
# the rows from the ledger's own world instead, and these lock the two things
# that mode adds: the line-up bookkeeping a match needs (the gym has episodes,
# a match does not) and the fact that adding it left the GYM path alone.


def _verdicts(los: float = 0.0, k: int = 3) -> list[Verdict]:
    from microduck_local.brain.kickselect import evaluate
    return [evaluate((0.2, 0.0), los + i * 0.3, foot, MODEL, PITCH, np.random.default_rng(5), 30)
            for i in range(-k, k + 1) for foot in ("kick_left", "kick_right")]


def test_the_gym_recorder_does_no_lineup_bookkeeping():
    """A Recorder with no clock is the one every gym dataset was collected
    with: the caller owns `explore`, nothing re-draws it, and `new_lineup` is
    never reached — so `--pitch` cannot have moved a gym number."""
    from kick_choice_data import Recorder
    rec = Recorder()
    rec.explore = (0.5, "kick_left")
    choose = rec.bind((0.0, 0.0, 0.0), 0.0, (1.5, 0.0))
    safe = _verdicts()
    for _ in range(5):
        choose((0.2, 0.0), PITCH, safe)
    assert rec.lineups == 0 and rec.explore == (0.5, "kick_left")
    assert rec.last is not None and rec.last_t is None
    assert rec.calls == 5


def test_a_pitch_lineup_holds_its_exploring_pick_and_a_gap_redraws_it():
    """A match has no episodes, so a LINE-UP is a run of ranking calls with no
    gap longer than `LINEUP_GAP_S`. Inside one the exploring pick is held
    fixed — a pick that jitters tick to tick is a spot the duck can never
    reach, and would measure the jitter (the gym's own reason). A gap, or an
    explicit `end_lineup` after a swing, starts a new one."""
    from kick_choice_data import LINEUP_GAP_S, Recorder
    t = [0.0]
    rec = Recorder(clock=lambda: t[0], rng=np.random.default_rng(0), explore=1.0)
    choose = rec.bind((0.0, 0.0, 0.0), 0.0, (1.5, 0.0))
    safe = _verdicts()
    picks = []
    for _ in range(10):                                  # one line-up, ten ticks
        choose((0.2, 0.0), PITCH, safe)
        picks.append(rec.explore)
        t[0] += 0.02
    assert rec.lineups == 1 and len(set(picks)) == 1 and picks[0] is not None
    first = picks[0]
    t[0] += LINEUP_GAP_S + 0.01                          # the duck stopped planning: a new line-up
    choose((0.2, 0.0), PITCH, safe)
    assert rec.lineups == 2
    second = rec.explore
    assert second is not None and second != first        # a new line-up re-draws (explore=1.0, seeded)
    rec.end_lineup()                                     # ...and a swing ends one outright
    assert rec.last is None and rec.last_t is None
    t[0] += 0.02                                         # …even with no gap at all
    choose((0.2, 0.0), PITCH, safe)
    assert rec.lineups == 3


def test_an_exploring_lineup_takes_a_candidate_the_rollout_did_not():
    """The point of `--explore`: the row's `pick` is not tied to `roll_pick`,
    and both indices address the same SAFE list the shipped selector ranked."""
    from kick_choice_data import Recorder
    safe = _verdicts()
    seen = set()
    for q in (0.0, 0.3, 0.6, 0.99):
        rec = Recorder()
        rec.explore = (q, "kick_left")
        rec.bind((0.0, 0.0, 0.0), 0.0, (1.5, 0.0))((0.2, 0.0), PITCH, safe)
        row = rec.last
        assert safe[row["pick"]].foot == "kick_left"
        assert 0 <= row["roll_pick"] < len(safe)
        seen.add(row["pick"])
    assert len(seen) > 1, "the quantile must actually move the pick"


def test_a_pitch_row_labels_advance_toward_the_mouth_that_team_attacks():
    """A pitch row carries `attack_sign` −1 for the team at +x, and the label
    is the signed displacement toward THAT mouth — `world/metrics.py`'s own
    `kickCarry` rule, not the gym's "+x is the attacked mouth". Getting this
    wrong would train the model to kick half the team the wrong way, and
    every number downstream would still look plausible."""
    row = {"swing": True, "arena": "pitch", "pick": 0,
           "cands": [[math.pi, "kick_left", 0.2, 0.0, 0.0, 0.0, 0.5]],
           "pitch": [1.7, 1.425, 0.7, -1.0], "ball": [0.5, -0.2], "los": math.pi,
           "goal": [-1.7, 0.0], "odom": [0.9, -0.2, math.pi],
           "advance": 0.9,                                # 0.9 m toward the −x mouth
           "exit_world": math.pi - 0.1, "swing_yaw": 0.0}
    X, Y, keep = design([row])
    assert keep == [row] and Y[0, 0] == 0.9
    assert Y[0, 1] == 0.0                                 # a line at the −x mouth is NOT backward for them
    assert design([{**row, "exit_world": 0.1}])[1][0, 1] == 1.0    # ...straight up the pitch is


def test_the_pitch_mode_defaults_are_the_ledgers():
    """The collector's world has to be the world the ledger is read on, or the
    box it fits is a third arena nobody measures."""
    import kick_choice_data as K
    assert K.PITCH_BALL_OUT_S == 5.0
    sig = __import__("inspect").signature(K.collect_pitch)
    assert sig.parameters["per_side"].default == 2
    assert sig.parameters["ball_out_s"].default == K.PITCH_BALL_OUT_S
    # …and `--pitch` is reachable from the command line with those defaults.
    import subprocess
    out = subprocess.run([sys.executable, str(Path(K.__file__)), "collect", "--help"],
                         capture_output=True, text=True).stdout
    assert "--pitch" in out and "--ball-out-s" in out and "--per-side" in out
