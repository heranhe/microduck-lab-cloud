"""The pitch ledger could not read a kick's DIRECTION, and now it can (12au).

`kicksBack` is `advance < 0` over the 2 s carry window, and 12at measured per
swing what that actually counts: among touches struck straight at the mouth it
fires 31-60% of the time below 1 m of travel and 0-5% above it. It is a
WEAK-TOUCH measure. So the 2v2 ledger's "13 -> 29% of kicks go back" could not
be read as "the kicks point backwards", and the arm that HALVES the backward
lines in the gym RAISES `kicksBack` (20.6 -> 23.5%) while whiff falls.

`kicksBackLine` is the direction column beside it: the line the ball left on
over its first `EXIT_S`, more than 90 deg from the mouth the kicking team
attacks. What is locked here:

  * the SIGN and the two TEAMS. A team attacks +x or -x depending on where it
    spawned, and reading the mouth off the wrong end would report every kick
    of one side as backward. Checked on a real World with a real ball rolled
    at a known heading, both sides, not against the formula that produced it.
  * that the two columns really are DIFFERENT questions: a ball that leaves
    forward and is walked back inside the window is `kicksBack` and NOT
    `kicksBackLine`. That case is the whole reason the column exists, so it
    is measured, not asserted in prose.
  * the SILENCE of a whiff: a ball nudged under `EXIT_MIN_M` has no line, and
    it is counted in `kickCount` and NOT in `kickLineCount` — the numerator
    and the denominator of a rate coming from the same population (AGENTS.md).
    Same for a kick settled before the window runs out: a goal or a ball-out
    teleports the ball, and a teleport has no direction.
  * the OLD columns, untouched: `kickCount`, `kicksBack` and `kickCarry` on
    the same events, and the three-field `_pending` entry another test writes
    by hand (`tests/test_ball_out.py`) still resolving.
  * BACKWARD COMPATIBILITY of the row: a battery written before the column
    loads (`eval_pitch.load_done` -> None, never 0) and still compares
    (`scripts/compare_pitch.py` prints a dash and says how many seeds carry
    it, rather than reporting "not measured" as "never happened").
  * the CONSTANTS, named rather than inherited: the window and the minimum
    travel are the gym's `EXIT_S` / `EXIT_MIN_M`, so the pitch column and the
    gym's `exit_play` are the same rule. A drift in either is one honest
    failure here instead of a quiet disagreement between two instruments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

from microduck_local.world import World, make_pitch
from microduck_local.world.metrics import (
    CARRY_S,
    EXIT_MIN_M,
    EXIT_S,
    GOAL_FIELDS,
    PitchMetrics,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _metrics(per_side: int = 1) -> tuple[World, PitchMetrics, int, int]:
    sc = make_pitch(per_side=per_side)
    w = World(sc, seed=0)
    m = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    j = w._ball_joint
    return w, m, int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])


def _roll(w: World, m: PitchMetrics, q: int, v: int, x0: float, y0: float,
          leg: list[tuple[float, float, float]]) -> None:
    """Drive the ball along a scripted path and tick the metrics with it.

    Each leg is (dx, dy, seconds): the ball is PLACED each control step, with
    its velocity zeroed, so the trajectory is the one the test wrote rather
    than one the physics negotiated — the same idiom `test_ball_out.py` uses
    to park a ball where it wants it."""
    r = w.scenario.balls[0].radius
    x, y = x0, y0
    for dx, dy, secs in leg:
        n = max(1, int(round(secs / 0.02)))
        for _ in range(n):
            x, y = x + dx / n, y + dy / n
            w.data.qpos[q:q + 7] = [x, y, r + 0.005, 1.0, 0.0, 0.0, 0.0]
            w.data.qvel[v:v + 6] = 0.0
            mujoco.mj_forward(w.model, w.data)
            w.step()
            m.tick()


def _teams(m: PitchMetrics) -> tuple[str, str]:
    """(the team attacking +x, the team attacking -x)."""
    fwd = [t for t in m.teams if m.sign[t] > 0]
    back = [t for t in m.teams if m.sign[t] < 0]
    assert len(fwd) == 1 and len(back) == 1
    return fwd[0], back[0]


def test_a_ball_rolled_forward_and_one_rolled_backward_read_back_their_line():
    """The sign, on a real ball, for both teams. `plus` attacks +x, so a kick
    that sends the ball toward +x left on a FORWARD line for it and on a
    BACKWARD line for the other side — the identical roll, read twice."""
    for toward_plus in (True, False):
        w, m, q, v = _metrics()
        plus, minus = _teams(m)
        x0, y0 = 0.0, 0.4
        dx = 0.30 if toward_plus else -0.30
        for tm in (plus, minus):
            m._pending.append((tm, w.t, (x0, y0), None))
        # 0.30 m over the exit window, then it sits: the line is unambiguous
        # and the carry (2 s) has the same sign, so this case cannot tell the
        # two rules apart — the next test is the one that does.
        _roll(w, m, q, v, x0, y0, [(dx, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
        assert not m._pending                                  # both settled
        assert m.kick_lines[plus] == m.kick_lines[minus] == 1   # both had a line
        assert m.kicks_back_line[plus] == (0 if toward_plus else 1)
        assert m.kicks_back_line[minus] == (1 if toward_plus else 0)
        # …and the old column agrees with it on this easy case.
        assert m.kicks_back[plus] == (0 if toward_plus else 1)
        assert m.kicks_back[minus] == (1 if toward_plus else 0)
        assert m.kick_carry[plus] == pytest.approx(dx, abs=0.02)


def test_a_kick_that_leaves_forward_and_is_walked_back_is_back_but_not_back_line():
    """The case the column exists for, and the one 12at measured in the gym:
    `advance < 0` over 2 s counts the short weak touch the duck walks back
    into, which is not a kick that pointed the wrong way. The two columns
    must disagree here, or the new one is measuring the old one."""
    w, m, q, v = _metrics()
    plus, _ = _teams(m)
    x0, y0 = -0.3, 0.2
    m._pending.append((plus, w.t, (x0, y0), None))
    # +0.25 m straight at the attacked mouth over the exit window, then
    # dribbled back past where it started, and STOPPED there before the carry
    # window closes — the carry is read at t0 + CARRY_S exactly, so the walk
    # back has to be over by then for the test to name the number it expects.
    _roll(w, m, q, v, x0, y0, [(0.25, 0.0, EXIT_S), (-0.45, 0.0, 1.2), (0.0, 0.0, 0.4)])
    assert not m._pending
    assert m.kick_count[plus] == 1 and m.kick_lines[plus] == 1
    assert m.kicks_back[plus] == 1              # the ledger's rule: it ended up behind
    assert m.kicks_back_line[plus] == 0         # …but it LEFT at the mouth
    assert m.kick_carry[plus] == pytest.approx(-0.20, abs=0.03)


def test_a_ball_that_barely_moves_has_no_line_and_is_not_in_the_denominator():
    """A whiff has no direction. Giving it one would put a uniformly
    distributed angle into the share; counting it in the denominator would
    dilute the share with events that cannot be in the numerator."""
    w, m, q, v = _metrics()
    plus, _ = _teams(m)
    x0, y0 = 0.1, -0.3
    m._pending.append((plus, w.t, (x0, y0), None))
    nudge = 0.6 * EXIT_MIN_M                    # under the floor, and it stays under it
    _roll(w, m, q, v, x0, y0, [(nudge, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
    assert m.kick_count[plus] == 1              # it was a kick…
    assert m.kick_lines[plus] == 0              # …with no line to read
    assert m.kicks_back_line[plus] == 0


def test_a_kick_settled_before_the_window_runs_out_has_no_line():
    """A goal or a ball-out settles every kick still in the air, because the
    World is about to teleport the ball. A kick 0.2 s old has no sample and
    must not borrow the teleport's direction."""
    w, m, q, v = _metrics()
    plus, _ = _teams(m)
    x0, y0 = 0.0, 0.0
    m._pending.append((plus, w.t, (x0, y0), None))
    _roll(w, m, q, v, x0, y0, [(0.3, 0.0, 0.2)])            # well inside EXIT_S
    assert len(m._pending) == 1 and m._pending[0][3] is None
    m._resolve_kicks(w.ball_xy(), force=True)               # what a goal does
    assert m.kick_count[plus] == 1 and m.kick_lines[plus] == 0
    assert m.kicks_back_line[plus] == 0


def test_the_old_three_field_pending_entry_still_resolves():
    """`tests/test_ball_out.py` hands this list a kick by hand in the shape it
    had before the exit sample existed. That entry has no line and must not
    raise — the column is additive, including for its own callers."""
    w, m, q, v = _metrics()
    plus, _ = _teams(m)
    m._pending = [(plus, w.t, (-0.2, 0.0))]                 # three fields, on purpose
    _roll(w, m, q, v, -0.2, 0.0, [(0.2, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
    assert m.kick_count[plus] == 1 and m.kicks_back[plus] == 0
    assert m.kick_carry[plus] == pytest.approx(0.2, abs=0.02)
    # A hand-written entry does get a sample once the window passes (the list
    # is normalised in place), so it reads a line like any other kick.
    assert m.kick_lines[plus] == 1 and m.kicks_back_line[plus] == 0


def test_the_row_carries_both_columns_and_their_denominators():
    w, m, _, _ = _metrics(per_side=2)
    row = m.row()
    for f in ("kickCount", "kicksBack", "kickLineCount", "kicksBackLine"):
        assert f in GOAL_FIELDS and f in row
        assert sorted(row[f]) == sorted(m.teams)
    # An event COUNT, not a rate: a row's columns must be addable across a
    # battery, so nothing here is divided by the run's length.
    assert all(isinstance(x, int) for x in row["kicksBackLine"].values())


def test_the_window_and_the_floor_are_the_gyms_own_constants():
    """A test that pins a number it does not name is a hostage (AGENTS.md).
    The pitch column and `kick_gym`'s `exit_play` answer the same question, so
    they must move together: if one instrument's window changes, this fails
    and says so instead of the two quietly disagreeing."""
    sys.path.insert(0, str(SCRIPTS))
    import kick_gym  # noqa: PLC0415

    assert EXIT_S == kick_gym.EXIT_S == 0.5
    assert EXIT_MIN_M == kick_gym.EXIT_MIN_M == 0.05


def _row(seed: int, kicks: int, back: int, line: int | None, back_line: int | None) -> dict:
    r = {"seed": seed, "tag": "t", "perSide": 2, "seconds": 30.0, "left": 0, "right": 0,
         "kickGoals": 0, "bumpGoals": 0, "falls": {"d0": 0},
         "possession": {"cream": 20.0, "graphite": 20.0},
         "kickCount": {"cream": kicks, "graphite": 0},
         "kicksBack": {"cream": back, "graphite": 0}}
    if line is not None:
        r["kickLineCount"] = {"cream": line, "graphite": 0}
        r["kicksBackLine"] = {"cream": back_line or 0, "graphite": 0}
    return r


def test_a_battery_written_before_the_column_still_loads_and_still_compares(tmp_path):
    """The roadmap's own rows (`runs/kickseed1/pitch-*-200.jsonl`) predate the
    column. They must keep loading as None — never 0.0, which would drag a
    mean toward zero — and they must keep COMPARING, with the tool saying the
    column is missing rather than printing a 0% back-line share."""
    from microduck_local.eval_pitch import load_done

    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    old.write_text("".join(json.dumps(_row(s, 10, 3, None, None)) + "\n" for s in range(4)))
    new.write_text("".join(json.dumps(_row(s, 10, 4, 8, 1)) + "\n" for s in range(4)))

    done = load_done(str(old), "t", 2, 30.0)
    assert len(done) == 4
    assert done[0]["kicksBackLine"] is None and done[0]["kickLineCount"] is None
    assert done[0]["kicksBack"] == {"cream": 3, "graphite": 0}

    out = subprocess.run([sys.executable, str(SCRIPTS / "compare_pitch.py"), str(old), str(new)],
                         capture_output=True, text=True, check=True).stdout
    assert "backward line" in out
    assert "not in these rows" in out                 # said, not silently zeroed
    assert "has 0 and" in out
    # …and the old proportion still reads, on the same run.
    assert "30/40 = 75.0%" not in out                 # (sanity: that is not this data)
    assert "own goal 2 s later" in out or "own goal" in out

    out2 = subprocess.run([sys.executable, str(SCRIPTS / "compare_pitch.py"), str(new), str(new)],
                          capture_output=True, text=True, check=True).stdout
    assert "4/32 = 12.5%" in out2                     # both arms carry it: the share prints
