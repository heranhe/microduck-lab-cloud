"""The pitch ledger kept no per-kick record, and now it does (roadmap E.2).

`kickCarry` is a run TOTAL of ~7 kicks, so E.2's registered per-swing claim
(+0.28 m a swing in the gym) could only be tested on the pitch through a
48-seed sum with a 35% MDE — while the ~300 events that answer it directly
were being summed away as they arrived. `kickEvents` keeps them:
`[t, carry, adv, line, back, foot]` per resolved kick, per team
(`world/metrics.py`, `KICK_EVENT`).

What is locked here:

  * the LIST AND THE TOTALS CANNOT DISAGREE. Every column of an event is read
    off the same numbers, in the same place, as the four totals beside it, so
    a real run's `kickCarry`, `kickCount`, `kicksBack`, `kickLineCount` and
    `kicksBackLine` are all recomputable from the list. That is the check that
    would catch the list and the columns being made of two different things —
    the failure mode a second instrument for the same quantity exists to have.
  * `adv` is None, NOT 0.0, when no exit sample was taken. A kick settled
    inside the window by a goal or a ball-out has no direction and no early
    travel; reporting "not measured" as "it went nowhere" is the one thing
    this repo engineers against.
  * the FOOT, which 12as's "the right foot alone on the 2v2 ledger" needed and
    the ledger could not say. "L"/"R" come off the skill the World started;
    anything else has no foot rather than a guessed one.
  * the LIST IS NOT IN `row()`. The /sim page puts `row()` in every streamed
    frame at 50 Hz and a list that grows for as long as the lab is up does not
    belong there; `events_row()` is the battery's door.
  * BACKWARD COMPATIBILITY, twice over: an entry another test appends to
    `_pending` by hand in the three-field shape still resolves (with no foot),
    and a battery row written before the list loads as None
    (`eval_pitch.load_done`) and still COMPARES — `scripts/compare_pitch.py`
    prints a dash and says which arm is missing it, rather than reporting an
    unmeasured column as "this arm took no kicks".
  * the PER-EVENT statistic itself: `compare_pitch.py` reads carry per kick
    over the pooled events, which is the reading E.2 had to do in a scratch
    script, and it is NOT paired (the arms do not take the same kicks, nor the
    same number of them).
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
    EXIT_S,
    KICK_EVENT,
    PitchMetrics,
    _foot,
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
    """Drive the ball along a scripted path and tick the metrics with it —
    the same idiom `test_metrics_kickline.py` and `test_ball_out.py` use: the
    ball is PLACED each control step with its velocity zeroed, so the
    trajectory is the one the test wrote."""
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


def _plus(m: PitchMetrics) -> str:
    """The team attacking +x."""
    return next(t for t in m.teams if m.sign[t] > 0)


# -- one kick, read back -------------------------------------------------------

def test_a_synthetic_kick_reads_back_its_carry_in_the_list():
    """The whole point: the metres this swing carried the ball are in the row
    as an EVENT, not only inside a run's sum."""
    w, m, q, v = _metrics()
    plus = _plus(m)
    t0, x0, y0 = w.t, -0.4, 0.15
    m._pending.append((plus, t0, (x0, y0), None, "R"))
    # 0.30 m by the exit sample, another 0.20 m by the time the carry window
    # closes, then still — the carry is read at t0 + CARRY_S exactly, so the
    # roll has to be over by then for the test to name the number it expects.
    _roll(w, m, q, v, x0, y0, [(0.30, 0.0, EXIT_S), (0.20, 0.0, 1.5), (0.0, 0.0, 0.5)])
    assert m.kick_count[plus] == 1
    (t, carry, adv, line, back, foot), = m.kick_events[plus]
    assert t == pytest.approx(t0, abs=1e-3)
    assert carry == pytest.approx(0.50, abs=0.03)          # the full 2 s window
    assert adv == pytest.approx(0.30, abs=0.03)            # …of which this much by EXIT_S
    assert (line, back, foot) == (1, 0, "R")
    # …and it is the SAME number the total is made of.
    assert carry == pytest.approx(m.kick_carry[plus], abs=1e-3)
    assert tuple(KICK_EVENT) == ("t", "carry", "adv", "line", "back", "foot")


def test_a_backward_kick_is_a_negative_carry_and_a_backward_line():
    w, m, q, v = _metrics()
    plus = _plus(m)
    x0, y0 = 0.2, -0.1
    m._pending.append((plus, w.t, (x0, y0), None, "L"))
    _roll(w, m, q, v, x0, y0, [(-0.30, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
    (_, carry, adv, line, back, foot), = m.kick_events[plus]
    assert carry < 0 and adv < 0
    assert (line, back, foot) == (1, 1, "L")
    assert m.kicks_back[plus] == 1 and m.kicks_back_line[plus] == 1


def test_a_kick_settled_before_the_window_has_no_exit_sample_and_says_None():
    """A goal or a ball-out teleports the ball, so a kick still in the air is
    settled early: it has no line and no early travel. `adv` is None, not 0.0
    — an unmeasured column reported as a zero is the failure this repo has
    made twice (AGENTS.md, and `load_done`'s own docstring)."""
    w, m, q, v = _metrics()
    plus = _plus(m)
    m._pending.append((plus, w.t, (0.0, 0.0), None, "L"))
    _roll(w, m, q, v, 0.0, 0.0, [(0.3, 0.0, 0.2)])          # well inside EXIT_S
    m._resolve_kicks(w.ball_xy(), force=True)               # what a goal does
    (_, carry, adv, line, back, _), = m.kick_events[plus]
    assert adv is None                                      # NOT 0.0
    assert (line, back) == (0, 0)
    assert carry == pytest.approx(0.3, abs=0.02)            # the carry is still real


def test_the_three_field_pending_entry_still_resolves_and_has_no_foot():
    """`tests/test_ball_out.py` appends a kick by hand in the shape the list
    had before the exit sample and the foot existed. It must not raise, and it
    must not be given a foot it never had."""
    w, m, q, v = _metrics()
    plus = _plus(m)
    m._pending = [(plus, w.t, (-0.2, 0.0))]                 # three fields, on purpose
    _roll(w, m, q, v, -0.2, 0.0, [(0.2, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
    (_, carry, _, line, back, foot), = m.kick_events[plus]
    assert foot == ""                                       # unknown, not guessed
    assert (line, back) == (1, 0)
    assert carry == pytest.approx(0.2, abs=0.02)


# -- the foot ------------------------------------------------------------------

def test_the_foot_comes_off_the_skill_the_world_started():
    """`_note_kick` reads the duck that is swinging, so the column says which
    foot took the kick — the reading 12as wanted on the 2v2 ledger ("the right
    foot alone") and the ledger had no way to give."""
    assert (_foot("kick_left"), _foot("kick_right")) == ("L", "R")
    assert _foot("ground_pick") == "" and _foot(None) == ""
    w, m, _, _ = _metrics(per_side=2)
    d = next(iter(w.ducks.values()))
    d.skill, d.skill_t0 = "kick_right", w.t
    m._note_kick((0.0, 0.0))
    assert [p[4] for p in m._pending] == ["R"]


# -- the list against the totals, on a real run --------------------------------

def test_the_list_and_the_columns_are_made_of_the_same_kicks():
    """Seeded runs end to end through `eval_pitch.run_one`: every kick column
    in the row is recomputable from the list. Two instruments for one quantity
    are only worth having if they cannot drift, and this is the test that
    would see them drift.

    The identity is checked on EVERY run this walks, kicks or none — a run
    where nobody swings still has to carry columns and a list that agree, at
    zero. What the scan underneath it is for is the NON-VACUITY guard: the
    identity only bites on a run that actually contains kicks, and how many a
    run contains moves whenever the brain, the kick sidecars or the camera
    model move. This test pinned seed 0 of a 90 s 2v2, which carried exactly
    two kicks at 90b01c7 and none at all the day the left sidecar was set to
    its in-play exit (9ca8d9d) — which is the day it started failing on main,
    on a behaviour change the sidecar's own note predicts ("~24% fewer swings
    for harder touches"). Measured at HEAD over seeds 0-5: 90 s of 2v2 gives
    0.33 kicks a seed against 1.17 before the correction, so no single seed of
    it is a safe premise. 120 s of 3v3 gives ~1.4 a run over seeds 0-7 (11
    kicks, 6 of 8 seeds non-empty), so the scan walks the denser world, adds
    up what it sees and stops as soon as it has swings to test on."""
    from microduck_local.eval_pitch import run_one

    seen = 0
    for seed in (0, 1, 2, 3):
        r = run_one(seed, 120.0, per_side=3)
        ev = r["kickEvents"]
        assert set(ev) == set(r["kickCount"])
        for t, kicks in ev.items():
            assert len(kicks) == r["kickCount"][t]
            assert sum(e[1] for e in kicks) == pytest.approx(r["kickCarry"][t], abs=2e-3)
            assert sum(1 for e in kicks if e[1] < 0) == r["kicksBack"][t]
            assert sum(e[3] for e in kicks) == r["kickLineCount"][t]
            assert sum(e[4] for e in kicks) == r["kicksBackLine"][t]
            assert all(e[5] in ("L", "R") for e in kicks)   # a real swing has a foot
            assert all(e[4] == 0 for e in kicks if e[3] == 0)   # no line, no direction
            seen += len(kicks)
        if seen >= 2:
            break
    assert seen >= 2, ("seeds 0-3 of a 120 s 3v3 took fewer than two kicks between them, so the "
                       "identity above never ran on a real swing: the pitch stopped kicking, or "
                       "the scan needs a denser world")


def test_the_streamed_row_does_not_carry_the_growing_list():
    """`row()` goes into every /sim frame; `events_row()` is the battery's
    door. A list that grows for as long as the lab is up is not a 50 Hz
    payload, and this is the line that keeps it out of one."""
    w, m, q, v = _metrics()
    plus = _plus(m)
    m._pending.append((plus, w.t, (0.0, 0.0), None, "L"))
    _roll(w, m, q, v, 0.0, 0.0, [(0.2, 0.0, EXIT_S), (0.0, 0.0, CARRY_S)])
    assert "kickEvents" not in m.row()
    row = m.events_row()
    assert set(row) == {"kickEvents"} and len(row["kickEvents"][plus]) == 1
    # A copy: a consumer that mutates what it got cannot corrupt the column
    # the run's own totals were made from.
    row["kickEvents"][plus][0][1] = 99.0
    assert m.kick_events[plus][0][1] != 99.0


# -- the row, old and new ------------------------------------------------------

def _row(seed: int, carries: list[float], events: bool = True) -> dict:
    """A battery row with (or without) the per-kick list, consistent with its
    own totals — `compare_pitch` reads both and they must agree."""
    r = {"seed": seed, "tag": "t", "perSide": 2, "seconds": 30.0, "left": 0, "right": 0,
         "kickGoals": 0, "bumpGoals": 0, "falls": {"d0": 0},
         "possession": {"cream": 20.0, "graphite": 20.0},
         "kickCount": {"cream": len(carries), "graphite": 0},
         "kicksBack": {"cream": sum(1 for c in carries if c < 0), "graphite": 0},
         "kickLineCount": {"cream": len(carries), "graphite": 0},
         "kicksBackLine": {"cream": sum(1 for c in carries if c < 0), "graphite": 0},
         "kickCarry": {"cream": round(sum(carries), 3), "graphite": 0.0}}
    if events:
        r["kickEvents"] = {"cream": [[1.0 + i, c, c, 1, int(c < 0), "L"] for i, c in enumerate(carries)],
                           "graphite": []}
    return r


def test_a_row_written_before_the_list_loads_as_none_and_still_compares(tmp_path):
    """The 48-seed E.2 rows (`runs/kickchoice/pitch-*.jsonl`) have no list.
    They must keep loading — as None, never as an empty list, which would read
    as "this arm took no kicks" — and keep comparing, with the tool SAYING the
    column is missing."""
    from microduck_local.eval_pitch import load_done

    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    old.write_text("".join(json.dumps(_row(s, [0.4, -0.1], events=False)) + "\n" for s in range(4)))
    new.write_text("".join(json.dumps(_row(s, [0.9, 0.5])) + "\n" for s in range(4)))

    done = load_done(str(old), "t", 2, 30.0)
    assert len(done) == 4 and done[0]["kickEvents"] is None      # not [], not {}

    out = subprocess.run([sys.executable, str(SCRIPTS / "compare_pitch.py"), str(old), str(new)],
                         capture_output=True, text=True, check=True).stdout
    assert "carry per kick" in out
    assert "— not in these rows" in out and "old has 0 and new has 4" in out
    assert "kickCarry" in out                                     # the run total still prints

    out2 = subprocess.run([sys.executable, str(SCRIPTS / "compare_pitch.py"), str(new), str(new)],
                          capture_output=True, text=True, check=True).stdout
    assert "+0.700 m over 8 kicks  →  +0.700 m over 8 kicks" in out2


def test_carry_per_kick_is_read_over_the_events_not_over_the_runs(tmp_path):
    """The statistic E.2 needed and did not have. Arm B's mean is the mean of
    ITS kicks, and the two arms need not take the same number of them — which
    is exactly why this cannot be the paired per-seed test above it."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(_row(s, [0.2, 0.4])) + "\n" for s in range(6)))
    # Arm B: fewer kicks a run, each worth more — the pattern E.2 had to
    # separate (its total fell while the rate held).
    b.write_text("".join(json.dumps(_row(s, [1.0])) + "\n" for s in range(6)))
    out = subprocess.run([sys.executable, str(SCRIPTS / "compare_pitch.py"), str(a), str(b),
                          "--label", "a", "b"], capture_output=True, text=True, check=True).stdout
    assert "+0.300 m over 12 kicks  →  +1.000 m over 6 kicks" in out
    assert "Δ +0.700" in out and "Welch" in out
    # …while the run TOTAL moved the other way: 0.6 -> 1.0 a run is up, and it
    # is the rate that says whether each swing was worth more.
    assert "kickCarry" in out
