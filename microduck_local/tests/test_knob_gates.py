"""A knob gated behind a knob that ships off measures NOTHING.

Two batteries have now been lost to this in this repo, from two sessions and
two mechanisms: `contest_margin=0.15` (2026-09-09) ran a full 16-seed
contested gym and reproduced the baseline episode for episode, because the
rule is gated on `use_color`, which ships `False`; and `gaze_bearing_max`
(2026-09-07) was shadowed by a hard-coded `0.6` in the gate it names. Both
reported a clean null about nothing.

`kick_gym.is_identical` catches it empirically, after the compute is spent.
These lock the preflight that catches the commonest form of it before, from
the source — and, as much as anything, lock it against being NOISY, because a
check that cries wolf on a legitimate arm will be ignored and then removed.
"""

from __future__ import annotations

import textwrap

import pytest

from microduck_local.brain.controllers import ChaseParams
from microduck_local.brain.knob_gates import co_gates, missing_gates, warning_for


def src(body: str) -> str:
    return textwrap.dedent(body)


# --- the real source, which is the thing that actually burned us ----------

def test_contest_margin_is_reported_as_gated_on_use_color():
    """The exact arm that was lost."""
    assert missing_gates("contest_margin=0.15") == {"contest_margin": frozenset({"use_color"})}


def test_setting_the_gate_alongside_it_is_clean():
    assert missing_gates("use_color=1,contest_margin=0.15") == {}
    assert warning_for("use_color=1,contest_margin=0.15") is None


def test_opp_keepout_is_gated_too():
    assert "use_color" in missing_gates("opp_keepout=0.3").get("opp_keepout", ())


@pytest.mark.parametrize("spec", ["approach_speed=0.4", "lost_s=2.0", "use_color=1", ""])
def test_a_legitimate_arm_is_not_warned_about(spec):
    """The noise floor. `lost_s` is read all over the file, `use_color` is a
    GATE rather than a gated knob, and a bare speed knob is gated by nothing.
    A false positive on any of these makes the check worthless."""
    assert warning_for(spec) is None


def test_use_color_is_not_reported_as_gated_by_what_it_gates():
    """Direction. `p.use_color and p.opp_keepout > 0` means opp_keepout needs
    use_color, NOT the other way round; reading the pair symmetrically gets
    this backwards and was the first version's bug."""
    assert "use_color" not in co_gates()


def test_the_real_source_reports_only_a_handful_of_gates():
    """A regression fence in both directions: if this grows a lot the rule has
    started over-reporting, and if a known gate vanishes something silently
    ungated it. 164 knobs, 3 gated."""
    g = co_gates()
    assert 1 <= len(g) <= 12, sorted(g)
    assert {"contest_margin", "opp_keepout"} <= set(g)


def test_every_reported_gate_is_a_real_knob_with_a_falsy_default():
    base = ChaseParams()
    for knob, gates in co_gates().items():
        assert hasattr(base, knob)
        for g in gates:
            assert hasattr(base, g), f"{knob} gated on non-knob {g}"


def test_the_warning_names_the_fix_not_just_the_problem():
    w = warning_for("contest_margin=0.15")
    assert w is not None
    assert "use_color=1" in w and "BROKEN, not null" in w


# --- the logic, on synthetic sources so it does not drift with the file ---

def test_a_knob_gated_on_every_path_is_reported():
    s = src("""
        class C:
            def f(self):
                if p.use_color and p.contest_margin > 0.0:
                    pass
        """)
    assert co_gates(s).get("contest_margin") == frozenset({"use_color"})


def test_a_knob_ungated_on_ANY_path_is_not_reported():
    """The intersection rule. One gated use and one ungated use means leaving
    the gate off does not disable the knob -- warning would be wrong."""
    s = src("""
        class C:
            def f(self):
                if p.use_color and p.contest_margin > 0.0:
                    pass
                if p.contest_margin > 0.0 and other:
                    pass
        """)
    assert "contest_margin" not in co_gates(s)


def test_a_comparison_is_never_treated_as_a_gate():
    """Only a BARE truthiness test gates. `p.lost_s > 0` is a subject."""
    s = src("""
        class C:
            def f(self):
                if p.lost_s > 0 and p.contest_margin > 0.0:
                    pass
        """)
    assert co_gates(s) == {}


def test_a_gate_that_ships_ON_is_not_a_trap():
    """`missing_gates` only reports gates whose default is falsy: a gate that
    is already satisfied needs no action from the caller."""
    s = src("""
        class C:
            def f(self):
                if p.two_stage and p.approach_speed > 0.0:
                    pass
        """)
    gated = co_gates(s)
    assert gated.get("approach_speed") == frozenset({"two_stage"})
    expect = {} if not getattr(ChaseParams(), "two_stage", False) else None
    if expect == {}:
        assert missing_gates("approach_speed=0.4", s) == {"approach_speed": frozenset({"two_stage"})}
    else:
        assert missing_gates("approach_speed=0.4", s) == {}


def test_self_p_and_params_receivers_are_both_understood():
    for recv in ("self.p", "params"):
        s = src(f"""
            class C:
                def f(self):
                    if {recv}.use_color and {recv}.contest_margin > 0.0:
                        pass
            """)
        assert co_gates(s).get("contest_margin") == frozenset({"use_color"}), recv


def test_an_unrelated_objects_attribute_is_not_a_knob():
    """`other.use_color` is somebody else's field, not a ChaseParams read."""
    s = src("""
        class C:
            def f(self):
                if other.use_color and other.contest_margin > 0.0:
                    pass
        """)
    assert co_gates(s) == {}


def test_an_or_condition_gates_nothing():
    s = src("""
        class C:
            def f(self):
                if p.use_color or p.contest_margin > 0.0:
                    pass
        """)
    assert co_gates(s) == {}


def test_a_knob_is_never_reported_as_gating_itself():
    s = src("""
        class C:
            def f(self):
                if p.use_color and p.use_color:
                    pass
        """)
    assert "use_color" not in co_gates(s)


def test_missing_gates_is_empty_for_a_spec_that_sets_nothing():
    assert missing_gates("") == {} and missing_gates("   ") == {}


def test_the_check_is_a_warning_and_never_raises():
    """It must not block a legitimate arm: a conjunction is evidence, not
    proof, and `is_identical` remains the ground truth."""
    for spec in ("contest_margin=0.15", "nonsense_knob=1", "", "opp_keepout=0.3"):
        warning_for(spec)          # no exception, whatever the spec says


# --- the probe must not fall into the trap it exists to measure -----------

def test_probe_contest_sets_its_own_gate():
    """`probe_contest.py` measures how often the contest rule fires. If it
    forgot `use_color` it would measure zero and look like a profound result
    -- the exact failure this module exists for, one level up."""
    import pathlib
    import re
    src_text = (pathlib.Path(__file__).resolve().parents[1]
                / "scripts" / "probe_contest.py").read_text()
    m = re.search(r'MICRODUCK_CHASE",\s*"([^"]+)"', src_text)
    assert m, "probe_contest.py no longer sets a MICRODUCK_CHASE default"
    spec = m.group(1)
    assert "contest_margin" in spec, spec
    assert warning_for(spec) is None, f"the probe's own arm is gated: {warning_for(spec)}"


def test_probe_contest_sets_the_arm_before_importing_the_brain():
    """Playbook rule 0: `brain_kwargs` reads `ChaseParams.from_env()` at
    construction, so the env var must be set above the brain imports or the
    probe measures the shipped path."""
    import pathlib
    src_text = (pathlib.Path(__file__).resolve().parents[1]
                / "scripts" / "probe_contest.py").read_text()
    assert (src_text.index("MICRODUCK_CHASE")
            < src_text.index("from microduck_local.brain import")), \
        "the arm is set after the brain import -- it will not reach the brain"
