"""What a resumed `eval-pitch` ledger is allowed to hold — the KICKS half.

`load_done` already refuses to mix tags, rosters, run lengths and the world's
physics knobs, because a resume that straddles two settings averages them in
one file and never says so. The kicks were the hole in that: they are not
flags but FILES — `policies/kick/kick_{left,right}.onnx` and the sidecar the
brain reads `exit_rad` out of, or a `MICRODUCK_SKILL_KICK_*` pin — so a
commit can change what a battery measures while the command line stays
identical. One did: `kick_left.json`'s `exit_rad` moved -0.225 -> +0.209
between a ledger's first 161 rows and its extension, and the resume would
have appended seeds from the new world onto seeds from the old one.

These tests pin a scratch copy of a kick so the mismatch is a real one on a
real World, not a hand-written field.
"""

import json
import shutil

import pytest

from microduck_local.eval_pitch import kick_provenance, load_done
from microduck_local.world import World, make_pitch


def _row(seed, **kw):
    """A ledger row with today's kick provenance in it, as `run_one` writes."""
    exits, skills = kick_provenance()
    return {"seed": seed, "tag": "", "perSide": 1, "seconds": 300.0,
            "left": 1, "right": 0, "kickGoals": 1, "bumpGoals": 0,
            "kicks": {"d0": 5, "d1": 4}, "pushes": {"d0": 1, "d1": 0},
            "falls": {"d0": 0, "d1": 1}, "simSeconds": 300.0,
            "kickExits": exits, "kickSkills": skills, **kw}


def _pin_a_kick(tmp_path, monkeypatch, *, exit_rad=None, name="kick_left.onnx"):
    """Copy the left kick into a scratch dir, give it a sidecar of our own,
    and point `MICRODUCK_SKILL_KICK_LEFT` at it. Same bytes and same
    basename by default, so ONLY the exit angle moves — which is the shape
    of the change that actually happened."""
    src = World.skill_path("kick_left")
    dst = tmp_path / name
    shutil.copyfile(src, dst)
    side = dict(json.loads(src.with_suffix(".json").read_text()))
    if exit_rad is not None:
        side["exit_rad"] = exit_rad
    dst.with_suffix(".json").write_text(json.dumps(side))
    monkeypatch.setenv("MICRODUCK_SKILL_KICK_LEFT", str(dst))
    return dst


def test_a_row_measured_with_the_kicks_running_now_resumes(tmp_path, capsys):
    f = tmp_path / "same.jsonl"
    f.write_text(json.dumps(_row(0)) + "\n")
    done = load_done(str(f), "", 1, 300.0)
    assert set(done) == {0}
    assert done[0]["kickExits"], "the shipped/local kicks carry sidecar exits"
    assert "provenance" not in capsys.readouterr().out          # nothing to warn about


def test_a_row_measured_with_a_different_exit_angle_refuses_and_names_both(tmp_path, monkeypatch):
    """The 9ca8d9d case, to the bit: the same ONNX, a sidecar that aims it
    somewhere else. `kickSkills` is unchanged and would not have caught it."""
    before_exits, before_skills = kick_provenance()
    f = tmp_path / "straddle.jsonl"
    f.write_text(json.dumps(_row(0)) + "\n")

    _pin_a_kick(tmp_path, monkeypatch, exit_rad=round(float(before_exits[0]) + 0.434, 3))
    after_exits, after_skills = kick_provenance()
    assert after_skills == before_skills                        # only the aim moved…
    assert after_exits != before_exits
    assert World(make_pitch(), seed=0).kick_exits() == tuple(after_exits), \
        "the pin has to reach the World, or the mismatch is fiction"

    with pytest.raises(SystemExit) as e:
        load_done(str(f), "", 1, 300.0)
    msg = str(e.value)
    assert f"kickExits={before_exits}" in msg and f"kickExits={after_exits}" in msg
    assert "straddle.jsonl:1" in msg


def test_a_row_measured_with_a_different_kick_FILE_refuses(tmp_path, monkeypatch):
    """…and the other half: the same aim out of a different policy file."""
    _, before_skills = kick_provenance()
    f = tmp_path / "swapped.jsonl"
    f.write_text(json.dumps(_row(0)) + "\n")

    _pin_a_kick(tmp_path, monkeypatch, name="kick_left_v2.onnx")   # same bytes, a name that says which
    after_exits, after_skills = kick_provenance()
    assert after_skills != before_skills

    with pytest.raises(SystemExit) as e:
        load_done(str(f), "", 1, 300.0)
    msg = str(e.value)
    assert f"kickSkills={before_skills}" in msg and f"kickSkills={after_skills}" in msg


def test_a_row_written_before_the_field_existed_resumes_with_one_warning(tmp_path, capsys):
    """Refusing every ledger written before this week would cost hours of
    seeds to recover provenance nobody recorded; pretending they match would
    be the silence the guard exists to break. So: resume, and say so once."""
    old = _row(0)
    del old["kickExits"], old["kickSkills"]
    f = tmp_path / "old.jsonl"
    f.write_text(json.dumps(old) + "\n" + json.dumps(_row(1)) + "\n")

    done = load_done(str(f), "", 1, 300.0)
    assert set(done) == {0, 1}                                  # neither seed is re-run
    assert done[0]["kickExits"] is None and done[0]["kickSkills"] is None   # unknown, not "the shipped ones"
    warnings = [ln for ln in capsys.readouterr().out.splitlines() if "provenance" in ln]
    assert len(warnings) == 1 and "1 row(s)" in warnings[0]
