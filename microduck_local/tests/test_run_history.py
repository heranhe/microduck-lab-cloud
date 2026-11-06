"""The history is read from what a run leaves behind, so these tests build runs the way the trainer does."""

import json
import os
import time

from microduck_local import run_history as RH


def make_run(root, name, *, records, behavior="spin", done=True, servo=None, log=None, policy=True, age=0.0):
    d = root / name
    d.mkdir()
    lines = [json.dumps(r) for r in records]
    if done:
        lines.append(json.dumps({"steps": records[-1]["steps"], "total": records[-1]["total"], "done": True}))
    (d / "progress.jsonl").write_text("\n".join(lines) + "\n")
    (d / "behavior.json").write_text(json.dumps({"behavior": behavior}))
    (d / "teach-job.json").write_text(json.dumps({"extraEnv": {"MICRODUCK_SERVO": servo} if servo else {}}))
    if log is not None:
        (d / "train.log").write_text(log)
    if policy:
        (d / "policy.onnx").write_bytes(b"x")
    if age:
        t = time.time() - age
        os.utime(d / "progress.jsonl", (t, t))
    return d


def rec(steps, total, score, elapsed, sps=5000):
    return {"steps": steps, "total": total, "ep_rew": score, "elapsed_s": elapsed, "sps": sps, "done": False}


def test_a_finished_run_reports_how_long_how_far_and_what_it_scored(tmp_path):
    make_run(tmp_path, "teach-spin-aaa", records=[rec(1000, 4000, 10.0, 1.0), rec(4000, 4000, 92.5, 61.0)], servo="feetech")
    [row] = RH.run_history(tmp_path)
    assert row["status"] == "done" and row["durationS"] == 61.0 and row["steps"] == 4000 and row["total"] == 4000
    assert row["score"] == 92.5 and row["servo"] == "feetech" and row["hasPolicy"] is True and row["behavior"] == "spin"


def test_the_closing_done_line_has_no_time_or_score_so_they_come_from_the_last_full_record(tmp_path):
    """The real trainer ends its log with a bare {steps,total,done} line; reading only that gave no duration."""
    d = tmp_path / "teach-real"
    d.mkdir()
    (d / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in [
        rec(1000, 2000, 5.0, 3.0), rec(2048, 2000, 77.0, 9.5), {"steps": 2000, "total": 2000, "done": True}]) + "\n")
    [row] = RH.run_history(tmp_path)
    assert row["status"] == "done" and row["durationS"] == 9.5 and row["score"] == 77.0
    assert row["steps"] == 2000      # the finished count, not the last rollout's overshoot


def test_an_unfinished_run_is_stopped_a_crashed_one_failed_and_a_live_one_training(tmp_path):
    make_run(tmp_path, "stopped", records=[rec(10, 100, 1.0, 5.0)], done=False, log="saved\n")
    make_run(tmp_path, "crashed", records=[rec(10, 100, 1.0, 5.0)], done=False, log="x\nTraceback (most recent call last):\n  boom\n")
    make_run(tmp_path, "running", records=[rec(10, 100, 1.0, 5.0)], done=False)
    status = {r["run"]: r["status"] for r in RH.run_history(tmp_path, live={"running"})}
    assert status == {"stopped": "stopped", "crashed": "failed", "running": "training"}


def test_a_curriculum_chain_is_one_row_with_summed_time_and_steps_and_the_last_stages_score(tmp_path):
    make_run(tmp_path, "teach-hs-abc-s1", records=[rec(100, 100, 5.0, 10.0)], age=300)
    make_run(tmp_path, "teach-hs-abc-s2", records=[rec(200, 200, 8.0, 20.0)], age=200)
    make_run(tmp_path, "teach-hs-abc-s3", records=[rec(300, 300, 99.0, 30.0)], age=100)
    make_run(tmp_path, "teach-other", records=[rec(50, 50, 1.0, 2.0)], age=10)
    rows = {r["run"]: r for r in RH.run_history(tmp_path)}
    chain = rows["teach-hs-abc-s3"]
    assert chain["stages"] == 3 and chain["durationS"] == 60.0 and chain["steps"] == 600 and chain["score"] == 99.0
    assert chain["chain"] == "teach-hs-abc" and chain["status"] == "done"
    assert "teach-hs-abc-s1" not in rows and "teach-other" in rows and "stages" not in rows["teach-other"]


def test_a_chain_that_lost_a_stage_to_a_crash_reads_failed_not_done(tmp_path):
    make_run(tmp_path, "c-s1", records=[rec(10, 10, 1.0, 1.0)], age=200)
    make_run(tmp_path, "c-s2", records=[rec(10, 10, 1.0, 1.0)], age=100, done=False, log="Traceback (most recent call last):\n")
    [row] = RH.run_history(tmp_path)
    assert row["status"] == "failed" and row["stages"] == 2


def test_a_number_the_run_never_wrote_stays_missing_instead_of_zero(tmp_path):
    d = make_run(tmp_path, "bare", records=[{"steps": 5, "total": 10}], done=False)
    (d / "teach-job.json").unlink()
    [row] = RH.run_history(tmp_path)
    assert row["durationS"] is None and row["score"] is None and row["sps"] is None and row["servo"] is None


def test_newest_first_limited_and_tolerant_of_junk(tmp_path):
    make_run(tmp_path, "old", records=[rec(1, 1, 1.0, 1.0)], age=500)
    make_run(tmp_path, "new", records=[rec(1, 1, 1.0, 1.0)], age=5)
    (tmp_path / "no-progress").mkdir()
    (tmp_path / "a-file.txt").write_text("x")
    broken = make_run(tmp_path, "broken", records=[rec(1, 1, 1.0, 1.0)], age=50, done=False)
    (broken / "progress.jsonl").write_text('{"steps": 1, "total"')   # a half-written line and nothing else
    t = time.time() - 50
    os.utime(broken / "progress.jsonl", (t, t))
    assert [r["run"] for r in RH.run_history(tmp_path)] == ["new", "broken", "old"]
    assert [r["run"] for r in RH.run_history(tmp_path, limit=1)] == ["new"]
    assert RH.run_history(tmp_path / "missing") == []


def test_a_servo_nobody_models_is_not_reported(tmp_path):
    make_run(tmp_path, "x", records=[rec(1, 1, 1.0, 1.0)], servo="dynamixel")
    assert RH.run_history(tmp_path)[0]["servo"] is None
