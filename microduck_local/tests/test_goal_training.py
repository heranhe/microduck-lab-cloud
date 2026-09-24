"""Goal stopping must certify actions, not a budget or a clean process exit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from microduck_local import goal_training as G
from microduck_local import train_behavior as TB


def test_goal_decisions():
    goal = G.record_evaluation(G.new_goal(), {"holds": [5.0] * 8 + [4.98] * 2}, 1)
    assert goal["status"] == "training" and goal["successes"] == 8
    passed = G.record_evaluation(goal, {"holds": [5.0] * 9 + [0]}, G.MAX_GOAL_STEPS)
    assert passed["status"] == "passed" and G.goal_job_status(passed) == "done"
    for _ in range(G.STALL_EVALUATIONS):
        goal = G.record_evaluation(goal, {"holds": [0.0] * 10}, 2)
    assert goal["status"] == "stalled" and G.goal_job_status(goal) == "stopped"
    limit = G.record_evaluation(G.new_goal(), {"holds": [0.0] * 10}, G.MAX_GOAL_STEPS)
    assert limit["status"] == "limit" and G.goal_job_status(limit) == "stopped"
    progress = G.record_evaluation(G.new_goal(), {"holds": [0.0] * 10}, 1)
    progress = G.record_evaluation(progress, {"holds": [0.0] * 10}, 2)
    assert progress["stale"] == 1
    progress = G.record_evaluation(progress, {"holds": [0.5] * 10}, 3)
    assert progress["stale"] == 0
    for invalid in ([5] * 9, [float("nan")] * 10, [21] * 10):
        with pytest.raises(ValueError):
            G.record_evaluation(G.new_goal(), {"holds": invalid}, 1)


def test_eval_error_stops_before_update(tmp_path, monkeypatch):
    from stable_baselines3.common.callbacks import BaseCallback

    cb = TB._progress_callback_cls(BaseCallback)(tmp_path, None, 1000, until_success=True)
    monkeypatch.setattr(cb, "_snapshot", lambda: None)
    def broken(*args):
        raise RuntimeError("invalid policy")
    monkeypatch.setattr(TB, "evaluate_policy", broken)
    assert cb._on_step() is False
    assert cb.goal["status"] == "error"
    assert G.goal_job_status(cb.goal) == "failed"
    assert cb._on_step() is False


def test_hop_drill_is_not_three_second_certification(monkeypatch):
    monkeypatch.setenv('MICRODUCK_HOP_GOAL_HOPS', '1')
    goal = G.new_goal('single_leg_hop')
    assert goal['required_hops'] == 1 and goal['hold_seconds'] == 3.
    report = {'holds': [0.]*20, 'stable_hops': [1]*18+[0]*2}
    assert G.record_evaluation(goal, report, 10)['status'] == 'passed'
    monkeypatch.setenv('MICRODUCK_HOP_GOAL_HOPS', '0')
    assert G.record_evaluation(G.new_goal('single_leg_hop'), report, 10)['status'] == 'training'
    monkeypatch.setenv('MICRODUCK_HOP_RECOVERY', '1')
    goal = G.new_goal('single_leg_hop')
    assert goal['recovery'] and goal['hold_seconds'] == .5 and goal['required'] == 18
    assert G.record_evaluation(goal, {'holds': [.5]*18+[0]*2}, 10)['status'] == 'passed'
    assert G.record_evaluation(goal, {'holds': [.5]*17+[0]*3}, 10)['status'] == 'training'
    monkeypatch.setenv('MICRODUCK_HOP_GOAL_HOPS', '1')
    with pytest.raises(ValueError):
        G.new_goal('single_leg_hop')


def test_real_training_extends_then_preserves_certified_onnx(tmp_path):
    # Real BAM/PPO/export, scripted evaluator results: this tests lifecycle,
    # not whether a 513-step policy has learned. Files stay in pytest's temp dir.
    script = '''
import hashlib, sys
from microduck_local import train_behavior as t
t.N_STEPS = 32
t.EVAL_INTERVAL = 256
def evaluate(path, round_index, weights, behavior_id):
    holds = [0.5 * round_index] * 10 if round_index < 2 else [5.0] * 9 + [0.0]
    return {"holds": holds, "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
t.evaluate_policy = evaluate
sys.argv = ["train", "one_leg_5s", "--until-success", "--envs", "1",
            "--steps", "256", "--run-name", "goal-test", "--update-device", "cpu"]
t.main()
'''
    result = subprocess.run([sys.executable, "-c", script],
        env={**os.environ, "MICRODUCK_RUNS_DIR": str(tmp_path), "MICRODUCK_OVERLAP": "0"},
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    run = tmp_path / "goal-test"
    goal = json.loads((run / "goal.json").read_text())
    records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    assert goal["status"] == "passed" and goal["round"] == 3
    assert goal["steps"] == 513  # continued past TWO 256-step practice blocks
    assert records[-1]["done"] is True and records[-1]["steps"] == 513
    assert records[-1]["total"] == 768
    assert [r["steps"] for r in records] == sorted(r["steps"] for r in records)
    for name in ("live.onnx", "policy.onnx"):
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == goal["policy_sha256"]
    assert (run / "model.zip").is_file() and (run / "vecnormalize.pkl").is_file()
