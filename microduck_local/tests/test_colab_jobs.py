"""Cloud orchestration tests do not allocate a real Colab runtime."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from microduck_local import colab_jobs

NO_SESSIONS = "[colab] No active sessions found on server.\n"
GPU = '{"device": "Tesla T4", "vram": "15 GB"}\n'
REAL_POLICY_CHECK = colab_jobs._policy_contract_error


def session_line(name, endpoint="gpu-t4-s-abc"):
    return f"[{name}] {endpoint} | Hardware: T4 | Shape: Standard | Variant: GPU\n"


def realistic(args, code=""):
    """What the real CLI prints for the calls a test does not care about."""
    if args[0] == "sessions":
        return CompletedProcess(args, 0, NO_SESSIONS, "")
    if args[0] == "exec" and "get_device_properties" in code:
        return CompletedProcess(args, 0, GPU, "")
    if args[0] == "exec" and "MICRODUCK_BOOTSTRAP_OK" in code:
        return CompletedProcess(args, 0, "MICRODUCK_BOOTSTRAP_OK\n", "")
    if args[0] == "exec" and "MICRODUCK_RESCUE_SENT" in code:
        return CompletedProcess(args, 0, "MICRODUCK_RESCUE_SENT\n", "")
    return CompletedProcess(args, 0, "", "")


def is_poll(args, kwargs):
    code = kwargs.get("code") or ""
    return args[0] == "exec" and "status.json" in code and "p.write_text" not in code


def provider_calls(calls):
    """Assertions about what we did, without the read-only `sessions` checks."""
    return [c for c in calls if (c[0] if isinstance(c, list) else c) not in ("sessions", "usage")]


@pytest.fixture(autouse=True)
def any_policy_is_valid(monkeypatch):
    # Fixtures write placeholder bytes; the contract check has its own tests.
    monkeypatch.setattr(colab_jobs, "_policy_contract_error", lambda path: None)


@pytest.fixture(autouse=True)
def fresh_account_cache():
    colab_jobs.invalidate_account_cache()
    yield
    colab_jobs.invalidate_account_cache()


class Clock:
    """Virtual wall clock: sleeping advances it, so a 3 h limit runs in ms."""

    def __init__(self, monkeypatch):
        self.now = 1_000_000.0
        monkeypatch.setattr(colab_jobs.time, "time", lambda: self.now)
        monkeypatch.setattr(colab_jobs.time, "sleep", self.sleep)

    def sleep(self, seconds):
        self.now += seconds


def test_cancelled_worker_does_not_release_twice(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job_id = "123456abcdef"
    directory = tmp_path / job_id
    directory.mkdir()
    job = {"id": job_id, "session": "test", "state": "allocating",
           "gpu": "T4", "task": "Mjlab-Velocity-Flat-MicroDuck",
           "released": False, "message": ""}
    jobs.jobs[job_id] = job
    stops = []

    def fake_cli(args, **kwargs):
        if args[0] == "stop":
            stops.append(args)
            return CompletedProcess(args, 0 if len(stops) == 1 else 1, "", "Not Found")
        return realistic(args, kwargs.get("code") or "")

    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    monkeypatch.setattr(colab_jobs.time, "sleep", lambda _: jobs.stop(job_id))
    jobs._run(job, directory, 1, 16)
    assert len(stops) == 1
    assert job["released"] and job["state"] == "stopped"
    assert job["message"] == ""


def test_concurrent_cleanup_preserves_success(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = {"session": "test", "released": False, "cancel_requested": True}
    entered = threading.Event()
    finish = threading.Event()
    calls = []

    def fake_cli(args, **kwargs):
        calls.append(args)
        entered.set()
        assert finish.wait(2)
        return realistic(args)

    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    workers = [threading.Thread(target=jobs._release, args=(job,)) for _ in range(2)]
    workers[0].start()
    assert entered.wait(2)
    workers[1].start()
    finish.set()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()
    assert provider_calls(calls) == [["stop", "-s", "test"]]
    assert job["released"] and job["released_at"] > 0


@pytest.mark.parametrize("status_code,output,expected", [
    (0, "[colab] Session 'test' not found.\n", True),
    (0, "test | BUSY", False),
    (1, "[colab] Session 'test' not found.\n", False),
])
def test_unassign_not_found_requires_confirmed_absence(tmp_path, monkeypatch, status_code, output, expected):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = {"session": "test", "released": False, "cancel_requested": True, "message": "old error"}
    calls = []

    def fake_cli(args, **kwargs):
        calls.append(args)
        if args[0] == "stop":
            return CompletedProcess(args, 1, "", "GET https://colab.research.google.com/tun/m/unassign/endpoint: Not Found")
        if args[0] == "sessions":
            return realistic(args)
        return CompletedProcess(args, status_code, output, "")

    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs._release(job)
    assert job["released"] is expected
    assert provider_calls(calls) == [["stop", "-s", "test"], ["status", "-s", "test"]]
    if expected:
        assert job["message"] == ""
    else:
        assert "released_at" not in job


def test_release_timeout_remains_retryable(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = {"session": "test", "released": False, "cancel_requested": True}

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(colab_jobs, "cli", timeout)
    jobs._release(job)
    assert not job["released"]
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kwargs: realistic(args))
    jobs._release(job)
    assert job["released"] and job["message"] == ""


@pytest.fixture(autouse=True)
def available_teachers(monkeypatch, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("teachers")
    for run, (variable, name) in colab_jobs.TEACHERS.items():
        path = tmp_path / name
        path.write_bytes(b"fixture checkpoint")
        monkeypatch.setenv(variable, str(path))


def test_missing_teacher_rejects_before_account_or_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("MICRODUCK_STAND_TEACHER", str(tmp_path / "missing.pt"))
    def unexpected(*a, **k):
        pytest.fail("CLI must not run without prerequisites")
    monkeypatch.setattr(colab_jobs, "cli", unexpected)
    with pytest.raises(RuntimeError, match="尚未分配 GPU"):
        colab_jobs.ColabJobs(tmp_path).start("Mjlab-VelStand-Flat-MicroDuck", "T4", 1, 16)



@pytest.mark.parametrize("recover", [True, False])
def test_poll_timeout_recovers_or_releases_after_bounded_failures(tmp_path, monkeypatch, recover):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = {"id": "123456abcdef", "session": "test", "state": "allocating",
           "gpu": "T4", "task": "Mjlab-VelStand-Flat-MicroDuck", "released": False, "message": ""}
    polls = 0
    stops = []
    def fake_cli(args, **kwargs):
        nonlocal polls
        if is_poll(args, kwargs):
            polls += 1
            if recover and polls == 2:
                return CompletedProcess(args, 0, '{"state":"done"}', "")
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if args[0] == "download" and args[-1].endswith("policy.onnx"):
            (tmp_path / "policy.onnx").write_bytes(b"test model")
        if args[0] == "stop":
            stops.append(polls)
        return realistic(args, kwargs.get("code") or "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    Clock(monkeypatch)
    jobs._run(job, tmp_path, 1, 16)
    assert job["released"]
    # An outage has to outlast the grace period before paid training is given up on.
    expected = 2 if recover else colab_jobs.POLL_GRACE_S // colab_jobs.POLL_INTERVAL_S + 1
    assert polls == expected
    assert stops == [polls]
    assert job["state"] == ("done" if recover else "failed")
    if recover:
        assert job["poll_failures"] == 0 and job["last_heartbeat_at"] > 0


def test_account_status_requires_existing_auth(monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    monkeypatch.setattr(colab_jobs, "cli", lambda *a, **k: CompletedProcess(a, 1, "", "not signed in"))
    status = colab_jobs.account_status()
    assert status["installed"] and not status["connected"]
    assert "colab usage" in status["message"]


def test_account_status_reads_real_balance(monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    monkeypatch.setattr(colab_jobs, "cli", lambda *a, **k: CompletedProcess(
        a, 0, "Current balance: 200.0 compute units\nUsage rate: 1.96/hr\n", ""))
    status = colab_jobs.account_status()
    assert status["connected"] and status["balance"] == 200.0
    assert status["rate"] == 1.96


def test_remote_runner_uses_official_cli_flags():
    import ast
    script = colab_jobs._remote_runner("Mjlab-Velocity-Flat-MicroDuck", 10, 64)
    assert 'str(entry), "train", "Mjlab-Velocity-Flat-MicroDuck"' in script
    assert 'str(entry), "export"' in script
    assert colab_jobs.UPSTREAM_REVISION in script
    assert '"--checkpoint-file"' in script
    assert 'root.mkdir(parents=True, exist_ok=True)' in script
    compile(script, "remote_runner.py", "exec")
    tree = ast.parse(script)
    entries = [node.args[0].value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "write_text" and node.args
               and isinstance(node.args[0], ast.Constant)
               and isinstance(node.args[0].value, str)]
    assert len(entries) == 1
    compile(entries[0], "entry.py", "exec")


def test_teacher_upload_failure_releases_without_training(tmp_path, monkeypatch):
    job = {"id": "abcdef123456", "session": "test", "state": "allocating",
           "gpu": "T4", "task": "Mjlab-VelStand-Flat-MicroDuck", "released": False, "message": ""}
    calls = []
    def fake_cli(args, **kwargs):
        calls.append(args[0])
        if args[0] == "upload":
            return CompletedProcess(args, 1, "", "")
        return realistic(args, kwargs.get("code") or "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 1, 16)
    assert provider_calls(calls) == ["new", "upload", "stop"]
    assert job["state"] == "failed" and job["released"]


def test_account_identity_uses_cli_email_only(monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    def fake_cli(args, **kwargs):
        output = ("Email:         duck@example.com\nAudience: secret\nScopes: private\n"
                  if args == ["whoami"] else "Current balance: 0\nUsage rate: 0/hr\n")
        return CompletedProcess(args, 0, output, "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    status = colab_jobs.account_status()
    assert status["email"] == "duck@example.com"
    assert "secret" not in str(status) and "private" not in str(status)


def test_identity_failure_keeps_connected_account(monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    def fake_cli(args, **kwargs):
        if args == ["whoami"]:
            return CompletedProcess(args, 1, "Email: wrong@example.com", "lookup failed")
        return CompletedProcess(args, 0, "Current balance: 0\nUsage rate: 0/hr\n", "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    status = colab_jobs.account_status()
    assert status["connected"] and status["email"] is None


def test_rejects_unlisted_task_before_any_gpu_allocation(tmp_path):
    jobs = colab_jobs.ColabJobs(tmp_path)
    try:
        jobs.start("malicious", "T4", 10, 64)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported task was accepted")
    assert not list(tmp_path.iterdir())


def test_recovered_job_can_still_release_gpu(tmp_path, monkeypatch):
    directory = tmp_path / "012345abcdef"
    directory.mkdir()
    (directory / "job.json").write_text('{"id":"012345abcdef","session":"microduck-012345abcdef","state":"running"}')
    stopped = []
    def fake_cli(args, **kwargs):
        stopped.append(args)
        return realistic(args)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs = colab_jobs.ColabJobs(tmp_path)
    assert jobs.list()[0]["state"] == "detached"
    assert jobs.stop("012345abcdef")["released"]
    assert provider_calls(stopped) == [["stop", "-s", "microduck-012345abcdef"]]


def test_zero_balance_allows_free_t4_but_rejects_paid_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 0})
    jobs = colab_jobs.ColabJobs(tmp_path)
    try:
        jobs.start("Mjlab-Velocity-Flat-MicroDuck", "A100", 10, 64)
    except RuntimeError as exc:
        assert "Select T4" in str(exc)
    else:
        raise AssertionError("paid GPU was accepted without paid compute units")
    assert not list(tmp_path.iterdir())

    # Test only the local decision. Do not allocate a real Colab GPU here.
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    job = jobs.start("Mjlab-VelStand-Flat-MicroDuck", "T4", 10, 16)
    assert job["gpu"] == "T4" and job["state"] == "allocating"
    assert job["released"] is False


def test_free_t4_quota_rejection_does_not_leave_phantom_session(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 0})
    calls = []
    def fake_cli(args, **kwargs):
        calls.append(args[0])
        if args[0] == "new":
            return CompletedProcess(args, 1, "", "Free GPU quota unavailable")
        return CompletedProcess(args, 1, "", "No such session")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = jobs.start("Mjlab-VelStand-Flat-MicroDuck", "T4", 10, 16)
    jobs.workers[job["id"]].join(timeout=2)
    failed = jobs.list()[0]
    assert failed["state"] == "failed" and failed["released"] is True
    assert "quota unavailable" in failed["message"]
    assert provider_calls(calls) == ["new", "stop"]


def test_stop_all_includes_finalizing_and_retries_failed_release(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    for job_id, state in (("111111111111", "finalizing"), ("222222222222", "failed")):
        directory = tmp_path / job_id
        directory.mkdir()
        jobs.jobs[job_id] = {"id": job_id, "session": f"microduck-{job_id}",
                             "state": state, "released": False, "message": ""}
    calls = []
    def fake_cli(args, **kwargs):
        calls.append(args)
        return realistic(args)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    stopped = jobs.stop_all()
    assert len(stopped) == 2
    assert all(job["released"] for job in stopped)
    assert provider_calls(calls) == [["stop", "-s", "microduck-111111111111"],
                                     ["stop", "-s", "microduck-222222222222"]]


def test_stop_during_allocation_stays_pending_until_allocation_thread_releases(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    finish_new = threading.Event()
    calls = []
    def fake_cli(args, **kwargs):
        calls.append(args)
        if args[0] == "new":
            entered.set()
            assert finish_new.wait(2)
        return realistic(args, kwargs.get("code") or "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 200})
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = jobs.start("Mjlab-Velocity-Flat-MicroDuck", "T4", 10, 64, confirm_paid=True)
    assert entered.wait(2)
    assert jobs.stop(job["id"])["state"] == "stopping"
    finish_new.set()
    for _ in range(100):
        current = jobs.list()[0]
        if current["state"] == "stopped" and current.get("released"):
            break
        threading.Event().wait(0.01)
    assert current["state"] == "stopped" and current["released"]
    assert current["allocated_at"] >= current["started"]
    assert provider_calls([call[0] for call in calls]) == ["new", "stop"]


def test_stop_during_gpu_probe_never_bootstraps_training(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job_id = "123456abcdef"
    directory = tmp_path / job_id
    directory.mkdir()
    job = {"id": job_id, "session": "test-session", "state": "allocating",
           "gpu": "T4", "task": "Mjlab-VelStand-Flat-MicroDuck",
           "released": False, "message": ""}
    jobs.jobs[job_id] = job
    calls = []
    def fake_cli(args, **kwargs):
        calls.append(args[0])
        if args[0] == "exec":
            assert "get_device_properties" in kwargs["code"], "training restarted after cancellation"
            jobs.stop(job_id)
            return CompletedProcess(args, 0, "", "")
        return realistic(args)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs._run(job, directory, 10, 8)
    assert provider_calls(calls) == ["new", "upload", "upload", "exec", "stop"]
    assert job["state"] == "stopped" and job["released"]


def test_shutdown_waits_for_allocating_session_to_be_released(tmp_path, monkeypatch):
    import threading

    entered = threading.Event()
    finish_new = threading.Event()
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 200})
    def fake_cli(args, **kwargs):
        if args[0] == "new":
            entered.set()
            assert finish_new.wait(2)
        return realistic(args, kwargs.get("code") or "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs = colab_jobs.ColabJobs(tmp_path)
    jobs.start("Mjlab-VelStand-Flat-MicroDuck", "T4", 10, 8, confirm_paid=True)
    assert entered.wait(2)
    original = jobs.stop_all
    def stop_all():
        result = original()
        finish_new.set()
        return result
    monkeypatch.setattr(jobs, "stop_all", stop_all)
    result = jobs.shutdown()
    assert result[0]["state"] == "stopped" and result[0]["released"]
    assert not any(worker.is_alive() for worker in jobs.workers.values())


# --- billing regressions ------------------------------------------------------
# Each of these reproduced against the previous orchestrator before it was fixed.


def job_dict(job_id="abcdefabcdef", **extra):
    return {"id": job_id, "session": f"microduck-{job_id}", "state": "allocating", "gpu": "T4",
            "task": "Mjlab-Velocity-Flat-MicroDuck", "released": False, "message": "", **extra}


def scripted(tmp_path, statuses, calls=None, on_download=None):
    """A CLI whose polls walk a list of remote statuses (the last one repeats)."""
    polls = iter(statuses)
    last = {"value": statuses[-1]}
    def fake_cli(args, **kwargs):
        code = kwargs.get("code") or ""
        if calls is not None:
            calls.append(args[0] if args[0] != "exec" else ("rescue" if "MICRODUCK_RESCUE_SENT" in code else "exec"))
        if is_poll(args, kwargs):
            last["value"] = next(polls, last["value"])
            value = last["value"]
            return CompletedProcess(args, 0, json.dumps(value) + "\n", "") if isinstance(value, dict) else value
        if args[0] == "download":
            if on_download:
                return on_download(args)
            Path(args[-1]).write_bytes(b"artifact")
        return realistic(args, code)
    return fake_cli


def test_lost_new_is_not_reported_released_while_google_lists_it(tmp_path, monkeypatch):
    # `colab new` killed mid-assignment leaves no local state, so `stop` exits 0
    # with "not found" while the runtime keeps billing as an unnamed session.
    asked = {"n": 0}
    def fake_cli(args, **kwargs):
        if args[0] == "new":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if args[0] == "stop":
            return CompletedProcess(args, 0, "[colab] Session 'x' not found.\n", "")
        if args[0] == "sessions":
            asked["n"] += 1
            return CompletedProcess(args, 0, NO_SESSIONS if asked["n"] == 1 else session_line("?"), "")
        return realistic(args)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 1, 16)
    assert job["state"] == "failed"
    assert job["released"] is False, "released while Google still bills an unnamed runtime"
    assert "仍显示" in job["message"]


def test_stop_success_alone_does_not_count_as_released(tmp_path, monkeypatch):
    listed = session_line("microduck-abcdefabcdef")
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw:
                        CompletedProcess(args, 0, listed, "") if args[0] == "sessions" else realistic(args))
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._release(job)
    assert job["released"] is False and "released_at" not in job


def test_an_unreachable_google_never_proves_a_release(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw:
                        CompletedProcess(args, 1, "", "") if args[0] == "sessions" else realistic(args))
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._release(job)
    assert job["released"] is False and "确认" in job["message"]


def test_failed_release_is_retried_by_the_reconciler_with_backoff(tmp_path, monkeypatch):
    calls = []
    outage = {"on": True}
    def fake_cli(args, **kwargs):
        calls.append(args[0])
        if args[0] == "stop" and outage["on"]:
            return CompletedProcess(args, 1, "", "network unreachable")
        if is_poll(args, kwargs):
            return CompletedProcess(args, 0, '{"state":"failed","message":"boom"}\n', "")
        return realistic(args, kwargs.get("code") or "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    clock = Clock(monkeypatch)
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = job_dict()
    jobs.jobs[job["id"]] = job
    jobs._run(job, tmp_path / job["id"], 1, 16)
    assert job["released"] is False
    assert calls.count("stop") == 1 + len(colab_jobs.RELEASE_RETRY_DELAYS_S), "the worker should retry first"
    attempts = calls.count("stop")
    jobs.reconcile()
    assert calls.count("stop") == attempts + 1, "nobody retried after the worker gave up"
    assert job["released"] is False and job["next_release_at"] > clock.now
    jobs.reconcile()
    assert calls.count("stop") == attempts + 1, "backoff ignored"
    outage["on"] = False
    clock.now = job["next_release_at"] + 1
    jobs.reconcile()
    assert job["released"] is True


def test_download_retries_before_paid_results_are_released(tmp_path, monkeypatch):
    calls = []
    attempts = {"policy.onnx": 0}
    def download(args):
        name = Path(args[-1]).name
        if name == "policy.onnx":
            attempts[name] += 1
            if attempts[name] == 1:
                raise subprocess.TimeoutExpired(args, 300)
        Path(args[-1]).write_bytes(b"artifact")
        return CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [{"state": "done"}], calls, download))
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 1, 16)
    assert job["state"] == "done" and job["released"] and attempts["policy.onnx"] == 2
    assert calls.index("stop") > len(calls) - 1 - calls[::-1].index("download"), "released before downloading"


def test_a_truncated_download_is_not_accepted(tmp_path, monkeypatch):
    status = {"state": "done", "files": {"policy.onnx": 100, "checkpoint.pt": 8}}
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [status]))  # writes 8 bytes
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 1, 16)
    assert job["state"] == "failed" and "policy.onnx" in job["message"]
    assert not (tmp_path / "policy.onnx").exists()
    assert job["artifacts"] == ["checkpoint.pt"]


def test_time_limit_saves_progress_then_releases(tmp_path, monkeypatch):
    calls = []
    training = {"state": "training", "log_age": 1.0, "iteration": 40, "iterations": 100}
    rescued = {"state": "rescued", "message": "budget", "files": {"checkpoint.pt": 8, "policy.onnx": 8}}
    limit_polls = 5 * 60 // colab_jobs.POLL_INTERVAL_S
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [training] * limit_polls + [rescued], calls))
    Clock(monkeypatch)
    job = job_dict(max_minutes=5)
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert "rescue" in calls, "the runner was never told to save"
    assert job["stop_reason"] == "budget" and job["state"] == "stopped" and job["released"] is True
    assert "5 分钟" in job["message"] and "policy.onnx" in job["artifacts"]
    assert calls.index("rescue") < calls.index("download") < calls.index("stop")


def test_time_limit_applies_even_if_the_save_never_completes(tmp_path, monkeypatch):
    training = {"state": "training", "log_age": 1.0}
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [training]))
    clock = Clock(monkeypatch)
    job = job_dict(max_minutes=5)
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["released"] is True and job["rescue_timed_out"]
    # 5 min budget + the rescue window + a little polling, and never hours.
    assert clock.now - job["allocated_at"] < 5 * 60 + colab_jobs.RESCUE_TIMEOUT_S + 120


def test_compute_unit_limit_stops_a_paid_job(tmp_path, monkeypatch):
    training = {"state": "training", "log_age": 1.0}
    rescued = {"state": "rescued", "files": {}}
    cli_steps = scripted(tmp_path, [training, rescued])
    def fake_cli(args, **kwargs):
        if args[0] == "usage":
            return CompletedProcess(args, 0, "Current balance: 37.50 compute units\nUsage rate: 11.77/hr\n", "")
        return cli_steps(args, **kwargs)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    Clock(monkeypatch)
    job = job_dict(balance_start=50.0, max_ccu=10.0)
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["ccu_spent"] == 12.5 and job["rate"] == 11.77
    assert job["stop_reason"] == "ccu" and job["released"] is True


def test_silent_training_log_is_a_stall(tmp_path, monkeypatch):
    stuck = {"state": "training", "log_age": colab_jobs.STALL_TIMEOUT_S + 1.0}
    rescued = {"state": "rescued", "files": {}}
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [stuck, rescued]))
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["stop_reason"] == "stall" and job["state"] == "failed" and job["released"] is True


def test_setup_that_never_finishes_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [{"state": "setup"}]))
    clock = Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["state"] == "failed" and job["released"] is True
    assert clock.now - job["allocated_at"] < colab_jobs.SETUP_TIMEOUT_S + 120


def test_runner_that_never_started_is_released(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [{"state": "missing"}]))
    clock = Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["state"] == "failed" and job["released"] is True
    assert clock.now - job["allocated_at"] < colab_jobs.BOOTSTRAP_GRACE_S + 120


def test_reclaimed_session_ends_the_job_at_once(tmp_path, monkeypatch):
    gone = CompletedProcess(["exec"], 1, "[colab] Session 'x' appears to be lost (404/401). Cleaning up.\n", "")
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [gone]))
    clock = Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["state"] == "failed" and "回收" in job["message"] and job["released"] is True
    assert clock.now - job["allocated_at"] < 120


def test_training_refuses_a_runtime_without_a_gpu(tmp_path, monkeypatch):
    # `colab exec` exits 0 even though the remote code raised.
    calls = []
    def fake_cli(args, **kwargs):
        code = kwargs.get("code") or ""
        calls.append("bootstrap" if "MICRODUCK_BOOTSTRAP_OK" in code else args[0])
        if args[0] == "exec":
            return CompletedProcess(args, 0, "", "RuntimeError: No CUDA GPUs are available")
        return realistic(args, code)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert "bootstrap" not in calls, "started training on a runtime with no GPU"
    assert job["state"] == "failed" and "GPU" in job["message"] and job["released"] is True


def test_bootstrap_failure_is_not_silently_treated_as_started(tmp_path, monkeypatch):
    def fake_cli(args, **kwargs):
        code = kwargs.get("code") or ""
        if "MICRODUCK_BOOTSTRAP_OK" in code:
            return CompletedProcess(args, 0, "", "PermissionError: /content")
        return realistic(args, code)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 100, 64)
    assert job["state"] == "failed" and "启动训练进程" in job["message"] and job["released"] is True


def recovered_jobs(tmp_path, **fields):
    job = job_dict(state="running", allocated_at=time.time(), **fields)
    directory = tmp_path / job["id"]
    directory.mkdir()
    (directory / "job.json").write_text(json.dumps(job))
    return colab_jobs.ColabJobs(tmp_path), job["id"]


def test_lab_restart_readopts_a_running_session_and_finishes_it(tmp_path, monkeypatch):
    jobs, job_id = recovered_jobs(tmp_path)
    assert jobs.jobs[job_id]["state"] == "detached"
    listed = {"on": True}
    steps = scripted(tmp_path, [{"state": "done"}])
    def fake_cli(args, **kwargs):
        if args[0] == "sessions":
            return CompletedProcess(args, 0, session_line(f"microduck-{job_id}") if listed["on"] else NO_SESSIONS, "")
        if args[0] == "stop":
            listed["on"] = False
        return steps(args, **kwargs)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    monkeypatch.setattr(colab_jobs.time, "sleep", lambda _: None)
    jobs.reconcile()
    jobs.workers[job_id].join(timeout=5)
    job = jobs.jobs[job_id]
    assert job["adopted_at"] and job["state"] == "done" and job["released"] is True
    assert (tmp_path / job_id / "policy.onnx").is_file()


def test_lab_restart_after_the_session_ended_settles_the_job(tmp_path, monkeypatch):
    jobs, job_id = recovered_jobs(tmp_path)
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: (
        CompletedProcess(args, 0, f"[colab] Session 'microduck-{job_id}' not found.\n", "")
        if args[0] == "stop" else realistic(args)))
    jobs.reconcile()
    assert jobs.jobs[job_id]["state"] == "failed" and jobs.jobs[job_id]["released"] is True


def test_lab_restart_without_google_keeps_the_job_visibly_unreleased(tmp_path, monkeypatch):
    jobs, job_id = recovered_jobs(tmp_path)
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, 1, "", "invalid_grant"))
    jobs.reconcile()
    job = jobs.jobs[job_id]
    assert job["state"] == "detached" and job["released"] is False and "计费" in job["message"]
    assert jobs.server["ok"] is False


def test_adopted_job_still_honours_the_time_limit_it_missed(tmp_path, monkeypatch):
    jobs, job_id = recovered_jobs(tmp_path, max_minutes=5)
    jobs.jobs[job_id]["allocated_at"] -= 3600  # the lab was down for an hour
    listed = {"on": True}
    steps = scripted(tmp_path, [{"state": "training", "log_age": 1.0}, {"state": "rescued", "files": {}}])
    def fake_cli(args, **kwargs):
        if args[0] == "sessions":
            return CompletedProcess(args, 0, session_line(f"microduck-{job_id}") if listed["on"] else NO_SESSIONS, "")
        if args[0] == "stop":
            listed["on"] = False
        return steps(args, **kwargs)
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    monkeypatch.setattr(colab_jobs.time, "sleep", lambda _: None)
    jobs.reconcile()
    jobs.workers[job_id].join(timeout=5)
    assert jobs.jobs[job_id]["stop_reason"] == "budget" and jobs.jobs[job_id]["released"] is True


def test_save_and_disconnect_keeps_the_checkpoint(tmp_path, monkeypatch):
    calls = []
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = job_dict()
    jobs.jobs[job["id"]] = job
    def stop_while_training(_):
        if job.get("remote_state") == "training" and not job.get("rescue_requested"):
            jobs.workers[job["id"]] = threading.current_thread()
            assert jobs.stop(job["id"], rescue=True)["state"] == "saving"
    monkeypatch.setattr(colab_jobs.time, "sleep", stop_while_training)
    statuses = [{"state": "training", "log_age": 1.0}, {"state": "training", "log_age": 1.0},
                {"state": "rescued", "files": {"checkpoint.pt": 8}}]
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, statuses, calls))
    jobs._run(job, tmp_path / job["id"], 100, 64)
    assert "rescue" in calls and calls.index("download") < calls.index("stop")
    assert job["state"] == "stopped" and job["released"] is True and job["artifacts"] == ["checkpoint.pt"]


def test_immediate_disconnect_wins_over_a_save_in_progress(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: (calls.append(args[0]), realistic(args))[1])
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = job_dict(state="saving", remote_state="saving", rescue_requested=True)
    jobs.jobs[job["id"]] = job
    assert jobs.stop(job["id"])["released"] is True
    assert "stop" in calls


@pytest.mark.parametrize("balance", [50.0, None])
def test_a_job_that_may_spend_money_needs_explicit_confirmation(tmp_path, monkeypatch, balance):
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": balance})
    monkeypatch.setattr(colab_jobs, "cli", lambda *a, **k: pytest.fail("allocated without confirmation"))
    with pytest.raises(colab_jobs.PaidConfirmationRequired):
        colab_jobs.ColabJobs(tmp_path).start("Mjlab-Velocity-Flat-MicroDuck", "T4", 10, 64)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("minutes", [0, 4, 721])
def test_time_limit_must_be_sane(tmp_path, monkeypatch, minutes):
    monkeypatch.setattr(colab_jobs, "account_status", lambda: pytest.fail("checked before validating"))
    with pytest.raises(ValueError, match="分钟"):
        colab_jobs.ColabJobs(tmp_path).start("Mjlab-Velocity-Flat-MicroDuck", "T4", 10, 64, minutes)


def test_job_record_survives_on_disk_as_valid_json_without_scratch_files(tmp_path):
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = job_dict()
    for state in ("allocating", "running"):
        job["state"] = state
        jobs._save(job)
    files = sorted(p.name for p in (tmp_path / job["id"]).iterdir())
    assert files == ["job.json"]
    assert json.loads((tmp_path / job["id"] / "job.json").read_text())["state"] == "running"


# --- account and Google's view ------------------------------------------------


REAL_UPDATE_NOTICE = ("[colab] A new version of Colab CLI is available: 0.7.5 (current: 0.7.4)\n"
                      "[colab] Run 'colab update' to update.\n")


def test_sessions_keep_could_not_ask_distinct_from_none(monkeypatch):
    results = {"out": REAL_UPDATE_NOTICE + NO_SESSIONS, "rc": 0}
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, results["rc"], results["out"], ""))
    assert colab_jobs.server_sessions() == []
    results["out"] = REAL_UPDATE_NOTICE + session_line("?", "gpu-a") + session_line("microduck-1", "gpu-b")
    assert colab_jobs.server_sessions() == [
        {"name": "?", "endpoint": "gpu-a", "hardware": "T4"},
        {"name": "microduck-1", "endpoint": "gpu-b", "hardware": "T4"}]
    results["out"] = REAL_UPDATE_NOTICE
    assert colab_jobs.server_sessions() is None, "an unparseable answer is not an empty list"
    results.update(out=NO_SESSIONS, rc=1)
    assert colab_jobs.server_sessions() is None


@pytest.mark.parametrize("stderr,stdout,reason", [
    ("To authorize colab-cli, visit this URL in any browser:", "Enter the authorization code:", "auth_expired"),
    ("('invalid_grant: Bad Request', {})", "", "auth_expired"),
    ("[colab] Warning: failed to fetch compute-unit info: Max retries exceeded", "", "network"),
    ("something odd", "", "unknown"),
])
def test_account_failures_say_what_actually_went_wrong(monkeypatch, stderr, stdout, reason):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, 1, stdout, stderr))
    status = colab_jobs.account_status()
    assert status["connected"] is False and status["reason"] == reason


def test_account_reports_active_assignments(monkeypatch):
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(
        args, 0, "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 2\n", ""))
    assert colab_jobs.account_status()["active_assignments"] == 2


def test_panels_polling_does_not_spawn_the_cli_each_time(monkeypatch):
    calls = []
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: "/tmp/colab")
    def fake_cli(args, **kw):
        calls.append(args[0])
        return CompletedProcess(args, 1, "", "To authorize colab-cli")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    for _ in range(5):
        assert colab_jobs.cached_account_status()["reason"] == "auth_expired"
    assert calls == ["usage"], "a dead token must not be retried on every poll"
    colab_jobs.cached_account_status(force=True)
    assert calls == ["usage", "usage"]


def test_friendly_error_drops_update_noise_and_tracebacks():
    assert colab_jobs.friendly_cli_error(REAL_UPDATE_NOTICE + "Free GPU quota unavailable") == "Free GPU quota unavailable"
    boxed = "╭─── Traceback (most recent call last) ───╮\n│ execution.py:230 │\n╰──────╯\nValueError: real cause\n"
    assert colab_jobs.friendly_cli_error(boxed) == "ValueError: real cause"
    assert "授权已失效" in colab_jobs.friendly_cli_error("Enter the authorization code:")


def test_reconcile_reports_runtimes_no_job_owns(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    jobs.jobs["aaaaaaaaaaaa"] = job_dict("aaaaaaaaaaaa", released=False, state="failed", next_release_at=time.time() + 999)
    output = (session_line("microduck-aaaaaaaaaaaa", "gpu-ours") + session_line("?", "gpu-unnamed")
              + session_line("microduck-ffffffffffff", "gpu-forgotten") + session_line("my-own-work", "gpu-mine"))
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, 0, output, ""))
    jobs.reconcile()
    assert jobs.server["ok"] and jobs.server["count"] == 4
    assert [o["endpoint"] for o in jobs.server["orphans"]] == ["gpu-unnamed", "gpu-forgotten"]
    assert [o["endpoint"] for o in jobs.server["others"]] == ["gpu-mine"]


def test_an_unexplained_active_runtime_triggers_a_prompt_reconcile(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    monkeypatch.setattr(colab_jobs, "RECONCILE_TICK_S", 0.01)
    monkeypatch.setattr(colab_jobs, "RECONCILE_IDLE_S", 3600)
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, 0, session_line("?", "gpu-x"), ""))
    jobs.start_reconciler(keep_awake=False)
    try:
        for _ in range(200):
            if jobs.server["count"] == 1:
                break
            time.sleep(0.01)
        assert jobs.server["count"] == 1, "first look never happened"
        monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(
            args, 0, session_line("?", "gpu-x") + session_line("?", "gpu-y"), ""))
        jobs.note_account({"connected": True, "active_assignments": 1})  # agrees: no extra look
        time.sleep(0.1)
        assert jobs.server["count"] == 1
        jobs.note_account({"connected": True, "active_assignments": 2})
        for _ in range(200):
            if jobs.server["count"] == 2:
                break
            time.sleep(0.01)
        assert jobs.server["count"] == 2, "waited for the idle interval instead of looking now"
    finally:
        jobs._stop_reconciler.set()


def test_orphan_release_only_touches_a_listed_unowned_runtime(tmp_path, monkeypatch):
    jobs = colab_jobs.ColabJobs(tmp_path)
    jobs.jobs["aaaaaaaaaaaa"] = job_dict("aaaaaaaaaaaa")
    output = session_line("microduck-aaaaaaaaaaaa", "gpu-ours") + session_line("?", "gpu-unnamed")
    monkeypatch.setattr(colab_jobs, "cli", lambda args, **kw: CompletedProcess(args, 0, output, ""))
    unassigned = []
    monkeypatch.setattr(colab_jobs.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(colab_jobs.subprocess, "run", lambda cmd, **kw: (unassigned.append(cmd[-1]), CompletedProcess(cmd, 0, "", ""))[1])
    with pytest.raises(ValueError):
        jobs.release_orphan("gpu-ours")
    jobs.release_orphan("gpu-never-listed")
    assert unassigned == []
    jobs.release_orphan("gpu-unnamed")
    assert unassigned == ["gpu-unnamed"]


def test_keep_awake_follows_unreleased_jobs(tmp_path, monkeypatch):
    spawned = []
    class Proc:
        def __init__(self, cmd, **kw):
            self.cmd, self.alive = cmd, True
            spawned.append(self)
        def poll(self):
            return None if self.alive else 0
        def terminate(self):
            self.alive = False
    monkeypatch.setattr(colab_jobs.sys, "platform", "darwin")
    monkeypatch.setattr(colab_jobs.shutil, "which", lambda name: "/usr/bin/caffeinate")
    monkeypatch.setattr(colab_jobs.subprocess, "Popen", Proc)
    jobs = colab_jobs.ColabJobs(tmp_path)
    jobs.keep_awake = True
    jobs._sync_awake()
    assert spawned == [], "kept the Mac awake with nothing running"
    job = job_dict()
    jobs.jobs[job["id"]] = job
    jobs._sync_awake()
    jobs._sync_awake()
    assert len(spawned) == 1 and spawned[0].cmd[:2] == ["caffeinate", "-i"]
    job["released"] = True
    jobs._sync_awake()
    assert spawned[0].alive is False


# --- policy contract ----------------------------------------------------------


def write_policy(path, obs=61, actions=14):
    import onnx
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["obs"], ["actions"])] if obs == actions else
        [helper.make_node("Gemm", ["obs", "weights"], ["actions"])],
        "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, actions])],
        [helper.make_tensor("weights", TensorProto.FLOAT, [obs, actions], [0.0] * (obs * actions))],
    )
    onnx.save(helper.make_model(graph), str(path))


def test_policy_contract_accepts_the_deployment_shape_and_rejects_others(tmp_path):
    good, junk = tmp_path / "good.onnx", tmp_path / "junk.onnx"
    write_policy(good)
    junk.write_bytes(b"not a model")
    assert REAL_POLICY_CHECK(good) is None
    assert "校验失败" in REAL_POLICY_CHECK(junk)
    # Each side alone must be enough to reject: a G1 policy is 99 -> 29.
    for obs, actions in ((99, 29), (99, 14), (61, 29)):
        wrong = tmp_path / f"wrong-{obs}-{actions}.onnx"
        write_policy(wrong, obs=obs, actions=actions)
        assert "部署约定" in REAL_POLICY_CHECK(wrong), (obs, actions)


def test_invalid_export_does_not_count_as_a_finished_job(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "_policy_contract_error", REAL_POLICY_CHECK)
    monkeypatch.setattr(colab_jobs, "cli", scripted(tmp_path, [{"state": "done"}]))  # 8 junk bytes
    Clock(monkeypatch)
    job = job_dict()
    colab_jobs.ColabJobs(tmp_path)._run(job, tmp_path, 1, 16)
    assert job["state"] == "failed" and "ONNX" in job["message"] and job["released"] is True
    assert "policy.onnx" not in job["artifacts"]


# --- the code that actually runs on the runtime -------------------------------
# Executed for real against a fake `uv` and `git`: a mistake in these strings
# would otherwise only surface on a GPU someone is paying for.


LOG_BLOCK = ("\x1b[1m                 Learning iteration 41/1000                 \x1b[0m \n"
             "                            Mean reward: -0.13\n"
             "                         Iteration time: 2.10s\n"
             "                           Time elapsed: 00:01:30\n"
             "                                    ETA: 00:33:12\n")


def run_poll(root):
    result = subprocess.run([sys.executable, "-c", colab_jobs._poll_code(str(root))],
                            capture_output=True, text=True, timeout=30, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, "one JSON line is the whole protocol"
    return json.loads(lines[0])


def test_poll_code_reads_live_progress_from_the_real_log_format(tmp_path):
    (tmp_path / "status.json").write_text('{"state":"training","message":""}')
    (tmp_path / "train.log").write_text(LOG_BLOCK.replace("41/", "40/") + LOG_BLOCK)
    out = run_poll(tmp_path)
    assert out["state"] == "training" and out["iteration"] == 41 and out["iterations"] == 1000
    assert out["reward"] == "-0.13" and out["iter_time"] == "2.10"
    assert out["eta"] == "00:33:12" and out["elapsed"] == "00:01:30"
    assert 0 <= out["log_age"] < 60
    assert out["files"]["train.log"] > 0


def test_poll_code_says_missing_before_the_runner_writes_anything(tmp_path):
    assert run_poll(tmp_path / "nowhere") == {"state": "missing", "files": {}}


FAKE_TOOL = """#!/bin/sh
case "$*" in
  *clone*) mkdir -p "$(eval echo \\${$#})"; echo cloned ;;
  *"git"*) ;;
esac
case "$0 $*" in
  *uv*" train "*)
    mkdir -p logs/run; : > logs/run/model_5.pt
    echo "Learning iteration 3/10"
    [ -n "$FAKE_TRAIN_HOLD" ] && exec sleep 60
    exit "${FAKE_TRAIN_RC:-0}" ;;
  *uv*" export "*)
    while [ $# -gt 0 ]; do
      if [ "$1" = "--onnx-file" ]; then : > "$2"; echo onnx > "$2"; fi
      shift
    done ;;
esac
exit 0
"""


def run_remote_runner(tmp_path, monkeypatch, *, seconds=3600, hold=False, rc=0, then=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("uv", "git"):
        path = bin_dir / tool
        path.write_text(FAKE_TOOL)
        path.chmod(0o755)
    content = tmp_path / "content"
    content.mkdir()
    script = colab_jobs._remote_runner("Mjlab-Velocity-Flat-MicroDuck", 10, 64, seconds).replace("/content", str(content))
    runner = tmp_path / "runner.py"
    runner.write_text(script)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_TRAIN_RC": str(rc)}
    if hold:
        env["FAKE_TRAIN_HOLD"] = "1"
    process = subprocess.Popen([sys.executable, str(runner)], env=env)
    root = content / "microduck-cloud-job"
    if then:
        for _ in range(200):
            if (root / "status.json").is_file() and json.loads((root / "status.json").read_text())["state"] == "training":
                break
            time.sleep(0.05)
        for _ in range(100):  # the fake trainer has saved its checkpoint
            if list((content / "microduck_rl").glob("logs/**/model_*.pt")):
                break
            time.sleep(0.05)
        then(root)
    assert process.wait(timeout=40) == 0
    return root, json.loads((root / "status.json").read_text())


def test_runner_finishes_normally_with_both_artifacts(tmp_path, monkeypatch):
    root, status = run_remote_runner(tmp_path, monkeypatch)
    assert status["state"] == "done"
    assert (root / "policy.onnx").stat().st_size > 0 and (root / "checkpoint.pt").is_file()
    assert "Learning iteration" in (root / "train.log").read_text()
    assert "git" in (root / "setup.log").read_text(), "setup output must be kept for diagnosis"


def test_runner_reports_a_training_failure(tmp_path, monkeypatch):
    root, status = run_remote_runner(tmp_path, monkeypatch, rc=3)
    assert status["state"] == "failed" and "exited with 3" in status["message"]
    assert "exited with 3" in (root / "error.log").read_text()


def test_runner_rescue_stops_training_and_keeps_the_result(tmp_path, monkeypatch):
    root, status = run_remote_runner(tmp_path, monkeypatch, hold=True, then=lambda r: (r / "rescue.request").write_text("budget"))
    assert status["state"] == "rescued" and "budget" in status["message"]
    assert (root / "checkpoint.pt").is_file() and (root / "policy.onnx").stat().st_size > 0


def test_runner_enforces_its_own_deadline_when_the_lab_is_gone(tmp_path, monkeypatch):
    # hold: trainer would run 60 s; a 1 s deadline must end it long before that.
    started = time.monotonic()
    root, status = run_remote_runner(tmp_path, monkeypatch, seconds=1, hold=True)
    assert status["state"] == "rescued" and "deadline" in status["message"]
    assert time.monotonic() - started < 30
    assert (root / "checkpoint.pt").is_file()


# --- web sign-in --------------------------------------------------------------


FAKE_CLI = """#!/bin/sh
# Mirrors the real CLI: URL on stderr, code on stdin, usage on stdout once signed in.
if [ -f "$FAKE_TOKEN" ]; then
  echo "Current balance: 0.00 compute units"; echo "Usage rate: 0.00/hr"; echo "Active assignments: 0"
  exit 0
fi
echo "To authorize colab-cli, visit this URL in any browser:" >&2
echo "  https://accounts.google.com/o/oauth2/auth?response_type=code&state=x" >&2
printf "Enter the authorization code: "
read -r code || exit 1
[ "$code" = "good-code" ] || { echo "invalid_grant" >&2; exit 1; }
: > "$FAKE_TOKEN"
echo "Current balance: 0.00 compute units"; echo "Usage rate: 0.00/hr"; echo "Active assignments: 0"
"""


@pytest.fixture
def fake_auth_cli(tmp_path, monkeypatch):
    binary = tmp_path / "colab"
    binary.write_text(FAKE_CLI)
    binary.chmod(0o755)
    token = tmp_path / "token"
    monkeypatch.setenv("FAKE_TOKEN", str(token))
    monkeypatch.setattr(colab_jobs, "cli_path", lambda: str(binary))
    return token


def test_web_sign_in_relays_the_code_and_reports_the_new_account(fake_auth_cli):
    flow = colab_jobs.ColabAuthFlow()
    started = flow.start()
    assert started["url"].startswith("https://accounts.google.com/") and not started["connected"]
    assert flow.finish("good-code")["connected"] is True
    assert fake_auth_cli.exists()


def test_web_sign_in_rejects_a_wrong_code_without_echoing_it(fake_auth_cli):
    flow = colab_jobs.ColabAuthFlow()
    flow.start()
    with pytest.raises(ValueError) as error:
        flow.finish("wrong-code")
    assert "wrong-code" not in str(error.value)
    assert not fake_auth_cli.exists()


def test_web_sign_in_validates_the_code_shape_and_needs_a_started_flow(fake_auth_cli):
    flow = colab_jobs.ColabAuthFlow()
    with pytest.raises(ValueError, match="过期"):
        flow.finish("good-code")
    flow.start()
    with pytest.raises(ValueError, match="格式"):
        flow.finish("two words")
    flow.cancel()
    assert flow.process is None


def test_web_sign_in_with_a_valid_token_needs_no_code(fake_auth_cli):
    fake_auth_cli.write_text("")
    assert colab_jobs.ColabAuthFlow().start() == {"connected": True}
