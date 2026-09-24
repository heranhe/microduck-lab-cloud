"""Cloud orchestration tests do not allocate a real Colab runtime."""

from subprocess import CompletedProcess

from microduck_local import colab_jobs


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
    script = colab_jobs._remote_runner("Mjlab-Velocity-Flat-MicroDuck", 10, 64)
    assert '"uv", "run", "train", "Mjlab-Velocity-Flat-MicroDuck"' in script
    assert '"--checkpoint-file"' in script
    compile(script, "remote_runner.py", "exec")


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
        return CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    jobs = colab_jobs.ColabJobs(tmp_path)
    assert jobs.list()[0]["state"] == "detached"
    assert jobs.stop("012345abcdef")["released"]
    assert stopped == [["stop", "-s", "microduck-012345abcdef"]]


def test_zero_balance_never_allocates_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 0})
    jobs = colab_jobs.ColabJobs(tmp_path)
    try:
        jobs.start("Mjlab-Velocity-Flat-MicroDuck", "T4", 10, 64)
    except RuntimeError as exc:
        assert "balance" in str(exc)
    else:
        raise AssertionError("zero balance was accepted")
    assert not list(tmp_path.iterdir())


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
        return CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    stopped = jobs.stop_all()
    assert len(stopped) == 2
    assert all(job["released"] for job in stopped)
    assert calls == [["stop", "-s", "microduck-111111111111"],
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
        return CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(colab_jobs, "cli", fake_cli)
    monkeypatch.setattr(colab_jobs, "account_status", lambda: {"connected": True, "balance": 200})
    jobs = colab_jobs.ColabJobs(tmp_path)
    job = jobs.start("Mjlab-Velocity-Flat-MicroDuck", "T4", 10, 64)
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
    assert [call[0] for call in calls] == ["new", "stop"]
