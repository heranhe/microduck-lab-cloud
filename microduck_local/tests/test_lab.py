"""Lab server units: roster persistence round-trip, helper spawn/remove guard
rules, TrainingJob launch/scale argv construction, staged-curriculum chaining,
stats payload shape, teach initFrom validation, reward-weight plumbing, the
user-chosen practice budget, one compiled mjModel per scene across the roster,
and the front door — who may open the WebSocket or upload a capture.
The training subprocess is faked throughout — the real --init-from
continuation lives in test_train_resume.py."""

import asyncio
import json
import types

import numpy as np
import pytest

from microduck_local import behaviors as B
from microduck_local import contract as C
from microduck_local import train_behavior as TB
from microduck_local import viz_server as V

# Above any real macOS pid, so psutil paths deterministically take their
# NoSuchProcess branches instead of touching a live process.
FAKE_PID = 4_194_304


class FakeProc:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = FAKE_PID
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch, tmp_path):
    """Record every trainer launch; keep runs/ and lab-state.json in tmp."""
    launches: list[FakeProc] = []

    def popen(cmd, **kwargs):
        proc = FakeProc(cmd, **kwargs)
        launches.append(proc)
        return proc

    monkeypatch.setattr(V.subprocess, "Popen", popen)
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return launches


def _fake_duck(duck_id, policy_id=None, onnx_path=None, label=None):
    return types.SimpleNamespace(id=duck_id, label=label or duck_id,
                                 policy_id=policy_id, onnx_path=onnx_path)


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def _endpoint(app, path: str, method: str):
    """Pull a handler straight off the FastAPI app (same trick as
    test_clips.py — no httpx in the project, so no TestClient)."""
    for r in app.routes:
        if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
            return r.endpoint
    raise AssertionError(f"no {method} {path} route")


# ------------------------------------------------------- TrainingJob argv

def test_training_job_launch_argv(fake_popen):
    job = V.TrainingJob("spin", helpers=1, steps=200_000, snap_steps=5000,
                        weights={"spin_fast": 3.0, "stay_put": -1.0})
    argv = fake_popen[0].cmd
    assert argv[1:4] == ["-m", "microduck_local.train_behavior", "spin"]
    assert _flag(argv, "--run-name") == job.run_name
    # Env count is BASE_ENVS + helpers * ENVS_PER_HELPER. ENVS_PER_HELPER is
    # 0: helpers are viewers, they do not resize the trainer. The test locks
    # the arithmetic so a future non-zero cannot silently drift.
    assert _flag(argv, "--envs") == str(V.BASE_ENVS + V.ENVS_PER_HELPER)
    assert _flag(argv, "--steps") == "200000"
    assert _flag(argv, "--snap-steps") == "5000"
    # Negative weights are clamped before they reach the subprocess — the
    # double-negation sign bug must not be reachable from the wire.
    assert json.loads(_flag(argv, "--weights-json")) == {"spin_fast": 3.0,
                                                         "stay_put": 0.0}
    assert "--init-from" not in argv
    assert job.payload()["weights"]["spin_fast"] == 3.0
    assert job.payload()["weights"]["stay_upright"] == 1.5  # default kept


def test_stop_sweeps_the_worker_tree(fake_popen, monkeypatch):
    """A stop must SIGTERM the SubprocVecEnv workers, not just the trainer.

    Regression: stop() terminated only the trainer, so each stop stranded ~10
    workers (reparented to init, ~322 MB) for the rest of the session. The
    child list must also be snapshotted BEFORE the trainer dies — afterwards
    there is no handle left to find them by.
    """
    killed, order = [], []

    class FakeChild:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            killed.append(self.pid)
            order.append(("child", self.pid))

    class FakeParent:
        def __init__(self, pid):
            assert pid == FAKE_PID

        def children(self, recursive=False):
            order.append(("snapshot", recursive))
            return [FakeChild(i) for i in range(10)]

    monkeypatch.setattr(V.psutil, "Process", FakeParent)
    job = V.TrainingJob("run")
    monkeypatch.setattr(job.proc, "terminate",
                        lambda: order.append(("parent", None)))
    job.stop(force=True)

    assert job.status == "stopped"
    assert killed == list(range(10)), "workers were not swept"
    assert order[0] == ("snapshot", True), "snapshot must precede the kill"
    assert order[1] == ("parent", None), "parent dies before its children"


def test_stop_does_not_block_the_event_loop(fake_popen, monkeypatch):
    """/teach/stop is an async handler, so stop() must not wait on the
    trainer — a blocking reap stalls the WebSocket stream for every duck."""
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    job = V.TrainingJob("run")

    def boom(timeout=None):
        raise AssertionError("stop() waited on the trainer")

    monkeypatch.setattr(job.proc, "wait", boom)
    job.stop()
    assert job.status == "stopping"
    assert job.proc.poll() is None
    assert (job.dir / "STOP").is_file()


def test_save_stop_is_idempotent_and_waits_for_trainer(fake_popen):
    job = V.TrainingJob("spin", steps=200_000)
    job.stop()
    deadline = job._stop_at
    job.stop()
    assert job._stop_at == deadline
    assert job.status == "stopping" and job.proc.poll() is None
    job.proc.returncode = 0
    job.poll()
    assert job.status == "stopped" and len(fake_popen) == 1


def test_save_timeout_forces_stop_and_reports_checkpoint_risk(fake_popen):
    job = V.TrainingJob("spin")
    job.stop()
    job._stop_at -= V.STOP_GRACE_S + 1
    job.poll()
    assert job.status == "stopped" and job.proc.returncode == -15
    assert job.payload()["stopWarning"]


def test_resume_uses_same_directory_budget_and_checkpoint(fake_popen):
    job = V.TrainingJob("spin", steps=200_000)
    for name in ("model.zip", "vecnormalize.pkl"):
        (job.dir / name).touch()
    (job.dir / "snapshot.json").write_text(json.dumps({"steps": 60_000, "next_snap": 110_000}))
    job.stop()
    job.proc.returncode = 0
    job.poll()
    assert job.payload()["canResume"]
    assert job.resume() is None
    assert not (job.dir / "STOP").exists()
    assert _flag(fake_popen[-1].cmd, "--init-from") == str(job.dir)
    assert _flag(fake_popen[-1].cmd, "--steps") == "200000"
    assert job.progress["steps"] == 60_000 and not job.progress["done"]
    assert job.status == "training"


def test_resume_refuses_without_checkpoint_or_after_budget(fake_popen):
    job = V.TrainingJob("spin", steps=1000)
    job.stop(force=True)
    assert job.resume() is not None
    for name in ("model.zip", "vecnormalize.pkl"):
        (job.dir / name).touch()
    job.progress["lastCheckpointStep"] = 1000
    assert not job.payload()["canResume"]
    assert len(fake_popen) == 1


def test_snapshot_progress_and_eta_match_saved_checkpoint(fake_popen):
    job = V.TrainingJob("spin", steps=200_000)
    (job.dir / "snapshot.json").write_text(json.dumps({"steps": 55_000, "next_snap": 105_000}))
    job.progress.update(steps=60_000, sps=1000)
    job.poll()
    progress = job.payload()["progress"]
    assert progress["lastCheckpointStep"] == 55_000
    assert progress["nextSnapStep"] == 105_000
    assert progress["etaSeconds"] == 140
    job.stop()
    assert job.payload()["progress"]["etaSeconds"] is None


def test_saving_job_keeps_directory_and_compute_slot(fake_popen):
    from fastapi import HTTPException
    app = V.make_app([])
    st = app.state.lab
    st.job = V.TrainingJob("spin")
    st.job.stop()
    assert st.job.run_name in V.training_run_names(st)
    clear = _endpoint(app, "/teach/clear", "POST")
    assert not asyncio.run(clear())["cleared"]
    status = _endpoint(app, "/teach/status", "GET")()
    assert status["running"] and status["status"] == "stopping"
    resume = _endpoint(app, "/teach/resume", "POST")
    with pytest.raises(HTTPException) as e:
        asyncio.run(resume(V.ResumeReq(runName=st.job.run_name)))
    assert e.value.status_code == 409


class FakeWorker:
    """A SubprocVecEnv worker that records its own SIGTERM."""

    def __init__(self, pid, killed, fail=None):
        self.pid = pid
        self._killed = killed
        self._fail = fail

    def terminate(self):
        if self._fail is not None:
            raise self._fail
        self._killed.append(self.pid)


def _fake_worker_tree(monkeypatch, current_proc, workers_for):
    """psutil.Process(pid).children() as the OS really behaves.

    The workers are findable only while the trainer is ALIVE; the moment it
    exits they are reparented to init and its child list is empty. Every
    sweep therefore has to work off a snapshot taken earlier — modelling that
    is the whole point of this fake.
    """

    class FakeParent:
        def __init__(self, pid):
            assert pid == FAKE_PID

        def children(self, recursive=False):
            proc = current_proc()
            return [] if proc.poll() is not None else list(workers_for(proc))

    monkeypatch.setattr(V.psutil, "Process", FakeParent)


def test_poll_sweeps_workers_a_dead_trainer_left_behind(fake_popen, monkeypatch):
    """A trainer the lab did NOT kill — OOM, a stray kill -9, a crash by
    signal — never runs multiprocessing's atexit hook, so it orphans its
    workers exactly as an un-swept stop did, and under the `fork` backend they
    never see EOF on their pipe either. poll() is the only witness left, and
    only because it keeps a snapshot from while the trainer was alive.
    """
    killed = []
    job = V.TrainingJob("run")
    workers = [FakeWorker(i, killed) for i in range(10)]
    _fake_worker_tree(monkeypatch, lambda: job.proc, lambda _p: workers)

    job.poll()               # ~1 Hz tick while training: snapshot, no kills
    assert killed == [], "a live trainer's fleet must be left alone"

    job.proc.returncode = -9  # killed from outside; workers now unfindable
    changed, _ = job.poll()

    assert changed and job.status == "failed"
    assert killed == list(range(10)), "orphans were not swept"


def test_sweep_survives_an_unreadable_worker(fake_popen, monkeypatch):
    """One handle that raises must not strand the rest of the fleet: psutil
    hands back AccessDenied/ZombieProcess as readily as NoSuchProcess."""
    killed = []
    job = V.TrainingJob("run")
    workers = [FakeWorker(0, killed),
               FakeWorker(1, killed, fail=V.psutil.AccessDenied(1)),
               FakeWorker(2, killed)]
    _fake_worker_tree(monkeypatch, lambda: job.proc, lambda _p: workers)

    job.stop(force=True)
    assert killed == [0, 2]


def test_stage_handoff_sweeps_only_the_finished_stage(fake_popen, monkeypatch):
    """The sweep must fire BEFORE _advance_stage() rebinds self.proc, or it
    would aim the finished stage's SIGTERMs at the incoming stage's fleet.
    (A clean stage exit already reaped its own workers via atexit, so the
    sweep is a no-op in the happy path — this pins where it points.)"""
    killed = []
    job = V.TrainingJob("backflip", steps=1000)
    s1 = [FakeWorker(i, killed) for i in range(3)]
    s2 = [FakeWorker(10 + i, killed) for i in range(3)]
    _fake_worker_tree(monkeypatch, lambda: job.proc,
                      lambda p: s2 if p is not fake_popen[0] else s1)

    job.poll()                       # snapshot stage 1's fleet
    _finish_stage(job, 1000)
    changed, _ = job.poll()          # stage 1 exits 0 → chain stage 2

    assert changed and job.status == "training" and len(fake_popen) == 2
    assert killed == [0, 1, 2], "stage 1's fleet was not the one swept"

    job.poll()                       # stage 2 alive: snapshot moves with it
    job.stop(force=True)
    assert killed == [0, 1, 2, 10, 11, 12]


def test_scale_restarts_warm_with_new_envs(fake_popen):
    job = V.TrainingJob("spin", steps=200_000, weights={"spin_fast": 3.0})
    (job.dir / "model.zip").touch()
    job.scale(2)
    assert fake_popen[0].returncode is not None  # old proc was terminated
    argv = fake_popen[1].cmd
    assert _flag(argv, "--init-from") == str(job.dir)
    two_helpers = V.BASE_ENVS + 2 * V.ENVS_PER_HELPER
    assert _flag(argv, "--envs") == str(two_helpers)
    assert _flag(argv, "--run-name") == job.run_name   # same progress.jsonl
    assert _flag(argv, "--steps") == "200000"          # absolute target kept
    # Weight overrides survive the restart — same scorecard, more envs.
    assert json.loads(_flag(argv, "--weights-json")) == {"spin_fast": 3.0}
    assert (job.envs, job.helpers) == (two_helpers, 2)
    assert job.restarting is False
    assert job.status == "training"

    job.scale(0)
    assert _flag(fake_popen[2].cmd, "--envs") == str(V.BASE_ENVS)


def test_scale_without_snapshot_relaunches_cold(fake_popen):
    job = V.TrainingJob("crouch", steps=100_000)
    job.scale(1)  # no model.zip on disk yet
    assert "--init-from" not in fake_popen[1].cmd
    assert _flag(fake_popen[1].cmd, "--envs") == str(
        V.BASE_ENVS + V.ENVS_PER_HELPER)


def test_poll_survives_scale_gap(fake_popen):
    """A dead proc during restart must not flip the job to failed."""
    job = V.TrainingJob("spin", steps=1000)
    job.proc.terminate()
    job.restarting = True
    job.poll()
    assert job.status == "training"
    job.restarting = False
    job.poll()
    assert job.status == "failed"


# ------------------------------------------------------- spawn/remove guards

def test_spawn_helper_guards(fake_popen):
    st = V.LabState([_fake_duck("d0")])
    assert "no active training" in V.spawn_helper_error(st)

    st.job = V.TrainingJob("spin", steps=1000)
    assert "first training snapshot" in V.spawn_helper_error(st)

    (st.job.dir / "model.zip").touch()
    assert V.spawn_helper_error(st) is None

    st.scaling = True
    assert "mid-restart" in V.spawn_helper_error(st)
    st.scaling = False

    st.ducks += [_fake_duck(f"helper{i}") for i in range(1, V.MAX_HELPERS)]
    assert V.spawn_helper_error(st) is None  # one below the cap: still allowed
    st.ducks.append(_fake_duck(f"helper{V.MAX_HELPERS}"))
    assert f"helper cap reached ({V.MAX_HELPERS})" in (V.spawn_helper_error(st) or "")

    st.job.status = "done"
    assert "no active training" in V.spawn_helper_error(st)


def test_a_helper_joins_the_job_it_was_ASKED_for(fake_popen):
    """The ＋ sits on every training row now, and the name is the only thing
    that tells two rows apart.

    Unnamed, the server took `st.job` — the NEWEST — so a click on the older
    run's row silently added a helper to the other run. `None` still means
    the newest, because that is what a caller written when the lab held one
    job means by it.
    """
    st = V.LabState([_fake_duck("d0")])
    older = V.TrainingJob("spin", steps=1000)
    newer = V.TrainingJob("spin", steps=1000)
    st.jobs[:] = [older, newer]
    for j in (older, newer):
        (j.dir / "model.zip").touch()
    assert older.run_name != newer.run_name

    assert V.spawn_helper_job(st, None) is newer            # old callers: unchanged
    assert V.spawn_helper_job(st, older.run_name) is older  # ...and this is the fix
    assert V.spawn_helper_job(st, newer.run_name) is newer
    assert V.spawn_helper_job(st, "nobody") is None

    assert V.spawn_helper_error(st, older.run_name) is None
    assert "no job named" in (V.spawn_helper_error(st, "nobody") or "")

    # A row whose run has STOPPED carries no ＋, and naming it is refused
    # rather than quietly falling back to whatever is still training.
    older.status = "done"
    assert V.spawn_helper_job(st, older.run_name) is None
    assert "no job named" in (V.spawn_helper_error(st, older.run_name) or "")
    assert V.spawn_helper_error(st, newer.run_name) is None


def test_remove_duck_guards(fake_popen):
    from types import SimpleNamespace
    st = V.LabState([_fake_duck("d0"), _fake_duck("trainee"),
                      _fake_duck("helper1")])
    # Any duck is removable now (roster declutter) …
    assert V.remove_duck_error(st, "d0") is None
    assert V.remove_duck_error(st, "trainee") is None  # no active run
    assert "no duck helper9" in V.remove_duck_error(st, "helper9")
    assert V.remove_duck_error(st, "helper1") is None
    # … except the trainee while ITS OWN run is live (the run's only window).
    # Spelled with `st.jobs` rather than two `st.job =` assignments because
    # the lab holds several jobs now and the guard asks about the one that
    # owns THIS duck — see test_only_a_training_trainee_is_unremovable.
    st.jobs = [SimpleNamespace(status="training", trainee_id="trainee")]
    assert "stop the run" in V.remove_duck_error(st, "trainee")
    st.jobs = [SimpleNamespace(status="done", trainee_id="trainee")]
    assert V.remove_duck_error(st, "trainee") is None
    # … and helpers still wait out a trainer restart (plain ducks don't).
    st.scaling = True
    assert "mid-restart" in V.remove_duck_error(st, "helper1")
    assert V.remove_duck_error(st, "d0") is None


def test_spawn_duck_guards_and_slots(fake_popen):
    st = V.LabState([_fake_duck(f"d{i}") for i in range(3)])
    assert "needs a policy" in V.spawn_duck_error(st, None)
    assert V.spawn_duck_error(st, "pollen:alpha_stand") is None
    st.ducks = [_fake_duck(f"d{i}") for i in range(V.MAX_DUCKS)]
    assert "full" in V.spawn_duck_error(st, "pollen:alpha_stand")
    assert V.next_duck_slot(
        [_fake_duck("d0"), _fake_duck("d2"), _fake_duck("helper1")]) == 1


def test_next_helper_slot_reuses_gaps():
    assert V.next_helper_slot([_fake_duck("d0")]) == 1
    assert V.next_helper_slot([_fake_duck("helper1"), _fake_duck("helper3")]) == 2


# ------------------------------------------------------- lab-state.json

def test_lab_state_round_trip(fake_popen, monkeypatch, tmp_path):
    live = tmp_path / "live.onnx"
    ducks = [
        _fake_duck("d0", onnx_path="/policies/alpha_walking.onnx",
                   label="alpha_walking"),
        _fake_duck("d1", policy_id="pollen:alpha_stand", label="alpha_stand"),
        _fake_duck("trainee", onnx_path=str(live), label="🎓 Spin in place @40k"),
    ]
    V.save_lab_state(ducks)
    data = json.loads((tmp_path / "lab-state.json").read_text())
    assert data["version"] == 1
    assert data["ducks"][0] == {"id": "d0", "label": "alpha_walking",
                                "policy": None,
                                "onnxPath": "/policies/alpha_walking.onnx",
                                # which body this roster slot is (robots/spec.py)
                                "robot": "microduck",
                                "showcase": False}
    assert data["ducks"][1]["policy"] == "pollen:alpha_stand"

    loaded_from: list[str] = []
    monkeypatch.setattr(V, "_onnx_infer",
                        lambda p: loaded_from.append(str(p)) or V._zero_infer)
    monkeypatch.setattr(V, "load_policy_infer",
                        lambda pid: loaded_from.append(pid) or V._zero_infer)
    restored = V.restore_ducks(tmp_path / "lab-state.json")
    assert [d.id for d in restored] == ["d0", "d1", "trainee"]
    assert [d.label for d in restored] == ["alpha_walking", "alpha_stand",
                                           "🎓 Spin in place @40k"]
    # Provenance survives the round trip, so the NEXT restart restores too.
    assert restored[1].policy_id == "pollen:alpha_stand"
    assert restored[2].onnx_path == str(live)
    assert loaded_from == ["/policies/alpha_walking.onnx", "pollen:alpha_stand",
                           str(live)]


def _trainee(duck_id, created_at, tmp_path):
    d = _fake_duck(duck_id, onnx_path=str(tmp_path / f"{duck_id}.onnx"))
    d.created_at = created_at
    return d


def test_a_restart_brings_back_only_the_newest_trainee(fake_popen, monkeypatch,
                                                       tmp_path, capsys):
    """Training dies with the server, so every trainee a restart restores is
    a FINISHED job's — and each one steps a full MuJoCo env in the 50 Hz
    loop. Restored all, they piled up one per job: on 2026-09-27 eight
    finished MOSS trainees pinned the lab at ~90% of a core and /sim ran at
    RTF 0.12 (7% once they were removed). The newest stays — a finished run's
    card is still looked at after a restart, and a G1 job once needed its
    restored trainee to guess the body — and "newest" is by creation time,
    not by where a reused slot happens to sit in the list."""
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)
    V.save_lab_state([
        _fake_duck("d0", onnx_path=str(tmp_path / "d0.onnx")),
        _trainee("trainee", 100.0, tmp_path),
        _trainee("trainee3", 400.0, tmp_path),     # newest, NOT listed last
        _trainee("trainee2", 300.0, tmp_path),
        _trainee("trainee4", 200.0, tmp_path),
    ])
    restored = V.restore_ducks(V.lab_state_path())
    assert [d.id for d in restored] == ["d0", "trainee3"]
    assert restored[1].created_at == 400.0         # survives the round trip
    assert "not restoring 3 finished trainee" in capsys.readouterr().out
    # ...and they stay gone: the file no longer lists them.
    kept = json.loads(V.lab_state_path().read_text())["ducks"]
    assert [e["id"] for e in kept] == ["d0", "trainee3"]


def test_an_old_lab_state_keeps_its_last_listed_trainee(fake_popen, monkeypatch,
                                                        tmp_path):
    """Files from before `createdAt` fall back to list order (a new trainee
    is appended to the roster when its job starts)."""
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)
    V.save_lab_state([_fake_duck("trainee", onnx_path=str(tmp_path / "a.onnx")),
                      _fake_duck("trainee2", onnx_path=str(tmp_path / "b.onnx")),
                      _fake_duck("d0", onnx_path=str(tmp_path / "c.onnx"))])
    assert "createdAt" not in V.lab_state_path().read_text()
    restored = V.restore_ducks(V.lab_state_path())
    assert [d.id for d in restored] == ["trainee2", "d0"]


def test_restore_skips_unloadable_entries(fake_popen, monkeypatch, tmp_path, capsys):
    def only_stand(pid):
        if pid != "pollen:alpha_stand":
            raise KeyError(pid)
        return V._zero_infer

    monkeypatch.setattr(V, "load_policy_infer", only_stand)
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)
    V.save_lab_state([
        _fake_duck("d0", policy_id="run:deleted-run"),   # registry entry gone
        _fake_duck("trainee"),                            # no brain recorded
        _fake_duck("d2", policy_id="pollen:alpha_stand"),
    ])
    restored = V.restore_ducks(V.lab_state_path())
    assert [d.id for d in restored] == ["d2"]
    out = capsys.readouterr().out
    assert "skipping d0" in out and "skipping trainee" in out


# ------------------------------------------------------- stats payload

def test_stats_payload_shape(fake_popen):
    sampler = V.StatsSampler()
    s = sampler.sample(None)
    assert set(s) == {"cpu", "mem", "lab", "trainer", "trainFps"}
    assert 0.0 <= s["cpu"] <= 100.0 and 0.0 < s["mem"] <= 100.0
    assert set(s["lab"]) == {"cpu", "memMb"} and s["lab"]["memMb"] > 0
    assert s["trainer"] is None and s["trainFps"] is None
    json.dumps(s)

    job = V.TrainingJob("spin", steps=200_000)
    (job.dir / "progress.jsonl").write_text(
        '{"steps": 3072, "total": 200000, "elapsed_s": 2.0}\n'
        '{"steps": 9216, "total": 200000, "elapsed_s": 4.0}\n')
    job.poll()
    s = sampler.sample(job)
    assert s["trainFps"] == pytest.approx(3072.0)   # 6144 steps / 2 s
    assert s["trainer"] is None  # fake pid resolves to no live process


def test_train_fps_none_across_restart_boundary(fake_popen):
    """elapsed_s restarts at 0 in a relaunched trainer — no negative fps."""
    job = V.TrainingJob("spin", steps=200_000)
    (job.dir / "progress.jsonl").write_text(
        '{"steps": 20480, "total": 200000, "elapsed_s": 30.0}\n')
    job.poll()
    assert job.train_fps() is None  # one point is not a rate
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 23552, "total": 200000, "elapsed_s": 1.5}\n')
    job.poll()
    assert job.train_fps() is None  # restart boundary
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 26624, "total": 200000, "elapsed_s": 3.0}\n')
    job.poll()
    assert job.train_fps() == pytest.approx(2048.0)


# ------------------------------------------------------- staged curriculum

def _finish_stage(job, steps):
    """Fake a stage completing: the trainer's final progress line, then a
    clean subprocess exit (poll() advances the chain on the EXIT, after the
    warm-start artifacts are guaranteed on disk)."""
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write(json.dumps({"steps": steps, "total": steps, "done": True}) + "\n")
    job.proc.returncode = 0


def test_curriculum_teach_chains_stages(fake_popen):
    # Follows whatever chain the behavior declares — the recipe's stage list
    # is live tuning surface and must be free to grow/shrink.
    stages = B.BEHAVIORS["backflip"].curriculum
    n = len(stages)
    assert n >= 2
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500)

    # Stage 1: its own run name/dir, no warm start, stage env on the process.
    assert job.run_name.endswith("-s1")
    argv = fake_popen[0].cmd
    assert _flag(argv, "--run-name") == job.run_name
    assert _flag(argv, "--steps") == "1000"
    assert "--init-from" not in argv
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in stages[0].env} == stages[0].env
    p = job.payload()
    assert p["stage"] == {"idx": 1, "count": n, "label": stages[0].label,
                          "detail": stages[0].detail, "start": 1}
    assert p["stage"]["detail"]  # the inspector's story is really there
    # The steps override shrinks EVERY stage; overall totals follow suit.
    assert (p["progress"]["overallSteps"], p["progress"]["overallTotal"]) == (0, 1000 * n)

    # Stage 1 done → stage 2 auto-launches, --init-from stage 1's dir.
    s1_dir = job.dir
    _finish_stage(job, 1000)
    changed, _ = job.poll()
    assert changed and job.status == "training"
    assert len(fake_popen) == 2
    argv2 = fake_popen[1].cmd
    assert _flag(argv2, "--init-from") == str(s1_dir)
    assert _flag(argv2, "--run-name") == job.run_name
    assert job.run_name.endswith("-s2")
    assert job.dir != s1_dir
    env2 = fake_popen[1].kwargs["env"]
    assert {k: env2[k] for k in stages[1].env} == stages[1].env
    p = job.payload()
    assert p["stage"] == {"idx": 2, "count": n, "label": stages[1].label,
                          "detail": stages[1].detail, "start": 1}
    # Per-stage fields reset for the new run; overall progress carries on.
    assert (p["progress"]["steps"], p["progress"]["total"]) == (0, 1000)
    assert p["progress"]["overallSteps"] == 1000

    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 400, "total": 1000}\n')
    job.poll()
    p = job.payload()
    assert (p["progress"]["steps"], p["progress"]["overallSteps"]) == (400, 1400)

    # Chain completes only when the FINAL stage does — with the ✔ run name.
    for i in range(2, n + 1):
        _finish_stage(job, 1000)
        job.poll()
    assert job.status == "done"
    assert job.run_name.endswith(f"-s{n}")  # relabel-on-done names the final stage
    assert len(fake_popen) == n
    assert job.payload()["progress"]["overallSteps"] == 1000 * n


def test_curriculum_scale_keeps_stage_warm_start(fake_popen):
    """A helper join before the current stage's first snapshot must not
    relaunch cold — that would silently drop the previous stage's brain."""
    job = V.TrainingJob("backflip", steps=1000)
    s1_dir = job.dir
    _finish_stage(job, 1000)
    job.poll()  # now on stage 2, no model.zip there yet
    job.scale(1)
    argv = fake_popen[2].cmd
    assert _flag(argv, "--init-from") == str(s1_dir)
    assert _flag(argv, "--envs") == str(V.BASE_ENVS + V.ENVS_PER_HELPER)
    stage2_env = B.BEHAVIORS["backflip"].curriculum[1].env
    env = fake_popen[2].kwargs["env"]
    assert {k: env[k] for k in stage2_env} == stage2_env

    # Once the stage has its own snapshot, scale resumes from its own dir.
    (job.dir / "model.zip").touch()
    job.scale(2)
    assert _flag(fake_popen[3].cmd, "--init-from") == str(job.dir)


def test_teach_stop_mid_chain_prevents_next_stage(fake_popen):
    job = V.TrainingJob("backflip", steps=1000)
    job.stop(force=True)
    job.poll()
    assert job.status == "stopped"
    assert len(fake_popen) == 1  # no stage 2

    # Even a stage that finished cleanly right as stop landed stays stopped.
    job2 = V.TrainingJob("backflip", steps=1000)
    _finish_stage(job2, 1000)
    job2.stop(force=True)
    job2.poll()
    assert job2.status == "stopped"
    assert len(fake_popen) == 2


def test_non_curriculum_behavior_stays_single_run(fake_popen):
    job = V.TrainingJob("spin", steps=1000)
    assert "-s1" not in job.run_name
    p = job.payload()
    assert p["stage"] is None
    # overall* mirror steps/total so consumers can read one pair everywhere.
    assert (p["progress"]["overallSteps"], p["progress"]["overallTotal"]) == (0, 1000)
    _finish_stage(job, 1000)
    job.poll()
    assert job.status == "done"
    assert len(fake_popen) == 1


def test_init_from_skips_curriculum_uses_final_window(fake_popen):
    """An explicit initFrom is a fine-tune of an existing run: single run,
    trained under the FINAL stage's env (the finished trick's spawn window)."""
    prev = V.RUNS_DIR / "teach-backflip-oldrun"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, init_from=prev)
    assert "-s1" not in job.run_name
    assert job.payload()["stage"] is None
    assert _flag(fake_popen[0].cmd, "--init-from") == str(prev)
    final_env = B.BEHAVIORS["backflip"].curriculum[-1].env
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in final_env} == final_env
    _finish_stage(job, 1000)
    job.poll()
    assert job.status == "done" and len(fake_popen) == 1


def test_behavior_card_exposes_curriculum():
    card = B.behavior_card(B.BEHAVIORS["backflip"])
    declared = B.BEHAVIORS["backflip"].curriculum
    assert card["curriculum"] == [
        {"label": s.label, "steps": s.steps, "detail": s.detail}
        for s in declared]
    assert len(card["curriculum"]) >= 2  # backflip really is staged
    # Every backflip stage tells the inspector its plain-English story.
    assert all(entry["detail"] for entry in card["curriculum"])
    json.dumps(card)  # stays JSON-friendly (env knobs are not in the card)
    assert B.behavior_card(B.BEHAVIORS["spin"])["curriculum"] == []


# ------------------------------------------------------- teach extras

def test_resolve_init_from_validation(fake_popen, monkeypatch):
    with pytest.raises(ValueError, match="does not exist"):
        V.resolve_init_from("no-such-run")
    with pytest.raises(ValueError, match="plain run name"):
        V.resolve_init_from("../../etc")

    run = V.RUNS_DIR / "half-run"
    run.mkdir(parents=True)
    (run / "model.zip").touch()
    with pytest.raises(ValueError, match="vecnormalize.pkl"):
        V.resolve_init_from("half-run")

    (run / "vecnormalize.pkl").touch()
    assert V.resolve_init_from("half-run") == run


def test_weight_overrides_reach_the_env():
    """The --weights-json path: make_env must hand overrides to BehaviorEnv,
    clamped, and the reward must actually be computed with them."""
    base = TB.make_env("spin", 0, 0, None)()
    tuned = TB.make_env("spin", 0, 0, {"spin_fast": 5.0, "stay_put": -2.0})()
    assert tuned.weight_overrides == {"spin_fast": 5.0, "stay_put": 0.0}

    import numpy as np
    base.reset(seed=0)
    tuned.reset(seed=0)
    action = np.full(14, 0.1, np.float32)
    for env in (base, tuned):
        for _ in range(5):
            env.step(action)
    # Same physics (same seed/actions), different scorecard: the tuned term
    # sum must scale by exactly new_weight / default_weight.
    assert tuned.reward_sums["spin_fast"] == pytest.approx(
        base.reward_sums["spin_fast"] * (5.0 / 2.5))
    assert tuned.reward_sums["stay_put_penalty"] == 0.0
    assert base.reward_sums["stay_put_penalty"] <= 0.0

# ------------------------------------------------------- sticky teach weights

def test_teach_weights_sticky_roundtrip(fake_popen):
    """A user's slider crank must survive into the next /teach that arrives
    without explicit weights — 'no weights' means 'same as I had it', not
    'back to defaults' (a fresh chat launch silently wiped a cranked
    legs_over and the user lost track of their setting). Both layers now:
    behavior-level weights AND per-stage overrides round-trip."""
    assert V.load_teach_weights() == {}
    entry = {"weights": {"legs_over": 2.5},
             "stageWeights": {"2": {"neck_kip": 3.0}}}
    V.save_teach_weights({"backflip": entry})
    sticky = V.load_teach_weights()
    # Loading fills in the layers the file predates (practice budget) with
    # "never set" — the panel then falls back to the recipe's own numbers.
    assert sticky == {"backflip": {**V.empty_sticky(), **entry}}
    # The inherit rule the /teach handler applies:
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500,
                        weights=sticky["backflip"]["weights"],
                        stage_weights=sticky["backflip"]["stageWeights"])
    assert job.weights == {"legs_over": 2.5}
    assert job.stage_weights == {2: {"neck_kip": 3.0}}
    assert "--weights-json" in fake_popen[0].cmd


def test_teach_weights_legacy_flat_shape_still_reads(fake_popen):
    """Files written before per-stage weights stored the flat behavior-level
    dict — they must keep loading, as the behavior-level layer."""
    V.teach_weights_path().write_text(json.dumps(
        {"backflip": {"legs_over": 2.5}}))
    sticky = V.load_teach_weights()
    assert sticky == {"backflip": {**V.empty_sticky(),
                                   "weights": {"legs_over": 2.5}}}


# ------------------------------------------------------- per-stage weights

def test_stage_weights_merge_into_each_stage_argv(fake_popen):
    """Stage overrides layer over behavior-level weights (stage wins per key)
    at each stage's LAUNCH; stages without overrides train on the
    behavior-level set alone. Catalog keys in a stage's overrides adopt that
    term like the behavior-level channel does."""
    job = V.TrainingJob("backflip", steps=1000,
                        weights={"legs_over": 2.0},
                        stage_weights={
                            "1": {"legs_over": 4.0, "calm_body": 2.0},
                            "99": {"legs_over": 9.0},   # out of range: dropped
                            "2": {"push_off": -1.0},    # clamped to 0
                        })
    assert json.loads(_flag(fake_popen[0].cmd, "--weights-json")) == {
        "legs_over": 4.0, "calm_body": 2.0}
    p = job.payload()
    assert p["stageWeights"] == {"1": {"legs_over": 4.0, "calm_body": 2.0},
                                 "2": {"push_off": 0.0}}
    # The adopted catalog term shows up as a card row (any layer counts).
    assert "calm_body" in job.extra_keys
    assert any(t["key"] == "calm_body" for t in p["behavior"]["terms"])

    _finish_stage(job, 1000)
    job.poll()  # stage 2: behavior-level + its own override
    assert json.loads(_flag(fake_popen[1].cmd, "--weights-json")) == {
        "legs_over": 2.0, "push_off": 0.0}
    _finish_stage(job, 1000)
    job.poll()  # stage 3 has no overrides -> behavior-level only
    assert json.loads(_flag(fake_popen[2].cmd, "--weights-json")) == {
        "legs_over": 2.0}


def test_set_stage_weights_live_edit(fake_popen):
    """Editing a FUTURE stage only records (no restart needed); editing the
    CURRENT stage reports changed, and the scale() warm-restart the endpoint
    then performs relaunches it under the new merged weights. The handoff
    launch re-reads the map, so recorded future edits land at their stage."""
    job = V.TrainingJob("backflip", steps=1000, weights={"legs_over": 2.0})
    assert job.set_stage_weights({"2": {"neck_kip": 3.0}}) is False
    assert job.stage_weights == {2: {"neck_kip": 3.0}}
    assert job.set_stage_weights({"2": {"neck_kip": 3.0},
                                  "1": {"legs_over": 5.0}}) is True
    (job.dir / "model.zip").touch()
    job.scale(job.helpers)  # the warm-restart /teach/weights performs
    argv = fake_popen[1].cmd
    assert _flag(argv, "--init-from") == str(job.dir)
    assert json.loads(_flag(argv, "--weights-json")) == {"legs_over": 5.0}
    _finish_stage(job, 1000)
    job.poll()  # stage 2 picks up the recorded future edit
    assert json.loads(_flag(fake_popen[2].cmd, "--weights-json")) == {
        "legs_over": 2.0, "neck_kip": 3.0}


def test_stage_weights_ignored_for_single_run_jobs(fake_popen):
    """A fine-tune (initFrom) collapses to a single run — per-stage overrides
    can't half-apply, so they're dropped whole."""
    prev = V.RUNS_DIR / "teach-backflip-oldrun"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, init_from=prev,
                        weights={"legs_over": 2.0},
                        stage_weights={"1": {"legs_over": 9.0}})
    assert job.stage_weights == {}
    assert json.loads(_flag(fake_popen[0].cmd, "--weights-json")) == {
        "legs_over": 2.0}
    assert job.set_stage_weights({"1": {"legs_over": 9.0}}) is False


# ------------------------------------------------------- startStage

def test_resolve_stage_init_picks_newest_trained(fake_popen):
    import os as _os
    with pytest.raises(ValueError, match="train the earlier stages first"):
        V.resolve_stage_init("backflip", 3)
    old = V.RUNS_DIR / "teach-backflip-aaaaaa-s2"
    new = V.RUNS_DIR / "teach-backflip-bbbbbb-s2"
    half = V.RUNS_DIR / "teach-backflip-cccccc-s2"  # no vecnormalize.pkl
    for d in (old, new, half):
        d.mkdir(parents=True)
        (d / "model.zip").touch()
    (old / "vecnormalize.pkl").touch()
    (new / "vecnormalize.pkl").touch()
    _os.utime(old / "model.zip", (1_000_000, 1_000_000))
    _os.utime(new / "model.zip", (2_000_000, 2_000_000))
    _os.utime(half / "model.zip", (3_000_000, 3_000_000))  # newest but unusable
    assert V.resolve_stage_init("backflip", 3) == new
    # A wrong stage number still errors even with s2 runs on disk.
    with pytest.raises(ValueError, match="stage 3"):
        V.resolve_stage_init("backflip", 4)


def test_start_stage_chain_construction(fake_popen):
    """startStage=N skips the earlier stages: the chain opens at -sN warm-
    started from the resolved prev-stage dir, overall totals count only the
    stages actually run, and the chain still walks to the real end."""
    stages = B.BEHAVIORS["backflip"].curriculum
    n = len(stages)
    prev = V.RUNS_DIR / "teach-backflip-oldrun-s2"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, start_stage=3,
                        stage_init_from=prev)
    assert job.run_name.endswith("-s3")
    argv = fake_popen[0].cmd
    assert _flag(argv, "--init-from") == str(prev)
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in stages[2].env} == stages[2].env
    p = job.payload()
    assert p["stage"] == {"idx": 3, "count": n, "label": stages[2].label,
                          "detail": stages[2].detail, "start": 3}
    assert (p["progress"]["overallSteps"],
            p["progress"]["overallTotal"]) == (0, 1000 * (n - 2))
    # A scale() before the first snapshot keeps the cross-chain warm start.
    job.scale(1)
    assert _flag(fake_popen[1].cmd, "--init-from") == str(prev)
    for _ in range(3, n + 1):
        _finish_stage(job, 1000)
        job.poll()
    assert job.status == "done"
    assert job.run_name.endswith(f"-s{n}")
    assert job.payload()["progress"]["overallSteps"] == 1000 * (n - 2)


# ------------------------------------------------------- one model per scene

def test_roster_shares_one_compiled_model():
    """Every duck on a scene steps the SAME mjModel object.

    A compiled model costs ~470 MB on a process's first compile and ~90-140 MB
    per extra copy; the mjData a duck actually owns is ~0.9 MB. A six-duck lab
    held ~1.4 GB of identical, never-written model — Duck._make_env pins
    domain_rand=False, so there is nothing per-duck in there to keep apart.
    """
    a = V.Duck("m0", "a", V._zero_infer, seed=1, env_kwargs={})
    b = V.Duck("m1", "b", V._zero_infer, seed=2, env_kwargs={})
    assert a.env.model is b.env.model
    assert a.env._model_shared and b.env._model_shared
    assert a.env.data is not b.env.data      # the state still is per duck

    # …and one duck's rollout leaves its neighbour's alone, which is the whole
    # risk of sharing (the frame loop steps them serially, in this order).
    b_qpos = b.env.data.qpos.copy()
    for _ in range(20):
        a.tick()
    np.testing.assert_array_equal(b.env.data.qpos, b_qpos)


def test_scene_swap_gets_its_own_shared_model():
    """rebuild_env onto the full-collision scene: that scene shares a model of
    its own (the cache is keyed by scene), never the walk scene's."""
    walk = V.Duck("m2", "w", V._zero_infer, seed=3, env_kwargs={})
    all_kw = {"scene_xml": str(C.SCENE_ALL_XML), "terminate_on_fall": False}
    a = V.Duck("m3", "a", V._zero_infer, seed=4, env_kwargs=all_kw)
    b = V.Duck("m4", "b", V._zero_infer, seed=5, env_kwargs={})
    b.rebuild_env(all_kw)

    assert b.env.scene_path == str(C.SCENE_ALL_XML)
    assert b.env.terminate_on_fall is False
    assert a.env.model is b.env.model
    assert a.env.model is not walk.env.model
    for _ in range(20):        # the swapped env really runs
        b.tick()
    assert np.isfinite(b.obs).all()


def test_bam_lab_keeps_private_models(monkeypatch):
    """MICRODUCK_ACTUATOR=bam opts out of sharing instead of blowing up: the
    BAM actuator rewrites model.dof_frictionloss every physics substep, so
    siblings on one model would retune each other's servos."""
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "bam")
    duck = V.Duck("m5", "bam", V._zero_infer, seed=6, env_kwargs={})
    assert duck.env.bam is not None
    assert duck.env._model_shared is False
    for _ in range(10):
        duck.tick()


# ------------------------------------------------------- trainee preview (A)

def test_trainee_preview_env_mirrors_stage(fake_popen):
    """The 🎓 preview env must practice what the stage practices — the
    behavior's spawn families under the ACTIVE stage's knobs, read per
    instance (os.environ belongs to the trainer subprocess; the shared lab
    process can't use it). Before this the trainee always spawned STANDING
    and the user watched stand-then-topple during 'learning to land'."""
    b = B.BEHAVIORS["backflip"]
    stage1 = b.curriculum[0]
    duck = V.Duck("trainee", "t", V._zero_infer, seed=97,
                  env_kwargs=V.trainee_env_kwargs(b, stage1.env))
    assert isinstance(duck.env, B.BehaviorEnv)
    ov = duck.env.spawn_overrides
    # Every knob the stage declares is mirrored verbatim — EXCEPT the spawn
    # MIX, which is deliberately LEANED to 85% of the stage's dominant family
    # (the trainer's exact mix includes spawns that look identical to plain
    # standing, and a watcher read that as "the mirroring is broken").
    MIX = "MICRODUCK_SPAWN_FAMILY_PROBS"
    for k, v in stage1.env.items():
        if k != MIX:
            assert ov[k] == v, k
    probs = [float(x) for x in ov[MIX].split(",")]
    declared = [float(x) for x in stage1.env[MIX].split(",")]
    if max(declared) >= 0.85:
        assert probs == declared        # already all-in: never watered down
    else:
        assert max(probs) == 0.85 and abs(sum(probs) - 1.0) < 0.01
    two_pi = 2 * 3.141592653589793
    # …and the resets really rehearse the stage: (almost) every episode is a
    # trick spawn, and any mid-roll drop lands inside the stage's own window
    # (stages without a declared window — the stand-steady one — spawn only
    # the credited standing pose, so there is no window to check).
    lo = float(stage1.env.get("MICRODUCK_BF_SPAWN_LO", 0.0))
    hi = float(stage1.env.get("MICRODUCK_BF_SPAWN_HI", two_pi))
    rots = []
    for seed in range(30):
        duck.env.reset(seed=seed)
        rots.append(duck.env._bf_rot)
    mid = [r for r in rots if 0.0 < r < two_pi - 1e-9]
    assert all(lo <= r <= hi for r in mid)
    assert sum(1 for r in rots if r > 0.0) >= 20
    # A handoff rebuild really swaps the knobs (kwargs differ -> new env).
    windowed = next(st for st in b.curriculum[1:]
                    if "MICRODUCK_BF_SPAWN_HI" in st.env)
    duck.rebuild_env(V.trainee_env_kwargs(b, windowed.env))
    assert (duck.env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == windowed.env["MICRODUCK_BF_SPAWN_HI"])


def test_on_stage_handoff_rebuilds_trainee(fake_popen):
    """The lab-loop hook: a stage advance narrates the handoff and re-mirrors
    the trainee's preview env onto the NEW stage's spawn knobs."""
    rebuilt: list[dict] = []
    trainee = types.SimpleNamespace(
        id="trainee", rebuild_env=lambda kw: rebuilt.append(kw))
    helper = types.SimpleNamespace(
        id="helper1", rebuild_env=lambda kw: rebuilt.append(kw))
    st = V.LabState([trainee, helper])
    st.job = V.TrainingJob("backflip", steps=1000)
    _finish_stage(st.job, 1000)
    st.job.poll()  # now on stage 2
    V.on_stage_handoff(st)
    assert len(rebuilt) == 2  # trainee AND helpers mirror the new stage
    assert rebuilt[0]["behavior_id"] == "backflip"
    stage2 = B.BEHAVIORS["backflip"].curriculum[1]
    assert (rebuilt[0]["spawn_overrides"]["MICRODUCK_BF_SPAWN_LO"]
            == stage2.env["MICRODUCK_BF_SPAWN_LO"])
    assert any("Training stage 2" in e for e in st.events)


# ------------------------------------------------------- showcase assign


def _mk_teach_run(name, behavior="backflip"):
    """A run dir shaped like a finished teach stage: assignable policy.onnx
    plus the behavior.json sibling env_kwargs/showcase resolution reads."""
    run = V.RUNS_DIR / name
    run.mkdir(parents=True)
    (run / "policy.onnx").touch()
    (run / "behavior.json").write_text(json.dumps({"behavior": behavior}))
    return run


def test_showcase_env_kwargs_uses_final_stage_knobs(fake_popen):
    """The "whole trick" assign rebuilds the duck's env as the behavior's own
    env under the LAST curriculum stage's knobs — the whole-trick stage by
    definition, so the server never learns any trick's knob names."""
    b = B.BEHAVIORS["backflip"]
    run = _mk_teach_run("teach-backflip-abc123-s4")
    kw = V.showcase_env_kwargs(str(run / "policy.onnx"))
    assert kw["behavior_id"] == "backflip"
    final = b.curriculum[-1].env
    ov = kw["spawn_overrides"]
    assert ov["MICRODUCK_BF_SPAWN_LO"] == final["MICRODUCK_BF_SPAWN_LO"]
    assert ov["MICRODUCK_BF_SPAWN_HI"] == final["MICRODUCK_BF_SPAWN_HI"]
    if b.spotter_fn is not None:
        # SPOTTED showcase: standing starts (the whole trick, front to back)
        # with the demo assist carrying the arc the actuators can't.
        assert kw["spotter"] is True
        assert ov["MICRODUCK_SPAWN_FAMILY_PROBS"] == "0.0,0.0"
    else:
        # Unspotted: lean the mix so the viewer sees the trick, not idling.
        probs = [float(x) for x in ov["MICRODUCK_SPAWN_FAMILY_PROBS"].split(",")]
        assert max(probs) == 0.85 and abs(sum(probs) - 1.0) < 0.01
    # A duck rebuilt with these kwargs really runs the behavior env.
    duck = V.Duck("d5", "x", V._zero_infer, seed=1, env_kwargs={})
    duck.rebuild_env(kw)
    assert isinstance(duck.env, B.BehaviorEnv)
    assert (duck.env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == final["MICRODUCK_BF_SPAWN_HI"])
    assert duck.env.spotter == (b.spotter_fn is not None)


def test_plain_assign_runs_the_behaviors_own_env(fake_popen):
    """A curriculum-stage policy assigned WITHOUT the showcase flag runs the
    behavior's OWN env — from ordinary standing starts, no stage overrides.

    The env class is the load-bearing half: an assigned trick policy used to
    land in the plain walking env, which resamples a random locomotion twist
    into the observation the policy acts on."""
    run = _mk_teach_run("teach-backflip-abc123-s4")
    kw = V.env_kwargs_for_policy_path(str(run / "policy.onnx"))
    assert kw == V.env_kwargs_for_behavior(B.BEHAVIORS["backflip"])
    assert kw["behavior_id"] == "backflip"
    assert kw["standing_spawns"] is True and "spawn_overrides" not in kw
    duck = V.Duck("d6", "x", V._zero_infer, seed=1, env_kwargs={})
    duck.rebuild_env(kw)
    assert isinstance(duck.env, B.BehaviorEnv)
    # ...and the standing pin really beats the recipe's spawn families: a
    # plain-assigned backflip must not start mid-roll (that is what the ✨
    # showcase assign is for), nor a stand duck lying on the floor.
    for _ in range(12):
        duck.env.reset()
        assert duck.env.last_spawn == "standing"


def test_assigned_trick_duck_never_sees_a_drive_command(fake_popen):
    """Regression: an assigned `run:teach-one_leg-*` duck runs one_leg's OWN
    env, so the twist command its policy reads stays zero for the whole
    episode — resets included.

    This is the mechanism behind the reported symptom (an assigned one_leg
    policy terminating ~5x/second, r-bar ~ -45, while the same weights held the
    pose for a full 20 s episode standalone): in the walking env the duck
    landed in, `_sample_commands` writes a fresh random locomotion twist into
    obs[48:51] at every reset and every resample window, and the lab's
    per-frame `set_cmd(zeros)` cannot take it back — the observation the policy
    acts on is already built. A trick policy trained on pinned-zero twist then
    gets a walk order it never trained for.
    """
    run = _mk_teach_run("teach-one_leg-abc123", behavior="one_leg")
    duck = V.Duck("d7", "one_leg", V._zero_infer, seed=1,
                  policy_id="run:teach-one_leg-abc123",
                  env_kwargs=V.env_kwargs_for_policy_path(str(run / "policy.onnx")))
    assert isinstance(duck.env, B.BehaviorEnv)
    assert duck.env.behavior.id == "one_leg"
    assert V.is_trick_duck(duck)      # ...which is why the lab sends it zeros
    for _ in range(int(duck.env.max_steps) + 200):   # past a reset or two
        assert np.allclose(duck.obs[48:51], 0.0), "drive command in a trick obs"
        duck.set_cmd(np.zeros(3, np.float32))  # what lab_loop sends every frame
        duck.tick()


def test_assigned_trick_duck_survives_a_full_episode(fake_popen):
    """The reported symptom itself: an assigned `run:teach-one_leg-*` duck runs
    a full episode without terminating.

    The brain is the shipped alpha_stand — a real balancing policy, since the
    stand pose is not a passive equilibrium here (see test_env_contract) and a
    zero brain topples on its own. Note that alpha_stand survives the walking
    env too, so this test pins the CONTRACT (nothing in the assigned duck's env
    kwargs may terminate a held pose) rather than reproducing the fall; the
    mechanism is pinned by the command-slot test above.
    """
    onnx_path = C.MICRODUCK_RL_DIR.parent / "microduck/policies/alpha_stand.onnx"
    if not onnx_path.exists():
        pytest.skip("microduck repo (shipped policies) not checked out next door")
    run = _mk_teach_run("teach-one_leg-abc123", behavior="one_leg")
    duck = V.Duck("d8", "one_leg", V._onnx_infer(onnx_path), seed=1,
                  policy_id="run:teach-one_leg-abc123",
                  env_kwargs=V.env_kwargs_for_policy_path(str(run / "policy.onnx")))
    steps = int(duck.env.max_steps) - 1
    for _ in range(steps):
        duck.set_cmd(np.zeros(3, np.float32))
        duck.tick()
    assert duck.falls == 0, f"assigned one_leg duck fell {duck.falls}x"
    assert duck.env.step_count == steps   # one episode, never reset


def test_showcase_is_noop_without_curriculum(fake_popen):
    """No curriculum (single-run behaviors, pollen policies, plain runs) =
    nothing for showcase to mean — callers fall back to a plain assign."""
    run = _mk_teach_run("teach-spin-abc123", behavior="spin")
    assert V.showcase_env_kwargs(str(run / "policy.onnx")) is None
    plain = V.RUNS_DIR / "plain-run"
    plain.mkdir(parents=True)  # no behavior.json sibling at all
    assert V.showcase_env_kwargs(str(plain / "policy.onnx")) is None
    assert V.showcase_env_kwargs(None) is None


def test_showcase_label_names_the_chain():
    """The roster label reads as the whole trick — the chain's name, not one
    stage's run name — with the ✨ showcase mark."""
    assert V.showcase_label("run:teach-backflip-ef471c-s4") == "backflip-ef471c ✨"
    # Non-chain ids (defensive: showcase only fires with a curriculum, but
    # the label must never crash on odd input) keep their name.
    assert V.showcase_label("run:some-run") == "some-run ✨"


def test_lab_state_showcase_round_trip(fake_popen, monkeypatch):
    """A showcase duck comes back showcasing after a restart — otherwise its
    persisted ✨ label would promise full-arc spawns its restored env no
    longer performs."""
    _mk_teach_run("teach-backflip-abc123-s4")
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    d = _fake_duck("d0", policy_id="run:teach-backflip-abc123-s4",
                   label="backflip-abc123 ✨")
    d.showcase = True
    V.save_lab_state([d])
    data = json.loads(V.lab_state_path().read_text())
    assert data["ducks"][0]["showcase"] is True
    restored = V.restore_ducks(V.lab_state_path())
    assert restored[0].showcase is True
    assert isinstance(restored[0].env, B.BehaviorEnv)
    final = B.BEHAVIORS["backflip"].curriculum[-1].env
    assert (restored[0].env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == final["MICRODUCK_BF_SPAWN_HI"])


# ------------------------------------------------------- policies palette

def _mkrun(name, mtime, artifact="policy.onnx"):
    d = V.RUNS_DIR / name
    d.mkdir(parents=True)
    (d / artifact).touch()
    import os as _os
    _os.utime(d / artifact, (mtime, mtime))
    return d


def test_discover_policies_timestamps_sort_and_chains(fake_popen):
    """Run entries carry mtime (epoch seconds), sort newest-first, and stage
    runs are annotated with their chain prefix + 1-based stage so the panel
    can fold a curriculum chain into one family."""
    _mkrun("teach-backflip-aaaaaa-s1", 1_000)
    _mkrun("teach-backflip-aaaaaa-s2", 5_000)
    _mkrun("plain-run", 3_000)
    _mkrun("no-policy-yet", 9_000, artifact="live.onnx")  # not assignable yet
    runs = [p for p in V.discover_policies() if p["group"] == "runs"]
    assert [p["label"] for p in runs] == [
        "teach-backflip-aaaaaa-s2", "plain-run", "teach-backflip-aaaaaa-s1"]
    by = {p["label"]: p for p in runs}
    assert by["plain-run"]["mtime"] == 3_000
    # Non-chain runs carry no chain/stage keys at all.
    assert "chain" not in by["plain-run"] and "stage" not in by["plain-run"]
    assert by["teach-backflip-aaaaaa-s1"]["chain"] == "teach-backflip-aaaaaa"
    assert by["teach-backflip-aaaaaa-s1"]["stage"] == 1
    assert by["teach-backflip-aaaaaa-s2"]["stage"] == 2
    # ids/paths keep the exact assignable shape (drag-to-assign unchanged).
    assert by["teach-backflip-aaaaaa-s2"]["id"] == "run:teach-backflip-aaaaaa-s2"
    assert by["teach-backflip-aaaaaa-s2"]["path"].endswith(
        "teach-backflip-aaaaaa-s2/policy.onnx")
    json.dumps(runs)


def test_policy_palette_marks_interrupted_and_unfinished_runs(fake_popen):
    stopped = _mkrun("teach-spin-stopped", 1000)
    unfinished = _mkrun("teach-spin-unfinished", 1000)
    _mkrun("teach-spin-legacy", 1000)
    (stopped / "progress.jsonl").write_text(
        '{"steps": 50, "done": false, "stopped": "user"}\n')
    (unfinished / "progress.jsonl").write_text(
        '{"steps": 50, "done": false}\n')
    by = {p["label"]: p for p in V.discover_policies() if p["group"] == "runs"}
    assert by["teach-spin-stopped"]["runStatus"] == "stopped"
    assert by["teach-spin-unfinished"]["runStatus"] == "unfinished"
    assert by["teach-spin-legacy"]["runStatus"] == "done"


def test_teach_checkpoint_onnx_appears_in_palette(fake_popen, monkeypatch):
    run = _mkrun("teach-spin-history", 1000)
    (run / "behavior.json").write_text('{"behavior": "spin"}')
    ckpt = run / "checkpoints"
    ckpt.mkdir()
    for name in ("model_000500000.zip", "vecnormalize_000500000.pkl",
                 "policy_000500000.onnx"):
        (ckpt / name).touch()
    entries = [p for p in V.discover_policies()
               if p["id"] == "ckpt:teach-spin-history@500k"]
    assert len(entries) == 1
    assert entries[0]["path"] == str(ckpt / "policy_000500000.onnx")
    sentinel = object()
    monkeypatch.setattr(V, "_onnx_infer", lambda path: sentinel)
    V._infer_cache.clear()
    assert V.load_policy_infer(entries[0]["id"]) is sentinel
    assert V.env_kwargs_for_policy_path(entries[0]["path"])["behavior_id"] == "spin"
    assert V.is_trick_duck(types.SimpleNamespace(
        id="d9", policy_id=entries[0]["id"]))


def test_run_mtime_fallbacks(fake_popen):
    """policy.onnx first, then live.onnx, then progress.jsonl — a run still
    training (or stopped before export) must still get a timestamp."""
    import os as _os
    run = V.RUNS_DIR / "r"
    run.mkdir(parents=True)
    assert V._run_mtime(run) is None
    (run / "progress.jsonl").touch()
    _os.utime(run / "progress.jsonl", (10, 10))
    assert V._run_mtime(run) == 10
    (run / "live.onnx").touch()
    _os.utime(run / "live.onnx", (20, 20))
    assert V._run_mtime(run) == 20
    (run / "policy.onnx").touch()
    _os.utime(run / "policy.onnx", (30, 30))
    assert V._run_mtime(run) == 30



def test_spotter_is_demo_only_and_self_clearing(fake_popen):
    """The spotter must never leak into training: it is off unless an env is
    built with spotter=True, it releases into the policy's own territory, and
    it never carries an applied force across an episode boundary."""
    plain = B.BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                          action_delay=False, random_yaw=False, seed=0)
    assert plain.spotter is False           # trainers build envs this way
    spotted = B.BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                            action_delay=False, random_yaw=False, seed=0,
                            spotter=True,  # as the spotted showcase builds it:
                            spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0.0,0.0"})
    import numpy as np
    spotted.reset(seed=0)
    for _ in range(40):
        spotted.step(np.zeros(14, np.float32))
    assert spotted.spotter_active            # assisting through the dead zone
    assert spotted.data.qfrc_applied[4] < 0  # backward pitch
    spotted._bf_rot = 5.0                    # past the assist's release point
    spotted.step(np.zeros(14, np.float32))
    assert not spotted.spotter_active and spotted.data.qfrc_applied[4] == 0.0
    spotted.reset(seed=1)                    # and never across episodes
    assert not spotted.spotter_active and float(
        np.max(np.abs(spotted.data.qfrc_applied))) == 0.0


def test_chain_warm_start_from_another_run(fake_popen, tmp_path):
    """initFrom + startStage runs the whole CHAIN warm-started from that run
    (initFrom alone keeps its single-run fine-tune meaning). The backflip
    could not learn to hold a stand on its own; the crouch/one-leg policies
    hold one for a full episode, and inheriting that is the point."""
    donor = V.RUNS_DIR / "teach-crouch-donor"
    donor.mkdir(parents=True)
    (donor / "model.zip").touch()
    (donor / "policy.onnx").touch()
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500,
                        start_stage=1, stage_init_from=donor)
    argv = fake_popen[0].cmd
    assert _flag(argv, "--init-from") == str(donor)   # stage 1 inherits it
    assert job.run_name.endswith("-s1")
    _finish_stage(job, 1000)
    job.poll()
    # …and stage 2 chains from stage 1 as usual, not from the donor again.
    assert _flag(fake_popen[1].cmd, "--init-from") != str(donor)
    assert job.run_name.endswith("-s2")


def test_showcase_duck_hands_off_to_a_standing_brain(fake_popen):
    """A finished trick hot-swaps to a brain that can stand — the robot's own
    pattern. The trick policy lands in a crouch it cannot rise from; the
    handoff turned 0/12 stands into 11/12 with 8.1 s holds."""
    run = _mk_teach_run("teach-backflip-abc123-s4")
    ho = V.handoff_for(str(run / "policy.onnx"))
    assert ho is not None and ho[1] == "alpha_stand"
    duck = V.Duck("d9", "x", V._zero_infer, seed=1, env_kwargs={})
    duck.handoff_infer, duck.handoff_label = ho
    # Before the trick completes, the duck stays on its own brain…
    duck.env._bf_rot = 0.0
    assert duck._handoff_due() is False
    # …and once the roll is done with both feet down, control moves over.
    duck.env._bf_rot = 6.0
    duck.env.foot_contact_state = {"left": True, "right": True}
    assert duck._handoff_due() is True
    duck.tick()
    assert duck.handed is True
    duck.reset()
    assert duck.handed is False   # every episode starts on the trick again


def test_find_ball_hands_off_to_the_kick_it_aims(fake_popen):
    """find_ball exists to aim the ball-BLIND kick policies, so its showcase
    handoff target is the kick, not the stand every trick lands into — and
    the condition comes from the behavior, so this duck and render-rollout
    ask the identical question instead of two copies of a rule."""
    run = _mk_teach_run("teach-find_ball-abc123-s3", behavior="find_ball")
    ho = V.handoff_for(str(run / "policy.onnx"))
    assert ho is not None and ho[1] == "ball_kick_right"
    duck = V.Duck("d7", "fb", V._zero_infer, seed=1,
                  env_kwargs={"behavior_id": "find_ball"})
    duck.handoff_infer, duck.handoff_label = ho
    duck.env._ball_aim_steps = 0
    assert duck._handoff_due() is False
    duck.env._ball_aim_steps = B._BALL_AIM_STEPS
    assert duck._handoff_due() is True


def test_find_ball_keeps_the_heading_it_turned_to(fake_popen):
    """The lab's post-handoff yaw correction undoes the drift a LANDING
    imparts. find_ball's turn is the deliverable, so the correction is off
    for it — otherwise the commander spins the duck off the ball it just
    squared up on, and hands the kick nothing."""
    flip = V.Duck("d5", "bf", V._zero_infer, seed=1,
                  env_kwargs={"behavior_id": "backflip"})
    ball = V.Duck("d6", "fb", V._zero_infer, seed=1,
                  env_kwargs={"behavior_id": "find_ball"})
    for duck in (flip, ball):
        duck.handed = True
        duck._settle = 100          # past the ~1 s landing settle
        duck.env.home_yaw = 1.2     # a heading 69 deg off where it is now
    assert flip._recenter_wz() is not None, "the flip still recentres"
    assert ball._recenter_wz() is None


def test_plain_assign_has_no_handoff(fake_popen):
    """Only showcase ducks hand off — a single-stage assign is being watched
    for what THAT policy does, unaided."""
    duck = V.Duck("d8", "x", V._zero_infer, seed=1, env_kwargs={})
    assert duck.handoff_infer is None
    duck.env._bf_rot = 6.0
    duck.env.foot_contact_state = {"left": True, "right": True}
    duck.tick()
    assert duck.handed is False


# --------------------------------------------------- practice-step budget

def _declared(behavior_id):
    """What the recipe itself asks for: per-stage steps, or the single-run
    default. Read from the library, never hardcoded — the budgets are live
    tuning surface."""
    b = B.BEHAVIORS[behavior_id]
    return [s.steps for s in b.curriculum] or [b.default_steps]


def test_split_step_budget_keeps_ratios_and_sums_exactly():
    """A chain's SHAPE is the curriculum's whole point, so a user's total is
    distributed in proportion to the declared stages — and the parts add up
    to exactly the number they typed, because a stage list that doesn't sum
    to the headline figure is the same misreport the control exists to fix."""
    declared = [1_000_000, 2_000_000, 1_000_000, 1_500_000, 1_500_000]
    for total in (500_000, 3_000_000, 7_000_000, 12_345_678):
        parts = V.split_step_budget(declared, total)
        assert sum(parts) == total
        scale = total / sum(declared)
        for want, got in zip(declared, parts):
            assert abs(got - want * scale) <= 1.0   # rounding only
    # The declared total reproduces the declared plan exactly.
    assert V.split_step_budget(declared, sum(declared)) == declared
    # Scaled to nothing, no stage is rounded out of existence.
    assert V.split_step_budget(declared, 5) == [1, 1, 1, 1, 1]
    assert V.split_step_budget([], 1000) == []


def test_no_budget_reproduces_the_declared_defaults(fake_popen):
    """Regression guard: the control is opt-in. Omitting it must train
    EXACTLY what the recipe declares, stage for stage."""
    job = V.TrainingJob("backflip")
    assert job.stage_steps == _declared("backflip")
    assert job.budget is None
    assert job.payload()["stepBudget"] == sum(_declared("backflip"))
    assert _flag(fake_popen[0].cmd, "--steps") == str(_declared("backflip")[0])

    single = V.TrainingJob("spin")
    assert single.stage_steps == [B.BEHAVIORS["spin"].default_steps]


def test_total_budget_splits_across_the_chain(fake_popen):
    """The user's number is the TOTAL for the whole trick. Reading it as
    "this many per stage" would multiply the real cost by the stage count —
    the exact surprise this control removes."""
    declared = _declared("backflip")
    job = V.TrainingJob("backflip", budget=3_500_000)
    assert sum(job.stage_steps) == 3_500_000
    scale = 3_500_000 / sum(declared)
    for want, got in zip(declared, job.stage_steps):
        assert abs(got - want * scale) <= 1.0
    # What the payload advertises is what the trainer is told, stage by
    # stage (the trainer copies --steps into the run's behavior.json, so the
    # provenance on disk follows automatically).
    p = job.payload()
    assert p["stageSteps"] == job.stage_steps
    assert p["stepBudget"] == 3_500_000
    assert p["progress"]["overallTotal"] == 3_500_000
    assert _flag(fake_popen[0].cmd, "--steps") == str(job.stage_steps[0])
    for i in range(1, len(declared)):
        _finish_stage(job, job.stage_steps[i - 1])
        job.poll()
        assert _flag(fake_popen[i].cmd, "--steps") == str(job.stage_steps[i])


def test_total_budget_on_a_single_run_behavior(fake_popen):
    job = V.TrainingJob("spin", budget=1_250_000)
    assert job.stage_steps == [1_250_000]
    assert _flag(fake_popen[0].cmd, "--steps") == "1250000"
    # A fine-tune is a single run too — it gets the whole budget, once.
    prev = V.RUNS_DIR / "prev"
    prev.mkdir(parents=True)
    ft = V.TrainingJob("backflip", init_from=prev, budget=800_000)
    assert ft.stage_steps == [800_000]


def test_budget_is_clamped_to_a_sane_range(fake_popen):
    """A typo in the millions field must not become an overnight run."""
    assert V.TrainingJob("spin", budget=1).stage_steps == [V.MIN_STEP_BUDGET]
    assert (V.TrainingJob("spin", budget=10 ** 12).stage_steps
            == [V.MAX_STEP_BUDGET])


def test_per_stage_budget_wins_over_the_split(fake_popen):
    """Per-stage step counts layer exactly like per-stage weights: explicit
    value wins, everything else keeps its proportional share."""
    declared = _declared("backflip")
    job = V.TrainingJob("backflip", budget=3_500_000,
                        stage_budgets={"2": 900_000, "0": 5, "99": 5,
                                       "nope": 5, "3": 0})
    split = V.split_step_budget(declared, 3_500_000)
    assert job.stage_budgets == {2: 900_000}          # junk keys drop
    assert job.stage_steps == [split[0], 900_000, *split[2:]]
    assert job.payload()["stageBudgets"] == {"2": 900_000}
    # The pinned stage is honest on the wire when it actually launches.
    _finish_stage(job, job.stage_steps[0])
    job.poll()
    assert _flag(fake_popen[1].cmd, "--steps") == "900000"


def test_per_stage_budget_ignored_for_single_run_jobs(fake_popen):
    """A fine-tune has no stages to key — the layer drops rather than
    half-applying one stage's number to the whole run."""
    prev = V.RUNS_DIR / "prev"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", init_from=prev, budget=800_000,
                        stage_budgets={"2": 900_000})
    assert job.stage_budgets == {}
    assert job.stage_steps == [800_000]


def test_steps_override_still_beats_a_chosen_budget(fake_popen):
    """TEACH_STEPS_OVERRIDE keeps its own meaning — that many steps for EVERY
    stage — so a probe runs a tiny job whatever the user last picked, and the
    probe's 1k never leaks back into the sticky budget."""
    job = V.TrainingJob("backflip", steps=1000, budget=8_000_000,
                        stage_budgets={"2": 5_000_000})
    assert job.stage_steps == [1000] * len(_declared("backflip"))
    assert job.budget is None
    assert _flag(fake_popen[0].cmd, "--steps") == "1000"


def test_teach_endpoint_makes_the_budget_sticky(fake_popen, monkeypatch):
    """End to end through the handler: the chosen budget reaches the job,
    lands in the sticky file, and a later ask with no number keeps it — the
    same rule the sliders follow, so 'always 4M' is typed once."""
    # The handler reloads behaviors.py to pick up recipe edits; in-process
    # that swaps class objects other tests hold, so neutralize it here.
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")

    out = asyncio.run(teach(V.TeachReq(text="do a backflip", steps=3_500_000)))
    assert out["matched"]
    assert out["job"]["stepBudget"] == 3_500_000
    assert V.load_teach_weights()["backflip"]["steps"] == 3_500_000

    asyncio.run(stop(force=True))
    again = asyncio.run(teach(V.TeachReq(text="do a backflip")))
    assert again["job"]["stepBudget"] == 3_500_000
    assert again["job"]["stageSteps"] == out["job"]["stageSteps"]

    # A per-stage pin sticks too, and a new total re-splits around it.
    asyncio.run(stop(force=True))
    pinned = asyncio.run(teach(V.TeachReq(text="do a backflip",
                                          stageSteps={"1": 400_000})))
    assert pinned["job"]["stageBudgets"] == {"1": 400_000}
    assert pinned["job"]["stageSteps"][0] == 400_000
    assert V.load_teach_weights()["backflip"]["stageSteps"] == {"1": 400_000}


# ------------------------------------------------- imitation runs keep their clip


def _seed_clip(monkeypatch, tmp_path, name):
    clips = tmp_path / "clips"
    clips.mkdir(exist_ok=True)
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", str(clips))
    (clips / f"{name}.json").write_text(json.dumps(
        {"version": 1, "name": name, "duration": 1.0, "loop": False,
         "keys": [{"t": 0.0, "joints": [float(j) for j in C.DEFAULT_POSE], "rootPitch": 0.0}]}))


def _launch_env(proc):
    return proc.kwargs.get("env") or {}


def test_teach_text_is_the_behavior_id(fake_popen, monkeypatch, tmp_path):
    """The panel's retrain/fine-tune buttons send the card's id. Before, they
    echoed its TITLE — and an imitation card is titled after its clip, so
    `Perform “backflip”` retrained the floor-roll recipe, `Perform
    “sprint-cycle”` the runner, and a clip named after nothing matched
    nothing at all."""
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    _seed_clip(monkeypatch, tmp_path, "backflip")
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")

    out = asyncio.run(teach(V.TeachReq(text="imitate", clip="backflip")))
    assert out["matched"]
    assert out["job"]["behavior"]["id"] == "imitate"
    assert out["job"]["behavior"]["clip"] == "backflip"
    assert out["job"]["behavior"]["title"] == "Perform “backflip”"
    assert _launch_env(fake_popen[-1])["MICRODUCK_CLIP"] == "backflip"
    asyncio.run(stop(force=True))

    # The lab's own display title still reaches the imitation recipe with
    # that clip (an older panel echoes it back verbatim) — it must never
    # pick the recipe that owns a keyword inside the clip's name.
    out = asyncio.run(teach(V.TeachReq(text="Perform “backflip”")))
    assert out["matched"], out
    assert out["job"]["behavior"]["id"] == "imitate"
    assert _launch_env(fake_popen[-1])["MICRODUCK_CLIP"] == "backflip"
    asyncio.run(stop(force=True))

    # ...and a title naming a clip that was never saved is refused, not
    # silently trained against the recipe's default clip.
    out = asyncio.run(teach(V.TeachReq(text="Perform “nope”")))
    assert not out["matched"] and "nope" in out["message"]

    # Every other card id round-trips too — the panel special-cases nothing.
    # Scoped to the DUCK: the registry also carries another body's tasks
    # (behaviors/g1_tasks.py), and this app's roster is a duck one, so a G1
    # id correctly matches nothing here.
    for b in B.for_robot("microduck"):
        out = asyncio.run(teach(V.TeachReq(text=b.id)))
        assert out["matched"] and out["job"]["behavior"]["id"] == b.id, b.id
        asyncio.run(stop(force=True))


def test_adopted_imitation_run_keeps_its_clip(fake_popen, monkeypatch, tmp_path):
    """Seating a finished imitation run (POST /teach/load — what clicking its
    duck does) must bring its clip back: the card is titled after it, and a
    ✨ fine-tune practices the SAME motion. train_behavior records the clip in
    behavior.json for exactly this; a fine-tune request that omits the clip
    inherits the run's, rather than the recipe's default."""
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    _seed_clip(monkeypatch, tmp_path, "hop")
    _seed_clip(monkeypatch, tmp_path, "backflip")
    run = _seed_finished_run("teach-imitate-hop-abc123", behavior="imitate")
    json_meta = json.loads((run / "behavior.json").read_text())
    (run / "behavior.json").write_text(json.dumps({**json_meta, "clip": "hop"}))
    for f in ("model.zip", "vecnormalize.pkl"):
        (run / f).write_bytes(b"x")
    app = V.make_app([])
    load = _endpoint(app, "/teach/load", "POST")
    teach = _endpoint(app, "/teach", "POST")

    seated = asyncio.run(load(V.LoadRunReq(policy="run:teach-imitate-hop-abc123")))
    assert seated["ok"], seated
    assert seated["job"]["behavior"]["clip"] == "hop"
    assert seated["job"]["behavior"]["title"] == "Perform “hop”"

    # Fine-tune the way the panel does now: id + the card's clip.
    out = asyncio.run(teach(V.TeachReq(text="imitate", clip="hop",
                                       initFrom="teach-imitate-hop-abc123")))
    assert out["matched"], out
    assert _launch_env(fake_popen[-1])["MICRODUCK_CLIP"] == "hop"
    assert _flag(fake_popen[-1].cmd, "--init-from") == str(run)
    asyncio.run(_endpoint(app, "/teach/stop", "POST")(force=True))

    # An older panel sends no clip at all: the run's own clip still wins over
    # the recipe default ("backflip").
    out = asyncio.run(teach(V.TeachReq(text="imitate",
                                       initFrom="teach-imitate-hop-abc123")))
    assert out["matched"], out
    assert _launch_env(fake_popen[-1])["MICRODUCK_CLIP"] == "hop"
    asyncio.run(_endpoint(app, "/teach/stop", "POST")(force=True))

    # A run that predates the record seats clip-less, like before.
    old = _seed_finished_run("teach-imitate-old111", behavior="imitate")
    seated = asyncio.run(load(V.LoadRunReq(policy="run:teach-imitate-old111")))
    assert seated["ok"] and "clip" not in seated["job"]["behavior"]
    assert V.run_clip(old) is None


def test_teach_steps_override_does_not_poison_the_sticky_budget(fake_popen,
                                                                monkeypatch):
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    V.save_teach_weights({"spin": {**V.empty_sticky(), "steps": 4_000_000}})
    monkeypatch.setenv("TEACH_STEPS_OVERRIDE", "1000")
    app = V.make_app([])
    out = asyncio.run(_endpoint(app, "/teach", "POST")(
        V.TeachReq(text="spin in place")))
    assert out["job"]["stageSteps"] == [1000]
    assert V.load_teach_weights()["spin"]["steps"] == 4_000_000


# --------------------------------------------------- teach/load (adopt)


def _seed_finished_run(name, behavior="spin", weights=None, steps=2000):
    run = V.RUNS_DIR / name
    run.mkdir(parents=True)
    (run / "behavior.json").write_text(json.dumps(
        {"behavior": behavior, "steps": steps, "weights": weights or {}}))
    (run / "progress.jsonl").write_text(json.dumps(
        {"steps": steps, "total": steps, "ep_rew": 4.2, "done": True}) + "\n")
    return run


def test_adopt_interrupted_export_is_resumable_not_finished(fake_popen):
    run = _seed_finished_run("teach-spin-interrupted", steps=100_000)
    for name in ("model.zip", "vecnormalize.pkl", "policy.onnx"):
        (run / name).touch()
    (run / "progress.jsonl").write_text(json.dumps({
        "steps": 50_000, "total": 100_000, "done": False,
        "stopped": "user"}) + "\n")
    job = V.TrainingJob.adopt(run.name)
    assert job.status == "stopped" and job.payload()["canResume"]

    # A crash after a checkpoint can leave an older export in the directory.
    # A terminal done:false record must not reclassify it as completed.
    (run / "progress.jsonl").write_text(json.dumps({
        "steps": 50_000, "total": 100_000, "done": False}) + "\n")
    job = V.TrainingJob.adopt(run.name)
    assert job.status == "failed" and job.payload()["canResume"]


def test_teach_load_seats_finished_run(fake_popen, monkeypatch):
    """POST /teach/load pulls a finished run into the panel: its payload
    streams in "done" state (sliders unlocked, fine-tune targeting that run)
    with NO subprocess launched — and is refused for non-runs and while a
    job actually trains."""
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    _seed_finished_run("teach-spin-adopt1", weights={"spin_fast": 3.5})
    app = V.make_app([])
    load = _endpoint(app, "/teach/load", "POST")
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")

    out = asyncio.run(load(V.LoadRunReq(policy="run:teach-spin-adopt1")))
    assert out["ok"]
    assert out["job"]["runName"] == "teach-spin-adopt1"
    assert out["job"]["status"] == "done"
    assert out["job"]["behavior"]["id"] == "spin"
    assert out["job"]["weights"]["spin_fast"] == 3.5
    assert fake_popen == []  # seated, not launched

    # Shipped policies / unknown names have no recipe → clean refusal.
    assert not asyncio.run(load(V.LoadRunReq(policy="pollen:alpha_stand")))["ok"]
    # A run predating behavior.json can't be reconstructed → clean refusal.
    (V.RUNS_DIR / "teach-spin-bare").mkdir(parents=True)
    bare = asyncio.run(load(V.LoadRunReq(policy="run:teach-spin-bare")))
    assert not bare["ok"] and "behavior.json" in bare["message"]

    # While a job trains, loading is refused outright — never yank a live job.
    assert asyncio.run(teach(V.TeachReq(text="spin in place")))["matched"]
    refused = asyncio.run(load(V.LoadRunReq(policy="run:teach-spin-adopt1")))
    assert not refused["ok"] and "stop it first" in refused["message"]
    asyncio.run(stop(force=True))


def test_fine_tune_merges_over_the_seated_runs_weights(fake_popen, monkeypatch):
    """Retrain/fine-tune of a seated run must carry the weights that run
    ACTUALLY trained under. Observed 2026-09-04: an imitate run seated via
    /teach/load with rotation_match at 6.05 was fine-tuned after adding one
    catalog term; the panel sent only the touched keys, the server took that
    dict as the complete override set, and rotation_match fell back to the
    recipe's 4.00 with no message. With initFrom the server now layers the
    request over the run's behavior.json — a delta can't reset the rest."""
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    run = _seed_finished_run("teach-spin-seat7", weights={"spin_fast": 6.05})
    (run / "model.zip").touch()
    (run / "vecnormalize.pkl").touch()
    # A stale sticky crank must NOT be the fine-tune's base either — the
    # brain being continued is the base.
    V.save_teach_weights({"spin": {**V.empty_sticky(),
                                   "weights": {"spin_fast": 9.0}}})
    app = V.make_app([])
    load = _endpoint(app, "/teach/load", "POST")
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")

    seated = asyncio.run(load(V.LoadRunReq(policy="run:teach-spin-seat7")))
    assert seated["ok"] and seated["job"]["weights"]["spin_fast"] == 6.05

    # The panel's old shape: only the added catalog term in `weights`.
    out = asyncio.run(teach(V.TeachReq(text="spin in place",
                                       initFrom="teach-spin-seat7",
                                       weights={"head_up": 1.0})))
    assert out["matched"]
    argv = fake_popen[0].cmd
    assert _flag(argv, "--init-from") == str(run)
    assert json.loads(_flag(argv, "--weights-json")) == {"spin_fast": 6.05,
                                                         "head_up": 1.0}
    assert out["job"]["weights"]["spin_fast"] == 6.05
    assert out["job"]["weights"]["head_up"] == 1.0
    # …and the merged set is what sticks for this behavior.
    assert V.load_teach_weights()["spin"]["weights"] == {"spin_fast": 6.05,
                                                         "head_up": 1.0}
    asyncio.run(stop(force=True))

    # An explicit value for a key the run trained under still wins (that is
    # the user dragging the slider), and no weights at all means "exactly
    # what the run had", not the sticky set or the recipe defaults.
    out = asyncio.run(teach(V.TeachReq(text="spin in place",
                                       initFrom="teach-spin-seat7",
                                       weights={"spin_fast": 3.0})))
    assert json.loads(_flag(fake_popen[1].cmd, "--weights-json")) == {
        "spin_fast": 3.0}
    asyncio.run(stop(force=True))
    out = asyncio.run(teach(V.TeachReq(text="spin in place",
                                       initFrom="teach-spin-seat7")))
    assert json.loads(_flag(fake_popen[2].cmd, "--weights-json")) == {
        "spin_fast": 6.05}
    asyncio.run(stop(force=True))

    # A run predating behavior.json fine-tunes under the request alone.
    bare = V.RUNS_DIR / "teach-spin-bare7"
    bare.mkdir(parents=True)
    (bare / "model.zip").touch()
    (bare / "vecnormalize.pkl").touch()
    out = asyncio.run(teach(V.TeachReq(text="spin in place",
                                       initFrom="teach-spin-bare7",
                                       weights={"head_up": 1.0})))
    assert out["matched"]
    assert json.loads(_flag(fake_popen[3].cmd, "--weights-json")) == {
        "head_up": 1.0}
    asyncio.run(stop(force=True))


def test_adopted_job_survives_the_lab_poll_cycle(fake_popen):
    """The adopted job has no subprocess: the lab loop's poll()/stop()/
    stats.sample() must all be safe with proc=None, and poll() replays the
    run's progress.jsonl so the panel shows its real final numbers."""
    _seed_finished_run("teach-spin-adopt2")
    job = V.TrainingJob.adopt("teach-spin-adopt2")
    assert job.proc is None and job.status == "done"
    job.poll()
    assert job.progress["ep_rew"] == 4.2 and job.progress["done"] is True
    assert job.train_fps() is None
    stats = V.StatsSampler().sample(job)
    assert stats["trainer"] is None and stats["trainFps"] is None
    job.stop(force=True)  # lifespan shutdown stops whatever job is seated
    assert fake_popen == []


# --------------------------------------------------- forward-speed readout


def _pose(env, quat, world_vel):
    """Impose a known heading and a known WORLD-frame trunk velocity.

    qvel[0:3] of a free joint is global linear velocity, so this is a clean
    "the duck is moving that way, facing this way" fixture; mj_forward
    refreshes the derived rotation matrices heading_lin_vel reads.
    """
    import mujoco
    env.data.qpos[3:7] = quat
    env.data.qvel[:] = 0.0
    env.data.qvel[0:3] = world_vel
    mujoco.mj_forward(env.model, env.data)


def _yaw(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, 0.0, np.sin(h)]


def _pitch(deg):
    """+deg = nose-down (right-handed about the trunk's +y, as POST /pose)."""
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, np.sin(h), 0.0]


def _inertial_forward(env) -> float:
    """The WRONG number: mj_objectVelocity(flg_local=1)'s "forward" slot."""
    import mujoco
    out = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                             env.trunk_body_id, out, 1)
    return float(out[3])  # res is rotation:translation, so [3] is local vx


def test_reported_speed_is_the_heading_frame():
    """The readout must project world velocity onto the duck's YAW facing —
    not onto mj_objectVelocity's local frame (which is the trunk's INERTIAL
    frame, whose axes are nowhere near the body's), and not onto the body's
    own x axis (which pays for diving). This test exists to FAIL if either
    substitution is ever made; see Duck.sample_speed."""
    duck = V.Duck("spd0", "x", V._zero_infer, seed=1, env_kwargs={})
    env = duck.env

    # Turned 90° left and moving world-+y at 0.6 m/s: the duck is running
    # straight ahead at 0.6.
    _pose(env, _yaw(90), [0.0, 0.6, 0.0])
    duck.speed_hist.clear()
    duck.sample_speed()
    assert duck.forward_speed() == pytest.approx(0.6, abs=1e-3)
    # World +x is 0 here, so this also pins that the readout is not just
    # qvel[0] with a heading-shaped comment on it.
    assert env.data.qvel[0] == pytest.approx(0.0, abs=1e-9)
    # And the inertial-frame call reads ~0 for this same motion — on this
    # trunk its x axis is close to PERPENDICULAR to forward, so a readout
    # built on it is blind to forward speed entirely.
    assert abs(_inertial_forward(env)) < 0.05

    # Nose-down 45° and falling straight down: no ground is being covered,
    # so forward speed is zero. The body-x projection would score +0.71 for
    # this dive — that is the other half of the trap.
    duck.speed_hist.clear()
    _pose(env, _pitch(45), [0.0, 0.0, -1.0])
    duck.sample_speed()
    assert duck.forward_speed() == pytest.approx(0.0, abs=1e-3)
    body_x = env.data.xmat[env.trunk_body_id].reshape(3, 3) @ np.array([1.0, 0, 0])
    assert float(body_x @ np.array([0.0, 0.0, -1.0])) > 0.5


def test_forward_speed_averages_half_a_second():
    """A stepping gait's instantaneous forward speed swings within each
    stride; the row shows the mean of the last SPEED_WINDOW control steps
    (0.5 s at 50 Hz) so the number is readable, and older samples age out."""
    assert V.SPEED_WINDOW == 25  # 0.5 s at TICK_HZ
    duck = V.Duck("spd1", "x", V._zero_infer, seed=2, env_kwargs={})
    duck.speed_hist.clear()
    assert duck.forward_speed() is None  # nothing sampled yet → "—", not 0.00

    duck.speed_hist.extend([0.0, 0.8] * (V.SPEED_WINDOW // 2))
    assert duck.forward_speed() == pytest.approx(0.4, abs=1e-3)
    # Fully overwriting the window drops every older sample.
    duck.speed_hist.extend([0.2] * V.SPEED_WINDOW)
    assert duck.forward_speed() == pytest.approx(0.2, abs=1e-3)


def test_speed_window_clears_on_reset_and_policy_swap():
    """Speed is per EPISODE and per BRAIN: half a second of the run that just
    ended in a faceplant must not follow the duck into its next episode, and
    the outgoing policy's speed must not be credited to the incoming one."""
    duck = V.Duck("spd2", "x", V._zero_infer, seed=3, env_kwargs={})
    duck.speed_hist.extend([0.3] * V.SPEED_WINDOW)
    duck.reset()
    assert duck.forward_speed() is None

    duck.speed_hist.extend([0.3] * V.SPEED_WINDOW)
    duck.swap_policy("other", V._zero_infer)
    assert duck.forward_speed() is None


def test_tick_records_a_speed_sample():
    """The window is filled by the lab loop's own tick, not only by tests."""
    duck = V.Duck("spd3", "x", V._zero_infer, seed=4, env_kwargs={})
    duck.speed_hist.clear()
    duck.tick()
    assert len(duck.speed_hist) == 1
    assert duck.forward_speed() is not None


# ------------------------------------------------ frame broadcast robustness


def _lab_state(app):
    """The handler's documented LabState, independent of closure structure."""
    return app.state.lab


def test_broadcast_survives_a_client_arriving_mid_send():
    """A browser connecting while the loop is suspended in `await
    send_text` mutates st.clients mid-iteration. That RuntimeError killed
    lab_loop outright — and since nothing ever retrieves that task's
    exception, the lab went on serving HTTP and accepting sockets while
    every duck froze, with no line in the log: the viewer's badge sat green
    over an empty scene. Iterate a snapshot instead.

    The frame payload is asserted here too: this is the only place the real
    per-duck dict is built."""
    walker = V.Duck("d0", "w", V._zero_infer, seed=1, env_kwargs={})
    trainee = V.Duck("trainee", "t", V._zero_infer, seed=2, env_kwargs={})
    app = V.make_app([walker, trainee])
    st = _lab_state(app)

    class Joiner:
        """A client whose send is slow enough for a second tab to connect."""

        def __init__(self):
            self.frames: list[str] = []

        async def send_text(self, frame):
            self.frames.append(frame)
            st.clients.add(object())  # another browser lands mid-broadcast

    joiner = Joiner()
    st.clients.add(joiner)

    async def run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.4)  # ~10 broadcasts at 25 Hz

    asyncio.run(run())

    # Before the fix this stopped at exactly one — the first send blew the
    # loop up and nothing was ever sent again.
    assert len(joiner.frames) > 2, "lab_loop stopped broadcasting"

    ducks = json.loads(joiner.frames[-1])["ducks"]
    assert [d["id"] for d in ducks] == ["d0", "trainee"]
    assert isinstance(ducks[0]["speed"], float)
    # A steerable duck reports what it was asked for; the trainee runs a
    # pinned-zero twist, so it reports achieved only.
    assert isinstance(ducks[0]["cmdSpeed"], float)
    assert ducks[1]["cmdSpeed"] is None


def test_scale_after_a_stop_does_not_strand_a_trainer(fake_popen, monkeypatch):
    """A stop that lands while a rescale is in flight must win.

    scale() runs on a worker thread while /teach/stop runs on the event loop.
    Without the stop_requested flag, stop() killed the already-dead old
    process and set status="stopped", then scale() launched a BRAND-NEW
    trainer and rebound self.proc — and poll() is gated on
    status == "training", so that trainer plus its 16-32 fork workers ran
    unreachable until the lab process exited.
    """
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    job = V.TrainingJob("run")
    before = len(fake_popen)

    job.stop(force=True)                      # the stop lands...
    job.scale(helpers=2)            # ...and the in-flight rescale must yield

    assert job.status == "stopped"
    assert len(fake_popen) == before, "scale() relaunched a trainer after a stop"
    assert not job.restarting, "restarting flag leaked"


# ------------------------------------------------------------------ /scene

def test_scene_geoms_carry_material_colors():
    """Every visual geom streams its MJCF material name + rgba — the viewer
    paints per-part colors (eye ring, mouth, shells) from these instead of
    guessing one color per body, so a missing/renamed field would silently
    regress the duck to a flat palette."""
    scene = V.extract_scene()
    assert scene["geoms"], "no visual geoms extracted"
    for g in scene["geoms"]:
        assert isinstance(g["mat"], str)
        rgba = g["rgba"]
        assert len(rgba) == 4
        assert all(0.0 <= v <= 1.0 for v in rgba)
    mats = {g["mat"] for g in scene["geoms"]}
    # Spot-check parts the viewer art-directs by name (a rename upstream
    # would orphan the overrides).
    assert {"noenoeil_material", "jaw_material", "sole_left_material",
            "top_head_shell_material"} <= mats


def test_preview_ducks_get_deployment_steering_and_drive(fake_popen):
    """The viewer must show DEPLOYED behavior: the policy is compass-blind, so
    an unsteered preview drifts into circles while steered videos run straight
    — the discrepancy the user caught twice. And locomotion trainees must get
    the drive command; zeroing it commanded the run trainee to STAND in the
    preview while the trainer practiced running."""
    duck = V.Duck("d0", "test", V._zero_infer, seed=0)
    # steer: forward drive with no explicit turn -> wz closes on measured yaw
    import mujoco
    yaw = 0.5
    duck.env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(duck.env.model, duck.env.data)
    duck.set_cmd(np.array([0.3, 0.0, 0.0], np.float32))
    # First straight tick ANCHORS to the current heading (no spin-back to
    # world zero — that whip knocked ducks over after turn segments)…
    assert duck.env.twist_cmd[2] == pytest.approx(0.0, abs=1e-6)
    # …then drifting off that anchor steers back.
    yaw2 = yaw + 0.3
    duck.env.data.qpos[3:7] = [np.cos(yaw2 / 2), 0.0, 0.0, np.sin(yaw2 / 2)]
    mujoco.mj_forward(duck.env.model, duck.env.data)
    duck.set_cmd(np.array([0.3, 0.0, 0.0], np.float32))
    # -4 * 0.3 = -1.2, clipped to the ±1.0 turn-rate limit
    assert duck.env.twist_cmd[2] == pytest.approx(-1.0, abs=0.05)
    # an explicit turn command wins over the hold
    duck.set_cmd(np.array([0.3, 0.0, 0.8], np.float32))
    assert duck.env.twist_cmd[2] == pytest.approx(0.8)
    # zero drive -> no steering injected
    duck.set_cmd(np.zeros(3, np.float32))
    assert duck.env.twist_cmd[2] == pytest.approx(0.0)


def test_locomotion_teach_runs_are_not_trick_ducks(fake_popen, tmp_path, monkeypatch):
    """A dragged-in RUN policy must receive drive commands. Name-based
    classification called every teach-* run a trick, so the user's dragged
    runner stood at cmd (0,0,0) — 'how come I don't see it move'."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path)
    for name, behavior, trick in (("teach-run-x", "run", False),
                                  ("teach-backflip-y", "backflip", True)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "behavior.json").write_text(
            json.dumps({"behavior": behavior}))
        d = types.SimpleNamespace(id="d3", policy_id=f"run:{name}")
        assert V.is_trick_duck(d) is trick, name
    # unknown run dir: conservative old rule
    d = types.SimpleNamespace(id="d3", policy_id="run:teach-mystery")
    assert V.is_trick_duck(d) is True


def test_handoff_waits_for_the_spin_to_brake(fake_popen, monkeypatch):
    """Handing off to alpha_stand while still spinning made the stand policy
    absorb the leftover momentum by pivoting — the user's 'it lands then
    turns to the side'. alpha_stand cannot fix heading (yaw unobservable), so
    the flip policy must brake before it lets go of the wheel."""
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    duck = V.Duck("d0", "bf", V._zero_infer, seed=0,
                  env_kwargs={"behavior_id": "backflip"})
    duck.env._bf_rot = 6.2
    duck.env.foot_contact_state = {"left": True, "right": True}
    duck.env.data.sensordata[duck.env.gyro_adr] = (0.0, -2.0, 0.0)  # still rolling
    assert not duck._handoff_due(), "handed off mid-spin"
    duck.env.data.sensordata[duck.env.gyro_adr] = (0.1, 0.2, 0.1)   # braked
    assert duck._handoff_due()


# ------------------------------------------------- deleting training data

def _run_with_files(root, name, sizes=(("policy.onnx", 100),
                                       ("checkpoints/model_1000_steps.zip", 900))):
    d = root / name
    for rel, n in sizes:
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * n)
    return d


def test_delete_run_erases_the_tree_and_reports_what_it_freed(fake_popen):
    keep = _run_with_files(V.RUNS_DIR, "teach-spin-keepme")
    doomed = _run_with_files(V.RUNS_DIR, "teach-spin-abc123")
    st = V.LabState([])

    res = V.delete_runs(["teach-spin-abc123"], st)

    assert res == {"deleted": ["teach-spin-abc123"], "freedBytes": 1000}
    assert not doomed.exists()
    assert keep.exists(), "deleting one run took a neighbour with it"


def test_delete_run_rejects_names_that_could_climb_out_of_runs(fake_popen):
    """The name arrives off the wire and selects a directory tree to erase —
    anything with a separator, a leading dot or .. must never resolve."""
    st = V.LabState([])
    for bad in ("../..", "..", "a/../../etc", "/etc", ".hidden", "a b", ""):
        with pytest.raises(ValueError):
            V.delete_runs([bad], st)


def test_delete_run_404s_on_a_name_with_no_run_dir(fake_popen):
    V.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        V.delete_runs(["teach-spin-nothere"], V.LabState([]))


def test_delete_refuses_any_run_of_the_job_training_right_now(fake_popen):
    """Not just the ACTIVE stage: a chain stage warm-starts from the previous
    stage's dir, so deleting a finished earlier stage mid-chain would break
    the next launch."""
    st = V.LabState([])
    st.job = V.TrainingJob("backflip", steps=1000)
    base = st.job._base_name
    done_stage = _run_with_files(V.RUNS_DIR, f"{base}-s1")
    active = st.job.dir                      # the stage training right now
    assert st.job.status == "training"

    for name in (active.name, done_stage.name):
        with pytest.raises(PermissionError, match="training right now"):
            V.delete_runs([name], st)
    assert active.exists() and done_stage.exists()

    # Once the job is over its runs are ordinary data again.
    st.job.status = "done"
    assert V.delete_runs([done_stage.name], st)["deleted"] == [done_stage.name]


def test_deleting_a_chain_is_all_or_nothing(fake_popen):
    """A half-deleted chain can't be resumed or fine-tuned from, so one bad
    target has to stop the whole delete before anything is touched."""
    st = V.LabState([])
    st.job = V.TrainingJob("backflip", steps=1000)
    base = st.job._base_name
    stages = [_run_with_files(V.RUNS_DIR, f"{base}-s{i}") for i in (1, 2)]

    with pytest.raises(PermissionError):
        V.delete_runs([s.name for s in stages] + [st.job.dir.name], st)
    assert all(s.exists() for s in stages), "a refused chain delete still bit"


def test_chain_run_names_finds_every_stage_in_order(fake_popen):
    for name in ("teach-backflip-abc-s2", "teach-backflip-abc-s10",
                 "teach-backflip-abc-s1", "teach-backflip-other-s1",
                 "teach-backflip-abc"):
        _run_with_files(V.RUNS_DIR, name)
    # Stage order is numeric (s10 last, not after s1), and a same-prefixed
    # NEIGHBOUR chain is not swept in.
    assert V.chain_run_names("teach-backflip-abc") == [
        "teach-backflip-abc", "teach-backflip-abc-s1",
        "teach-backflip-abc-s2", "teach-backflip-abc-s10"]
    assert V.chain_run_names("teach-nothing") == []


def test_chain_delete_takes_stages_that_never_exported_a_policy(fake_popen):
    """The palette only shows stages with a policy.onnx; a chain delete must
    still clear the stage that died before exporting, or the run dir is
    orphaned where nothing can see it."""
    _run_with_files(V.RUNS_DIR, "teach-backflip-abc-s1")
    _run_with_files(V.RUNS_DIR, "teach-backflip-abc-s2",
                    sizes=(("progress.jsonl", 40),))  # no policy.onnx
    res = V.delete_runs(V.chain_run_names("teach-backflip-abc"), V.LabState([]))
    assert res["deleted"] == ["teach-backflip-abc-s1", "teach-backflip-abc-s2"]
    assert res["freedBytes"] == 1040
    assert not (V.RUNS_DIR / "teach-backflip-abc-s2").exists()


def test_delete_drops_the_cached_infer_so_a_stale_chip_cannot_resurrect_it(fake_popen):
    _run_with_files(V.RUNS_DIR, "teach-spin-abc123")
    V._infer_cache["run:teach-spin-abc123"] = V._zero_infer
    V._infer_cache["ckpt:teach-spin-abc123@1k"] = V._zero_infer
    V._infer_cache["run:teach-spin-keepme"] = V._zero_infer
    try:
        V.delete_runs(["teach-spin-abc123"], V.LabState([]))
        assert "run:teach-spin-abc123" not in V._infer_cache
        assert "ckpt:teach-spin-abc123@1k" not in V._infer_cache
        assert "run:teach-spin-keepme" in V._infer_cache
    finally:
        V._infer_cache.pop("run:teach-spin-keepme", None)


def test_policies_carry_the_size_a_delete_would_free(fake_popen):
    """The confirm dialog quotes this number — it has to count checkpoints,
    not just policy.onnx."""
    _run_with_files(V.RUNS_DIR, "teach-spin-abc123")
    entry = next(p for p in V.discover_policies() if p["id"] == "run:teach-spin-abc123")
    assert entry["sizeBytes"] == 1000


def test_hf_token_settings_roundtrip_never_leaks_the_token(fake_popen, monkeypatch, tmp_path):
    """BYOK contract: the token is validated via whoami() BEFORE persisting,
    lands 0600 on disk, and no response after the POST ever contains it —
    the browser gets a mask + username only."""
    import huggingface_hub
    from fastapi import HTTPException

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            if self.token != "hf_good_token_1234567890":
                raise RuntimeError("Invalid user token")
            return {"name": "testduck"}

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(V, "HF_TOKEN_PATH", tmp_path / "hf-token.json")
    app = V.make_app([])
    get = _endpoint(app, "/settings/hf", "GET")
    post = _endpoint(app, "/settings/hf", "POST")
    delete = _endpoint(app, "/settings/hf", "DELETE")

    assert get() == {"configured": False}

    with pytest.raises(HTTPException) as e:
        post(V.HfTokenReq(token="hf_wrong"))
    assert e.value.status_code == 401
    assert not (tmp_path / "hf-token.json").exists()  # rejected ≠ persisted

    res = post(V.HfTokenReq(token="hf_good_token_1234567890"))
    assert res["configured"] and res["username"] == "testduck"
    assert "hf_good_token_1234567890" not in str(res)  # mask only
    mode = (tmp_path / "hf-token.json").stat().st_mode & 0o777
    assert mode == 0o600
    shown = get()
    assert shown["username"] == "testduck"
    assert "hf_good_token_1234567890" not in str(shown)

    assert delete() == {"configured": False}
    assert not (tmp_path / "hf-token.json").exists()
    assert delete() == {"configured": False}  # idempotent


def test_active_cloud_job_blocks_local_training_and_token_removal(fake_popen, monkeypatch, tmp_path):
    from fastapi import HTTPException

    from microduck_local import colab_jobs, hf_jobs

    hf = hf_jobs.HfJobs(tmp_path / "hf")
    hf.jobs["active"] = {"id": "active", "state": "running", "released": False,
                         "repo_id": "testduck/results"}
    colab = colab_jobs.ColabJobs(tmp_path / "colab")
    monkeypatch.setattr(hf_jobs, "HfJobs", lambda: hf)
    monkeypatch.setattr(colab_jobs, "ColabJobs", lambda: colab)
    monkeypatch.setattr(V, "HF_TOKEN_PATH", tmp_path / "token.json")
    V.HF_TOKEN_PATH.write_text(json.dumps({"token": "hf_original", "username": "testduck"}))
    app = V.make_app([])
    with pytest.raises(HTTPException) as error:
        asyncio.run(_endpoint(app, "/teach", "POST")(V.TeachReq(text="stand still")))
    assert error.value.status_code == 409
    assert not fake_popen
    for method, args in (("DELETE", ()), ("POST", (V.HfTokenReq(token="hf_replacement"),))):
        with pytest.raises(HTTPException) as error:
            _endpoint(app, "/settings/hf", method)(*args)
        assert error.value.status_code == 409
    assert json.loads(V.HF_TOKEN_PATH.read_text())["token"] == "hf_original"


def test_cloud_start_reserves_slot_before_provider_returns(fake_popen, monkeypatch, tmp_path):
    from fastapi import HTTPException

    from microduck_local import colab_jobs, hf_jobs

    hf = hf_jobs.HfJobs(tmp_path / "hf")
    colab = colab_jobs.ColabJobs(tmp_path / "colab")
    monkeypatch.setattr(hf_jobs, "HfJobs", lambda: hf)
    monkeypatch.setattr(colab_jobs, "ColabJobs", lambda: colab)
    app = V.make_app([])
    request = types.SimpleNamespace(headers={})
    def start(*args):
        # Simulate another request while the provider has not returned an ID.
        with pytest.raises(HTTPException) as error:
            asyncio.run(_endpoint(app, "/teach", "POST")(V.TeachReq(text="stand still")))
        assert error.value.status_code == 409
        raise RuntimeError("allocation failed")
    monkeypatch.setattr(colab, "start", start)
    with pytest.raises(HTTPException):
        _endpoint(app, "/cloud/colab/jobs", "POST")(
            V.ColabStartReq(task="Mjlab-VelStand-Flat-MicroDuck", gpu="T4"), request)
    assert not fake_popen
    # A failed allocation must release the reservation.
    monkeypatch.setattr(colab, "start", lambda *args: {"id": "next"})
    assert _endpoint(app, "/cloud/colab/jobs", "POST")(
        V.ColabStartReq(task="Mjlab-VelStand-Flat-MicroDuck", gpu="T4"), request)["id"] == "next"


def test_shutdown_attempts_hf_even_when_colab_release_fails(fake_popen, monkeypatch, tmp_path):
    from microduck_local import colab_jobs, hf_jobs

    hf = hf_jobs.HfJobs(tmp_path / "hf")
    colab = colab_jobs.ColabJobs(tmp_path / "colab")
    monkeypatch.setattr(hf_jobs, "HfJobs", lambda: hf)
    monkeypatch.setattr(colab_jobs, "ColabJobs", lambda: colab)
    attempts = []
    def fail():
        attempts.append("colab")
        raise RuntimeError("network error")
    def stop_hf(token):
        attempts.append("hf")
        return []
    monkeypatch.setattr(colab, "stop_all", fail)
    monkeypatch.setattr(hf, "stop_all", stop_hf)
    app = V.make_app([])
    async def run():
        async with app.router.lifespan_context(app):
            pass
    asyncio.run(run())
    assert sorted(attempts) == ["colab", "hf"]


def test_showcase_env_kwargs_serves_an_unspotted_curriculum(fake_popen, tmp_path):
    """The ✨ whole-trick chip on a curriculum WITHOUT a spotter (the headstand
    ladder) must return real env kwargs. The non-spotter tail used to sit
    unreachable after handoff_for's return, so showcase_env_kwargs fell through
    to None and do_assign silently downgraded to a plain standing assign."""
    b = V.behaviors_mod.BEHAVIORS["headstand"]
    assert b.curriculum and b.spotter_fn is None      # the case that regressed
    run = V.RUNS_DIR / "teach-headstand-abc-s5"
    run.mkdir(parents=True, exist_ok=True)
    (run / "behavior.json").write_text(json.dumps({"behavior": "headstand"}))
    (run / "policy.onnx").write_bytes(b"x")
    kw = V.showcase_env_kwargs(str(run / "policy.onnx"))
    assert kw is not None and kw["behavior_id"] == "headstand"
    assert kw["standing_spawns"] is False             # it rehearses the arc


def test_trainee_preview_mirrors_stage_physics(fake_popen):
    """Stage knobs are not only spawns: a stage that ladders the ACTUATOR or
    the servo current must reach the in-process preview env too, or the watched
    duck runs different physics than the trainer (xml stage 1 / bam@1.3)."""
    b = V.behaviors_mod.BEHAVIORS["headstand"]
    kw = V.trainee_env_kwargs(b, {"MICRODUCK_ACTUATOR": "xml",
                                  "MICRODUCK_BAM_CURRENT_SCALE": "1.3"})
    assert kw["actuator_force"] == "xml"
    assert kw["bam_current_scale"] == 1.3


def test_stage_env_carries_the_runs_clip(fake_popen):
    """The clip rides extra_env into the trainer; the preview mirror reads
    stage_env, so an imitate run's preview tracked the recipe's DEFAULT clip."""
    job = V.TrainingJob("imitate", steps=1000,
                        extra_env={"MICRODUCK_CLIP": "my walk"})
    assert job.stage_env()["MICRODUCK_CLIP"] == "my walk"
    kw = V.trainee_env_kwargs(job.behavior, job.stage_env())
    assert kw["clip_name"] == "my walk"


def test_seating_another_run_keeps_ownership_of_live_preview_ducks(fake_popen, monkeypatch):
    """Selecting another run preserves the original card's preview ownership."""
    monkeypatch.setattr(V, "save_lab_state", lambda ducks: None)
    app = V.make_app([])
    st = app.state.lab
    _run_with_files(V.RUNS_DIR, "teach-spin-other")
    (V.RUNS_DIR / "teach-spin-other" / "behavior.json").write_text(
        json.dumps({"behavior": "spin", "steps": 1000, "weights": {}}))
    st.ducks = [_fake_duck("trainee"), _fake_duck("helper 1"), _fake_duck("d0")]

    st.job = V.TrainingJob("spin", steps=1000)      # launched: owns the ducks
    st.job.status = "done"
    assert st.job.owns_preview_ducks is True
    original = st.job
    asyncio.run(_endpoint(app, "/teach/load", "POST")(
        V.LoadRunReq(policy="run:teach-spin-other")))
    assert st.jobs == [original, st.job]
    assert original.owns_preview_ducks and not st.job.owns_preview_ducks

    asyncio.run(_endpoint(app, "/teach/clear", "POST")())
    assert [d.id for d in st.ducks] == ["trainee", "helper 1", "d0"]
    asyncio.run(_endpoint(app, "/teach/clear", "POST")({"run": original.run_name}))
    assert [d.id for d in st.ducks] == ["d0"]


def test_teach_clear_never_sweeps_a_roster_the_card_did_not_build(fake_popen, monkeypatch):
    """A SEATED run (POST /teach/load, which the panel fires on duck selection)
    creates no preview ducks, so dismissing its card must leave the roster
    alone. This has bitten twice: first by purging whenever st.job was None
    (which is the ordinary post-restart state), then by purging for any job at
    all — and after a restart restore_ducks has legitimately brought a trainee
    and helpers back."""
    saved = []
    monkeypatch.setattr(V, "save_lab_state", lambda ducks: saved.append(list(ducks)))
    app = V.make_app([])
    st = app.state.lab
    _run_with_files(V.RUNS_DIR, "teach-spin-seated")
    (V.RUNS_DIR / "teach-spin-seated" / "behavior.json").write_text(
        json.dumps({"behavior": "spin", "steps": 1000, "weights": {}}))
    st.ducks = [_fake_duck("trainee"), _fake_duck("helper 1"), _fake_duck("d0")]

    st.job = V.TrainingJob.adopt("teach-spin-seated")
    assert st.job.owns_preview_ducks is False
    asyncio.run(_endpoint(app, "/teach/clear", "POST")())
    assert [d.id for d in st.ducks] == ["trainee", "helper 1", "d0"]
    assert not saved                       # nothing persisted, nothing lost

    # A LAUNCHED job does own them, and dismissing its card takes them along.
    st.job = V.TrainingJob("spin", steps=1000)
    assert st.job.owns_preview_ducks is True
    st.job.status = "done"
    asyncio.run(_endpoint(app, "/teach/clear", "POST")())
    assert [d.id for d in st.ducks] == ["d0"]
    assert saved and [d.id for d in saved[-1]] == ["d0"]


def test_teach_clear_persists_the_roster(fake_popen, monkeypatch):
    """Every other roster mutation saves; without this the cleared trainee and
    helpers came back on the next lab start."""
    saved = []
    monkeypatch.setattr(V, "save_lab_state", lambda ducks: saved.append(list(ducks)))
    app = V.make_app([])
    st = app.state.lab
    st.job = V.TrainingJob("spin", steps=1000)
    st.job.status = "done"
    st.ducks = [_fake_duck("trainee"), _fake_duck("helper 1"), _fake_duck("d0")]
    asyncio.run(_endpoint(app, "/teach/clear", "POST")())
    assert [d.id for d in st.ducks] == ["d0"]
    assert saved and [d.id for d in saved[-1]] == ["d0"]


def test_delete_clears_a_card_showing_that_run(fake_popen):
    """A finished run seated by /teach/load is deletable (it is ordinary data),
    but the card must go with it — it used to keep streaming, and ✨ fine-tune
    pointed at an rmtree'd dir."""
    app = V.make_app([])
    st = app.state.lab
    _run_with_files(V.RUNS_DIR, "teach-spin-seated")
    (V.RUNS_DIR / "teach-spin-seated" / "behavior.json").write_text(
        json.dumps({"behavior": "spin", "steps": 1000, "weights": {}}))
    st.job = V.TrainingJob.adopt("teach-spin-seated")
    assert st.job.status == "done"
    _endpoint(app, "/runs/{name}", "DELETE")("teach-spin-seated")
    assert st.job is None


def test_teach_refuses_an_unknown_or_unsafe_clip(fake_popen):
    """The clip name becomes a path in BOTH the trainer and the preview envs,
    so it gets the same validation every clip endpoint uses — and must exist,
    or the job reports healthy while every worker dies at env construction."""
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    for bad in ("../secrets", "no-such-clip"):
        res = asyncio.run(teach(V.TeachReq(text="imitate this", clip=bad)))
        assert res["matched"] is False


def test_download_route_never_serves_a_live_preview(fake_popen):
    """GET /runs/{name}/policy.onnx hands out the DEPLOYABLE artifact: the
    baked policy.onnx when available, and never a live preview or raw
    checkpoint. Same name guard as
    DELETE — the string picks a directory."""
    from fastapi import HTTPException
    app = V.make_app([])
    download = _endpoint(app, "/runs/{name}/policy.onnx", "GET")

    for name, code in (("../etc", 422), ("teach-spin-nothere", 404)):
        with pytest.raises(HTTPException) as e:
            download(name)
        assert e.value.status_code == code

    # A run with only checkpoints has nothing worth handing out: 404.
    _run_with_files(V.RUNS_DIR, "teach-spin-raw",
                    sizes=(("checkpoints/model_1000_steps.zip", 900),))
    with pytest.raises(HTTPException) as e:
        download("teach-spin-raw")
    assert e.value.status_code == 404

    # Finished run: the baked export wins, served under a download filename.
    _run_with_files(V.RUNS_DIR, "teach-spin-done",
                    sizes=(("policy.onnx", 100), ("live.onnx", 50)))
    res = download("teach-spin-done")
    assert str(res.path).endswith("teach-spin-done/policy.onnx")
    assert "teach-spin-done.onnx" in res.headers["content-disposition"]

    # An interrupted run may retain an export for preview and resumption,
    # but that artifact is not a finished policy for deployment.
    stopped = _run_with_files(V.RUNS_DIR, "teach-spin-stopped",
                              sizes=(("policy.onnx", 100),))
    (stopped / "progress.jsonl").write_text(
        '{"steps": 50000, "total": 100000, "done": false, "stopped": "user"}\n')
    with pytest.raises(HTTPException) as e:
        download("teach-spin-stopped")
    assert e.value.status_code == 409

    # Mid-training preview is never silently downloaded as a final policy.
    _run_with_files(V.RUNS_DIR, "teach-spin-live",
                    sizes=(("live.onnx", 50),))
    with pytest.raises(HTTPException) as e:
        download("teach-spin-live")
    assert e.value.status_code == 404


def test_delete_route_maps_guards_to_status_codes_and_toasts(fake_popen):
    from fastapi import HTTPException
    app = V.make_app([])
    delete = _endpoint(app, "/runs/{name}", "DELETE")
    _run_with_files(V.RUNS_DIR, "teach-spin-abc-s1")
    _run_with_files(V.RUNS_DIR, "teach-spin-abc-s2")

    for name, kwargs, code in (("../etc", {}, 422),
                               ("teach-spin-nothere", {}, 404)):
        with pytest.raises(HTTPException) as e:
            delete(name, **kwargs)
        assert e.value.status_code == code

    res = delete("teach-spin-abc", chain=True)
    assert res["deleted"] == ["teach-spin-abc-s1", "teach-spin-abc-s2"]
    assert not list(V.RUNS_DIR.iterdir())
    # The lab's own event stream narrates it, so every viewer sees the delete
    # — not just the browser that asked for it.
    assert any("deleted teach-spin-abc" in e for e in app.state.lab.events)


# ------------------------------------------------- the front door (origins)

class _FakeSock:
    """Enough WebSocket for the /ws handler: an Origin header in, accept/close
    out, then queued messages before the client goes away."""

    def __init__(self, origin=None, msgs=()):
        self.headers = {} if origin is None else {"origin": origin}
        self.msgs = list(msgs)
        self.accepted = False
        self.closed = None
        self.reads = 0

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed = code

    async def receive_text(self):
        self.reads += 1
        if self.msgs:
            return json.dumps(self.msgs.pop(0))
        raise V.WebSocketDisconnect(1000)


def _ws_endpoint(app, path: str = "/ws"):
    """WebSocket routes carry no .methods, so _endpoint cannot find them."""
    for r in app.routes:
        if getattr(r, "path", None) == path and not getattr(r, "methods", None):
            return r.endpoint
    raise AssertionError(f"no WebSocket {path} route")


class _FakeReq:
    """Enough Request for POST /captures: headers, and a body that records
    whether the guards let anyone reach it."""

    def __init__(self, headers=None, body=b"not-really-a-webm"):
        self.headers = headers or {}
        self._body = body
        self.reads = 0

    async def body(self):
        self.reads += 1
        return self._body


def test_ws_handshake_refuses_a_non_loopback_origin(fake_popen):
    """A WebSocket handshake is NOT a CORS request: no preflight is sent and
    CORSMiddleware never sees it, so the allowlist bought this socket nothing
    while the socket itself takes assign/spawn_duck/remove_duck/reset. Any page
    the user was browsing while the lab ran could open ws://127.0.0.1:8788/ws
    and wipe the roster.
    """
    app = V.make_app([])
    ws = _ws_endpoint(app)
    st = app.state.lab

    # None = curl / the CLI / these tests: browsers always stamp Origin on a
    # handshake, so an absent header cannot be a forged cross-site request.
    for origin in (None, "http://localhost:63317", "http://127.0.0.1:8788",
                   "https://localhost", "http://[::1]:63317"):
        sock = _FakeSock(origin, msgs=[{"cmd": [0.1, 0.0, 0.0]}])
        asyncio.run(ws(sock))
        assert sock.accepted, f"{origin!r} is a legitimate local client"
        assert sock.closed is None
        assert sock.reads == 2, "the command loop never ran"

    # "localhost.evil.example" and "127.0.0.1.evil.example" are the reason the
    # rule is an anchored regex and not a substring/startswith test.
    for origin in ("http://evil.example", "http://localhost.evil.example",
                   "http://127.0.0.1.evil.example", "http://192.168.1.5:63317",
                   "https://duck-lab.example:8788", "null", "file://"):
        sock = _FakeSock(origin, msgs=[{"remove_duck": "walker"}])
        asyncio.run(ws(sock))
        assert not sock.accepted, f"{origin!r} was handed a live socket"
        assert sock.closed == 1008          # policy violation
        assert sock.reads == 0, f"{origin!r} reached the command loop"
        assert sock not in st.clients


def test_ws_commands_work_when_launch_terminal_is_unavailable(fake_popen, monkeypatch):
    app = V.make_app([])

    def unavailable_terminal(*args, **kwargs):
        raise OSError("launch terminal is no longer available")

    monkeypatch.setattr(V, "print", unavailable_terminal, raising=False)
    sock = _FakeSock("http://localhost:63317", msgs=[{"cmd": [0.2, 0.0, 0.0]}])
    asyncio.run(_ws_endpoint(app)(sock))
    assert sock.accepted and sock.reads == 2
    assert app.state.lab.override[0] == pytest.approx(0.2)


def test_cors_and_ws_enforce_one_shared_origin_rule(fake_popen):
    """Two enforcement points, one regex. A second literal in the middleware
    would drift from the socket's check the next time the allowlist changes —
    which is exactly how /ws ended up with no check at all."""
    app = V.make_app([])
    cors = [m for m in app.user_middleware if m.cls is V.CORSMiddleware]
    assert cors, "the CORS middleware is gone"
    assert cors[0].kwargs["allow_origin_regex"] == V.LOCAL_ORIGIN_RE.pattern

    # Sharing the PATTERN is not enough, and asserting only that was how the
    # first version of this test passed over a real split: Starlette calls
    # fullmatch, origin_allowed called match, and Python's `$` also matches
    # just before a trailing newline — so one regex produced two verdicts on
    # "http://localhost\n". Lock the ANSWERS, not the source text.
    for origin in ("http://localhost:63317", "http://127.0.0.1:8788",
                   "https://localhost", "http://[::1]:63317",
                   "http://evil.example", "http://localhost.evil.example",
                   "null", "file://", "http://localhost\n",
                   "http://localhost:63317\nX-Injected: 1"):
        assert V.origin_allowed(origin) is bool(
            V.LOCAL_ORIGIN_RE.fullmatch(origin)), (
                f"{origin!r}: the socket and the middleware disagree")


def test_capture_upload_refuses_forgeable_cross_origin_posts(fake_popen, monkeypatch,
                                                             tmp_path):
    """POST /captures reads a raw body with no schema, writes up to
    CAPTURE_MAX_BYTES to disk and spawns ffmpeg. Sent as
    `Content-Type: text/plain` it was a CORS-SIMPLE request — dispatched
    cross-origin with NO preflight, so CORSMiddleware never got a veto and the
    side effect landed even though the reply was opaque to the attacker.
    """
    from fastapi import HTTPException

    caps = tmp_path / "captures"
    monkeypatch.setenv("MICRODUCK_CAPTURES_DIR", str(caps))
    converted = []

    def fake_convert(src, base):
        converted.append(base)
        return {"name": base, "mp4": f"/captures/{base}.mp4"}

    monkeypatch.setattr(V, "convert_capture", fake_convert)
    app = V.make_app([])
    post = _endpoint(app, "/captures", "POST")

    # Every safelisted content type, including one with a charset parameter —
    # the guard must compare the bare type, not the whole header.
    for ctype in ("text/plain", "text/plain;charset=UTF-8", "TEXT/PLAIN",
                  "application/x-www-form-urlencoded", "multipart/form-data"):
        req = _FakeReq({"content-type": ctype, "origin": "http://localhost:63317"})
        with pytest.raises(HTTPException) as e:
            asyncio.run(post(req, name="duck"))
        assert e.value.status_code == 415
        assert req.reads == 0, "the body was read before the guard ran"

    # A typeless Blob sends no Content-Type at all, which is simple too — the
    # Origin door is what stops that one.
    req = _FakeReq({"origin": "http://evil.example"})
    with pytest.raises(HTTPException) as e:
        asyncio.run(post(req, name="duck"))
    assert e.value.status_code == 403
    assert req.reads == 0

    assert converted == [], "a refused upload still spawned the converter"
    assert not caps.exists(), "a refused upload still touched the disk"

    # The real viewer: RecordPanel posts the MediaRecorder Blob itself, so
    # fetch stamps video/webm (Chrome/Firefox) or video/mp4 (Safari).
    for ctype in ("video/webm;codecs=vp9", "video/mp4"):
        req = _FakeReq({"content-type": ctype, "origin": "http://localhost:63317"})
        assert asyncio.run(post(req, name="🎓 trainee"))["name"].startswith("trainee-")
    # curl / the CLI: no Origin, no Content-Type.
    req = _FakeReq()
    assert asyncio.run(post(req, name="duck"))["name"].startswith("duck-")
    assert len(converted) == 3
    assert not list(caps.glob("*.upload")), "the scratch upload was left behind"


# ------------------------------------------------- atomic json persistence

def test_overlapping_saves_never_share_one_scratch_file(fake_popen, monkeypatch,
                                                        tmp_path):
    """save_teach_weights / save_lab_state / save_clip each wrote through ONE
    fixed "<name>.tmp". FastAPI runs sync handlers in a threadpool and the sim
    loop saves the roster too, so two overlapping saves poured their json into
    one inode and renamed the garbled result into place.

    Force the overlap deterministically: save A is suspended with its scratch
    file written but not yet renamed, and save B of the SAME path runs start to
    finish inside that window. With a shared scratch name B rewrites and then
    renames away the file A is still holding, and A's own rename explodes.
    """
    path = tmp_path / "lab-state.json"
    real_replace = V.Path.replace
    scratch, reentered = [], []

    def replace(self, target):
        scratch.append(self)
        if not reentered:
            reentered.append(True)
            V._atomic_write_json(path, {"who": "B"})
        return real_replace(self, target)

    monkeypatch.setattr(V.Path, "replace", replace)
    V._atomic_write_json(path, {"who": "A"})
    monkeypatch.undo()

    assert len(scratch) == 2
    assert scratch[0] != scratch[1], "both saves wrote through one scratch path"
    # A renamed last, so A wins — the point is that the survivor PARSES and is
    # exactly one payload rather than a splice of both.
    assert json.loads(path.read_text()) == {"who": "A"}
    assert not list(tmp_path.glob("*.tmp")), "a scratch file was left behind"


def test_atomic_write_cleans_up_after_a_failed_serialization(tmp_path):
    """The scratch file must never outlive a failed save: for hf-token.json it
    holds a live token, at a path .gitignore does not cover."""
    path = tmp_path / "hf-token.json"
    with pytest.raises(TypeError):
        V._atomic_write_json(path, {"token": object()}, mode=0o600)
    assert not path.exists()
    assert not list(tmp_path.iterdir()), "a scratch file survived the failure"


def test_every_json_writer_goes_through_the_atomic_helper(fake_popen, monkeypatch,
                                                          tmp_path):
    """The repeated failure mode here is fixing one site and missing its
    siblings, and the fixed-tmp write had FOUR. Route check: no writer may roll
    its own tmp+replace, and the token keeps its 0600-from-creation mode."""
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            pass

        def whoami(self):
            return {"name": "testduck"}

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(V, "HF_TOKEN_PATH", tmp_path / "hf-token.json")
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", str(tmp_path / "clips"))
    calls = []
    monkeypatch.setattr(V, "_atomic_write_json",
                        lambda path, obj, mode=0o644: calls.append((path.name, mode)))
    app = V.make_app([])

    V.save_teach_weights({})
    V.save_lab_state([])
    V.save_clip("demo", {"version": V.CLIP_VERSION, "name": "demo", "keys": []})
    _endpoint(app, "/settings/hf", "POST")(V.HfTokenReq(token="hf_good_token"))

    assert calls == [("teach-weights.json", 0o644), ("lab-state.json", 0o644),
                     ("demo.json", 0o644), ("hf-token.json", 0o600)]


# ------------------------------------------------- run-name validation

def test_init_from_uses_the_one_run_name_validator(fake_popen):
    """initFrom hand-rolled `Path(name).name != name`, strictly looser than
    run_dir()/RUN_NAME_RE: it waved through leading dots, spaces and unbounded
    length, and the value goes straight onto a spawned trainer's --init-from."""
    for bad in ("../../etc", ".ssh", ".hidden", "a b", "-dash", "x" * 200, ""):
        with pytest.raises(ValueError, match="plain run name"):
            V.resolve_init_from(bad)

    # …and a name RUN_NAME_RE accepts still has to be a warm-startable run.
    with pytest.raises(ValueError, match="does not exist"):
        V.resolve_init_from("teach-spin-nothere")
    run = V.RUNS_DIR / "teach-spin-abc123"
    run.mkdir(parents=True)
    (run / "model.zip").touch()
    (run / "vecnormalize.pkl").touch()
    assert V.resolve_init_from("teach-spin-abc123") == run


def test_cli_run_dir_is_tagged_so_the_lab_stops_driving_a_trick_brain(
        fake_popen, tmp_path, monkeypatch):
    """`duck-lab runs/find_ball` must behave like dropping the SAME dir from
    the palette. build_ducks used to set only a label, so policy_id stayed
    None, is_trick_duck() said False on the missing "run:" prefix, and the lab
    posted its WASD velocity command to a ball brain that trained on zero
    twist: 1058 falls and mean reward -3.2 in four minutes of
    `duck-lab runs/find_ball`, against a duck that stood there quite happily
    when the identical dir was assigned from the palette."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path)
    V._TRICK_DUCK_CACHE.clear()
    run = tmp_path / "find_ball"
    run.mkdir()
    (run / "policy.onnx").touch()
    (run / "behavior.json").write_text(json.dumps({"behavior": "find_ball"}))
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)

    duck, = V.build_ducks(types.SimpleNamespace(policies=[str(run)],
                                                checkpoints=None))
    assert duck.policy_id == "run:find_ball"
    assert V.is_trick_duck(duck) is True

    # A LOCOMOTION run tagged the same way still gets driven — the tag says
    # "look me up", not "hold still". test_locomotion_teach_runs_are_not_
    # trick_ducks is the other half of that rule.
    V._TRICK_DUCK_CACHE.clear()
    runner = tmp_path / "teach-run-z"
    runner.mkdir()
    (runner / "policy.onnx").touch()
    (runner / "behavior.json").write_text(json.dumps({"behavior": "run"}))
    duck, = V.build_ducks(types.SimpleNamespace(policies=[str(runner)],
                                                checkpoints=None))
    assert duck.policy_id == "run:teach-run-z"
    assert V.is_trick_duck(duck) is False


def test_a_run_dir_from_somewhere_else_is_not_claimed_as_a_named_run(
        fake_popen, tmp_path, monkeypatch):
    """`run:<name>` is resolved by BARE NAME under RUNS_DIR, so only a dir that
    really is RUNS_DIR/<name> may claim it. A scratch copy that happens to
    share a basename must not inherit an unrelated run's behavior.json."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    elsewhere = tmp_path / "scratch" / "find_ball"
    elsewhere.mkdir(parents=True)
    (elsewhere / "policy.onnx").touch()
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)

    duck, = V.build_ducks(types.SimpleNamespace(policies=[str(elsewhere)],
                                               checkpoints=None))
    assert duck.policy_id is None
    assert duck.label == "find_ball"


def test_a_successful_termination_is_not_counted_as_a_fall():
    """A task env ends an episode by SUCCEEDING as readily as by failing.

    The lab's counter was named for a walking duck, where the only early end
    is on the floor. MOSS's pick env terminates on `knocked or picked`, so
    every counted "fall" on a working pickup policy was a pickup: the panel
    showed `moss-pick-v1` a red 921 for a policy measured at 20 picks out of
    20. An env that reports `info["success"]` gets its wins counted apart;
    one that does not is unchanged and every termination is still a fall.
    """
    d = V.Duck("m9", "outcomes", V._zero_infer, seed=1, env_kwargs={})

    class _Env:
        """Ends every episode, alternating success and failure."""

        def __init__(self):
            self.n = 0

        def step(self, _a):
            self.n += 1
            return (d.obs, 0.0, True, False, {"success": self.n % 2 == 1})

        def reset(self, **_kw):
            return d.obs, {}

    d.env = _Env()
    d.sample_speed = lambda: None          # needs a real model otherwise
    for _ in range(6):
        d.tick()
    assert (d.wins, d.falls) == (3, 3), (d.wins, d.falls)

    # An env that says nothing about the outcome keeps the old meaning.
    quiet = V.Duck("m10", "quiet", V._zero_infer, seed=1, env_kwargs={})

    class _Quiet(_Env):
        def step(self, _a):
            return (quiet.obs, 0.0, True, False, {})

    quiet.env = _Quiet()
    quiet.sample_speed = lambda: None
    for _ in range(4):
        quiet.tick()
    assert (quiet.wins, quiet.falls) == (0, 4), (quiet.wins, quiet.falls)


def test_the_lab_trains_several_jobs_at_once(fake_popen):
    """Concurrency, and the env budget that bounds it.

    The lab used to hold exactly one job and refuse any second /teach. That
    was a software limit, not a machine one: one 32-env job measures ~463%
    CPU, about 4.6 of an 18-core machine, so more fit beside it. What is
    scarce is ENVS, so jobs draw from one budget and the refusal is about
    that rather than about a count.
    """
    st = V.LabState([types.SimpleNamespace(id="d0")])
    assert st.job is None and st.jobs == []

    a = V.TrainingJob("backflip", steps=1000, envs=32, trainee_id="trainee")
    st.job = a
    b = V.TrainingJob("one_leg", steps=1000, envs=32, trainee_id="trainee2")
    st.job = b

    # BOTH are live, and each has its own preview duck: a shared "trainee"
    # would have them writing over each other's window.
    assert st.jobs == [a, b]
    assert {j.trainee_id for j in st.jobs} == {"trainee", "trainee2"}
    assert st.envs_in_use() == 64
    assert len(st.training_jobs()) == 2

    # `st.job` keeps its old single-job meaning for every caller written
    # before concurrency: the newest job.
    assert st.job is b
    st.job = None
    assert st.jobs == [a] and st.job is a


def test_trainee_slots_are_reused_and_recognised(fake_popen):
    ducks = [types.SimpleNamespace(id="d0")]
    # A job CLAIMS a slot at construction and creates its duck only after, so
    # asking twice before either duck exists must not hand out the same id —
    # it did, and two concurrent jobs shared one preview window.
    jobs = [V.TrainingJob("backflip", steps=1000, trainee_id="trainee")]
    assert V.next_trainee_slot(ducks, jobs) == "trainee2"
    assert V.next_trainee_slot(ducks) == "trainee"
    ducks.append(types.SimpleNamespace(id="trainee"))
    assert V.next_trainee_slot(ducks) == "trainee2"
    ducks.append(types.SimpleNamespace(id="trainee2"))
    assert V.next_trainee_slot(ducks) == "trainee3"
    # A freed slot is reused, so ids stay stable as jobs come and go.
    ducks = [d for d in ducks if d.id != "trainee"]
    assert V.next_trainee_slot(ducks) == "trainee"
    assert V.is_trainee("trainee") and V.is_trainee("trainee3")
    assert not V.is_trainee("helper1") and not V.is_trainee("d0")


def test_env_budget_is_what_refuses_a_job(fake_popen):
    """A job is refused for want of ENVS, not because another one exists."""
    st = V.LabState([types.SimpleNamespace(id="d0")])
    budget = V.env_budget()
    assert budget >= V.BASE_ENVS
    # Fill the budget exactly.
    st.job = V.TrainingJob("backflip", steps=1000, envs=budget)
    assert st.envs_in_use() == budget
    assert budget - st.envs_in_use() < V.MIN_JOB_ENVS   # the refusal condition
    # A job that has FINISHED frees its envs again.
    st.jobs[0].status = "done"
    assert st.envs_in_use() == 0
    assert budget - st.envs_in_use() >= V.MIN_JOB_ENVS


def test_an_unnamed_stop_refuses_when_several_jobs_train(fake_popen):
    """It used to mean "stop the one job"; concurrency made it "stop all".

    The viewer's stop button sends no name, so one click on one card stopped
    every live run — including another session's. An unnamed stop now refuses
    unless there is exactly one, or the caller says `all`.
    """
    import asyncio
    app = V.make_app([])
    st = app.state.lab
    st.jobs = [V.TrainingJob("backflip", steps=1000),
               V.TrainingJob("one_leg", steps=1000)]
    route = {r.name: r for r in app.routes if getattr(r, "name", "")}["teach_stop"]

    out = asyncio.run(route.endpoint(None))
    assert out["ok"] is False and out["stopped"] == []
    assert "2 jobs are training" in out["message"]
    assert all(j.status == "training" for j in st.jobs)   # nothing was stopped

    # Naming one stops exactly that one.
    out = asyncio.run(route.endpoint({"run": st.jobs[0].run_name}))
    assert out["ok"] and out["stopped"] == [st.jobs[0].run_name]

    # And `all` still stops the rest.
    out = asyncio.run(route.endpoint({"all": True}))
    assert out["ok"]


def test_finished_trainees_are_reaped(fake_popen):
    """A trainee duck per job, but not one per job FOREVER.

    With a single job the lab reused one duck called "trainee". Per-job ids
    allocate a fresh one each time and nothing removed the old, so after nine
    starts the stage carried nine trainees — six of them identical copies of
    a stopped run, each still stepping a MuJoCo env in the 50 Hz loop.
    """
    ducks = [types.SimpleNamespace(id="d0"),
             types.SimpleNamespace(id="trainee"),
             types.SimpleNamespace(id="trainee2"),
             types.SimpleNamespace(id="trainee3")]
    st = V.LabState(ducks)
    dead = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    dead.status = "stopped"
    liveJob = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee3")
    st.jobs = [dead, liveJob]

    gone = V.reap_trainee_ducks(st)
    # `trainee` is claimed by a stopped job that is not the newest, so it
    # goes. `trainee3` is training. `trainee2` is claimed by NO job — a
    # restored duck, left alone. `d0` is not a trainee at all.
    assert gone == ["trainee"]
    assert [d.id for d in st.ducks] == ["d0", "trainee2", "trainee3"]
    assert st.jobs == [liveJob]

    # The NEWEST job keeps its duck even when finished, so its card survives.
    st2 = V.LabState([types.SimpleNamespace(id="trainee")])
    done = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    done.status = "done"
    st2.jobs = [done]
    assert V.reap_trainee_ducks(st2) == []
    assert [d.id for d in st2.ducks] == ["trainee"]


def test_a_restored_trainee_is_never_reaped(fake_popen):
    """No job claims it, so it is a restored duck, not an orphan.

    After a lab restart `st.jobs` is empty while `restore_ducks` has brought
    the roster back. A sweep that went by NAME deleted those — the trap
    `teach_clear` documents — and a G1 trainee vanishing that way made the
    body guess fall back to the duck, so the next job trained on the wrong
    robot entirely.
    """
    st = V.LabState([types.SimpleNamespace(id="trainee"),
                     types.SimpleNamespace(id="trainee2")])
    assert st.jobs == []
    assert V.reap_trainee_ducks(st) == []
    assert [d.id for d in st.ducks] == ["trainee", "trainee2"]


def test_only_a_training_trainee_is_unremovable(fake_popen):
    """The guard asks about THIS duck's job, not the newest one.

    Asking about the newest job refused every trainee on the stage the moment
    anything trained — including orphans from jobs that had already stopped,
    which is exactly the pile a user wants to clear.
    """
    ducks = [types.SimpleNamespace(id="trainee"),
             types.SimpleNamespace(id="trainee2")]
    st = V.LabState(ducks)
    dead = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    dead.status = "stopped"
    liveJob = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    st.jobs = [dead, liveJob]

    assert V.remove_duck_error(st, "trainee") is None          # orphan: go
    assert "stop the run first" in V.remove_duck_error(st, "trainee2")


def test_every_job_is_polled_not_just_the_newest(fake_popen):
    """`poll()` is what launches a chain's next stage.

    The lab loop polled `st.job` — the newest — so any other job stranded
    the moment its current stage finished: a stow chain read "training" with
    no trainer alive, stage 1 complete and stage 2 never launched, because a
    later job had taken over `st.job`. Polling is per job now.
    """
    older = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    newer = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    st = V.LabState([types.SimpleNamespace(id="d0")])
    st.jobs = [older, newer]
    assert st.job is newer, "the OLDER job is the one at risk"

    # Finish the older job's first stage; only a poll advances it.
    _finish_stage(older, 1000)
    before = older.stage_idx
    V.poll_all(st)          # what the lab loop calls, once per tick
    assert older.stage_idx == before + 1, (
        "the older job never advanced — this is the stranding bug")
    assert any("Training stage" in e for e in st.events)


def test_the_stats_strip_follows_a_training_job(fake_popen):
    """It reported the NEWEST job, which said "done" while another trained.

    With concurrency the newest job is often a finished one, and the HUD's
    `train ✔ done` then reads as "nothing is running" while a chain is still
    going — which is exactly how it was read.
    """
    st = V.LabState([types.SimpleNamespace(id="d0")])
    training = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    finished = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    finished.status = "done"
    st.jobs = [training, finished]
    assert st.job is finished, "the newest job is the finished one"
    live = st.training_jobs()
    assert live and live[0] is training, (
        "the strip must follow the job that is actually training")


def test_a_snapshot_only_reaches_its_own_jobs_duck(fake_popen):
    """One run's brain must not be loaded into another run's window.

    The snapshot used to be taken from the NEWEST job and fanned onto every
    duck whose id looked like a trainee. With two jobs training that put the
    approach policy into the stow trainee's window — the preview showed a
    run practising someone else's task.
    """
    a_duck = types.SimpleNamespace(id="trainee")
    b_duck = types.SimpleNamespace(id="trainee2")
    helper = types.SimpleNamespace(id="helper1")
    other = types.SimpleNamespace(id="d0")
    st = V.LabState([a_duck, b_duck, helper, other])
    a = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    b = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    st.jobs = [a, b]

    helper.onnx_path = str(a.dir / "live.onnx")
    assert V.snapshot_targets(st, a) == [helper, a_duck]
    assert V.snapshot_targets(st, b) == [b_duck]
    # the crux: B's snapshot never reaches A's duck
    assert a_duck not in V.snapshot_targets(st, b)
    assert b_duck not in V.snapshot_targets(st, a)


def test_teach_env_knobs_are_micro_duck_only():
    """`/teach` may carry MICRODUCK_* env knobs for its trainer, nothing else.

    It exists because an `initFrom` fine-tune has no curriculum stage, so it
    inherits the LAB's environment and therefore every env's DEFAULT rung —
    continuing MOSS's shipped pickup (12/12 at rung 2) would have retrained
    it at rung 0. The allow-list is the point: this value reaches a
    subprocess environment, so a request must not be able to set PATH.

    Planted the regression to check this bites: drop the `startswith` guard
    in `teach_env_knobs` and the PATH case below stops raising.
    """
    import microduck_local.viz_server as V

    assert "env" in V.TeachReq.model_fields, "TeachReq must carry env knobs"
    assert V.teach_env_knobs(None) == {}
    assert V.teach_env_knobs({"MICRODUCK_MOSS_PICK_RUNG": 2}) == {
        "MICRODUCK_MOSS_PICK_RUNG": "2"}, "values are stringified for the env"

    for evil in ({"PATH": "/tmp/evil"}, {"PYTHONPATH": "/tmp/evil"},
                 {"LD_PRELOAD": "x.so"}, {"microduck_lower": "1"}):
        with pytest.raises(Exception) as got:
            V.teach_env_knobs(evil)
        assert getattr(got.value, "status_code", None) == 422, (
            f"{evil} must be refused with 422, not passed to a subprocess")


def test_the_watched_trainee_mirrors_the_trainers_env_knobs():
    """A /teach `env` knob must reach the PREVIEW, not just the trainer.

    The trainer subprocess inherits the variables and the body's
    `train_env_kwargs` reads them. The preview is in-process and mirrored
    only two hardcoded knobs, so MOSS's curriculum trained rung 2 while the
    watched trainee ran the default rung 0 — silently, because both look
    plausible. A peer session hit the identical shape on the duck and lost an
    hour arguing with its user about what the robot was doing.

    Planted the regression to check this bites: drop the `_body_env_kwargs`
    call from `trainee_env_kwargs` and `pick_rung` disappears below.
    """
    import os

    import microduck_local.viz_server as V
    from microduck_local.behaviors.moss_tasks import MOSS_PICK, MOSS_STOW

    kw = V.trainee_env_kwargs(MOSS_PICK, {"MICRODUCK_MOSS_PICK_RUNG": "2"})
    assert kw.get("pick_rung") == 2, (
        "the trainee preview must run the rung the trainer was given — "
        f"got {kw!r}")

    kw = V.trainee_env_kwargs(MOSS_STOW, {"MICRODUCK_MOSS_STOW_RUNG": "1"})
    assert kw.get("stow_rung") == 1

    # And it must not leak: the preview stages the variables for the length
    # of one call, in a process that is also running other ducks.
    before = os.environ.get("MICRODUCK_MOSS_PICK_RUNG")
    V.trainee_env_kwargs(MOSS_PICK, {"MICRODUCK_MOSS_PICK_RUNG": "0"})
    assert os.environ.get("MICRODUCK_MOSS_PICK_RUNG") == before, (
        "staging a knob for the preview must not change the lab's own "
        "environment — every other duck in the roster reads it too")


def test_the_trainee_preview_is_not_an_easier_task_than_the_trainer():
    """What you WATCH must be what is being TRAINED.

    Three separate knobs were found wrong this way in one session, each
    patched by name, each time leaving the next one broken:

      * `/teach` env knobs reached the trainer and not the preview, so MOSS
        previewed rung 0 while training rung 2;
      * the preview inherited `domain_rand=False`, so the can spawned upright
        EVERY episode while the trainer saw a 50/50 upright/lying mix;
      * a peer session lost an hour to the same shape on the duck.

    The person watching reasonably concluded the robot only ever met one
    case, and said so. This asserts the RULE rather than the three knobs.
    """
    import microduck_local.viz_server as V
    from microduck_local.behaviors.moss_tasks import MOSS_PICK

    kw = V.trainee_env_kwargs(MOSS_PICK, {"MICRODUCK_MOSS_PICK_RUNG": "2"})
    assert kw.get("domain_rand") is True, (
        "the trainee preview must randomise the way the trainer does — "
        "without this the watched can is always upright")
    assert kw.get("obs_noise") is True
    assert kw.get("pick_rung") == 2

    # And the reason the preview could not randomise before: it shared one
    # compiled model with the roster, which domain randomisation writes to.
    # A duck that randomises has to take a private compile.
    import inspect
    src = inspect.getsource(V.Duck._make_env)
    assert "writes_model" in src, (
        "a preview that randomises must not share the roster's mjModel")


def test_concurrent_endpoints_keep_saving_jobs_and_resume_the_named_run(fake_popen, monkeypatch):
    """Save/resume from the fork must coexist with upstream's job list."""
    monkeypatch.setattr(V, "env_budget", lambda: V.BASE_ENVS * 4)
    app = V.make_app([])
    st = app.state.lab
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")
    resume = _endpoint(app, "/teach/resume", "POST")
    first = asyncio.run(teach(V.TeachReq(text="spin")))
    second = asyncio.run(teach(V.TeachReq(text="one_leg")))
    assert first["matched"] and second["matched"]
    older, newer = st.jobs
    assert older.trainee_id != newer.trainee_id
    assert {d.id for d in st.ducks} >= {older.trainee_id, newer.trainee_id}
    asyncio.run(stop({"run": older.run_name}))
    assert older.status == "stopping" and newer.status == "training"
    assert older.run_name in V.training_run_names(st)
    assert st.envs_in_use() == older.envs + newer.envs
    assert older.trainee_id not in V.reap_trainee_ducks(st)
    assert V.spawn_helper_job(st, older.run_name) is None
    (older.dir / "model.zip").touch()
    (older.dir / "vecnormalize.pkl").touch()
    older.proc.returncode = 0
    older.poll()
    assert older.status == "stopped"
    out = asyncio.run(resume(V.ResumeReq(runName=older.run_name)))
    assert out["ok"] and out["job"]["trainee"] == older.trainee_id
    assert st.jobs == [newer, older]
    assert newer.status == "training"
    assert _flag(fake_popen[-1].cmd, "--init-from") == str(older.dir)


def test_saving_jobs_reserve_capacity_and_cloud_checks_every_local_job(fake_popen, monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(V, "env_budget", lambda: V.BASE_ENVS)
    app = V.make_app([])
    st = app.state.lab
    teach = _endpoint(app, "/teach", "POST")
    assert asyncio.run(teach(V.TeachReq(text="spin")))["matched"]
    live = st.job
    live.stop()
    refused = asyncio.run(teach(V.TeachReq(text="one_leg")))
    assert not refused["matched"] and refused["busy"]
    assert st.jobs == [live]
    finished = V.TrainingJob("one_leg", trainee_id="trainee2")
    finished.status = "done"
    st.jobs.append(finished)
    with pytest.raises(HTTPException) as exc:
        _endpoint(app, "/cloud/colab/jobs", "POST")(
            V.ColabStartReq(task="Mjlab-Velocity-Flat-MicroDuck", gpu="T4"),
            types.SimpleNamespace(headers={}))
    assert exc.value.status_code == 409
    status = _endpoint(app, "/teach/status", "GET")()
    assert status["running"] and status["envBudget"]["used"] == live.envs


def test_named_force_stop_does_not_kill_another_training_job(fake_popen):
    app = V.make_app([])
    a = V.TrainingJob("spin", trainee_id="trainee")
    b = V.TrainingJob("one_leg", trainee_id="trainee2")
    app.state.lab.jobs = [a, b]
    out = asyncio.run(_endpoint(app, "/teach/stop", "POST")({"run": a.run_name}, force=True))
    assert out["ok"] and a.status == "stopped" and b.status == "training"
    assert a.proc.returncode is not None and b.proc.returncode is None


def test_helpers_follow_only_their_own_chain_across_stage_handoffs(fake_popen):
    a = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    b = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    rebuilt = []
    helper_a = types.SimpleNamespace(id="helper1", onnx_path=str(a.dir / "live.onnx"),
                                     rebuild_env=lambda kw: rebuilt.append("a"))
    helper_b = types.SimpleNamespace(id="helper2", onnx_path=str(b.dir / "live.onnx"),
                                     rebuild_env=lambda kw: rebuilt.append("b"))
    st = V.LabState([helper_a, helper_b])
    st.jobs = [a, b]
    _finish_stage(a, 1000)
    V.poll_all(st)
    assert rebuilt == ["a"]
    assert V.snapshot_targets(st, a) == [helper_a]
    assert V.snapshot_targets(st, b) == [helper_b]


def test_clearing_named_card_preserves_other_jobs_and_helpers(fake_popen):
    app = V.make_app([])
    st = app.state.lab
    a = V.TrainingJob("spin", trainee_id="trainee")
    b = V.TrainingJob("one_leg", trainee_id="trainee2")
    b.stop(force=True)
    st.jobs = [a, b]
    st.ducks = [_fake_duck("trainee"), _fake_duck("trainee2"),
                _fake_duck("helper1", onnx_path=str(a.dir / "live.onnx")),
                _fake_duck("helper2", onnx_path=str(b.dir / "live.onnx"))]
    clear = _endpoint(app, "/teach/clear", "POST")
    assert not asyncio.run(clear({"run": a.run_name}))["cleared"]
    assert asyncio.run(clear({"run": b.run_name}))["cleared"]
    assert st.jobs == [a] and a.status == "training"
    assert {d.id for d in st.ducks} == {"trainee", "helper1"}


def test_stage_weights_route_to_the_named_older_job(fake_popen):
    app = V.make_app([])
    a = V.TrainingJob("backflip", steps=1000, trainee_id="trainee")
    b = V.TrainingJob("one_leg", steps=1000, trainee_id="trainee2")
    app.state.lab.jobs = [a, b]
    route = _endpoint(app, "/teach/weights", "POST")
    key = a.behavior.terms[0].key
    weight = a.behavior.terms[0].weight * 0.5
    out = asyncio.run(route(V.StageWeightsReq(run=a.run_name, stageWeights={"1": {key: weight}})))
    assert out["ok"] and out["job"]["runName"] == a.run_name
    assert a.stage_weights[1][key] == weight and b.stage_weights == {}
    assert b.proc.returncode is None


def test_loaded_run_resumes_into_its_own_preview(fake_popen):
    app = V.make_app([])
    st = app.state.lab
    outgoing = V.TrainingJob("one_leg", trainee_id="trainee")
    outgoing.stop(force=True)
    st.jobs = [outgoing]
    old_duck = V.Duck("trainee", "old", V._zero_infer, seed=1)
    st.ducks = [old_duck]
    name = "teach-spin-loaded-resume"
    _seed_finished_run(name)
    run = V.RUNS_DIR / name
    (run / "model.zip").touch()
    (run / "vecnormalize.pkl").touch()
    (run / "progress.jsonl").write_text(json.dumps({
        "steps": 50, "total": 100000, "done": False, "stopped": "user"}) + "\n")
    load = _endpoint(app, "/teach/load", "POST")
    assert asyncio.run(load(V.LoadRunReq(policy=f"run:{name}")))["ok"]
    assert st.jobs[0] is outgoing and not st.job.owns_preview_ducks
    out = asyncio.run(_endpoint(app, "/teach/resume", "POST")(V.ResumeReq(runName=name)))
    assert out["ok"] and st.job.trainee_id != "trainee"
    assert st.duck("trainee") is old_duck
    assert st.job.owns_preview_ducks


def test_adopted_stage_keeps_its_helpers(fake_popen):
    job = V.TrainingJob("backflip", trainee_id="trainee")
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "behavior.json").write_text(json.dumps({
        "behavior": "backflip", "steps": 1000, "weights": {}}))
    adopted = V.TrainingJob.adopt(job.run_name)
    helper = _fake_duck("helper1", onnx_path=str(job.dir / "live.onnx"))
    st = V.LabState([helper])
    st.jobs = [adopted]
    assert V.helpers_for_job(st, adopted) == [helper]


def test_new_job_counts_only_helpers_it_can_take(fake_popen, monkeypatch):
    monkeypatch.setattr(V, "env_budget", lambda: V.BASE_ENVS * 4)
    app = V.make_app([])
    st = app.state.lab
    teach = _endpoint(app, "/teach", "POST")
    assert asyncio.run(teach(V.TeachReq(text="spin")))["matched"]
    first = st.job
    helper = V.Duck("helper1", "helper", V._zero_infer, seed=1,
                    onnx_path=str(first.dir / "live.onnx"))
    st.ducks.append(helper)
    original_env = helper.env
    assert asyncio.run(teach(V.TeachReq(text="one_leg")))["matched"]
    assert st.job.helpers == 0
    assert helper.env is original_env
    assert V.helpers_for_job(st, first) == [helper]
