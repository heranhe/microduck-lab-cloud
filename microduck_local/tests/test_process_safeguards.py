"""Nothing this repo starts may outlive what it was started for.

Two leaks, both found running on 2026-09-28:
* vec-env workers of a trainer killed with SIGKILL (or OOM, or a crash)
  were reparented to launchd and waited on their semaphore forever;
* a scratch lab from a filming session was still stepping its roster three
  days later — and, started with --fresh on the shared state file, had
  deleted the main lab's roster on the way in.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from microduck_local import viz_server as V


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); ask ps whether it is really running
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _wait_gone(pids, timeout: float) -> list[int]:
    end = time.monotonic() + timeout
    left = list(pids)
    while left and time.monotonic() < end:
        left = [p for p in left if _alive(p)]
        time.sleep(0.1)
    return left


def test_vec_env_workers_exit_when_their_trainer_is_killed():
    script = textwrap.dedent("""
        import time, numpy as np, gymnasium as gym
        from microduck_local.vec_env import ForkVecEnv
        class Toy(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (2,), np.float32)
            action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
            def reset(self, *, seed=None, options=None):
                return np.zeros(2, np.float32), {}
            def step(self, a):
                return np.zeros(2, np.float32), 0.0, False, False, {}
        v = ForkVecEnv([Toy for _ in range(3)])
        v.reset()
        print(*[p.pid for p in v.processes], flush=True)
        time.sleep(120)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script],
                              stdout=subprocess.PIPE, text=True)
    workers: list[int] = []
    try:
        workers = [int(x) for x in parent.stdout.readline().split()]
        assert len(workers) == 3 and all(_alive(p) for p in workers)
        parent.send_signal(signal.SIGKILL)     # no atexit, no "close"
        parent.wait(timeout=10)
        assert _wait_gone(workers, timeout=10) == [], "orphaned vec-env workers"
    finally:
        parent.kill()
        for p in workers:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_a_later_fork_child_does_not_keep_the_workers_alive():
    """The lifeline's write end must live ONLY in the process that made it: a
    helper the trainer forks afterwards (an eval pool, another vec env's
    workers) would otherwise hold it open after the trainer is killed."""
    script = textwrap.dedent("""
        import os, time, numpy as np, gymnasium as gym
        from microduck_local.vec_env import ForkVecEnv
        class Toy(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (2,), np.float32)
            action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
            def reset(self, *, seed=None, options=None):
                return np.zeros(2, np.float32), {}
            def step(self, a):
                return np.zeros(2, np.float32), 0.0, False, False, {}
        v = ForkVecEnv([Toy for _ in range(2)])
        v.reset()
        helper = os.fork()
        if helper == 0:
            time.sleep(60)
            os._exit(0)
        print(helper, *[p.pid for p in v.processes], flush=True)
        time.sleep(120)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script],
                              stdout=subprocess.PIPE, text=True)
    pids: list[int] = []
    try:
        pids = [int(x) for x in parent.stdout.readline().split()]
        helper, workers = pids[0], pids[1:]
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)
        assert _alive(helper)                 # the helper is still there...
        assert _wait_gone(workers, timeout=10) == [],             "a later fork child kept the vec-env workers alive"
    finally:
        parent.kill()
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_an_http_request_counts_as_someone_using_the_lab(tmp_path, monkeypatch):
    """An agent driving /world over REST never opens a socket; its requests
    must keep a scratch lab from idling out under it."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    before = app.state.last_request
    time.sleep(0.01)
    with TestClient(app) as c:
        c.get("/teach/status")
    assert app.state.last_request > before


def test_idle_watch_needs_an_unbroken_quiet_spell():
    w = V.IdleWatch(10.0)
    assert not w.update(False, 0.0)
    assert not w.update(False, 9.9)
    assert not w.update(True, 10.0)        # someone looked: the clock restarts
    assert not w.update(False, 11.0)
    assert not w.update(False, 20.9)
    assert w.update(False, 21.0)


def test_only_scratch_ports_exit_by_default():
    assert V.idle_exit_minutes(V.DEFAULT_LAB_PORT, None) == 0.0
    assert V.idle_exit_minutes(8799, None) == V.SCRATCH_IDLE_EXIT_MIN
    assert V.idle_exit_minutes(8799, 0) == 0.0          # opted out
    assert V.idle_exit_minutes(V.DEFAULT_LAB_PORT, 5) == 5.0


def test_a_scratch_port_never_touches_the_main_roster(monkeypatch):
    monkeypatch.delenv("LAB_STATE_PATH", raising=False)
    assert V.scratch_state_path(V.DEFAULT_LAB_PORT) is None
    p = V.scratch_state_path(8799)
    assert p is not None and p.name == "lab-state-8799.json"
    assert p != V.lab_state_path()
    monkeypatch.setenv("LAB_STATE_PATH", "/tmp/x.json")
    assert V.scratch_state_path(8799) is None           # an explicit path wins


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_scratch_lab_nobody_watches_shuts_itself_down(tmp_path):
    walker = V.POLICIES_DIR / "alpha_walking.onnx"
    if not walker.exists():
        pytest.skip("shipped policies not downloaded — scripts/setup.sh")
    port = _free_port()
    env = {**os.environ, "LAB_STATE_PATH": str(tmp_path / "lab-state.json")}
    lab = subprocess.Popen(
        [sys.executable, "-c", "from microduck_local.viz_server import main; main()",
         str(walker), "--port", str(port), "--idle-exit", "0.05"],     # 3 s
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = lab.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        lab.kill()
        pytest.fail("the scratch lab was still running after 120 s")
    assert "exiting (--idle-exit)" in out, out[-2000:]
