"""Colab GPU jobs for the local lab.

The Google account belongs to the person running the lab.  Authentication is
performed by google-colab-cli; this module never reads or stores OAuth tokens.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

GPU_TYPES = {"T4", "L4", "A100", "H100"}
TASKS = {"Mjlab-Velocity-Flat-MicroDuck", "Mjlab-VelStand-Flat-MicroDuck"}
RUNS = Path(__file__).resolve().parents[2] / "runs" / "colab"
POLL_TIMEOUT = 60
MAX_POLL_FAILURES = 6
UPSTREAM_REVISION = "cb70b792312d559a4da09064d92009079671815f"
TEACHERS = {
    "69u48n8l": ("MICRODUCK_STAND_TEACHER", "model_9750.pt"),
    "441tzs6d": ("MICRODUCK_WALK_TEACHER", "model_3750.pt"),
}


def teacher_checkpoints(task: str) -> dict[str, Path]:
    """Check prerequisites before any GPU allocation; never read credentials."""
    if task != "Mjlab-VelStand-Flat-MicroDuck":
        return {}
    found = {}
    missing = []
    for run, (variable, filename) in TEACHERS.items():
        path = Path(os.environ.get(variable) or str(RUNS.parent.parent / "teachers" / run / filename)).expanduser()
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"{run}/{filename}（可用 {variable} 指定路径）")
        else:
            found[run] = path.resolve()
    if missing:
        raise RuntimeError("VelStand 缺少教师模型，尚未分配 GPU。请从有访问权限的 W&B 项目 pollen-robotics/mjlab_microduck 下载：" + "；".join(missing) + "，放入 microduck_local/teachers/<run>/ 对应目录后重试。")
    return found


def friendly_cli_error(value: str) -> str:
    """Turn known provider/client failures into an actionable UI message."""
    raw = value.strip()
    if "JupyterSubprotocol" in raw or "jupyter_kernel_client" in raw:
        return (
            "Colab CLI runtime is incompatible with its kernel client. "
            "Update the project environment with `uv sync` (google-colab-cli >= 0.7.4), then retry."
        )
    return raw


def cli_path() -> str | None:
    bundled = Path(os.sys.executable).parent / "colab"
    if bundled.is_file():
        return str(bundled)
    return shutil.which("colab")


def cli(args: list[str], *, code: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    binary = cli_path()
    if not binary:
        raise RuntimeError("Colab CLI is not installed. Run: uv tool install google-colab-cli")
    return subprocess.run(
        [binary, *args], input=code, text=True, capture_output=True,
        stdin=subprocess.DEVNULL if code is None else None, timeout=timeout,
    )


def account_status() -> dict:
    """Read only: never trigger an interactive OAuth flow from an HTTP request."""
    if not cli_path():
        return {"installed": False, "connected": False, "message": "Install google-colab-cli"}
    try:
        result = cli(["usage"], timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": True, "connected": False, "message": str(exc)}
    if result.returncode:
        return {"installed": True, "connected": False,
                "message": "Run `colab usage` in Terminal to connect your Google account."}
    balance = re.search(r"Current balance:\s*([\d.,]+)", result.stdout)
    rate = re.search(r"Usage rate:\s*([\d.,]+)", result.stdout)
    # Identity must come from the same CLI credentials used to allocate GPUs.
    # Return only the email, never the diagnostic scopes, audience or tokens.
    email = None
    try:
        identity = cli(["whoami"], timeout=15)
        match = re.search(r"^Email:\s*([^\s<>]+@[^\s<>]+)\s*$", identity.stdout, re.MULTILINE)
        if identity.returncode == 0 and match:
            email = match.group(1)
    except (OSError, subprocess.TimeoutExpired):
        pass  # An identity lookup failure does not invalidate a working account.
    return {"installed": True, "connected": True,
            "email": email,
            "balance": float(balance.group(1).replace(",", "")) if balance else None,
            "rate": float(rate.group(1).replace(",", "")) if rate else None}


def _remote_runner(task: str, iterations: int, envs: int) -> str:
    """Create a self-contained Colab script; the official stack owns training."""
    return f'''import json, os, pathlib, shutil, subprocess, time, traceback
root = pathlib.Path("/content/microduck-cloud-job")
root.mkdir(parents=True, exist_ok=True)
status = root / "status.json"
def mark(state, message=""):
    status.write_text(json.dumps({{"state": state, "message": message, "time": time.time()}}))
try:
    mark("setup")
    if not shutil.which("uv"):
        subprocess.run(["python", "-m", "pip", "install", "uv"], check=True)
    repo = pathlib.Path("/content/microduck_rl")
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/pollen-robotics/microduck_rl.git", str(repo)], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", "{UPSTREAM_REVISION}"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "--detach", "{UPSTREAM_REVISION}"], cwd=repo, check=True)
    subprocess.run(["uv", "sync", "--no-dev"], cwd=repo, check=True)
    # Both training and export construct the expert-backed runner. Redirect only
    # the pinned teachers to uploaded checkpoints; keep the algorithm unchanged.
    entry = root / "entry.py"
    entry.write_text("""import sys
from pathlib import Path
import mjlab_microduck.tasks.distill as distill
original = distill._resolve_checkpoint
def resolve(cfg, prefix=""):
    run = str(cfg.get(prefix + "wandb_run_path", "")).rsplit("/", 1)[-1]
    if run in ("69u48n8l", "441tzs6d") and not cfg.get(prefix + "checkpoint_path"):
        path = Path("/content/microduck-teacher-" + run + ".pt")
        if not path.is_file():
            raise RuntimeError("Missing uploaded teacher: " + run)
        return path
    return original(cfg, prefix)
distill._resolve_checkpoint = resolve
mode = sys.argv.pop(1)
if mode == "train":
    from mjlab.scripts.train import main
else:
    from mjlab_microduck.export import main
main()
""")
    mark("training")
    with (root / "train.log").open("w") as log:
        result = subprocess.run(["uv", "run", "python", str(entry), "train", "{task}",
            "--env.scene.num-envs", "{envs}", "--agent.max_iterations", "{iterations}",
            "--agent.save_interval", str(max(1, min(250, {iterations})))],
            cwd=repo, env={{**os.environ, "WANDB_MODE": "disabled"}},
            stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"training exited with {{result.returncode}}")
    checkpoints = sorted(repo.glob("logs/**/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError("training finished without a checkpoint")
    mark("exporting")
    subprocess.run(["uv", "run", "python", str(entry), "export", "{task}",
        "--checkpoint-file", str(checkpoints[-1]), "--num-envs", "1",
        "--onnx-file", str(root / "policy.onnx")],
        cwd=repo, check=True)
    shutil.copy2(checkpoints[-1], root / "checkpoint.pt")
    mark("done")
except Exception as exc:
    (root / "error.log").write_text(traceback.format_exc())
    mark("failed", str(exc))
'''


class ColabJobs:
    def __init__(self, runs: Path = RUNS):
        self.runs = runs
        self.jobs: dict[str, dict] = {}
        self.workers: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.release_lock = threading.Lock()
        for path in runs.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if re.fullmatch(r"[0-9a-f]{12}", job["id"]):
                    if job["state"] in {"allocating", "starting", "running", "finalizing", "stopping"}:
                        job["state"] = "detached"
                        job["message"] = "Lab restarted; check and stop this Colab session."
                    self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def start(self, task: str, gpu: str, iterations: int, envs: int) -> dict:
        if task not in TASKS or gpu not in GPU_TYPES:
            raise ValueError("Unsupported task or GPU")
        if not 1 <= iterations <= 100_000 or not 1 <= envs <= 4096:
            raise ValueError("Training limits are out of range")
        teacher_checkpoints(task)
        account = account_status()
        if not account["connected"]:
            raise RuntimeError("Connect your Google account with `colab usage` first")
        # Compute units describe paid access. A free account can still be
        # assigned a standard T4 when Colab has capacity; `colab new` is the
        # authority on current eligibility and quota, not the CCU balance.
        balance = account.get("balance")
        if gpu != "T4" and (not isinstance(balance, (int, float)) or balance <= 0):
            raise RuntimeError("No paid Colab compute units are available. Select T4 to try the free tier")
        job_id = uuid.uuid4().hex[:12]
        session = f"microduck-{job_id}"
        directory = self.runs / job_id
        directory.mkdir(parents=True)
        job = {"id": job_id, "session": session, "task": task, "gpu": gpu,
               "state": "allocating", "started": time.time(), "released": False, "message": ""}
        with self.lock:
            self.jobs[job_id] = job
        (directory / "job.json").write_text(json.dumps(job, indent=2))
        worker = threading.Thread(target=self._run, args=(job, directory, iterations, envs), daemon=True)
        self.workers[job_id] = worker
        worker.start()
        return dict(job)

    def _run(self, job: dict, directory: Path, iterations: int, envs: int) -> None:
        session = job["session"]
        try:
            teachers = teacher_checkpoints(job["task"])
            result = cli(["new", "-s", session, "--gpu", job["gpu"]], timeout=180)
            if result.returncode:
                # The provider rejected the assignment before a runtime was
                # allocated. A failed best-effort `stop` below must not turn
                # this into a phantom billable session that blocks retrying.
                job["allocation_rejected"] = True
                raise RuntimeError(friendly_cli_error(result.stderr.strip() or result.stdout.strip()))
            job["allocated_at"] = time.time()
            (directory / "job.json").write_text(json.dumps(job, indent=2))
            if job["state"] in {"stopping", "stopped"}:
                return
            for run, path in teachers.items():
                if job.get("cancel_requested"):
                    return
                uploaded = cli(["upload", "-s", session, str(path),
                                f"/content/microduck-teacher-{run}.pt"], timeout=120)
                if uploaded.returncode:
                    raise RuntimeError(f"教师模型 {run} 上传失败，请检查连接后重试。")
            # Ask the allocated runtime instead of guessing from the requested
            # tier: Colab may serve different A100 memory variants.
            probe = cli(["exec", "-s", session], code=(
                "import json,torch; p=torch.cuda.get_device_properties(0); "
                "print(json.dumps({'device':p.name,'vram':"
                "f'{p.total_memory / 1024**3:.0f} GB'}))"), timeout=30)
            if probe.returncode == 0:
                for line in probe.stdout.splitlines():
                    if line.strip().startswith("{"):
                        try:
                            device = json.loads(line)
                            job["device"] = device.get("device")
                            job["vram"] = device.get("vram")
                        except json.JSONDecodeError:
                            pass
            if job.get("cancel_requested"):
                return
            job["state"] = "starting"
            runner = _remote_runner(job["task"], iterations, envs)
            # The kernel returns immediately; training runs in a separate process.
            bootstrap = ("import pathlib,subprocess,sys\n"
                         f"p=pathlib.Path('/content/microduck-cloud-runner-{job['id']}.py')\n"
                         f"p.write_text({runner!r})\n"
                         "subprocess.Popen([sys.executable,str(p)],"
                         "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n")
            result = cli(["exec", "-s", session], code=bootstrap, timeout=60)
            if result.returncode:
                raise RuntimeError(friendly_cli_error(result.stderr.strip() or result.stdout.strip()))
            if job.get("cancel_requested"):
                return
            job["state"] = "running"
            poll_failures = 0
            while job["state"] == "running":
                time.sleep(15)
                if job.get("cancel_requested"):
                    return
                code = "import pathlib; p=pathlib.Path('/content/microduck-cloud-job/status.json'); print(p.read_text() if p.exists() else '{}')"
                remote = None
                try:
                    result = cli(["exec", "-s", session], code=code, timeout=POLL_TIMEOUT)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            try:
                                candidate = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(candidate, dict) and candidate.get("state") in {"setup", "training", "exporting", "done", "failed"}:
                                remote = candidate
                except (subprocess.TimeoutExpired, OSError):
                    pass  # A lost progress request is not a failed remote training run.
                if job.get("cancel_requested"):
                    return
                if remote is None:
                    poll_failures += 1
                    job["poll_failures"] = poll_failures
                    job["message"] = f"云端进度暂时无法获取，正在重试（{poll_failures}/{MAX_POLL_FAILURES}）；训练可能仍在运行。"
                    if poll_failures >= MAX_POLL_FAILURES:
                        raise RuntimeError("连续多次无法获取云端进度，已停止本次任务并尝试释放算力；无法确认远端训练结果。")
                    continue
                poll_failures = 0
                job["poll_failures"] = 0
                job["last_heartbeat_at"] = time.time()
                job["remote_state"] = remote["state"]
                job["message"] = remote.get("message", "")
                if remote["state"] in {"done", "failed"}:
                    job["state"] = "finalizing"
                if job["state"] == "finalizing":
                    for name in ("policy.onnx", "checkpoint.pt", "train.log", "error.log"):
                        if job.get("cancel_requested"):
                            return
                        cli(["download", "-s", session,
                             f"/content/microduck-cloud-job/{name}", str(directory / name)], timeout=120)
                    if job.get("cancel_requested"):
                        return
                    policy = directory / "policy.onnx"
                    if job.get("remote_state") == "done" and policy.is_file() and policy.stat().st_size > 0:
                        job["state"] = "done"
                    else:
                        job["state"] = "failed"
                        job["message"] = job["message"] or "Training or ONNX export did not complete"
        except Exception as exc:
            if not job.get("cancel_requested"):
                job["state"] = "failed"
                job["message"] = str(exc)
        finally:
            self._release(job)
            if job.get("cancel_requested"):
                job["state"] = "stopped"
            (directory / "job.json").write_text(json.dumps(job, indent=2))

    def _release(self, job: dict) -> None:
        # HTTP cancellation and worker cleanup can arrive together. Check the
        # success flag inside the lock so a second caller cannot undo success.
        with self.release_lock:
            if job.get("released") is True:
                return
            reason = ""
            try:
                result = cli(["stop", "-s", job["session"]], timeout=60)
                released = result.returncode == 0
                if not released:
                    reason = result.stderr.strip() or result.stdout.strip()
                    if "/unassign/" in reason and "Not Found" in reason:
                        status = cli(["status", "-s", job["session"]], timeout=30)
                        missing = f"[colab] Session '{job['session']}' not found."
                        released = status.returncode == 0 and missing in status.stdout.splitlines()
            except Exception as exc:
                released = False
                reason = str(exc)
            job["released"] = released or job.get("allocation_rejected") is True
            if job["released"]:
                job.setdefault("released_at", time.time())
                # A cancelled job should not keep a stale release traceback.
                if job.get("cancel_requested"):
                    job["message"] = ""
            else:
                job["message"] = reason or "Colab resource release could not be confirmed."

    def stop(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("released") is True:
            return dict(job)
        job["cancel_requested"] = True
        if job["state"] in {"allocating", "stopping"}:
            # The allocation thread must finish `new` before it can release
            # the session. Keep the pending state visible to the UI.
            job["state"] = "stopping"
            (self.runs / job_id / "job.json").write_text(json.dumps(job, indent=2))
            return dict(job)
        job["state"] = "stopped" if job["state"] not in {"done", "failed"} else job["state"]
        self._release(job)
        (self.runs / job_id / "job.json").write_text(json.dumps(job, indent=2))
        return dict(job)

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(job) for job in self.jobs.values()]

    def stop_all(self) -> list[dict]:
        """Release every session started by this lab, leaving other Colab work alone."""
        return [self.stop(job_id) for job_id, job in list(self.jobs.items())
                if job["state"] in {"allocating", "starting", "running", "finalizing", "detached", "stopping"}
                or job.get("released") is False]

    def shutdown(self) -> list[dict]:
        """Let in-flight allocation finish and release before daemon exit."""
        self.stop_all()
        deadline = time.monotonic() + 250
        for worker in list(self.workers.values()):
            worker.join(timeout=max(0, deadline - time.monotonic()))
        return self.list()
