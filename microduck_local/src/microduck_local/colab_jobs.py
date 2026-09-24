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
    return {"installed": True, "connected": True,
            "balance": float(balance.group(1).replace(",", "")) if balance else None,
            "rate": float(rate.group(1).replace(",", "")) if rate else None}


def _remote_runner(task: str, iterations: int, envs: int) -> str:
    """Create a self-contained Colab script; the official stack owns training."""
    return f'''import json, os, pathlib, shutil, subprocess, time, traceback
root = pathlib.Path("/content/microduck-cloud-job")
root.mkdir(exist_ok=True)
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
    subprocess.run(["uv", "sync", "--no-dev"], cwd=repo, check=True)
    mark("training")
    with (root / "train.log").open("w") as log:
        result = subprocess.run(["uv", "run", "train", "{task}",
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
    subprocess.run(["uv", "run", "python", "scripts/export.py", "{task}",
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
        self.lock = threading.Lock()
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
        account = account_status()
        if not account["connected"]:
            raise RuntimeError("Connect your Google account with `colab usage` first")
        balance = account.get("balance")
        if not isinstance(balance, (int, float)) or balance <= 0:
            raise RuntimeError("No positive Colab compute-unit balance was found. Check your subscription first")
        job_id = uuid.uuid4().hex[:12]
        session = f"microduck-{job_id}"
        directory = self.runs / job_id
        directory.mkdir(parents=True)
        job = {"id": job_id, "session": session, "task": task, "gpu": gpu,
               "state": "allocating", "started": time.time(), "released": False, "message": ""}
        with self.lock:
            self.jobs[job_id] = job
        (directory / "job.json").write_text(json.dumps(job, indent=2))
        threading.Thread(target=self._run, args=(job, directory, iterations, envs), daemon=True).start()
        return dict(job)

    def _run(self, job: dict, directory: Path, iterations: int, envs: int) -> None:
        session = job["session"]
        try:
            result = cli(["new", "-s", session, "--gpu", job["gpu"]], timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            job["allocated_at"] = time.time()
            if job["state"] in {"stopping", "stopped"}:
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
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            if job["state"] in {"stopping", "stopped"}:
                return
            job["state"] = "running"
            while job["state"] == "running":
                time.sleep(15)
                code = "import pathlib; p=pathlib.Path('/content/microduck-cloud-job/status.json'); print(p.read_text() if p.exists() else '{}')"
                result = cli(["exec", "-s", session], code=code, timeout=30)
                if result.returncode:
                    continue
                for line in result.stdout.splitlines():
                    if line.strip().startswith('{'):
                        try:
                            remote = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if remote.get("state") in {"setup", "training", "exporting", "done", "failed"}:
                            job["remote_state"] = remote["state"]
                            job["message"] = remote.get("message", "")
                            if remote["state"] in {"done", "failed"}:
                                job["state"] = "finalizing"
                if job["state"] == "finalizing":
                    for name in ("policy.onnx", "checkpoint.pt", "train.log", "error.log"):
                        cli(["download", "-s", session,
                             f"/content/microduck-cloud-job/{name}", str(directory / name)], timeout=120)
                    policy = directory / "policy.onnx"
                    if job.get("remote_state") == "done" and policy.is_file() and policy.stat().st_size > 0:
                        job["state"] = "done"
                    else:
                        job["state"] = "failed"
                        job["message"] = job["message"] or "Training or ONNX export did not complete"
        except Exception as exc:
            if job["state"] not in {"stopping", "stopped"}:
                job["state"] = "failed"
                job["message"] = str(exc)
        finally:
            if not job.get("released"):
                try:
                    result = cli(["stop", "-s", session], timeout=60)
                    job["released"] = result.returncode == 0
                    if not job["released"]:
                        reason = result.stderr.strip() or result.stdout.strip()
                        job["message"] = f"{job['message']} · Release: {reason}" if job["message"] else reason
                except Exception as exc:
                    job["released"] = False
                    job["message"] = f"{job['message']} · Release: {exc}" if job["message"] else str(exc)
            if job["state"] == "stopping":
                job["state"] = "stopped"
            (directory / "job.json").write_text(json.dumps(job, indent=2))

    def stop(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("released") is True:
            return dict(job)
        if job["state"] in {"allocating", "stopping"}:
            # The allocation thread must finish `new` before it can release
            # the session. Keep the pending state visible to the UI.
            job["state"] = "stopping"
            (self.runs / job_id / "job.json").write_text(json.dumps(job, indent=2))
            return dict(job)
        job["state"] = "stopped" if job["state"] not in {"done", "failed"} else job["state"]
        try:
            result = cli(["stop", "-s", job["session"]], timeout=60)
            job["released"] = result.returncode == 0
            if not job["released"]:
                job["message"] = result.stderr.strip() or result.stdout.strip()
        except Exception as exc:
            job["released"] = False
            job["message"] = str(exc)
        (self.runs / job_id / "job.json").write_text(json.dumps(job, indent=2))
        return dict(job)

    def list(self) -> list[dict]:
        return [dict(job) for job in self.jobs.values()]

    def stop_all(self) -> list[dict]:
        """Release every session started by this lab, leaving other Colab work alone."""
        return [self.stop(job_id) for job_id, job in list(self.jobs.items())
                if job["state"] in {"allocating", "starting", "running", "finalizing", "detached", "stopping"}
                or job.get("released") is False]
