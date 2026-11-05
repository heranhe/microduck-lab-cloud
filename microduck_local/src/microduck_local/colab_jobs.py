"""Colab GPU jobs for the local lab.

The Google account belongs to the person running the lab.  Authentication is
performed by google-colab-cli; this module never reads or stores OAuth tokens.
The web sign-in only relays a one-time authorization code to the CLI's stdin.

Billing rules this module exists to keep:

* A runtime costs money from assignment until unassignment, so every job has a
  wall-clock budget and every stage has a stall limit.
* "Released" is what Google's assignment list says, never what `stop` returned.
* The lab can crash, sleep or lose the network while a runtime keeps billing, so
  a background reconciler retries releases and re-adopts orphaned jobs.
* Results that were paid for are downloaded, with retries, BEFORE unassigning.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

GPU_TYPES = {"T4", "L4", "A100", "H100"}
TASKS = {"Mjlab-Velocity-Flat-MicroDuck", "Mjlab-VelStand-Flat-MicroDuck"}
RUNS = Path(__file__).resolve().parents[2] / "runs" / "colab"
REMOTE_ROOT = "/content/microduck-cloud-job"
UPSTREAM_REVISION = "cb70b792312d559a4da09064d92009079671815f"
TEACHERS = {
    "69u48n8l": ("MICRODUCK_STAND_TEACHER", "model_9750.pt"),
    "441tzs6d": ("MICRODUCK_WALK_TEACHER", "model_3750.pt"),
}

# --- budgets -----------------------------------------------------------------
# microduck_rl's README puts a usable gait at about 1-2 h on 4096 envs, so the
# default leaves room for that without letting a forgotten job run overnight.
DEFAULT_MAX_MINUTES = 180
MIN_MAX_MINUTES = 5
MAX_MAX_MINUTES = 720
# The runtime enforces its own deadline a little later than the lab does, so the
# lab normally wins and saves results; the remote one only fires if the lab died.
REMOTE_DEADLINE_SLACK_S = 300

# --- monitoring --------------------------------------------------------------
POLL_INTERVAL_S = 15
POLL_TIMEOUT = 60
# A lost progress request is not a failed training run. Billing continues either
# way and nothing can be released while offline, so wait out ordinary outages.
POLL_GRACE_S = 900
BOOTSTRAP_GRACE_S = 180   # runner never wrote its first status
SETUP_TIMEOUT_S = 1500    # clone + uv sync; about 4 min when healthy
STALL_TIMEOUT_S = 900     # train.log silent while "training"
EXPORT_TIMEOUT_S = 900
RESCUE_TIMEOUT_S = 420    # asked to save; give up and release after this
USAGE_SAMPLE_S = 60
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_S = 10
ARTIFACTS = ("policy.onnx", "checkpoint.pt", "train.log", "setup.log", "error.log")
REQUIRED_ARTIFACTS = ("policy.onnx", "checkpoint.pt")
SERVED_ARTIFACTS = ("checkpoint.pt", "train.log", "setup.log", "error.log")
POLICY_OBS = 61
POLICY_ACTIONS = 14

# --- release / reconcile -----------------------------------------------------
RELEASE_RETRY_DELAYS_S = (5, 15)
RELEASE_CONFIRM_ATTEMPTS = 3
RELEASE_CONFIRM_DELAY_S = 3
RECONCILE_TICK_S = 5
RECONCILE_ACTIVE_S = 60
RECONCILE_IDLE_S = 300
RELEASE_BACKOFF_MAX_S = 600
NEW_TIMEOUT_S = 300

ACTIVE_STATES = {"allocating", "starting", "running", "saving", "finalizing", "detached", "stopping"}
REMOTE_STATES = {"setup", "training", "saving", "exporting", "rescued", "done", "failed", "missing"}
ACCOUNT_TTL_CONNECTED_S = 60
ACCOUNT_TTL_NETWORK_S = 30
ACCOUNT_TTL_AUTH_EXPIRED_S = 300
AUTH_FLOW_TTL_S = 600
_SESSION_LINE = re.compile(r"^\[([^\]]+)\] (\S+) \| Hardware: ([^|]+?) \|", re.MULTILINE)
_AUTH_URL = re.compile(r"https://accounts\.google\.com/\S+")
_NOISE = re.compile(r"new version of Colab CLI|'colab update|'pip install --upgrade google-colab-cli"
                    r"|To silence this check")


class PaidConfirmationRequired(RuntimeError):
    """A job could spend the account's paid compute units and nobody agreed yet."""


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
    raw = "\n".join(line for line in value.strip().splitlines() if not _NOISE.search(line)).strip()
    if "JupyterSubprotocol" in raw or "jupyter_kernel_client" in raw:
        return (
            "Colab CLI runtime is incompatible with its kernel client. "
            "Update the project environment with `uv sync` (google-colab-cli >= 0.7.4), then retry."
        )
    if _classify_failure(raw) == "auth_expired":
        return "Google 授权已失效，无法操作 Colab。请在云算力面板重新连接账号。"
    if "╭" in raw or "Traceback" in raw:
        # A rich traceback is kilobytes of box drawing around one useful line.
        lines = [line.strip(" │╰─╯╭") for line in raw.splitlines()]
        useful = [line for line in lines if line]
        return useful[-1][:400] if useful else "Colab CLI 执行失败。"
    return raw[:800]


def _classify_failure(text: str) -> str:
    if "authorization code" in text or "To authorize" in text or "invalid_grant" in text:
        return "auth_expired"
    if re.search(r"Max retries|ConnectionError|SSLError|Failed to establish|Temporary failure|"
                 r"Name or service not known|Network is unreachable|failed to fetch compute-unit", text):
        return "network"
    return "unknown"


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


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1).replace(",", "")) if match else None


def account_status() -> dict:
    """Read only: never trigger an interactive OAuth flow from an HTTP request.

    stdin is closed, so an expired token ends the CLI's code prompt with an
    error instead of hanging. A failure is classified rather than folded into
    "not connected": re-authorizing cannot fix a network outage.
    """
    checked_at = time.time()
    if not cli_path():
        return {"installed": False, "connected": False, "reason": "not_installed",
                "message": "Install google-colab-cli", "checked_at": checked_at}
    try:
        result = cli(["usage"], timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"installed": True, "connected": False, "reason": "network",
                "message": str(exc), "checked_at": checked_at}
    if result.returncode:
        reason = _classify_failure(f"{result.stdout}\n{result.stderr}")
        message = {
            "auth_expired": "Google 授权已失效，请重新连接账号。",
            "network": "无法连接 Google Colab，请检查网络或代理后重试。",
        }.get(reason, "Run `colab usage` in Terminal to connect your Google account.")
        return {"installed": True, "connected": False, "reason": reason,
                "message": message, "checked_at": checked_at}
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
    assignments = _number(r"Active assignments:\s*([\d.,]+)", result.stdout)
    return {"installed": True, "connected": True,
            "reason": "ok",
            "email": email,
            "balance": _number(r"Current balance:\s*([\d.,]+)", result.stdout),
            "rate": _number(r"Usage rate:\s*([\d.,]+)", result.stdout),
            "active_assignments": int(assignments) if assignments is not None else None,
            "checked_at": checked_at}


_account_cache: dict = {"value": None, "at": 0.0}
_account_lock = threading.Lock()


def cached_account_status(force: bool = False) -> dict:
    """The panels poll; the CLI should not.

    Before this cache one open page spawned `colab usage` every 15 s, and a dead
    token logged 2311 failed refreshes in one day. Concurrent callers share one
    lookup, and an expired token is re-checked rarely until someone re-authorizes.
    """
    with _account_lock:
        value = _account_cache["value"]
        if value is not None and not force:
            ttl = (ACCOUNT_TTL_CONNECTED_S if value.get("connected") else
                   ACCOUNT_TTL_AUTH_EXPIRED_S if value.get("reason") == "auth_expired" else
                   ACCOUNT_TTL_NETWORK_S)
            if time.time() - _account_cache["at"] < ttl:
                return dict(value)
        value = account_status()
        _account_cache.update(value=value, at=time.time())
        return dict(value)


def invalidate_account_cache() -> None:
    with _account_lock:
        _account_cache.update(value=None, at=0.0)


def server_sessions() -> list[dict] | None:
    """Every assignment Google reports for this account; None if it cannot say.

    None and [] must stay different: "could not ask" never proves a release. An
    unnamed entry ("?") has no local CLI state, which is what a `colab new`
    killed mid-assignment leaves behind, or a session from the Colab website.
    """
    try:
        result = cli(["sessions"], timeout=30)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    entries = [{"name": name, "endpoint": endpoint, "hardware": hardware.strip()}
               for name, endpoint, hardware in _SESSION_LINE.findall(result.stdout)]
    if not entries and "No active sessions" not in result.stdout:
        return None
    return entries


class ColabAuthFlow:
    """Runs the CLI's copy-paste sign-in on the person's behalf.

    The CLI prints a Google URL and waits on stdin for the code Google shows.
    The code is written straight to that process and never logged or stored;
    the CLI exchanges it and writes its own token file.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.started_at = 0.0
        self.stderr: list[str] = []
        self.stdout: list[str] = []

    def _alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _kill(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None

    def start(self) -> dict:
        binary = cli_path()
        if not binary:
            raise RuntimeError("Colab CLI is not installed. Run: uv tool install google-colab-cli")
        with self.lock:
            self._kill()
            self.stderr, self.stdout = [], []
            self.started_at = time.time()
            self.process = subprocess.Popen([binary, "usage"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for stream, sink in ((self.process.stderr, self.stderr), (self.process.stdout, self.stdout)):
                threading.Thread(target=self._drain, args=(stream, sink), daemon=True).start()
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                match = _AUTH_URL.search("".join(self.stderr))
                if match:
                    return {"connected": False, "url": match.group(0)}
                if self.process.poll() is not None:
                    # Give the drain threads a moment to deliver the last lines.
                    time.sleep(0.2)
                    match = _AUTH_URL.search("".join(self.stderr))
                    if match:
                        return {"connected": False, "url": match.group(0)}
                    break
                time.sleep(0.1)
            failed = self.process.poll() not in (0,)
            detail = "".join(self.stderr)
            self._kill()
        invalidate_account_cache()
        if failed:
            raise RuntimeError(friendly_cli_error(detail) or "无法启动 Google 授权流程。")
        return {"connected": True}

    @staticmethod
    def _drain(stream, sink: list[str]) -> None:
        for line in stream:
            sink.append(line)

    def finish(self, code: str) -> dict:
        code = code.strip()
        if not code or len(code) > 2048 or re.search(r"\s", code):
            raise ValueError("授权码格式不正确，请复制 Google 页面显示的完整授权码。")
        with self.lock:
            if not self._alive() or time.time() - self.started_at > AUTH_FLOW_TTL_S:
                self._kill()
                raise ValueError("授权流程已过期，请重新点击连接。")
            process = self.process
            process.stdin.write(code + "\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                returncode = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._kill()
                raise RuntimeError("Google 授权超时，请检查网络后重试。") from None
            self.process = None
        invalidate_account_cache()
        if returncode:
            raise ValueError("Google 未接受这个授权码，请重新点击连接并使用新的授权码。")
        return cached_account_status(force=True)

    def cancel(self) -> None:
        with self.lock:
            self._kill()


def _remote_runner(task: str, iterations: int, envs: int, max_seconds: int = DEFAULT_MAX_MINUTES * 60) -> str:
    """Create a self-contained Colab script; the official stack owns training.

    The runner also owns a deadline and a rescue path. A runtime cannot unassign
    itself without the owner's OAuth token, so if the lab is gone the most it can
    do is stop burning GPU, keep the result on disk and go idle for Google to
    reclaim; a live lab picks up the "rescued" state and finishes the job.
    """
    return f'''import json, os, pathlib, shutil, signal, subprocess, time, traceback
root = pathlib.Path("{REMOTE_ROOT}")
root.mkdir(parents=True, exist_ok=True)
status = root / "status.json"
rescue_flag = root / "rescue.request"
hard_deadline = time.time() + {max_seconds}
def mark(state, message=""):
    # Atomic: a poll must never read a half-written status as "missing".
    scratch = root / "status.tmp"
    scratch.write_text(json.dumps({{"state": state, "message": message, "time": time.time()}}))
    os.replace(scratch, status)
def setup(cmd, **kwargs):
    with (root / "setup.log").open("a") as log:
        log.write("$ " + " ".join(cmd) + "\\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout={SETUP_TIMEOUT_S}, **kwargs)
try:
    mark("setup")
    if not shutil.which("uv"):
        setup(["python", "-m", "pip", "install", "uv"])
    repo = pathlib.Path("/content/microduck_rl")
    if not repo.exists():
        setup(["git", "clone", "--depth", "1",
               "https://github.com/pollen-robotics/microduck_rl.git", str(repo)])
    setup(["git", "fetch", "--depth", "1", "origin", "{UPSTREAM_REVISION}"], cwd=repo)
    setup(["git", "checkout", "--detach", "{UPSTREAM_REVISION}"], cwd=repo)
    setup(["uv", "sync", "--no-dev"], cwd=repo)
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
    def export(checkpoint):
        with (root / "export.log").open("a") as log:
            subprocess.run(["uv", "run", "python", str(entry), "export", "{task}",
                "--checkpoint-file", str(checkpoint), "--num-envs", "1",
                "--onnx-file", str(root / "policy.onnx")],
                cwd=repo, check=True, stdout=log, stderr=subprocess.STDOUT, timeout={EXPORT_TIMEOUT_S})
    mark("training")
    # Unbuffered, so train.log is a live progress feed and a stall detector.
    # Its own session, so a rescue can signal uv and its Python child together.
    with (root / "train.log").open("w") as log:
        trainer = subprocess.Popen(["uv", "run", "python", str(entry), "train", "{task}",
            "--env.scene.num-envs", "{envs}", "--agent.max_iterations", "{iterations}",
            "--agent.save_interval", str(max(1, min(250, {iterations})))],
            cwd=repo, env={{**os.environ, "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stopping = None
        while trainer.poll() is None:
            if rescue_flag.exists():
                stopping = rescue_flag.read_text().strip() or "stop requested"
            elif time.time() >= hard_deadline:
                stopping = "remote deadline"
            if stopping:
                break
            time.sleep(2)
        if stopping:
            mark("saving", stopping)
            try:
                os.killpg(trainer.pid, signal.SIGTERM)
                trainer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(trainer.pid, signal.SIGKILL)
                trainer.wait()
            except ProcessLookupError:
                pass
    if not stopping and trainer.returncode:
        raise RuntimeError(f"training exited with {{trainer.returncode}}")
    checkpoints = sorted(repo.glob("logs/**/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if stopping:
        note = stopping
        if checkpoints:
            shutil.copy2(checkpoints[-1], root / "checkpoint.pt")
            try:
                export(checkpoints[-1])
            except Exception as exc:
                note = f"{{stopping}}; export failed: {{exc}}"
        else:
            note = f"{{stopping}}; no checkpoint yet"
        mark("rescued", note)
    else:
        if not checkpoints:
            raise RuntimeError("training finished without a checkpoint")
        mark("exporting")
        shutil.copy2(checkpoints[-1], root / "checkpoint.pt")
        export(checkpoints[-1])
        mark("done")
except Exception as exc:
    (root / "error.log").write_text(traceback.format_exc())
    mark("failed", str(exc))
'''


def _poll_code(root: str = REMOTE_ROOT) -> str:
    """One JSON line: stage, live training numbers, GPU use and artifact sizes.

    Parsed on the runtime so only a few hundred bytes cross the wire. The log's
    age is read on the runtime's own clock, so the Mac's sleep cannot fake a stall.
    """
    return f'''import json, pathlib, re, subprocess, time
root = pathlib.Path({root!r})
try:
    out = json.loads((root / "status.json").read_text())
except Exception:
    out = {{"state": "missing"}}
log = root / "train.log"
if log.is_file():
    stat = log.stat()
    out["log_age"] = round(time.time() - stat.st_mtime, 1)
    with log.open("rb") as fh:
        fh.seek(max(0, stat.st_size - 16000))
        tail = re.sub(r"\\x1b\\[[0-9;]*m", "", fh.read().decode("utf-8", "replace"))
    found = re.findall(r"Learning iteration (\\d+)/(\\d+)", tail)
    if found:
        out["iteration"], out["iterations"] = int(found[-1][0]), int(found[-1][1])
    for key, pattern in (("reward", r"Mean reward: *(-?[\\d.]+)"), ("iter_time", r"Iteration time: *([\\d.]+)s"),
                         ("elapsed", r"Time elapsed: *(\\S+)"), ("eta", r"ETA: *(\\S+)")):
        hits = re.findall(pattern, tail)
        if hits:
            out[key] = hits[-1]
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    util, used, total = [int(v) for v in gpu.stdout.splitlines()[0].split(",")]
    out["gpu_util"], out["gpu_mem_used"], out["gpu_mem_total"] = util, used, total
except Exception:
    pass
out["files"] = {{p.name: p.stat().st_size for p in root.iterdir() if p.is_file()}} if root.is_dir() else {{}}
print(json.dumps(out, separators=(",", ":")))
'''


def _policy_contract_error(path: Path) -> str | None:
    """Reject an export that cannot run behind the duck's ONNX contract."""
    try:
        import onnx
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        def dims(value):
            return [d.dim_value for d in value.type.tensor_type.shape.dim]
        obs, actions = dims(model.graph.input[0]), dims(model.graph.output[0])
    except Exception as exc:
        return f"ONNX 校验失败：{exc}"
    if not obs or obs[-1] != POLICY_OBS or not actions or actions[-1] != POLICY_ACTIONS:
        return f"ONNX 输入输出不符合部署约定：obs {obs}，actions {actions}"
    return None


class ColabJobs:
    def __init__(self, runs: Path = RUNS):
        self.runs = runs
        self.jobs: dict[str, dict] = {}
        self.workers: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.release_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.server: dict = {"ok": None, "checked_at": None, "count": None, "orphans": [], "others": []}
        self.keep_awake = False
        self._awake: subprocess.Popen | None = None
        self._stop_reconciler = threading.Event()
        self._reconcile_soon = threading.Event()
        self._reconciler: threading.Thread | None = None
        for path in runs.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if re.fullmatch(r"[0-9a-f]{12}", job["id"]):
                    if job["state"] in ACTIVE_STATES:
                        job["state"] = "detached"
                        job["message"] = "后端已重启，正在向 Google 确认这个 Colab 会话的状态。"
                    self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    # --- persistence ---------------------------------------------------------
    def _save(self, job: dict) -> None:
        """Atomic: a crash mid-write must not lose the record of a billing session."""
        if "id" not in job:
            return
        directory = self.runs / job["id"]
        with self.save_lock:
            directory.mkdir(parents=True, exist_ok=True)
            fd, scratch = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(dict(job), fh, indent=2)
                os.replace(scratch, directory / "job.json")
            except BaseException:
                Path(scratch).unlink(missing_ok=True)
                raise

    # --- starting a job ------------------------------------------------------
    def start(self, task: str, gpu: str, iterations: int, envs: int,
              max_minutes: int = DEFAULT_MAX_MINUTES, max_ccu: float | None = None,
              confirm_paid: bool = False) -> dict:
        if task not in TASKS or gpu not in GPU_TYPES:
            raise ValueError("Unsupported task or GPU")
        if not 1 <= iterations <= 100_000 or not 1 <= envs <= 4096:
            raise ValueError("Training limits are out of range")
        if not MIN_MAX_MINUTES <= max_minutes <= MAX_MAX_MINUTES:
            raise ValueError(f"时长上限需要在 {MIN_MAX_MINUTES}–{MAX_MAX_MINUTES} 分钟之间")
        if max_ccu is not None and max_ccu <= 0:
            raise ValueError("计算单元上限必须大于 0")
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
        # With a balance on the account any GPU may draw on it, T4 included.
        if not confirm_paid and (not isinstance(balance, (int, float)) or balance > 0):
            raise PaidConfirmationRequired(
                f"账户余额 {balance if balance is not None else '未知'} 计算单元，本次训练可能产生扣费，请确认后再启动。")
        job_id = uuid.uuid4().hex[:12]
        session = f"microduck-{job_id}"
        directory = self.runs / job_id
        directory.mkdir(parents=True)
        job = {"id": job_id, "session": session, "task": task, "gpu": gpu,
               "state": "allocating", "started": time.time(), "released": False, "message": "",
               "iterations": iterations, "envs": envs,
               "max_minutes": max_minutes, "max_ccu": max_ccu,
               "balance_start": balance if isinstance(balance, (int, float)) else None}
        with self.lock:
            self.jobs[job_id] = job
        self._save(job)
        worker = threading.Thread(target=self._run, args=(job, directory, iterations, envs), daemon=True)
        self.workers[job_id] = worker
        worker.start()
        self._sync_awake()
        self._reconcile_soon.set()
        return dict(job)

    def _probe_gpu(self, job: dict) -> None:
        """A GPU runtime is what is being paid for, so refuse to train without one.

        `colab exec` exits 0 even when the remote code raised, so success is the
        JSON line appearing, not the exit code.
        """
        code = ("import json,torch; p=torch.cuda.get_device_properties(0); "
                "print(json.dumps({'device':p.name,'vram':"
                "f'{p.total_memory / 1024**3:.0f} GB'}))")
        detail = ""
        for attempt in range(3):
            if job.get("cancel_requested"):
                return
            try:
                probe = cli(["exec", "-s", job["session"], "--timeout", "90"], code=code, timeout=120)
            except subprocess.TimeoutExpired:
                detail = "GPU 探测超时"
                continue
            detail = probe.stderr.strip() or detail
            for line in probe.stdout.splitlines():
                if line.strip().startswith("{"):
                    try:
                        device = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if device.get("device"):
                        job["device"] = device["device"]
                        job["vram"] = device.get("vram")
                        return
            if job.get("cancel_requested"):
                return
            if attempt < 2:
                time.sleep(5)
        raise RuntimeError(f"未能确认 Colab 分配了 GPU，已取消训练以避免空耗算力。{friendly_cli_error(detail)}")

    def _launch(self, job: dict, directory: Path, iterations: int, envs: int) -> bool:
        """Allocate, verify and start the remote runner. False means cancelled."""
        session = job["session"]
        teachers = teacher_checkpoints(job["task"])
        # What Google already lists before our request, so an unnamed entry that
        # appears afterwards can be attributed to a `new` we lost track of.
        before = server_sessions()
        job["baseline_endpoints"] = None if before is None else [e["endpoint"] for e in before]
        try:
            result = cli(["new", "-s", session, "--gpu", job["gpu"]], timeout=NEW_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            job["allocation_unknown"] = True
            raise RuntimeError("Colab 分配超时。Google 侧可能仍会完成分配，正在核对；如有遗留会话会显示在云算力面板。") from None
        if result.returncode:
            # The provider rejected the assignment before a runtime was
            # allocated. A failed best-effort `stop` below must not turn
            # this into a phantom billable session that blocks retrying.
            job["allocation_rejected"] = True
            raise RuntimeError(friendly_cli_error(result.stderr.strip() or result.stdout.strip()))
        job["allocated_at"] = time.time()
        now_listed = server_sessions()
        for entry in now_listed or []:
            if entry["name"] == session:
                job["endpoint"] = entry["endpoint"]
        self._save(job)
        if job["state"] in {"stopping", "stopped"}:
            return False
        for run, path in teachers.items():
            if job.get("cancel_requested"):
                return False
            uploaded = cli(["upload", "-s", session, str(path),
                            f"/content/microduck-teacher-{run}.pt"], timeout=120)
            if uploaded.returncode:
                raise RuntimeError(f"教师模型 {run} 上传失败，请检查连接后重试。")
        # Ask the allocated runtime instead of guessing from the requested
        # tier: Colab may serve different A100 memory variants.
        self._probe_gpu(job)
        if job.get("cancel_requested"):
            return False
        job["state"] = "starting"
        self._save(job)
        runner = _remote_runner(job["task"], iterations, envs,
                                int(job.get("max_minutes", DEFAULT_MAX_MINUTES)) * 60 + REMOTE_DEADLINE_SLACK_S)
        # The kernel returns immediately; training runs in a separate process.
        bootstrap = ("import pathlib,subprocess,sys\n"
                     f"p=pathlib.Path('/content/microduck-cloud-runner-{job['id']}.py')\n"
                     f"p.write_text({runner!r})\n"
                     "subprocess.Popen([sys.executable,str(p)],"
                     "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                     "print('MICRODUCK_BOOTSTRAP_OK')\n")
        result = cli(["exec", "-s", session], code=bootstrap, timeout=60)
        if result.returncode or "MICRODUCK_BOOTSTRAP_OK" not in result.stdout:
            raise RuntimeError("无法在 Colab 启动训练进程：" + friendly_cli_error(result.stderr.strip() or result.stdout.strip()))
        if job.get("cancel_requested"):
            return False
        job["state"] = "running"
        self._save(job)
        return True

    def _run(self, job: dict, directory: Path, iterations: int, envs: int) -> None:
        try:
            if self._launch(job, directory, iterations, envs):
                self._monitor(job, directory)
        except Exception as exc:
            if not job.get("cancel_requested"):
                job["state"] = "failed"
                job["message"] = str(exc)
        finally:
            self._finish(job)

    # --- monitoring ----------------------------------------------------------
    def _poll(self, job: dict) -> tuple[dict | None, bool]:
        """Returns (remote status, session gone). Absent is not the same as gone."""
        try:
            result = cli(["exec", "-s", job["session"], "--timeout", "50"],
                         code=_poll_code(), timeout=POLL_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError):
            return None, False  # A lost progress request is not a failed remote training run.
        output = f"{result.stdout}\n{result.stderr}"
        if "appears to be lost" in output or f"Session '{job['session']}' not found" in output:
            return None, True
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and candidate.get("state") in REMOTE_STATES:
                    return candidate, False
        return None, False

    def _sample_usage(self, job: dict) -> None:
        """What the account is spending right now, from Google's own numbers."""
        try:
            result = cli(["usage"], timeout=15)
        except (OSError, subprocess.TimeoutExpired, RuntimeError):
            return
        if result.returncode:
            return
        balance = _number(r"Current balance:\s*([\d.,]+)", result.stdout)
        rate = _number(r"Usage rate:\s*([\d.,]+)", result.stdout)
        if rate is not None:
            job["rate"] = rate
        if balance is not None:
            job["balance"] = balance
            start = job.get("balance_start")
            if isinstance(start, (int, float)):
                job["ccu_spent"] = round(max(0.0, start - balance), 3)

    def _limit_reason(self, job: dict, remote: dict, now: float) -> tuple[str, str] | None:
        minutes = job.get("max_minutes") or DEFAULT_MAX_MINUTES
        allocated = job.get("allocated_at") or now
        if now - allocated >= minutes * 60:
            return "budget", f"已达到 {minutes} 分钟时长上限，已自动停止并释放算力。"
        max_ccu = job.get("max_ccu")
        if max_ccu and (job.get("ccu_spent") or 0) >= max_ccu:
            return "ccu", f"已达到 {max_ccu} 计算单元上限，已自动停止并释放算力。"
        age = remote.get("log_age")
        if remote["state"] == "training" and isinstance(age, (int, float)) and age >= STALL_TIMEOUT_S:
            return "stall", f"训练日志已 {int(age // 60)} 分钟没有更新，判定为卡死，已停止并释放算力。"
        return None

    def _send_rescue(self, job: dict) -> bool:
        reason = job.get("stop_reason", "user")
        code = ("import pathlib\n"
                f"pathlib.Path({REMOTE_ROOT!r} + '/rescue.request').write_text({reason!r})\n"
                "print('MICRODUCK_RESCUE_SENT')\n")
        try:
            result = cli(["exec", "-s", job["session"]], code=code, timeout=POLL_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return result.returncode == 0 and "MICRODUCK_RESCUE_SENT" in result.stdout

    def _monitor(self, job: dict, directory: Path) -> None:
        job["started_monitor_at"] = time.time()
        last_usage = 0.0
        last_save = 0.0
        missing_since = None
        setup_since = None
        failing_since = None
        rescue_at = None
        poll_failures = 0
        remote = None
        while True:
            time.sleep(POLL_INTERVAL_S)
            if job.get("cancel_requested"):
                return
            polled, gone = self._poll(job)
            if job.get("cancel_requested"):
                return
            now = time.time()
            if gone:
                raise RuntimeError("Colab 会话已结束（可能被 Google 回收），本次未取回训练结果。")
            if polled is None:
                poll_failures += 1
                failing_since = failing_since or now
                job["poll_failures"] = poll_failures
                job["message"] = f"云端进度暂时无法获取（第 {poll_failures} 次），正在重试；训练可能仍在运行，算力仍在计费。"
                if now - failing_since >= POLL_GRACE_S:
                    raise RuntimeError(f"{POLL_GRACE_S // 60} 分钟无法联系云端，已尝试释放算力；无法确认远端训练结果。")
                continue
            remote = polled
            poll_failures = 0
            failing_since = None
            job["poll_failures"] = 0
            job["last_heartbeat_at"] = now
            job["remote_state"] = remote["state"]
            if not job.get("rescue_requested") or remote.get("message"):
                job["remote_message"] = remote.get("message", "")
            job["message"] = ""
            job["progress"] = {key: remote[key] for key in
                               ("iteration", "iterations", "reward", "iter_time", "elapsed", "eta",
                                "log_age", "gpu_util", "gpu_mem_used", "gpu_mem_total")
                               if key in remote}
            if now - last_usage >= USAGE_SAMPLE_S:
                last_usage = now
                self._sample_usage(job)
            if now - last_save >= USAGE_SAMPLE_S:
                last_save = now
                self._save(job)
            state = remote["state"]
            if state in {"done", "failed", "rescued"}:
                break
            if state == "missing":
                missing_since = missing_since or now
                if now - missing_since >= BOOTSTRAP_GRACE_S:
                    raise RuntimeError("云端训练进程没有启动，已释放算力。")
                continue
            missing_since = None
            if state == "setup":
                setup_since = setup_since or now
                if now - setup_since >= SETUP_TIMEOUT_S:
                    raise RuntimeError(f"环境安装超过 {SETUP_TIMEOUT_S // 60} 分钟仍未完成，已释放算力。")
            limit = self._limit_reason(job, remote, now)
            if limit and not job.get("rescue_requested"):
                job["stop_reason"], job["stop_message"] = limit
                if state != "training":
                    # Nothing worth saving before training starts.
                    raise RuntimeError(limit[1])
                job["rescue_requested"] = True
            if job.get("rescue_requested"):
                if job["state"] != "saving":
                    job["state"] = "saving"
                    self._save(job)
                rescue_at = rescue_at or now
                if not job.get("rescue_sent"):
                    job["rescue_sent"] = self._send_rescue(job)
                if now - rescue_at >= RESCUE_TIMEOUT_S:
                    job["rescue_timed_out"] = True
                    break
        self._collect(job, directory, remote)

    def _download(self, job: dict, directory: Path, name: str, expected: int | None, attempts: int) -> bool:
        destination = directory / name
        for attempt in range(attempts):
            if job.get("cancel_requested"):
                return False
            try:
                result = cli(["download", "-s", job["session"], f"{REMOTE_ROOT}/{name}", str(destination)], timeout=300)
                size = destination.stat().st_size if destination.is_file() else 0
                if result.returncode == 0 and size > 0 and (expected is None or size == expected):
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass
            if attempt + 1 < attempts:
                time.sleep(DOWNLOAD_RETRY_S * (attempt + 1))
        destination.unlink(missing_ok=True)  # never leave a truncated artifact behind
        return False

    def _collect(self, job: dict, directory: Path, remote: dict | None) -> None:
        """Bring back what was paid for, then decide the job's final state."""
        job["state"] = "finalizing"
        self._save(job)
        remote_state = job.get("remote_state")
        listed = (remote or {}).get("files")
        artifacts, missing = [], []
        for name in ARTIFACTS:
            if job.get("cancel_requested"):
                return
            required = name in REQUIRED_ARTIFACTS
            if isinstance(listed, dict) and name not in listed:
                if required and remote_state == "done":
                    missing.append(name)
                continue
            expected = listed.get(name) if isinstance(listed, dict) else None
            ok = self._download(job, directory, name, expected, DOWNLOAD_ATTEMPTS if required else 1)
            if ok:
                artifacts.append(name)
            elif required:
                missing.append(name)
        if job.get("cancel_requested"):
            return
        job["artifacts"] = artifacts
        policy = directory / "policy.onnx"
        contract = _policy_contract_error(policy) if "policy.onnx" in artifacts else None
        if contract:
            job["artifacts"] = [name for name in artifacts if name != "policy.onnx"]
        reason = job.get("stop_reason")
        saved = "、".join(name for name in job["artifacts"] if name in REQUIRED_ARTIFACTS)
        if reason:
            note = job.get("stop_message") or "已保存进度并释放算力。"
            if job.get("rescue_timed_out"):
                note += " 保存超时，未能确认云端已保存。"
            note += f" 已取回：{saved}。" if saved else " 没有可取回的检查点。"
            job["state"] = "failed" if reason == "stall" else "stopped"
            job["message"] = note
        elif remote_state == "done" and "policy.onnx" in job["artifacts"]:
            job["state"] = "done"
            job["message"] = ""
            if missing:
                job["message"] = f"ONNX 已取回，但 {'、'.join(missing)} 下载失败。"
        else:
            job["state"] = "failed"
            job["message"] = (contract
                              or (f"训练已完成，但 {'、'.join(missing)} 多次下载失败。" if remote_state == "done" and missing else "")
                              or job.get("remote_message")
                              or "Training or ONNX export did not complete")

    # --- releasing -----------------------------------------------------------
    def _finish(self, job: dict) -> None:
        for delay in (0, *RELEASE_RETRY_DELAYS_S):
            if delay:
                time.sleep(delay)
            self._release(job)
            if job.get("released") is True:
                break
        if job.get("cancel_requested"):
            job["state"] = "stopped"
        self._save(job)
        self._sync_awake()
        self._reconcile_soon.set()
        invalidate_account_cache()  # the panel's "active runtimes" should move now

    def _confirm_gone(self, job: dict) -> bool | None:
        """Ask Google whether this job's runtime is gone. None: could not ask."""
        for attempt in range(RELEASE_CONFIRM_ATTEMPTS):
            entries = server_sessions()
            if entries is None:
                return None
            endpoint = job.get("endpoint")
            still = [e for e in entries
                     if e["name"] == job["session"] or (endpoint and e["endpoint"] == endpoint)]
            baseline = job.get("baseline_endpoints")
            if not still and baseline is not None and not endpoint:
                # We never learned this job's endpoint, so a nameless newcomer
                # could be the runtime our `new` lost track of.
                still = [e for e in entries if e["name"] == "?" and e["endpoint"] not in baseline]
            if not still:
                return True
            if attempt + 1 < RELEASE_CONFIRM_ATTEMPTS:
                time.sleep(RELEASE_CONFIRM_DELAY_S)
        return False

    def _release(self, job: dict) -> None:
        # HTTP cancellation and worker cleanup can arrive together. Check the
        # success flag inside the lock so a second caller cannot undo success.
        with self.release_lock:
            if job.get("released") is True:
                return
            reason = ""
            requested = False
            try:
                result = cli(["stop", "-s", job["session"]], timeout=60)
                requested = result.returncode == 0
                if not requested:
                    reason = result.stderr.strip() or result.stdout.strip()
                    if "/unassign/" in reason and "Not Found" in reason:
                        status = cli(["status", "-s", job["session"]], timeout=30)
                        missing = f"[colab] Session '{job['session']}' not found."
                        requested = status.returncode == 0 and missing in status.stdout.splitlines()
            except Exception as exc:
                reason = str(exc)
            # `stop` exits 0 for a session it merely has no local record of, so
            # the exit code alone cannot prove the runtime stopped billing.
            confirmed = self._confirm_gone(job) if requested else None
            rejected = job.get("allocation_rejected") is True
            released = confirmed is True or (rejected and confirmed is not False)
            job["released"] = released
            job["release_attempts"] = job.get("release_attempts", 0) + 1
            if released:
                job.setdefault("released_at", time.time())
                job.pop("next_release_at", None)
                # A cancelled job should not keep a stale release traceback.
                if job.get("cancel_requested"):
                    job["message"] = ""
            elif confirmed is False:
                job["message"] = "已请求释放，但 Google 仍显示这个会话，算力可能仍在计费，将自动重试。"
            elif requested:
                job["message"] = "已请求释放，但暂时无法向 Google 确认，将自动重试。"
            else:
                detail = friendly_cli_error(reason) if reason else ""
                job["message"] = (detail + " " if detail else "") + "算力释放未成功，算力可能仍在计费，将自动重试。"

    def stop(self, job_id: str, rescue: bool = False) -> dict:
        """Disconnect. rescue=True saves the latest progress first (bounded)."""
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("released") is True:
            return dict(job)
        worker = self.workers.get(job_id)
        worker_alive = worker is not None and worker.is_alive()
        if (rescue and worker_alive and job["state"] in {"running", "saving"}
                and job.get("remote_state") in {"training", "saving"}):
            job.setdefault("stop_reason", "user")
            job.setdefault("stop_message", "已按要求保存进度并释放算力。")
            job["rescue_requested"] = True
            job["state"] = "saving"
            self._save(job)
            return dict(job)
        job["cancel_requested"] = True
        if job["state"] in {"allocating", "stopping"}:
            # The allocation thread must finish `new` before it can release
            # the session. Keep the pending state visible to the UI.
            job["state"] = "stopping"
            self._save(job)
            return dict(job)
        job["state"] = "stopped" if job["state"] not in {"done", "failed"} else job["state"]
        self._release(job)
        self._save(job)
        self._sync_awake()
        invalidate_account_cache()
        return dict(job)

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(job) for job in self.jobs.values()]

    def has_unreleased(self) -> bool:
        return any(job.get("released") is not True for job in self.jobs.values())

    def stop_all(self) -> list[dict]:
        """Release every session started by this lab, leaving other Colab work alone."""
        return [self.stop(job_id) for job_id, job in list(self.jobs.items())
                if job["state"] in ACTIVE_STATES
                or job.get("released") is False]

    def shutdown(self) -> list[dict]:
        """Let in-flight allocation finish and release before daemon exit."""
        self._stop_reconciler.set()
        self.stop_all()
        deadline = time.monotonic() + 250
        for worker in list(self.workers.values()):
            worker.join(timeout=max(0, deadline - time.monotonic()))
        self._sync_awake(force_off=True)
        return self.list()

    # --- keeping the Mac awake -----------------------------------------------
    def _sync_awake(self, force_off: bool = False) -> None:
        """A sleeping Mac cannot notice training finished, while the runtime bills."""
        if not self.keep_awake:
            return
        need = not force_off and self.has_unreleased()
        running = self._awake is not None and self._awake.poll() is None
        if need and not running and sys.platform == "darwin" and shutil.which("caffeinate"):
            # -w: never outlives this process, however the lab dies.
            self._awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif not need and running:
            self._awake.terminate()
            self._awake = None

    # --- reconciling with Google ---------------------------------------------
    def start_reconciler(self, keep_awake: bool = True) -> None:
        self.keep_awake = keep_awake
        if self._reconciler is None:
            self._reconciler = threading.Thread(target=self._reconcile_loop, daemon=True)
            self._reconciler.start()

    def note_account(self, account: dict) -> None:
        """Google's count disagrees with the last list: look now, not in 5 min.

        Otherwise someone who just signed in sees "1 active runtime" with no way
        to tell whose it is, while it may be billing.
        """
        if account.get("connected") and account.get("active_assignments") != self.server.get("count"):
            self._reconcile_soon.set()

    def _reconcile_loop(self) -> None:
        next_run = 0.0
        while not self._stop_reconciler.wait(RECONCILE_TICK_S):
            if time.time() < next_run and not self._reconcile_soon.is_set():
                continue
            self._reconcile_soon.clear()
            try:
                self.reconcile()
            except Exception as exc:  # A bug here must not stop later releases.
                print(f"[colab] reconcile failed: {exc}", flush=True)
            next_run = time.time() + (RECONCILE_ACTIVE_S if self.has_unreleased() else RECONCILE_IDLE_S)

    def _worker_alive(self, job_id: str) -> bool:
        worker = self.workers.get(job_id)
        return worker is not None and worker.is_alive()

    def reconcile(self) -> None:
        """Compare what we believe with what Google is billing, and act on it."""
        self._sync_awake()
        unreleased = [job for job in list(self.jobs.values()) if job.get("released") is not True]
        # Nothing to protect and a dead token: asking again only fills the CLI's
        # log. The next successful sign-in invalidates the cached account state.
        cached = _account_cache["value"]
        if not unreleased and cached and cached.get("reason") == "auth_expired":
            self.server = {**self.server, "ok": False}
            return
        entries = server_sessions()
        self.server = self._summarize(entries)
        for job in unreleased:
            if self._worker_alive(job["id"]):
                continue
            if job["state"] == "detached":
                self._adopt(job, entries)
            elif time.time() >= job.get("next_release_at", 0):
                self._release(job)
                if job.get("released") is not True:
                    delay = min(RELEASE_BACKOFF_MAX_S, 60 * 2 ** min(job.get("release_attempts", 1), 6))
                    job["next_release_at"] = time.time() + delay
                self._save(job)
        self._sync_awake()

    def _summarize(self, entries: list[dict] | None) -> dict:
        if entries is None:
            return {"ok": False, "checked_at": time.time(), "count": None,
                    "orphans": self.server.get("orphans", []), "others": self.server.get("others", [])}
        tracked = {job["session"] for job in self.jobs.values() if job.get("released") is not True}
        known = {job["session"] for job in self.jobs.values()}
        orphans, others = [], []
        for entry in entries:
            if entry["name"] in tracked:
                continue
            public = {"endpoint": entry["endpoint"], "hardware": entry["hardware"], "name": entry["name"]}
            # An unnamed runtime, or one of ours whose release we believed in.
            if entry["name"] == "?" or entry["name"] in known or entry["name"].startswith("microduck-"):
                orphans.append(public)
            else:
                others.append(public)
        return {"ok": True, "checked_at": time.time(), "count": len(entries),
                "orphans": orphans, "others": others}

    def _adopt(self, job: dict, entries: list[dict] | None) -> None:
        """Pick a job back up after the lab restarted; the runtime never noticed."""
        if entries is None:
            job["message"] = "后端重启后暂时无法向 Google 确认这个会话，它可能仍在计费，请确认账号已连接。"
            return
        if not any(entry["name"] == job["session"] for entry in entries):
            job["cancel_requested"] = False
            self._release(job)
            if job.get("released") is True:
                job["state"] = "failed"
                job["message"] = "后端重启期间这个 Colab 会话已结束，训练结果未能取回。"
                self._save(job)
            return
        job["state"] = "saving" if job.get("rescue_requested") else "running"
        job["adopted_at"] = time.time()
        job.setdefault("allocated_at", job["adopted_at"])
        job["message"] = ""
        self._save(job)
        worker = threading.Thread(target=self._resume, args=(job,), daemon=True)
        self.workers[job["id"]] = worker
        worker.start()

    def _resume(self, job: dict) -> None:
        directory = self.runs / job["id"]
        try:
            self._monitor(job, directory)
        except Exception as exc:
            if not job.get("cancel_requested"):
                job["state"] = "failed"
                job["message"] = str(exc)
        finally:
            self._finish(job)

    def release_orphan(self, endpoint: str) -> dict:
        """Unassign a runtime Google lists but no job owns, at the person's request.

        `colab stop` needs local CLI state these runtimes do not have, and an
        unnamed entry may be the person's own Colab tab, which is why this only
        ever runs on an explicit click and only on an entry currently listed.
        """
        entries = server_sessions()
        if entries is None:
            raise RuntimeError("暂时无法向 Google 查询会话列表，请确认账号已连接后重试。")
        entry = next((e for e in entries if e["endpoint"] == endpoint), None)
        if entry is None:
            self.server = self._summarize(entries)
            return self.server
        tracked = {job["session"] for job in self.jobs.values() if job.get("released") is not True}
        if entry["name"] in tracked:
            raise ValueError("这个会话属于正在运行的任务，请使用任务的断开按钮。")
        if importlib.util.find_spec("colab_cli") is None:
            raise RuntimeError("当前环境无法直接释放这个会话，请在 Colab 网页的“管理会话”中终止。")
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; from colab_cli.common import state; state.client.unassign(sys.argv[1])", endpoint],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(friendly_cli_error(result.stderr) or "释放失败，请在 Colab 网页的“管理会话”中终止。")
        invalidate_account_cache()
        self.server = self._summarize(server_sessions())
        return self.server
