"""Hugging Face Jobs orchestration for the local lab.

Only jobs carrying this lab's labels are listed or cancelled.  The user's
token remains in the lab process and is passed to the remote container as a
secret so completed artifacts can be copied into a private Hub repository.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from .colab_jobs import TASKS, _remote_runner

RUNS = Path(__file__).resolve().parents[2] / "runs" / "huggingface"
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
LABELS = {"app": "microduck-lab"}
ACTIVE_STAGES = {"SCHEDULING", "RUNNING"}


def _api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _iso_epoch(value) -> float | None:
    return value.timestamp() if value is not None else None


def _hardware(api) -> list[dict]:
    rows = []
    for item in api.list_jobs_hardware():
        accelerator = item.accelerator
        if accelerator is None:
            continue
        rows.append({
            "id": item.name,
            "label": item.pretty_name,
            "gpu": accelerator.model,
            "quantity": accelerator.quantity,
            "vram": accelerator.vram,
            "cost": item.unit_cost_usd,
            "unit": item.unit_label,
        })
    return rows


def account_status(token_data: dict | None) -> dict:
    if not token_data:
        return {"connected": False, "hardware": []}
    try:
        api = _api(token_data["token"])
        return {
            "connected": True,
            "username": token_data.get("username", ""),
            "hardware": _hardware(api),
        }
    except Exception as exc:
        return {"connected": False, "hardware": [], "message": str(exc)}


def _hf_runner(task: str, iterations: int, envs: int, repo_id: str,
               local_id: str) -> str:
    base = _remote_runner(task, iterations, envs)
    return base + f'''
# Persist the useful artifacts after either success or failure.  HF_TOKEN is a
# Jobs secret and is never embedded in the visible command.
subprocess.run(["python", "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    folder_path=str(root), repo_id={repo_id!r}, repo_type="model",
    path_in_repo={f"jobs/{local_id}"!r})
'''


class HfJobs:
    def __init__(self, runs: Path = RUNS):
        self.runs = runs
        self.jobs: dict[str, dict] = {}
        for path in runs.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("id") and job.get("remote_id"):
                    self.jobs[job["id"]] = job
            except (OSError, ValueError, TypeError):
                continue

    def _save(self, job: dict) -> None:
        directory = self.runs / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "job.json").write_text(json.dumps(job, indent=2))

    def start(self, token_data: dict | None, task: str, flavor: str,
              iterations: int, envs: int) -> dict:
        if not token_data:
            raise RuntimeError("Connect a Hugging Face account first")
        if task not in TASKS:
            raise ValueError("Unsupported task")
        if not 1 <= iterations <= 100_000 or not 1 <= envs <= 4096:
            raise ValueError("Training limits are out of range")
        api = _api(token_data["token"])
        hardware = {row["id"]: row for row in _hardware(api)}
        if flavor not in hardware:
            raise ValueError("Unsupported Hugging Face hardware")

        local_id = uuid.uuid4().hex[:12]
        username = token_data.get("username", "")
        repo_id = f"{username}/microduck-cloud-results"
        api.create_repo(repo_id=repo_id, repo_type="model", private=True,
                        exist_ok=True)
        info = api.run_job(
            image=IMAGE,
            command=["python", "-c",
                     _hf_runner(task, iterations, envs, repo_id, local_id)],
            secrets={"HF_TOKEN": token_data["token"]}, flavor=flavor,
            timeout="24h", name=f"microduck-{local_id}",
            labels={**LABELS, "microduck-id": local_id},
        )
        spec = hardware[flavor]
        job = {
            "id": local_id, "remote_id": info.id, "platform": "huggingface",
            "task": task, "gpu": spec["gpu"], "vram": spec["vram"],
            "flavor": flavor, "state": "allocating", "started": time.time(),
            "allocated_at": None, "released": False, "message": "",
            "url": info.url, "repo_id": repo_id,
        }
        self.jobs[local_id] = job
        self._save(job)
        return dict(job)

    def _download_result(self, api, job: dict) -> None:
        if job.get("downloaded"):
            return
        from huggingface_hub import hf_hub_download
        directory = self.runs / job["id"]
        for name in ("policy.onnx", "checkpoint.pt", "train.log", "error.log",
                     "status.json"):
            try:
                source = hf_hub_download(
                    repo_id=job["repo_id"], repo_type="model",
                    filename=f"jobs/{job['id']}/{name}", token=api.token)
                shutil.copy2(source, directory / name)
            except Exception:
                continue
        status_path = directory / "status.json"
        remote = json.loads(status_path.read_text()) if status_path.is_file() else {}
        job["remote_state"] = remote.get("state")
        job["message"] = remote.get("message", "")
        job["state"] = "done" if (directory / "policy.onnx").is_file() else "failed"
        if job["state"] == "failed" and not job["message"]:
            job["message"] = "Training or ONNX export did not complete"
        job["downloaded"] = True

    def refresh(self, token_data: dict | None) -> list[dict]:
        if not token_data:
            return [dict(job) for job in self.jobs.values()]
        api = _api(token_data["token"])
        for job in self.jobs.values():
            if job.get("released") is True and job["state"] in {"done", "failed", "stopped"}:
                continue
            try:
                info = api.inspect_job(job_id=job["remote_id"])
                stage_value = info.status.stage
                stage = getattr(stage_value, "value", stage_value)
                stage = str(stage)
                job["remote_state"] = stage.lower()
                if info.started_at:
                    job["allocated_at"] = _iso_epoch(info.started_at)
                if stage == "SCHEDULING":
                    job["state"] = "allocating"
                elif stage == "RUNNING":
                    job["state"] = "running"
                elif stage == "COMPLETED":
                    job["released"] = True
                    job["state"] = "finalizing"
                    self._download_result(api, job)
                elif stage in {"CANCELED", "DELETED"}:
                    job["state"] = "stopped"
                    job["released"] = True
                elif stage == "ERROR":
                    job["state"] = "failed"
                    job["released"] = True
                    job["message"] = info.status.message or "Hugging Face job failed"
                self._save(job)
            except Exception as exc:
                job["message"] = str(exc)
                self._save(job)
        return [dict(job) for job in self.jobs.values()]

    def stop(self, token_data: dict | None, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if not token_data:
            raise RuntimeError("Reconnect Hugging Face before stopping this job")
        if job.get("released") is True:
            return dict(job)
        api = _api(token_data["token"])
        api.cancel_job(job_id=job["remote_id"])
        job["state"] = "stopped"
        job["released"] = True
        self._save(job)
        return dict(job)

    def stop_all(self, token_data: dict | None) -> list[dict]:
        return [self.stop(token_data, job_id) for job_id, job in list(self.jobs.items())
                if job.get("released") is not True]
