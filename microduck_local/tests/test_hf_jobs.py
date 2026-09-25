from dataclasses import dataclass
from types import SimpleNamespace

from microduck_local import hf_jobs
from huggingface_hub._jobs_api import JobStage


@dataclass
class Accelerator:
    model: str = "NVIDIA L4"
    quantity: str = "1"
    vram: str = "24 GB"


class FakeApi:
    token = "hidden"

    def list_jobs_hardware(self):
        return [SimpleNamespace(
            name="l4x1", pretty_name="NVIDIA L4", accelerator=Accelerator(),
            unit_cost_usd=0.8, unit_label="hour")]

    def create_repo(self, **kwargs):
        self.repo = kwargs

    def run_job(self, **kwargs):
        self.started = kwargs
        return SimpleNamespace(id="remote-1", url="https://huggingface.co/jobs/me/remote-1")

    def cancel_job(self, **kwargs):
        self.cancelled = kwargs["job_id"]

    def inspect_job(self, **kwargs):
        return SimpleNamespace(
            status=SimpleNamespace(stage=JobStage.RUNNING, message=None),
            started_at=None)


def test_start_uses_secret_token_and_records_real_hardware(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path)
    job = jobs.start({"token": "hf_secret", "username": "me"},
                     "Mjlab-Velocity-Flat-MicroDuck", "l4x1", 10, 8)
    assert job["gpu"] == "NVIDIA L4" and job["vram"] == "24 GB"
    assert api.started["secrets"] == {"HF_TOKEN": "hf_secret"}
    assert "hf_secret" not in " ".join(api.started["command"])
    assert api.started["labels"]["app"] == "microduck-lab"


def test_stop_only_cancels_the_recorded_remote_job(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path)
    job = jobs.start({"token": "hf_secret", "username": "me"},
                     "Mjlab-Velocity-Flat-MicroDuck", "l4x1", 10, 8)
    stopped = jobs.stop({"token": "hf_secret"}, job["id"])
    assert stopped["released"] and stopped["state"] == "stopped"
    assert api.cancelled == "remote-1"


def test_refresh_understands_hub_job_stage_enum(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path)
    job = jobs.start({"token": "hf_secret", "username": "me"},
                     "Mjlab-Velocity-Flat-MicroDuck", "l4x1", 10, 8)
    refreshed = jobs.refresh({"token": "hf_secret"})[0]
    assert refreshed["id"] == job["id"]
    assert refreshed["state"] == "running"
