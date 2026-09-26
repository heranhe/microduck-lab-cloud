from dataclasses import dataclass
from types import SimpleNamespace

from huggingface_hub._jobs_api import JobStage

from microduck_local import hf_jobs


@dataclass
class Accelerator:
    model: str = "NVIDIA L4"
    quantity: str = "1"
    vram: str = "24 GB"


class FakeApi:
    token = "hidden"
    stage = JobStage.RUNNING

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
            status=SimpleNamespace(stage=self.stage, message=None),
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
    assert not stopped["released"] and stopped["state"] == "stopping"
    assert api.cancelled == "remote-1"
    assert jobs.refresh({"token": "hf_secret"})[0]["state"] == "stopping"
    api.stage = JobStage.CANCELED
    confirmed = jobs.refresh({"token": "hf_secret"})[0]
    assert confirmed["released"] and confirmed["state"] == "stopped"
    assert confirmed["released_at"] >= confirmed["started"]


def test_refresh_understands_hub_job_stage_enum(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path)
    job = jobs.start({"token": "hf_secret", "username": "me"},
                     "Mjlab-Velocity-Flat-MicroDuck", "l4x1", 10, 8)
    refreshed = jobs.refresh({"token": "hf_secret"})[0]
    assert refreshed["id"] == job["id"]
    assert refreshed["state"] == "running"


def test_stop_all_attempts_remaining_jobs_after_provider_error(tmp_path, monkeypatch):
    api = FakeApi()
    attempted = []
    def cancel_job(*, job_id):
        attempted.append(job_id)
        if job_id == "remote-1":
            raise RuntimeError("temporary outage")
    api.cancel_job = cancel_job
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path)
    jobs.jobs = {
        "first": {"id": "first", "remote_id": "remote-1", "state": "running", "released": False},
        "second": {"id": "second", "remote_id": "remote-2", "state": "running", "released": False},
    }
    results = jobs.stop_all({"token": "hidden"})
    assert attempted == ["remote-1", "remote-2"]
    assert "temporary outage" in results[0]["message"]
    assert results[1]["state"] == "stopping"
    assert not any(job["released"] for job in results)


def test_completed_job_retries_artifacts_after_download_outage(tmp_path, monkeypatch):
    import huggingface_hub

    api = FakeApi()
    api.stage = JobStage.COMPLETED
    monkeypatch.setattr(hf_jobs, "_api", lambda token: api)
    jobs = hf_jobs.HfJobs(tmp_path / "jobs")
    jobs.start({"token": "hidden", "username": "me"},
               "Mjlab-VelStand-Flat-MicroDuck", "l4x1", 10, 8)
    def unavailable(**kwargs):
        raise OSError("temporary download failure")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    pending = jobs.refresh({"token": "hidden"})[0]
    assert pending["released"] and pending["state"] == "finalizing"
    assert not pending["downloaded"]
    released_at = pending["released_at"]
    status = tmp_path / "status.json"
    status.write_text('{"state":"done"}')
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"example model")
    def download(**kwargs):
        return str(status if kwargs["filename"].endswith("status.json") else policy)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    done = jobs.refresh({"token": "hidden"})[0]
    assert done["state"] == "done" and done["downloaded"]
    assert done["released_at"] == released_at
