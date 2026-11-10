import json
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    pdf_filename: str | None = None
    output_file: str | None = None
    error: str | None = None
    progress: int = 0  # 0-100
    created_at: str = ""
    updated_at: str = ""


def _now() -> str:
    return datetime.now().isoformat()


class JobManager:
    """
    In-memory job registry. Persisted to disk as JSON.
    """

    def __init__(self, statefile_path: Path):
        self.statefile_path = statefile_path
        self.jobs: dict[str, JobRecord] = {}
        self.jobs = {k: JobRecord(**v) for k, v in self.load_jobs_from_disk().items()}

    # Create

    def add_job(self, job_id: str, pdf_filename: str):
        self.jobs[job_id] = JobRecord(
            id=job_id,
            status=JobStatus.PENDING,
            pdf_filename=pdf_filename,
            created_at=_now(),
            updated_at=_now(),
        )
        self.save_jobs_to_disk()

    # Read

    def has_job(self, job_id: str) -> bool:
        return job_id in self.jobs

    def all_jobs(self) -> list[JobRecord]:
        return list(self.jobs.values())

    def job_with_id(self, job_id: str) -> JobRecord:
        return self.jobs[job_id]

    # Update

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            for k, v in kwargs.items():
                setattr(self.jobs[job_id], k, v)
            self.jobs[job_id].updated_at = _now()
            self.save_jobs_to_disk()

    # Delete

    def delete_job_with_id(self, job_id: str) -> JobRecord:
        deleted_job = self.jobs.pop(job_id)
        self.save_jobs_to_disk()
        return deleted_job

    # I/O

    def load_jobs_from_disk(self) -> dict:
        if self.statefile_path.exists():
            with open(self.statefile_path) as f:
                return json.load(f)
        print(
            f"JobManager: no state file found at {self.statefile_path}. "
            "Assuming no jobs."
        )
        return {}

    def save_jobs_to_disk(self):
        with open(self.statefile_path, "w") as f:
            json.dump({
                k: vars(v) for k, v in self.jobs.items()
            }, f, indent=2)
