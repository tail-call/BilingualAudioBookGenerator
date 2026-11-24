import asyncio
import logging
import traceback

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import Job, JobType

from .protocols import Runnable

logger = logging.getLogger(__name__)


class BaseWorker:
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        job_type: JobType,
        idle_delay: float,
        max_concurrency: int,
    ):
        self.parent = parent
        self.jobs_repository = jobs_repository
        self.job_type = job_type
        self.idle_delay = idle_delay
        if max_concurrency > 0:
            self._semaphore = asyncio.Semaphore(max_concurrency)
        else:
            self._semaphore = None

    async def dispatch(self):
        while self.parent.running:
            await self.dispatch_one()

    async def dispatch_one(self):
        if self._semaphore:
            await self._semaphore.acquire()

        job_id = await self.jobs_repository.claim_next_job(self.job_type)

        if not job_id:
            if self._semaphore:
                self._semaphore.release()

            await asyncio.sleep(self.idle_delay)
            return

        asyncio.create_task(self._try_run_job(job_id))

    async def _try_run_job(self, job_id: str):
        try:
            await self._find_and_run_job(job_id)
        except Exception as exc:
            logger.error(
                f"Worker {self}: job {job_id} failed: {exc}\n{traceback.format_exc()}"
            )
            await self.jobs_repository.fail_job(job_id, str(exc))
        finally:
            if self._semaphore:
                self._semaphore.release()

    async def _find_and_run_job(self, job_id: str):
        job = await self.jobs_repository.get_job(job_id)
        if not job:
            logger.warning(f"Worker {self}: no job with {job_id} found, skipping")
            return
        return await self.run(job)

    async def run(self, job: Job):
        raise NotImplementedError()
