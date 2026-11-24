import asyncio
import logging
import traceback
from pathlib import Path

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import JobType
from bookaudio.tts import render_sentence_audio
from bookaudio.workers.protocols import Runnable

logger = logging.getLogger(__name__)


class TTSWorker:
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        output_dir: Path,
        max_concurrency: int = 2,
    ):
        self.parent = parent
        self.jobs_repository = jobs_repository
        self.output_dir = output_dir
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def dispatch(self):
        while self.parent.running:
            await self._semaphore.acquire()
            job_id = await self.jobs_repository.claim_next_job(JobType.VOICER)
            if not job_id:
                self._semaphore.release()
                await asyncio.sleep(0.5)
                continue

            asyncio.create_task(self._wrapped_voicer(job_id))

    async def _wrapped_voicer(self, job_id: str):
        try:
            await self._process_voicer_job(job_id)
        except Exception as exc:
            logger.error(f"Voicer {job_id} failed: {exc}\n{traceback.format_exc()}")
            await self.jobs_repository.fail_job(job_id, str(exc))
        finally:
            self._semaphore.release()

    async def _process_voicer_job(self, job_id: str):
        job = await self.jobs_repository.get_job(job_id)
        if not job:
            return
        book_id = job.book_id
        seq_num = job.sequence_num
        payload = job.payload

        if not payload:
            raise RuntimeError(f"No payload in a job with id {job_id}")

        part_filename = self.output_dir / book_id / f"part_{seq_num:05d}.mp3"

        # Check if snippet exists on disk (resumability)
        if not (part_filename.exists() and part_filename.stat().st_size > 0):
            await render_sentence_audio(payload["translation_data"], part_filename)

        # Mark voicer completed and notify orchestrator
        await self.jobs_repository.complete_voicer_job(
            job_id=job_id,
            book_id=book_id,
            audio_file=str(part_filename),
        )

        logger.info(f"Voiced sentence #{seq_num} saved to {part_filename.name}")

        # Check if all sentences are ready for gluing
        if await self.jobs_repository.try_spawn_glue_job(book_id):
            logger.info(
                f"All voicer jobs completed for Book {book_id}. Spawned final GLUE job."
            )
