import logging
from pathlib import Path

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import Job, JobType
from bookaudio.tts import render_sentence_audio
from bookaudio.workers.base import BaseWorker
from bookaudio.workers.protocols import Runnable

logger = logging.getLogger(__name__)


class TTSWorker(BaseWorker):
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        output_dir: Path,
        max_concurrency: int = 2,
    ):
        super().__init__(
            parent,
            jobs_repository,
            JobType.VOICER,
            idle_delay=0.5,
            max_concurrency=max_concurrency,
        )
        self.output_dir = output_dir

    async def run(self, job: Job):
        book_id = job.book_id
        seq_num = job.sequence_num
        payload = job.payload

        if not payload:
            raise RuntimeError(f"Worker {self}: no payload in a job {job.id}")

        part_filename = self.output_dir / book_id / f"part_{seq_num:05d}.mp3"

        # Check if snippet exists on disk (resumability)
        if not (part_filename.exists() and part_filename.stat().st_size > 0):
            await render_sentence_audio(payload["translation_data"], part_filename)

        # Mark voicer completed and notify orchestrator
        await self.jobs_repository.complete_voicer_job(
            job_id=job.id,
            book_id=book_id,
            audio_file=str(part_filename),
        )

        logger.info(f"Voiced sentence #{seq_num} saved to {part_filename.name}")

        # Check if all sentences are ready for gluing
        if await self.jobs_repository.try_spawn_glue_job(book_id):
            logger.info(
                f"All voicer jobs completed for Book {book_id}. Spawned final GLUE job."
            )
