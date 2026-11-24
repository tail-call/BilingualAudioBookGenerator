import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from bookaudio.jobs_repository import JobsRepository
from bookaudio.nlp import NLPService
from bookaudio.workers.audio_concatenation import AudioConcatenationWorker
from bookaudio.workers.protocols import Worker
from bookaudio.workers.sentence_annotation import SentenceAnnotationWorker
from bookaudio.workers.sentence_collection import SentenceCollectionWorker
from bookaudio.workers.tts import TTSWorker

logger = logging.getLogger(__name__)


class JobOrchestrator:
    def __init__(
        self,
        base_dir: Path,
        nlp_service: NLPService,
        jobs_repository: JobsRepository,
        max_preprocessor_concurrency: int = 2,
        max_voicer_concurrency: int = 2,
    ):
        self.base_dir = base_dir
        self.nlp_service = nlp_service
        self.jobs_repository = jobs_repository
        self.output_dir = base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        self.running = False

        # Must use Sequence instead of list because it's covariant
        self.workers: Sequence[Worker] = [
            SentenceCollectionWorker(self, self.jobs_repository, self.nlp_service),
            SentenceAnnotationWorker(
                self,
                self.jobs_repository,
                self.nlp_service,
                max_preprocessor_concurrency,
            ),
            TTSWorker(
                self, self.jobs_repository, self.output_dir, max_voicer_concurrency
            ),
            AudioConcatenationWorker(self, self.jobs_repository, self.output_dir),
        ]

        self._worker_tasks: list[asyncio.Task] = []

    async def start(self):
        """Recover state from database and launch worker dispatchers."""
        self.running = True
        await self.recover_interrupted_jobs()

        self._worker_tasks = [
            asyncio.create_task(worker.dispatch()) for worker in self.workers
        ]

        self._worker_tasks.extend(
            [
                asyncio.create_task(self._glue_dispatcher()),
            ]
        )

        logger.info("Orchestrator started with workers.")

    async def stop(self):
        """Cancel worker tasks."""
        self.running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        logger.info("Orchestrator stopped.")

    async def recover_interrupted_jobs(self):
        """
        Allow jobs to resume from where they stopped.
        Any job that was in 'processing' state when the system was interrupted
        is safely reset to 'pending' to be picked up immediately.
        """
        recovered_count = await self.jobs_repository.recover_interrupted_jobs()
        if recovered_count > 0:
            logger.info(
                f"Resumption: Reset {recovered_count} interrupted in-flight jobs to PENDING."
            )
