import logging

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import Job, JobType
from bookaudio.nlp import NLPService
from bookaudio.workers.base import BaseWorker
from bookaudio.workers.protocols import Runnable

logger = logging.getLogger(__name__)


class SentenceAnnotationWorker(BaseWorker):
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        nlp_service: NLPService,
        max_concurrency: int = 2,
    ):
        super().__init__(
            parent,
            jobs_repository,
            JobType.PREPROCESSOR,
            idle_delay=0.5,
            max_concurrency=max_concurrency,
        )
        self.nlp_service = nlp_service

    async def run(self, job: Job):
        payload = job.payload
        if not payload:
            raise RuntimeError(f"No payload in a job with id {job.id}")

        translation = await self.nlp_service.translate_sentence(
            source_language=payload["source_language"],
            target_language=payload["target_language"],
            context_text=payload["context_text"],
            sentence_text=payload["sentence_text"],
        )

        # Preprocessor finishes & spawns VOICER process
        await self.jobs_repository.complete_preprocessor_job(
            job_id=job.id,
            book_id=job.book_id,
            sequence_num=job.sequence_num,
            sentence_id=job.sentence_id,
            translation_data=translation.model_dump(),
        )
        logger.info(
            f"Preprocessor completed sentence #{job.sequence_num}. Spawned voicer job."
        )
