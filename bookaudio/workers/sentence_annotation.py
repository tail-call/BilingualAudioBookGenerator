import asyncio
import logging
import traceback

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import JobType
from bookaudio.nlp import NLPService
from bookaudio.workers.protocols import Runnable

logger = logging.getLogger(__name__)


class SentenceAnnotationWorker:
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        nlp_service: NLPService,
        max_concurrency: int = 2,
    ):
        self.parent = parent
        self.jobs_repository = jobs_repository
        self.nlp_service = nlp_service
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def dispatch(self):
        while self.parent.running:
            await self._semaphore.acquire()
            job_id = await self.jobs_repository.claim_next_job(JobType.PREPROCESSOR)
            if not job_id:
                self._semaphore.release()
                await asyncio.sleep(0.5)
                continue

            asyncio.create_task(self._wrapped_preprocessor(job_id))

    async def _wrapped_preprocessor(self, job_id: str):
        try:
            await self._process_preprocessor_job(job_id)
        except Exception as exc:
            logger.error(
                f"Preprocessor {job_id} failed: {exc}\n{traceback.format_exc()}"
            )
            await self.jobs_repository.fail_job(job_id, str(exc))
        finally:
            self._semaphore.release()

    async def _process_preprocessor_job(self, job_id: str):
        job = await self.jobs_repository.get_job(job_id)
        if not job:
            return
        payload = job.payload
        if not payload:
            raise RuntimeError(f"No payload in a job with id {job_id}")

        translation = await self.nlp_service.translate_sentence(
            source_language=payload["source_language"],
            target_language=payload["target_language"],
            context_text=payload["context_text"],
            sentence_text=payload["sentence_text"],
        )

        # Preprocessor finishes & spawns VOICER process
        await self.jobs_repository.complete_preprocessor_job(
            job_id=job_id,
            book_id=job.book_id,
            sequence_num=job.sequence_num,
            sentence_id=job.sentence_id,
            translation_data=translation.model_dump(),
        )
        logger.info(
            f"Preprocessor completed sentence #{job.sequence_num}. Spawned voicer job."
        )
