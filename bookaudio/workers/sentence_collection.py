import logging

from pypdf import PdfReader

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import Job, JobType
from bookaudio.nlp import NLPService
from bookaudio.util import batch_indices
from bookaudio.workers.base import BaseWorker
from bookaudio.workers.protocols import Runnable

logger = logging.getLogger(__name__)


class SentenceCollectionWorker(BaseWorker):
    def __init__(
        self,
        parent: Runnable,
        jobs_repository: JobsRepository,
        nlp_service: NLPService,
    ):
        super().__init__(
            parent,
            jobs_repository,
            JobType.COLLECTOR,
            idle_delay=1.0,
            max_concurrency=0,
        )
        self.nlp_service = nlp_service

    async def run(self, job: Job):
        # XXX: should rewrite get_job_with_book() as get_book()
        data = await self.jobs_repository.get_job_with_book(job.id)
        if not data:
            return
        job, book = data

        reader = PdfReader(book.pdf_path)
        total_pages = len(reader.pages)
        batch_size = 5

        # Resumable pagination from payload
        current_page = (job.payload or {}).get("current_page", 0)

        # Check existing collected sentences in case of resumption
        seq_counter = await self.jobs_repository.get_sentence_count(book.id)

        while current_page < total_pages:
            logger.info(
                f"Book {book.id}: Collecting pages {current_page}..<{min(current_page + batch_size, total_pages)}"
            )
            page_text = "".join(
                p.extract_text() or ""
                for p in reader.pages[current_page : current_page + batch_size]
            )

            sentences = await self.nlp_service.extract_sentences(page_text)
            current_page += batch_size
            seq_counter = await self.jobs_repository.save_collected_sentences_batch(
                job_id=job.id,
                book_id=book.id,
                sentence_texts=[s.content.strip() for s in sentences],
                start_seq=seq_counter,
                next_page=current_page,
            )

        # Collector done: Spawn a PREPROCESSOR job for each extracted sentence
        sentences = await self.jobs_repository.get_book_sentences(book.id)
        total_sentences = len(sentences)
        preprocessor_specs = []

        for i, model in enumerate(sentences):
            c_indices = batch_indices(total_sentences, i, batch_size=5)
            context_str = " ".join(sentences[j].text for j in c_indices)
            preprocessor_specs.append(
                {
                    "sequence_num": model.sequence_num,
                    "sentence_id": model.id,
                    "payload": {
                        "sentence_text": model.text,
                        "context_text": context_str,
                        "source_language": book.source_language,
                        "target_language": book.target_language,
                    },
                }
            )

        await self.jobs_repository.complete_collector_job(
            job_id=job.id,
            book_id=book.id,
            preprocessor_specs=preprocessor_specs,
            total_sentences=total_sentences,
        )

        logger.info(
            f"Book {book.id}: Collector finished. Spawned {total_sentences} preprocessor jobs."
        )
