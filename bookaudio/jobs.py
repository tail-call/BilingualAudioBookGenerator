import asyncio
import json
import logging
import traceback
import uuid
from datetime import datetime, UTC
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import func, select, update

from bookaudio.tts import (
    concat_files_with_ffmpeg,
    render_sentence_audio,
)
from bookaudio.database import AsyncSessionLocal
from bookaudio.nlp import NLPService, TranslationOutput
from bookaudio.models import Book, Job, JobStatus, JobType, Sentence
from bookaudio.util import batch_indices

logger = logging.getLogger("orchestrator")


class JobOrchestrator:
    def __init__(
        self,
        base_dir: Path,
        nlp_service: NLPService,
        max_preprocessor_concurrency: int = 2,
        max_voicer_concurrency: int = 2,
    ):
        self.base_dir = base_dir
        self.nlp_service = nlp_service
        self.output_dir = base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        self.running = False
        self.preprocessor_sem = asyncio.Semaphore(max_preprocessor_concurrency)
        self.voicer_sem = asyncio.Semaphore(max_voicer_concurrency)
        self._worker_tasks: list[asyncio.Task] = []

    async def start(self):
        """Recover state from database and launch worker dispatchers."""
        self.running = True
        await self.recover_interrupted_jobs()

        self._worker_tasks = [
            asyncio.create_task(self._collector_dispatcher()),
            asyncio.create_task(self._preprocessor_dispatcher()),
            asyncio.create_task(self._voicer_dispatcher()),
            asyncio.create_task(self._glue_dispatcher()),
        ]
        logger.info("Orchestrator started with workers.")

    async def stop(self):
        """Cancel worker tasks."""
        self.running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        logger.info("Orchestrator stopped.")

    # -------------------------------------------------------------------------
    # RESUMPTION / RECOVERY LOGIC
    # -------------------------------------------------------------------------

    async def recover_interrupted_jobs(self):
        """
        Primary Goal: allow jobs to resume from where they stopped.
        Any job that was in 'processing' state when the system was interrupted
        is safely reset to 'pending' to be picked up immediately.
        """
        async with AsyncSessionLocal() as session, session.begin():
            stmt = (
                update(Job)
                .where(Job.status == JobStatus.PROCESSING)
                .values(status=JobStatus.PENDING)
            )
            res = await session.execute(stmt)
            if res.rowcount > 0:
                logger.info(
                    f"Resumption: Reset {res.rowcount} interrupted in-flight jobs to PENDING."
                )

    async def _claim_next_job(self, job_type: JobType) -> str | None:
        """
        Claims next job using PostgreSQL's `SKIP LOCKED`.
        """
        async with AsyncSessionLocal() as session, session.begin():
            stmt = (
                select(Job.id)
                .where(Job.job_type == job_type, Job.status == JobStatus.PENDING)
                .order_by(Job.sequence_num.asc().nulls_first(), Job.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            job_id = await session.scalar(stmt)
            if job_id:
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(
                        status=JobStatus.PROCESSING,
                        updated_at=datetime.now(UTC),
                    )
                )
                return job_id
            return None

    # -------------------------------------------------------------------------
    # WORKER 1: SENTENCE COLLECTOR
    # -------------------------------------------------------------------------

    async def _collector_dispatcher(self):
        while self.running:
            job_id = await self._claim_next_job(JobType.COLLECTOR)
            if not job_id:
                await asyncio.sleep(1.0)
                continue
            try:
                await self._process_collector_job(job_id)
            except Exception as exc:
                logger.error(f"Collector job {job_id} failed: {exc}\n{traceback.format_exc()}")
                await self._fail_job(job_id, str(exc))

    async def _process_collector_job(self, job_id: str):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            book = await session.get(Book, job.book_id)
            if not book:
                return

        reader = PdfReader(book.pdf_path)
        total_pages = len(reader.pages)
        batch_size = 5

        # Resumable pagination from payload
        current_page = (job.payload or {}).get("current_page", 0)

        # Check existing collected sentences in case of resumption
        async with AsyncSessionLocal() as session:
            existing_count = await session.scalar(
                select(func.count(Sentence.id)).where(Sentence.book_id == book.id)
            )
            seq_counter = existing_count or 0

        while current_page < total_pages:
            logger.info(f"Book {book.id}: Collecting pages {current_page}..<{min(current_page + batch_size, total_pages)}")
            page_text = "".join(
                p.extract_text() or ""
                for p in reader.pages[current_page : current_page + batch_size]
            )

            sentences = await self.nlp_service.extract_sentences(page_text)

            async with AsyncSessionLocal() as session:
                for sentence in sentences:
                    model = Sentence(
                        book_id=book.id,
                        sequence_num=seq_counter,
                        text=sentence.content.strip(),
                    )
                    session.add(model)
                    seq_counter += 1

                current_page += batch_size
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(payload={"current_page": current_page})
                )
                await session.commit()

        # Collector done: Spawn a PREPROCESSOR job for each extracted sentence
        async with AsyncSessionLocal() as session:
            sentences = (
                await session.scalars(
                    select(Sentence)
                    .where(Sentence.book_id == book.id)
                    .order_by(Sentence.sequence_num)
                )
            ).all()

            total_sentences = len(sentences)
            for i, model in enumerate(sentences):
                # Calculate sliding window context
                c_indices = batch_indices(total_sentences, i, batch_size=5)
                context_str = " ".join([sentences[j].text for j in c_indices])

                prep_job = Job(
                    id=uuid.uuid4().hex,
                    book_id=book.id,
                    job_type=JobType.PREPROCESSOR,
                    status=JobStatus.PENDING,
                    sequence_num=model.sequence_num,
                    sentence_id=model.id,
                    payload={
                        "sentence_text": model.text,
                        "context_text": context_str,
                        "source_language": book.source_language,
                        "target_language": book.target_language,
                    },
                )
                session.add(prep_job)

            # Update book & collector status
            await session.execute(
                update(Book)
                .where(Book.id == book.id)
                .values(total_sentences=total_sentences)
            )
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status=JobStatus.DONE)
            )
            await session.commit()
            logger.info(f"Book {book.id}: Collector finished. Spawned {total_sentences} preprocessor jobs.")

    # -------------------------------------------------------------------------
    # WORKER 2: SENTENCE PREPROCESSOR
    # -------------------------------------------------------------------------

    async def _preprocessor_dispatcher(self):
        while self.running:
            await self.preprocessor_sem.acquire()
            job_id = await self._claim_next_job(JobType.PREPROCESSOR)
            if not job_id:
                self.preprocessor_sem.release()
                await asyncio.sleep(0.5)
                continue

            asyncio.create_task(self._wrapped_preprocessor(job_id))

    async def _wrapped_preprocessor(self, job_id: str):
        try:
            await self._process_preprocessor_job(job_id)
        except Exception as exc:
            logger.error(f"Preprocessor {job_id} failed: {exc}\n{traceback.format_exc()}")
            await self._fail_job(job_id, str(exc))
        finally:
            self.preprocessor_sem.release()

    async def _process_preprocessor_job(self, job_id: str):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            payload = job.payload

        translation = await self.nlp_service.translate_sentence(
            source_language=payload["source_language"],
            target_language=payload["target_language"],
            context_text=payload['context_text'],
            sentence_text=payload['sentence_text'],
        )

        # Preprocessor finishes & spawns VOICER process
        async with AsyncSessionLocal() as session:
            voicer_job = Job(
                id=uuid.uuid4().hex,
                book_id=job.book_id,
                job_type=JobType.VOICER,
                status=JobStatus.PENDING,
                sequence_num=job.sequence_num,
                sentence_id=job.sentence_id,
                payload={"translation_data": translation.model_dump()},
            )
            session.add(voicer_job)

            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status=JobStatus.DONE, result=translation.model_dump())
            )
            await session.commit()
            logger.info(f"Preprocessor completed sentence #{job.sequence_num}. Spawned voicer job.")

    # -------------------------------------------------------------------------
    # WORKER 3: SENTENCE VOICER
    # -------------------------------------------------------------------------

    async def _voicer_dispatcher(self):
        while self.running:
            await self.voicer_sem.acquire()
            job_id = await self._claim_next_job(JobType.VOICER)
            if not job_id:
                self.voicer_sem.release()
                await asyncio.sleep(0.5)
                continue

            asyncio.create_task(self._wrapped_voicer(job_id))

    async def _wrapped_voicer(self, job_id: str):
        try:
            await self._process_voicer_job(job_id)
        except Exception as exc:
            logger.error(f"Voicer {job_id} failed: {exc}\n{traceback.format_exc()}")
            await self._fail_job(job_id, str(exc))
        finally:
            self.voicer_sem.release()

    async def _process_voicer_job(self, job_id: str):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            book_id = job.book_id
            seq_num = job.sequence_num
            payload = job.payload

        part_filename = self.output_dir / book_id / f"part_{seq_num:05d}.mp3"

        # Check if snippet exists on disk (resumability)
        if not (part_filename.exists() and part_filename.stat().st_size > 0):
            await render_sentence_audio(payload["translation_data"], part_filename)

        # Mark voicer completed and notify orchestrator
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status=JobStatus.DONE,
                    result={"audio_file": str(part_filename)},
                )
            )

            # Update book progress
            done_voicers = await session.scalar(
                select(func.count(Job.id)).where(
                    Job.book_id == book_id,
                    Job.job_type == JobType.VOICER,
                    Job.status == JobStatus.DONE,
                )
            )
            await session.execute(
                update(Book)
                .where(Book.id == book_id)
                .values(completed_sentences=done_voicers)
            )
            await session.commit()
            logger.info(f"Voiced sentence #{seq_num} saved to {part_filename.name}")

            # Orchestrator checks if all sentences are ready for gluing
            await self._check_and_trigger_glue(session, book_id)

    async def _check_and_trigger_glue(self, session, book_id: str):
        book = await session.get(Book, book_id)
        if not book or book.total_sentences == 0:
            return

        uncompleted = await session.scalar(
            select(func.count(Job.id)).where(
                Job.book_id == book_id,
                Job.job_type == JobType.VOICER,
                Job.status != JobStatus.DONE,
            )
        )
        existing_glue = await session.scalar(
            select(func.count(Job.id)).where(
                Job.book_id == book_id,
                Job.job_type == JobType.GLUE,
            )
        )

        if uncompleted == 0 and existing_glue == 0:
            glue_job = Job(
                id=uuid.uuid4().hex,
                book_id=book_id,
                job_type=JobType.GLUE,
                status=JobStatus.PENDING,
            )
            session.add(glue_job)
            await session.commit()
            logger.info(f"All voicer jobs completed for Book {book_id}. Spawned final GLUE job.")

    # -------------------------------------------------------------------------
    # WORKER 4: AUDIO GLUE
    # -------------------------------------------------------------------------

    async def _glue_dispatcher(self):
        while self.running:
            job_id = await self._claim_next_job(JobType.GLUE)
            if not job_id:
                await asyncio.sleep(1.0)
                continue
            try:
                await self._process_glue_job(job_id)
            except Exception as exc:
                logger.error(f"Glue job {job_id} failed: {exc}\n{traceback.format_exc()}")
                await self._fail_job(job_id, str(exc))

    async def _process_glue_job(self, job_id: str):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            book = await session.get(Book, job.book_id)
            if not book:
                return

            voicer_jobs = (
                await session.scalars(
                    select(Job)
                    .where(Job.book_id == book.id, Job.job_type == JobType.VOICER)
                    .order_by(Job.sequence_num)
                )
            ).all()

        part_files = [Path(j.result["audio_file"]) for j in voicer_jobs]
        final_mp3 = self.output_dir / f"audiobook_{book.id}.mp3"

        logger.info(f"Gluing {len(part_files)} audio samples into {final_mp3.name}...")
        await concat_files_with_ffmpeg(final_mp3, part_files)

        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Job).where(Job.id == job_id).values(status=JobStatus.DONE)
            )
            await session.execute(
                update(Book)
                .where(Book.id == book.id)
                .values(
                    status=JobStatus.DONE,
                    output_file=final_mp3.name,
                )
            )
            await session.commit()
            logger.info(f"Book {book.id} successfully completed: {final_mp3.name}")

    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------

    async def _fail_job(self, job_id: str, error_msg: str):
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            job.retry_count += 1
            if job.retry_count < 3:
                # Retry transient failures
                job.status = JobStatus.PENDING
                job.error = f"Retrying: {error_msg}"
            else:
                job.status = JobStatus.FAILED
                job.error = error_msg
                await session.execute(
                    update(Book)
                    .where(Book.id == job.book_id)
                    .values(status=JobStatus.FAILED, error=error_msg)
                )
            await session.commit()
