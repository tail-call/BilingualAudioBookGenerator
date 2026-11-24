import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select, update

from bookaudio.database import AsyncSessionLocal
from bookaudio.models import Book, Job, JobStatus, JobType, Sentence


# XXX: currently it's a repository of all kinds of things
class JobsRepository:
    def __init__(self, session_factory=AsyncSessionLocal):
        self.session_factory = session_factory

    async def recover_interrupted_jobs(self) -> int:
        """Reset any in-flight 'processing' jobs back to 'pending'."""
        async with self.session_factory() as session, session.begin():
            stmt = (
                update(Job)
                .where(Job.status == JobStatus.PROCESSING)
                .values(status=JobStatus.PENDING)
            )
            res = await session.execute(stmt)
            return res.rowcount or 0

    async def create_job(
        self,
        book_id: str,
        pdf_filename: str,
        pdf_path: Path,
        source_language: str,
        target_language: str,
    ):
        async with self.session_factory() as session, session.begin():
            book = Book(
                id=book_id,
                pdf_filename=pdf_filename,
                pdf_path=str(pdf_path),
                source_language=source_language,
                target_language=target_language,
                status=JobStatus.PENDING,
            )
            session.add(book)

    async def claim_next_job(self, job_type: JobType) -> str | None:
        """Claim the next pending job of the given type using SKIP LOCKED."""
        async with self.session_factory() as session, session.begin():
            stmt = (
                select(Job.id)
                .where(Job.job_type == job_type, Job.status == JobStatus.PENDING)
                .order_by(Job.sequence_num.asc().nulls_first(), Job.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            job_id = await session.scalar(stmt)
            if not job_id:
                return None

            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status=JobStatus.PROCESSING,
                    updated_at=datetime.now(UTC),
                )
            )
            return job_id

    async def get_job(self, job_id: str) -> Job | None:
        async with self.session_factory() as session:
            return await session.get(Job, job_id)

    async def get_job_with_book(self, job_id: str) -> tuple[Job, Book] | None:
        async with self.session_factory() as session:
            job = await session.get(Job, job_id)
            if not job:
                return None
            book = await session.get(Book, job.book_id)
            if not book:
                return None
            return job, book

    async def get_sentence_count(self, book_id: str) -> int:
        async with self.session_factory() as session:
            count = await session.scalar(
                select(func.count(Sentence.id)).where(Sentence.book_id == book_id)
            )
            return count or 0

    async def save_collected_sentences_batch(
        self,
        job_id: str,
        book_id: str,
        sentence_texts: list[str],
        start_seq: int,
        next_page: int,
    ) -> int:
        async with self.session_factory() as session, session.begin():
            seq_counter = start_seq
            for text in sentence_texts:
                session.add(
                    Sentence(
                        book_id=book_id,
                        sequence_num=seq_counter,
                        text=text,
                    )
                )
                seq_counter += 1

            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(payload={"current_page": next_page})
            )
            return seq_counter

    async def get_book_sentences(self, book_id: str) -> list[Sentence]:
        async with self.session_factory() as session:
            return list(
                (
                    await session.scalars(
                        select(Sentence)
                        .where(Sentence.book_id == book_id)
                        .order_by(Sentence.sequence_num)
                    )
                ).all()
            )

    async def complete_collector_job(
        self,
        job_id: str,
        book_id: str,
        preprocessor_specs: list[dict],
        total_sentences: int,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            for spec in preprocessor_specs:
                session.add(
                    Job(
                        id=uuid.uuid4().hex,
                        book_id=book_id,
                        job_type=JobType.PREPROCESSOR,
                        status=JobStatus.PENDING,
                        sequence_num=spec["sequence_num"],
                        sentence_id=spec["sentence_id"],
                        payload=spec["payload"],
                    )
                )

            await session.execute(
                update(Book)
                .where(Book.id == book_id)
                .values(total_sentences=total_sentences)
            )
            await session.execute(
                update(Job).where(Job.id == job_id).values(status=JobStatus.DONE)
            )

    async def complete_preprocessor_job(
        self,
        job_id: str,
        book_id: str,
        sequence_num: int | None,
        sentence_id: int | None,
        translation_data: dict,
    ) -> None:
        async with self.session_factory() as session, session.begin():
            session.add(
                Job(
                    id=uuid.uuid4().hex,
                    book_id=book_id,
                    job_type=JobType.VOICER,
                    status=JobStatus.PENDING,
                    sequence_num=sequence_num,
                    sentence_id=sentence_id,
                    payload={"translation_data": translation_data},
                )
            )
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status=JobStatus.DONE, result=translation_data)
            )

    async def complete_voicer_job(
        self, job_id: str, book_id: str, audio_file: str
    ) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status=JobStatus.DONE,
                    result={"audio_file": audio_file},
                )
            )

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

    async def try_spawn_glue_job(self, book_id: str) -> bool:
        async with self.session_factory() as session, session.begin():
            book = await session.scalar(
                select(Book).where(Book.id == book_id).with_for_update()
            )
            if not book or book.total_sentences == 0:
                return False

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
                session.add(
                    Job(
                        id=uuid.uuid4().hex,
                        book_id=book_id,
                        job_type=JobType.GLUE,
                        status=JobStatus.PENDING,
                    )
                )
                return True
            return False

    async def get_glue_job_data(
        self, job_id: str
    ) -> tuple[Job, Book, list[Job]] | None:
        async with self.session_factory() as session:
            job = await session.get(Job, job_id)
            if not job:
                return None
            book = await session.get(Book, job.book_id)
            if not book:
                return None

            voicer_jobs = list(
                (
                    await session.scalars(
                        select(Job)
                        .where(Job.book_id == book.id, Job.job_type == JobType.VOICER)
                        .order_by(Job.sequence_num)
                    )
                ).all()
            )
            return job, book, voicer_jobs

    async def complete_glue_job(
        self, job_id: str, book_id: str, output_filename: str
    ) -> None:
        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(Job).where(Job.id == job_id).values(status=JobStatus.DONE)
            )
            await session.execute(
                update(Book)
                .where(Book.id == book_id)
                .values(
                    status=JobStatus.DONE,
                    output_file=output_filename,
                )
            )

    async def fail_job(self, job_id: str, error_msg: str, max_retries: int = 3) -> None:
        async with self.session_factory() as session, session.begin():
            job = await session.get(Job, job_id)
            if not job:
                return

            job.retry_count += 1
            if job.retry_count < max_retries:
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
