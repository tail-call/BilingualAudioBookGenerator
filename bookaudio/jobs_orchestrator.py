import asyncio
import logging
import traceback
from pathlib import Path

from pypdf import PdfReader

from bookaudio.jobs_repository import JobsRepository
from bookaudio.models import JobType
from bookaudio.nlp import NLPService
from bookaudio.tts import (
    concat_files_with_ffmpeg,
    render_sentence_audio,
)
from bookaudio.util import batch_indices

logger = logging.getLogger("orchestrator")


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

    # -------------------------------------------------------------------------
    # WORKER 1: SENTENCE COLLECTOR
    # -------------------------------------------------------------------------

    async def _collector_dispatcher(self):
        while self.running:
            job_id = await self.jobs_repository.claim_next_job(JobType.COLLECTOR)
            if not job_id:
                await asyncio.sleep(1.0)
                continue
            try:
                await self._process_collector_job(job_id)
            except Exception as exc:
                logger.error(
                    f"Collector job {job_id} failed: {exc}\n{traceback.format_exc()}"
                )
                await self.jobs_repository.fail_job(job_id, str(exc))

    async def _process_collector_job(self, job_id: str):
        data = await self.jobs_repository.get_job_with_book(job_id)
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
                job_id=job_id,
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
            job_id=job_id,
            book_id=book.id,
            preprocessor_specs=preprocessor_specs,
            total_sentences=total_sentences,
        )
        logger.info(
            f"Book {book.id}: Collector finished. Spawned {total_sentences} preprocessor jobs."
        )

    # -------------------------------------------------------------------------
    # WORKER 2: SENTENCE PREPROCESSOR
    # -------------------------------------------------------------------------

    async def _preprocessor_dispatcher(self):
        while self.running:
            await self.preprocessor_sem.acquire()
            job_id = await self.jobs_repository.claim_next_job(JobType.PREPROCESSOR)
            if not job_id:
                self.preprocessor_sem.release()
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
            self.preprocessor_sem.release()

    async def _process_preprocessor_job(self, job_id: str):
        job = await self.jobs_repository.get_job(job_id)
        if not job:
            return
        payload = job.payload

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

    # -------------------------------------------------------------------------
    # WORKER 3: SENTENCE VOICER
    # -------------------------------------------------------------------------

    async def _voicer_dispatcher(self):
        while self.running:
            await self.voicer_sem.acquire()
            job_id = await self.jobs_repository.claim_next_job(JobType.VOICER)
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
            await self.jobs_repository.fail_job(job_id, str(exc))
        finally:
            self.voicer_sem.release()

    async def _process_voicer_job(self, job_id: str):
        job = await self.jobs_repository.get_job(job_id)
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

    # -------------------------------------------------------------------------
    # WORKER 4: AUDIO GLUE
    # -------------------------------------------------------------------------

    async def _glue_dispatcher(self):
        while self.running:
            job_id = await self.jobs_repository.claim_next_job(JobType.GLUE)
            if not job_id:
                await asyncio.sleep(1.0)
                continue
            try:
                await self._process_glue_job(job_id)
            except Exception as exc:
                logger.error(f"Glue job {job_id} failed: {exc}\n{traceback.format_exc()}")
                await self.jobs_repository.fail_job(job_id, str(exc))

    async def _process_glue_job(self, job_id: str):
        data = await self.jobs_repository.get_glue_job_data(job_id)
        if not data:
            return
        _, book, voicer_jobs = data

        part_files = [Path(j.result["audio_file"]) for j in voicer_jobs]
        final_mp3 = self.output_dir / f"audiobook_{book.id}.mp3"

        logger.info(f"Gluing {len(part_files)} audio samples into {final_mp3.name}...")
        await concat_files_with_ffmpeg(final_mp3, part_files)

        await self.jobs_repository.complete_glue_job(
            job_id=job_id,
            book_id=book.id,
            output_filename=final_mp3.name,
        )
        logger.info(f"Book {book.id} successfully completed: {final_mp3.name}")
