import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bookaudio.jobs_repository import JobsRepository
from bookaudio.database import get_db, init_db
from bookaudio.nlp import NLPService
from bookaudio.models import Book, Job, JobStatus, JobType
from bookaudio.jobs import JobOrchestrator

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

orchestrator = JobOrchestrator(
    BASE_DIR,
    nlp_service=NLPService(),
    jobs_repository=JobsRepository(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    await init_db()
    # Start background orchestrator and job resumption
    await orchestrator.start()
    yield
    # Graceful shutdown
    await orchestrator.stop()


app = FastAPI(title="BookAudio Resumable Engine", version="0.2.0", lifespan=lifespan)


def calculate_progress(book: Book) -> int:
    if book.status == JobStatus.DONE:
        return 100
    if book.status == JobStatus.PENDING:
        return 0
    if book.total_sentences == 0:
        return 10
    return 10 + int(85 * (book.completed_sentences / book.total_sentences))


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job(
    pdf: UploadFile = File(..., description="PDF file to convert"),
    source_language: str = Form("English"),
    target_language: str = Form("Russian"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF and register the book."""
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    book_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{book_id}_{pdf.filename}"

    content = await pdf.read()
    pdf_path.write_bytes(content)

    book = Book(
        id=book_id,
        pdf_filename=pdf.filename,
        pdf_path=str(pdf_path),
        source_language=source_language,
        target_language=target_language,
        status=JobStatus.PENDING,
    )
    db.add(book)
    await db.commit()

    return {
        "job_id": book_id,
        "status": JobStatus.PENDING,
        "pdf_filename": pdf.filename,
    }


@app.post("/jobs/{job_id}/start")
async def start_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Start or resume pipeline processing for a book."""
    book = await db.get(Book, job_id)
    if not book:
        raise HTTPException(404, "Book not found.")

    # Check existing collector job
    collector = await db.scalar(
        select(Job).where(Job.book_id == job_id, Job.job_type == JobType.COLLECTOR)
    )

    if not collector:
        collector = Job(
            id=uuid.uuid4().hex,
            book_id=book.id,
            job_type=JobType.COLLECTOR,
            status=JobStatus.PENDING,
            payload={"current_page": 0},
        )
        db.add(collector)

    book.status = JobStatus.PROCESSING
    await db.commit()

    return {"job_id": job_id, "status": JobStatus.PROCESSING}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Check granular progress and stage."""
    book = await db.get(Book, job_id)
    if not book:
        raise HTTPException(404, "Job not found.")

    return {
        "id": book.id,
        "status": book.status,
        "pdf_filename": book.pdf_filename,
        "progress": calculate_progress(book),
        "total_sentences": book.total_sentences,
        "completed_sentences": book.completed_sentences,
        "output_file": book.output_file,
        "error": book.error,
        "created_at": book.created_at.isoformat(),
        "updated_at": book.updated_at.isoformat(),
    }


@app.get("/jobs")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    """List all registered books."""
    books = (await db.scalars(select(Book).order_by(Book.created_at.desc()))).all()
    return [
        {
            "id": b.id,
            "status": b.status,
            "pdf_filename": b.pdf_filename,
            "progress": calculate_progress(b),
            "output_file": b.output_file,
            "created_at": b.created_at.isoformat(),
        }
        for b in books
    ]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a book, its database records, and disk files."""
    book = await db.get(Book, job_id)
    if not book:
        raise HTTPException(404, "Job not found.")

    # Remove upload and output files
    Path(book.pdf_path).unlink(missing_ok=True)
    if book.output_file:
        (OUTPUT_DIR / book.output_file).unlink(missing_ok=True)
    shutil.rmtree(OUTPUT_DIR / book.id, ignore_errors=True)

    await db.delete(book)
    await db.commit()
    return {"deleted": job_id}


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Download generated audiobook file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@app.get("/", response_class=FileResponse)
async def frontend():
    return FileResponse("index.html")


def main():
    import uvicorn

    # uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=True)
    uvicorn.run("bookaudio.app:app", host="0.0.0.0", port=8100)
