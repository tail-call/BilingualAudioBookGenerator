import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audiobook import (
    extract_sentences,
    generate_annotations,
    generate_tts,
)
from jobs import JobManager, JobStatus

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
STATE_FILE = BASE_DIR / ".app_state.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def concat_files_with_ffmpeg(final_out: str, output_files: list[Path]):
    # Use ffmpeg to concat (or just pick the first if single)
    import subprocess
    import tempfile

    filelist = Path(tempfile.mktemp(suffix=".txt"))
    with open(filelist, "w") as f:
        f.writelines(f"file '{p}'\n" for p in output_files)

    dest = OUTPUT_DIR / final_out
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist), "-c", "copy", str(dest),
        ],
        check=True,
    )
    filelist.unlink(missing_ok=True)
    return dest

async def run_pipeline(job_id: str, pdf_path: Path):
    """Three-stage pipeline: extract → annotate → generate TTS."""

    jsonl_out = BASE_DIR / f"scratch_{job_id}.jsonl"
    annotated_out = BASE_DIR / f"scratch_{job_id}-annotated.json"
    final_out = f"audiobook_{job_id}.mp3"

    try:
        # Stage 1: extract sentences
        job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=10)
        extract_sentences(str(pdf_path), str(jsonl_out))
        job_manager.update_job(job_id, progress=40)

        # Stage 2: generate annotations
        generate_annotations(str(jsonl_out), str(annotated_out))
        job_manager.update_job(job_id, progress=70)

        # Stage 3: generate TTS (returns list of files in output/)
        generate_tts(str(annotated_out))

        # Bundle all output/*.mp3 into a single file
        output_files = sorted(OUTPUT_DIR.glob("gputonic-*.mp3"))
        if not output_files:
            # try older naming
            output_files = sorted(OUTPUT_DIR.glob("*.mp3"))

        if output_files:
            dest = concat_files_with_ffmpeg(final_out, output_files)

            job_manager.update_job(
                job_id,
                status=JobStatus.DONE,
                progress=100,
                output_file=dest.name
            )
        else:
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error="No audio files were generated."
            )

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED, error=str(exc) + str(tb),
            output_file=None
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


app = FastAPI(title="BookAudio", version="0.1.0")
job_manager = JobManager(STATE_FILE)


@app.post("/jobs")
async def create_job(
    pdf: UploadFile = File(..., description="PDF file to convert"),
    source_language: str = Form("English"),
    target_language: str = Form("Russian"),
):
    """Upload a PDF and start the audiobook generation pipeline."""

    # Validate
    if not pdf.filename or pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    job_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{job_id}_{pdf.filename}"

    content = await pdf.read()
    pdf_path.write_bytes(content)

    job_manager.add_job(job_id, pdf.filename)

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "pdf_filename": pdf.filename
    }


@app.post("/jobs/{job_id}/start")
async def start_job(job_id: str, background_tasks: BackgroundTasks):
    """Start (or restart) processing a job in the background."""
    if not job_manager.has_job(job_id):
        raise HTTPException(404, "Job not found.")

    job = job_manager.job_with_id(job_id)
    pdf_path = UPLOAD_DIR / f"{job_id}_{job.pdf_filename}"
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found: {job.pdf_filename}")

    background_tasks.add_task(run_pipeline, job_id, pdf_path)
    return {"job_id": job_id, "status": JobStatus.PROCESSING}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    if not job_manager.has_job(job_id):
        raise HTTPException(404, "Job not found.")
    return job_manager.job_with_id(job_id)


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return job_manager.all_jobs()


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if not job_manager.has_job(job_id):
        raise HTTPException(404, "Job not found.")
    job = job_manager.delete_job_with_id(job_id)
    # Clean up files
    for p in [
        UPLOAD_DIR / f"{job_id}_{job.pdf_filename}",
        BASE_DIR / f"scratch_{job_id}.jsonl",
        BASE_DIR / f"scratch_{job_id}-annotated.json",
    ]:
        p.unlink(missing_ok=True)
    return {"deleted": job_id}


@app.get("/output/{filename}")
async def get_output(filename: str):
    """Download a generated audio file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@app.get("/", response_class=FileResponse)
async def frontend():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=True)
