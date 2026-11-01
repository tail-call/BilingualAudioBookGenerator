import asyncio
import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from audiobook import (
    extract_sentences,
    generate_annotations,
    generate_tts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
STATE_FILE = BASE_DIR / ".app_state.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# In-memory job registry (persisted to disk as JSON)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    pdf_filename: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    created_at: str = ""
    updated_at: str = ""


_jobs: dict[str, JobRecord] = {}


def _load_jobs() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_jobs():
    with open(STATE_FILE, "w") as f:
        json.dump({ k: vars(v) for k, v in _jobs.items()}, f, indent=2)


# Initialise from disk
_jobs = {k: JobRecord(**v) for k, v in _load_jobs().items()}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BookAudio", version="0.1.0")


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update(job_id: str, **kwargs):
    if job_id in _jobs:
        for k, v in kwargs.items():
            setattr(_jobs[job_id], k, v)
        _jobs[job_id].updated_at = _now()
        _save_jobs()


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------


async def _run_pipeline(job_id: str, pdf_path: Path):
    """Three-stage pipeline: extract → annotate → generate TTS."""

    jsonl_out = BASE_DIR / f"scratch_{job_id}.jsonl"
    annotated_out = BASE_DIR / f"scratch_{job_id}-annotated.json"
    final_out = f"audiobook_{job_id}.mp3"

    try:
        # Stage 1: extract sentences
        _update(job_id, status=JobStatus.PROCESSING, progress=10)
        extract_sentences(str(pdf_path), str(jsonl_out))
        _update(job_id, progress=40)

        # Stage 2: generate annotations
        generate_annotations(str(jsonl_out), str(annotated_out))
        _update(job_id, progress=70)

        # Stage 3: generate TTS (returns list of files in output/)
        generate_tts(str(annotated_out))

        # Bundle all output/*.mp3 into a single file
        output_files = sorted(OUTPUT_DIR.glob("gputonic-*.mp3"))
        if not output_files:
            # try older naming
            output_files = sorted(OUTPUT_DIR.glob("*.mp3"))

        if output_files:
            # Use ffmpeg to concat (or just pick the first if single)
            import subprocess
            import tempfile

            filelist = Path(tempfile.mktemp(suffix=".txt"))
            with open(filelist, "w") as f:
                for p in output_files:
                    f.write(f"file '{p}'\n")

            dest = OUTPUT_DIR / final_out
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(filelist), "-c", "copy", str(dest),
                ],
                check=True,
            )
            filelist.unlink(missing_ok=True)

            _update(job_id, status=JobStatus.DONE, progress=100, output_file=dest.name)
        else:
            _update(job_id, status=JobStatus.FAILED, error="No audio files were generated.")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _update(job_id, status=JobStatus.FAILED, error=str(exc), output_file=None)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job(
    pdf: UploadFile = File(..., description="PDF file to convert"),
    source_language: str = Form("English"),
    target_language: str = Form("Russian"),
):
    """Upload a PDF and start the audiobook generation pipeline."""

    # Validate
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    job_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{job_id}_{pdf.filename}"

    content = await pdf.read()
    pdf_path.write_bytes(content)

    _jobs[job_id] = JobRecord(
        id=job_id,
        status=JobStatus.PENDING,
        pdf_filename=pdf.filename,
        created_at=_now(),
        updated_at=_now(),
    )
    _save_jobs()

    return {"job_id": job_id, "status": JobStatus.PENDING, "pdf_filename": pdf.filename}


@app.post("/jobs/{job_id}/start")
async def start_job(job_id: str, background_tasks: BackgroundTasks):
    """Start (or restart) processing a job in the background."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")

    job = _jobs[job_id]
    pdf_path = UPLOAD_DIR / f"{job_id}_{job.pdf_filename}"
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF file not found: {job.pdf_filename}")

    background_tasks.add_task(_run_pipeline, job_id, pdf_path)
    return {"job_id": job_id, "status": JobStatus.PROCESSING}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")
    return _jobs[job_id]


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return list(_jobs.values())


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found.")
    job = _jobs.pop(job_id)
    _save_jobs()
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


# ---------------------------------------------------------------------------
# Frontend — single-page HTML served at root
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def frontend():
    return HTMLResponse(content=_index_html())


def _index_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BookAudio — Audiobook Generator</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e2e4e9;
    --muted: #8b8fa3;
    --accent: #6c8cff;
    --accent-hover: #8ba4ff;
    --green: #4ade80;
    --red: #f87171;
    --yellow: #fbbf24;
    --radius: 10px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
  }
  .container { max-width: 800px; margin: 0 auto; padding: 2rem 1.5rem; }

  header { text-align: center; margin-bottom: 2.5rem; }
  header h1 { font-size: 2rem; font-weight: 700; }
  header h1 span { color: var(--accent); }
  header p { color: var(--muted); margin-top: .4rem; }

  /* Card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1.15rem; margin-bottom: 1rem; }

  /* Form */
  .form-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem; }
  .form-group { flex:1; min-width:200px; }
  label { display:block; font-size:.85rem; color:var(--muted); margin-bottom:.3rem; }
  input[type="text"], select {
    width:100%; padding:.6rem .8rem; border:1px solid var(--border);
    border-radius:6px; background:var(--bg); color:var(--text); font-size:.95rem;
  }
  input[type="file"] {
    width:100%; padding:.6rem; border:2px dashed var(--border);
    border-radius:6px; background:var(--bg); color:var(--muted); cursor:pointer;
    text-align:center;
  }
  input[type="file"]:hover { border-color: var(--accent); }

  button {
    padding: .7rem 1.6rem; border:none; border-radius:6px;
    background: var(--accent); color:#fff; font-size:.95rem; font-weight:600;
    cursor:pointer; transition: background .2s;
  }
  button:hover { background: var(--accent-hover); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  button.danger { background: var(--red); }
  button.danger:hover { background: #e05555; }
  button.small { padding:.4rem .8rem; font-size:.8rem; }

  /* Progress bar */
  .progress-wrap {
    margin-top:1rem; background:var(--bg); border-radius:20px; height:10px; overflow:hidden;
  }
  .progress-bar {
    height:100%; width:0%; background:var(--accent); border-radius:20px;
    transition: width .4s ease;
  }
  .progress-bar.green { background: var(--green); }
  .progress-bar.red { background: var(--red); }

  /* Job list */
  .job-list { list-style:none; }
  .job-item {
    display:flex; align-items:center; gap:1rem; padding:.9rem 0;
    border-bottom:1px solid var(--border);
  }
  .job-item:last-child { border-bottom:none; }
  .job-info { flex:1; min-width:0; }
  .job-name { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .job-meta { font-size:.8rem; color:var(--muted); }

  .badge {
    display:inline-block; padding:.2rem .6rem; border-radius:20px;
    font-size:.75rem; font-weight:600; text-transform:uppercase;
  }
  .badge.pending { background:#374151; color:var(--muted); }
  .badge.processing { background:#1e3a5f; color:var(--accent); }
  .badge.done { background:#14532d; color:var(--green); }
  .badge.failed { background:#7f1d1d; color:var(--red); }

  .actions { display:flex; gap:.5rem; flex-shrink:0; }
  .actions a { text-decoration:none; }

  /* Empty state */
  .empty { text-align:center; color:var(--muted); padding:2rem 0; }

  /* Log */
  .log {
    background:var(--bg); border-radius:6px; padding:.8rem;
    font-family:monospace; font-size:.8rem; color:var(--muted);
    max-height:200px; overflow-y:auto; margin-top:.8rem;
  }

  /* Responsive */
  @media (max-width:500px) {
    .form-row { flex-direction:column; }
    .job-item { flex-direction:column; align-items:flex-start; }
    .actions { width:100%; justify-content:flex-end; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>🎧 <span>Book</span>Audio</h1>
    <p>Upload a PDF → get a bilingual audiobook</p>
  </header>

  <!-- Upload card -->
  <div class="card">
    <h2>Upload &amp; Generate</h2>
    <form id="upload-form">
      <div class="form-row">
        <div class="form-group">
          <label for="pdf-input">PDF File</label>
          <input type="file" id="pdf-input" accept=".pdf" required>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="src-lang">Source Language</label>
          <select id="src-lang">
            <option value="English" selected>English</option>
            <option value="Russian">Russian</option>
          </select>
        </div>
        <div class="form-group">
          <label for="tgt-lang">Target Language</label>
          <select id="tgt-lang">
            <option value="Russian" selected>Russian</option>
            <option value="English">English</option>
          </select>
        </div>
      </div>
      <button type="submit" id="submit-btn">Generate Audiobook</button>
    </form>
    <div class="progress-wrap" id="progress-wrap" style="display:none;">
      <div class="progress-bar" id="progress-bar"></div>
    </div>
    <div class="log" id="status-log" style="display:none;"></div>
  </div>

  <!-- Jobs card -->
  <div class="card">
    <h2>Recent Jobs</h2>
    <ul class="job-list" id="job-list">
      <li class="empty">No jobs yet. Upload a PDF to get started.</li>
    </ul>
  </div>

</div>

<script>
// ── Helpers ──────────────────────────────────────────────────────────
const API = "";
const $ = id => document.getElementById(id);

async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

async function post(path, body, headers) {
  const r = await fetch(API + path, { method: "POST", body, headers });
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

// ── Render jobs list ─────────────────────────────────────────────────
async function renderJobs() {
  const jobs = await get("/jobs");
  const list = $("job-list");
  if (!jobs.length) {
    list.innerHTML = '<li class="empty">No jobs yet. Upload a PDF to get started.</li>';
    return;
  }
  list.innerHTML = jobs.sort((a,b) => b.created_at.localeCompare(a.created_at))
    .map(j => {
      const badgeClass = j.status;
      const canDownload = j.status === "done" && j.output_file;
      const canStart = j.status === "pending";
      return `<li class="job-item">
        <div class="job-info">
          <div class="job-name">${escapeHtml(j.pdf_filename || j.id)}</div>
          <div class="job-meta">${j.created_at} · ${j.progress}%</div>
        </div>
        <span class="badge ${badgeClass}">${j.status}</span>
        <div class="actions">
          ${canStart ? `<button class="small" onclick="startJob('${j.id}')">▶ Start</button>` : ""}
          ${canDownload ? `<a href="/output/${escapeHtml(j.output_file)}" download><button class="small">⬇ Download</button></a>` : ""}
          <button class="small danger" onclick="deleteJob('${j.id}')">✕</button>
        </div>
      </li>`;
    }).join("");
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ── Upload form ──────────────────────────────────────────────────────
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("pdf-input").files[0];
  const src = $("src-lang").value;
  const tgt = $("tgt-lang").value;

  const btn = $("submit-btn");
  btn.disabled = true;
  btn.textContent = "Uploading…";

  try {
    const fd = new FormData();
    fd.append("pdf", file);
    fd.append("source_language", src);
    fd.append("target_language", tgt);

    const result = await post("/jobs", fd);
    const jobId = result.job_id;

    $("progress-wrap").style.display = "block";
    $("status-log").style.display = "block";
    $("status-log").textContent = `Job created: ${jobId}\n`;

    // Auto-start
    await post(`/jobs/${jobId}/start`, null);
    pollJob(jobId);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Audiobook";
  }
});

// ── Start / Delete ───────────────────────────────────────────────────
async function startJob(id) {
  try {
    await post(`/jobs/${id}/start`, null);
    pollJob(id);
    renderJobs();
  } catch (e) {
    alert(e.message);
  }
}

async function deleteJob(id) {
  if (!confirm("Delete this job?")) return;
  try {
    await fetch(API + `/jobs/${id}`, { method: "DELETE" });
    renderJobs();
  } catch (e) {
    alert(e.message);
  }
}

// ── Polling ──────────────────────────────────────────────────────────
let pollTimer = null;

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const j = await get(`/jobs/${jobId}`);
      // Update progress bar
      const bar = $("progress-bar");
      bar.style.width = j.progress + "%";
      bar.className = "progress-bar";
      if (j.status === "done") bar.classList.add("green");
      if (j.status === "failed") bar.classList.add("red");

      // Update log
      const log = $("status-log");
      log.textContent += `[${j.progress}%] ${j.status}\n`;
      log.scrollTop = log.scrollHeight;

      renderJobs();

      if (j.status === "done" || j.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch {
      // ignore transient errors
    }
  }, 2000);
}

// ── Init ─────────────────────────────────────────────────────────────
renderJobs();
setInterval(renderJobs, 10000); // auto-refresh
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8100, reload=True)
