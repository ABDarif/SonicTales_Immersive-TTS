"""Web API + static front end.

    uvicorn app.server:app          (or: python main.py serve)

There are no accounts: each browser gets an anonymous random token in a cookie and only sees its own jobs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT, Settings, get_settings
from app.db import Database
from app.pdf_extract import PdfError, load_book
from app.pipeline import PipelineOptions, run_pipeline, select_chapters
from app.tts_engine import available_engines

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
VISITOR_COOKIE = "visitor"
VISITOR_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
VISITOR_COOKIE_MAX_AGE = 365 * 24 * 3600
MAX_ACTIVE_PER_VISITOR = 2  # queued + running jobs one browser may have at once
MAX_ACTIVE_TOTAL = 20  # whole-server cap so an open service can't be flooded
MEDIA_TYPES = {"mp3": "audio/mpeg", "m4b": "audio/mp4", "wav": "audio/wav"}
WEB_DIR = ROOT / "web"


def parse_chapter_spec(spec: str | None) -> list[int] | None:
    """'1,3,5-7' -> [1,3,5,6,7]; blank -> None (all chapters)."""
    if not spec or not spec.strip():
        return None
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d{1,4})(?:\s*-\s*(\d{1,4}))?", part)
        if not m:
            raise ValueError(f"bad chapter selection {part!r}; use e.g. 1,3,5-7")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if hi < lo or hi - lo > 500:
            raise ValueError(f"bad chapter range {part!r}")
        result.update(range(lo, hi + 1))
    return sorted(result) or None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings.data_dir / "sonictales.db")
    jobs_dir = settings.data_dir / "jobs"
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")  # TTS is CPU-bound

    def remove_orphan_folders() -> None:
        """Delete job folders with no DB row (legacy accounts, crashes between mkdir and insert)."""
        known = db.all_job_ids()
        for d in jobs_dir.iterdir():
            if d.is_dir() and d.name not in known:
                shutil.rmtree(d, ignore_errors=True)

    def purge_expired() -> None:
        """Anonymous audiobooks are not kept forever: delete finished jobs older than the retention window."""
        if settings.retention_days > 0:
            for job_id in db.delete_finished_before(time.time() - settings.retention_days * 86400):
                shutil.rmtree(jobs_dir / job_id, ignore_errors=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        db.init()
        remove_orphan_folders()
        purge_expired()
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="SonicTales", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.db = db
    app.state.settings = settings

    # ------------------------------------------------------------------ middleware

    @app.middleware("http")
    async def security(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith("/api/docs"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; media-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
        return response

    # ------------------------------------------------------------------ anonymous visitors

    def visitor(request: Request, response: Response) -> str:
        """Identify this browser without any login: a random private token in a cookie.

        Returns the sha256 of the token (what we store), and issues a fresh cookie if there is none.
        """
        token = request.cookies.get(VISITOR_COOKIE)
        if not token or not VISITOR_TOKEN_RE.fullmatch(token):
            token = secrets.token_urlsafe(24)
            response.set_cookie(
                VISITOR_COOKIE,
                token,
                max_age=VISITOR_COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=settings.secure_cookies,
                path="/",
            )
        return hashlib.sha256(token.encode()).hexdigest()

    @app.get("/api/engines")
    def engines():
        return {
            "engines": available_engines(settings),
            "builders": {"heuristic": True, "llm": bool(settings.anthropic_api_key)},
        }

    # ------------------------------------------------------------------ jobs

    def job_dir(job_id: str) -> Path:
        return jobs_dir / job_id  # job_id is a server-generated uuid hex, never user input

    def job_json(row) -> dict:
        d = {
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "error": row["error"],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "duration_s": row["duration_s"],
            "engine": row["engine"],
            "fmt": row["fmt"],
            "chapters": [],
        }
        script = job_dir(row["id"]) / "script.json"
        if row["status"] == "done" and script.is_file():
            data = json.loads(script.read_text(encoding="utf-8"))
            d["chapters"] = [{"index": c["index"], "title": c["title"]} for c in data["chapters"]]
        return d

    def run_job(job_id: str, source: Path, opts: PipelineOptions) -> None:
        last = [0.0]

        def progress(frac: float, msg: str) -> None:
            now = time.monotonic()
            if now - last[0] > 0.4 or frac >= 1.0:
                last[0] = now
                db.update_job(job_id, progress=round(frac, 3), message=msg)

        db.update_job(job_id, status="running", message="Starting")
        try:
            result = run_pipeline(source, job_dir(job_id), opts, settings, progress=progress)
            db.update_job(
                job_id,
                status="done",
                progress=1.0,
                message="Done",
                engine=result.engine,
                duration_s=result.duration_s,
                finished_at=time.time(),
            )
        except (ValueError, PdfError, RuntimeError, OSError) as exc:
            log.warning("job %s failed: %s", job_id, exc)
            db.update_job(job_id, status="failed", error=str(exc)[:500], finished_at=time.time())
        except Exception:  # never leave a job stuck in "running"
            log.exception("job %s crashed", job_id)
            db.update_job(job_id, status="failed", error="Unexpected server error", finished_at=time.time())

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        file: UploadFile | None = File(default=None),
        text: str = Form(default=""),
        title: str = Form(default=""),
        tts: str = Form(default="auto"),
        builder: str = Form(default="auto"),
        ambience: bool = Form(default=True),
        ambience_db: float = Form(default=-12.0),
        fmt: str = Form(default="mp3"),
        chapters: str = Form(default=""),
        owner: str = Depends(visitor),
    ):
        if tts not in {"auto", "kokoro", "openai", "espeak", "synthetic"}:
            raise HTTPException(422, "Unknown TTS engine")
        if builder not in {"auto", "heuristic", "llm"}:
            raise HTTPException(422, "Unknown script builder")
        if fmt not in {"mp3", "m4b", "wav"}:
            raise HTTPException(422, "Format must be mp3, m4b or wav")
        if not -30.0 <= ambience_db <= -3.0:
            raise HTTPException(422, "Ambience level must be between -30 and -3 dB")
        if tts != "auto" and not available_engines(settings).get(tts):
            raise HTTPException(422, f"The {tts} engine is not available on this server")
        if builder == "llm" and not settings.anthropic_api_key:
            raise HTTPException(422, "The LLM builder needs ANTHROPIC_API_KEY on the server")
        try:
            chapter_list = parse_chapter_spec(chapters)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        if db.count_active(owner) >= MAX_ACTIVE_PER_VISITOR:
            raise HTTPException(429, "You already have audiobooks being generated; wait for one to finish")
        if db.count_active() >= MAX_ACTIVE_TOTAL:
            raise HTTPException(429, "The server is busy right now; please try again in a few minutes")
        purge_expired()

        job_id = uuid.uuid4().hex
        folder = job_dir(job_id)
        folder.mkdir(parents=True)
        try:
            if file is not None and file.filename:
                ext = Path(file.filename).suffix.lower()
                if ext not in {".pdf", ".txt"}:
                    raise HTTPException(422, "Upload a .pdf or .txt file")
                source = folder / f"source{ext}"
                size = 0
                with source.open("wb") as fh:
                    while chunk := await file.read(1 << 20):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "File is larger than 25 MB")
                        fh.write(chunk)
                if ext == ".pdf":
                    with source.open("rb") as fh:
                        if not fh.read(1024).lstrip().startswith(b"%PDF-"):
                            raise HTTPException(422, "That file is not a valid PDF")
                default_title = Path(file.filename).stem[:80]
            elif text.strip():
                if len(text) > settings.max_chars:
                    raise HTTPException(413, f"Text is longer than {settings.max_chars:,} characters")
                source = folder / "source.txt"
                source.write_text(text, encoding="utf-8")
                default_title = "Pasted text"
            else:
                raise HTTPException(422, "Upload a PDF/TXT file or paste some text")

            # Parse now so obviously bad input fails fast with a clear message instead of a failed job later.
            try:
                book = await run_in_threadpool(load_book, source, title.strip()[:120] or default_title)
                select_chapters(book, chapter_list, settings.max_chars)
            except (PdfError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

        opts = PipelineOptions(
            tts=tts,
            builder=builder,
            ambience=ambience,
            ambience_db=ambience_db,
            fmt=fmt,
            chapters=chapter_list,
            max_chars=settings.max_chars,
        )
        db.create_job(job_id, owner, book.title, json.dumps({"tts": tts, "builder": builder}), fmt)
        executor.submit(run_job, job_id, source, opts)
        return {"id": job_id}

    @app.get("/api/jobs")
    def list_jobs(owner: str = Depends(visitor)):
        return [job_json(r) for r in db.list_jobs(owner)]

    def owned_job(job_id: str, owner: str):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Not found")
        row = db.get_job(job_id, owner)
        if row is None:
            raise HTTPException(404, "Not found")
        return row

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, owner: str = Depends(visitor)):
        return job_json(owned_job(job_id, owner))

    @app.get("/api/jobs/{job_id}/script")
    def get_script(job_id: str, owner: str = Depends(visitor)):
        owned_job(job_id, owner)
        path = job_dir(job_id) / "script.json"
        if not path.is_file():
            raise HTTPException(404, "Script is not ready yet")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str, chapter: int | None = None, download: bool = False, owner: str = Depends(visitor)):
        row = owned_job(job_id, owner)
        if row["status"] != "done":
            raise HTTPException(409, "Audio is not ready yet")
        name = f"chapter_{chapter:03d}.{row['fmt']}" if chapter else f"audiobook.{row['fmt']}"
        path = job_dir(job_id) / name
        if not path.is_file():
            raise HTTPException(404, "No such audio file")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", row["title"]).strip("_") or "audiobook"
        return FileResponse(
            path,
            media_type=MEDIA_TYPES[row["fmt"]],
            filename=f"{safe}{'_ch' + str(chapter) if chapter else ''}.{row['fmt']}" if download else None,
        )

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, owner: str = Depends(visitor)):
        row = owned_job(job_id, owner)
        if row["status"] in {"queued", "running"}:
            raise HTTPException(409, "Wait for the job to finish before deleting it")
        db.delete_job(job_id, owner)
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        return {"ok": True}

    # ------------------------------------------------------------------ static front end

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
