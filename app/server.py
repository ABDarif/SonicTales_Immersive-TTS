"""Web API + static front end.

    uvicorn app.server:app          (or: python main.py serve)

Auth is a cookie session; every job belongs to one user and is only visible to them.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, Settings, get_settings
from app.db import Database
from app.pdf_extract import PdfError, load_book
from app.pipeline import PipelineOptions, run_pipeline, select_chapters
from app.tts_engine import available_engines

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SESSION_COOKIE = "session"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MEDIA_TYPES = {"mp3": "audio/mpeg", "m4b": "audio/mp4", "wav": "audio/wav"}
WEB_DIR = ROOT / "web"


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=200)


class SignupBody(Credentials):
    email: str = Field(max_length=120)


class LoginThrottle:
    """Blocks brute force: N failed logins per (client, username) within a window."""

    def __init__(self, limit: int = 8, window_s: float = 600.0) -> None:
        self.limit, self.window = limit, window_s
        self._fails: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: tuple[str, str]) -> deque[float]:
        q = self._fails[key]
        while q and q[0] < time.monotonic() - self.window:
            q.popleft()
        return q

    def blocked(self, key: tuple[str, str]) -> bool:
        with self._lock:
            return len(self._prune(key)) >= self.limit

    def record_failure(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._prune(key).append(time.monotonic())

    def reset(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._fails.pop(key, None)


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
    throttle = LoginThrottle()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        db.init()
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

    # ------------------------------------------------------------------ auth

    def current_user(request: Request):
        user = db.user_for_session(request.cookies.get(SESSION_COOKIE))
        if user is None:
            raise HTTPException(401, "Please log in")
        return user

    def set_session(response: Response, user_id: int) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            db.create_session(user_id),
            max_age=14 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=settings.secure_cookies,
            path="/",
        )

    @app.post("/api/signup", status_code=201)
    def signup(body: SignupBody, response: Response):
        username = body.username.strip()
        if not USERNAME_RE.match(username):
            raise HTTPException(422, "Username must be 3-32 characters: letters, digits, _ . -")
        if not EMAIL_RE.match(body.email.strip()):
            raise HTTPException(422, "Enter a valid email address")
        user_id = db.create_user(username, body.email.strip(), body.password)
        if user_id is None:
            raise HTTPException(409, "That username is taken")
        set_session(response, user_id)
        return {"id": user_id, "username": username}

    @app.post("/api/login")
    def login(body: Credentials, request: Request, response: Response):
        key = (request.client.host if request.client else "?", body.username.lower())
        if throttle.blocked(key):
            raise HTTPException(429, "Too many failed attempts; try again in a few minutes")
        user_id = db.authenticate(body.username.strip(), body.password)
        if user_id is None:
            throttle.record_failure(key)
            raise HTTPException(401, "Invalid username or password")
        throttle.reset(key)
        set_session(response, user_id)
        return {"id": user_id, "username": body.username.strip()}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        db.delete_session(request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request):
        """Who am I? 200 with `user: null` when logged out, so a fresh page load isn't a console error."""
        user = db.user_for_session(request.cookies.get(SESSION_COOKIE))
        return {"user": {"id": user["id"], "username": user["username"]} if user else None}

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
        user=Depends(current_user),
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
        db.create_job(job_id, user["id"], book.title, json.dumps({"tts": tts, "builder": builder}), fmt)
        executor.submit(run_job, job_id, source, opts)
        return {"id": job_id}

    @app.get("/api/jobs")
    def list_jobs(user=Depends(current_user)):
        return [job_json(r) for r in db.list_jobs(user["id"])]

    def owned_job(job_id: str, user):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Not found")
        row = db.get_job(job_id, user["id"])
        if row is None:
            raise HTTPException(404, "Not found")
        return row

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, user=Depends(current_user)):
        return job_json(owned_job(job_id, user))

    @app.get("/api/jobs/{job_id}/script")
    def get_script(job_id: str, user=Depends(current_user)):
        owned_job(job_id, user)
        path = job_dir(job_id) / "script.json"
        if not path.is_file():
            raise HTTPException(404, "Script is not ready yet")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str, chapter: int | None = None, download: bool = False, user=Depends(current_user)):
        row = owned_job(job_id, user)
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
    def delete_job(job_id: str, user=Depends(current_user)):
        row = owned_job(job_id, user)
        if row["status"] in {"queued", "running"}:
            raise HTTPException(409, "Wait for the job to finish before deleting it")
        db.delete_job(job_id, user["id"])
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        return {"ok": True}

    # ------------------------------------------------------------------ static front end

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
