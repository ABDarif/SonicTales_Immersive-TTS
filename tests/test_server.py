import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import VISITOR_COOKIE, create_app, parse_chapter_spec

from .conftest import STORY


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def submit(client, **form):
    form.setdefault("tts", "synthetic")
    form.setdefault("fmt", "wav")
    files = form.pop("files", None)
    return client.post("/api/jobs", data=form, files=files)


def wait_done(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def jobs_dir(client) -> Path:
    return client.app.state.settings.data_dir / "jobs"


# ---------------------------------------------------------------- no accounts


def test_no_login_is_needed_to_use_the_app(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/jobs").json() == []
    r = submit(client, text=STORY)
    assert r.status_code == 202, "a brand-new visitor can generate an audiobook immediately"


@pytest.mark.parametrize("path", ["/api/signup", "/api/login", "/api/logout", "/api/me"])
def test_account_endpoints_are_gone(client, path):
    assert client.post(path, json={}).status_code in {404, 405}
    assert client.get(path).status_code in {404, 405} or client.get(path).headers["content-type"].startswith(
        "text/html"
    )


def test_frontend_has_no_login_or_signup(client):
    html = client.get("/").text.lower()
    js = client.get("/scripts.js").text.lower()
    for word in ("log in", "login", "sign up", "signup", "password", "logout"):
        assert word not in html and word not in js, word


# ---------------------------------------------------------------- anonymous visitors


def test_visitor_cookie_is_private_and_reused(client):
    r = client.get("/api/jobs")
    header = r.headers["set-cookie"].lower()
    assert f"{VISITOR_COOKIE}=" in header and "httponly" in header and "samesite=lax" in header
    token = client.cookies.get(VISITOR_COOKIE)
    assert len(token) >= 20
    assert client.get("/api/jobs").headers.get("set-cookie") is None, "an existing cookie is not reissued"
    assert client.cookies.get(VISITOR_COOKIE) == token


def test_garbage_cookie_is_replaced(client):
    client.cookies.set(VISITOR_COOKIE, "short", domain="testserver.local")
    r = client.get("/api/jobs")
    assert r.status_code == 200 and VISITOR_COOKIE in r.headers["set-cookie"]


def test_database_stores_only_a_hash_of_the_token(client):
    submit(client, text="A short story about a quiet garden.")
    token = client.cookies.get(VISITOR_COOKIE)
    with sqlite3.connect(client.app.state.db.path) as conn:
        owner = conn.execute("SELECT owner FROM jobs").fetchone()[0]
    assert owner == hashlib.sha256(token.encode()).hexdigest() and token not in owner


def test_history_follows_the_browser_cookie(client):
    job_id = submit(client, text="A short story about a quiet garden.").json()["id"]
    wait_done(client, job_id)
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id], "same browser sees its history"
    other_browser = TestClient(client.app)  # a second browser (no lifespan: must not re-run startup)
    assert other_browser.get("/api/jobs").json() == []


def test_visitors_cannot_see_each_others_jobs(client):
    job_id = submit(client, text="A short test story about a quiet garden.").json()["id"]
    wait_done(client, job_id)
    other = TestClient(client.app)  # a second browser (no lifespan: must not re-run startup)
    for path in ("", "/audio", "/script"):
        assert other.get(f"/api/jobs/{job_id}{path}").status_code == 404
    assert other.delete(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}").status_code == 200


def test_cookieless_request_cannot_fetch_audio(client):
    job_id = submit(client, text="A short test story about a quiet garden.").json()["id"]
    wait_done(client, job_id)
    stranger = TestClient(client.app)  # a second browser (no lifespan: must not re-run startup)
    assert stranger.get(f"/api/jobs/{job_id}/audio").status_code == 404


# ---------------------------------------------------------------- abuse limits


@pytest.fixture
def blocked_pipeline(monkeypatch):
    """Make jobs hang until released so we can observe 'active' limits deterministically."""
    release = threading.Event()

    def slow(*args, **kwargs):
        release.wait(30)
        raise RuntimeError("released")

    monkeypatch.setattr("app.server.run_pipeline", slow)
    yield release
    release.set()


def test_per_visitor_active_job_limit(client, blocked_pipeline):
    assert submit(client, text="First short story text here.").status_code == 202
    assert submit(client, text="Second short story text here.").status_code == 202
    third = submit(client, text="Third short story text here.")
    assert third.status_code == 429 and "wait" in third.json()["detail"]
    assert len(list(jobs_dir(client).iterdir())) == 2, "a refused request leaves nothing behind"
    other = TestClient(client.app)  # a second browser (no lifespan: must not re-run startup)
    assert submit(other, text="Another visitor's story text.").status_code == 202


def test_global_active_job_limit(client, blocked_pipeline, monkeypatch):
    monkeypatch.setattr("app.server.MAX_ACTIVE_TOTAL", 1)
    assert submit(client, text="First short story text here.").status_code == 202
    other = TestClient(client.app)  # a second browser (no lifespan: must not re-run startup)
    r = submit(other, text="Another visitor's story text.")
    assert r.status_code == 429 and "busy" in r.json()["detail"]


# ---------------------------------------------------------------- retention and migration


def test_old_finished_jobs_are_purged_on_startup(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        job_id = submit(c, text="A short test story about a quiet garden.").json()["id"]
        wait_done(c, job_id)
        with sqlite3.connect(app.state.db.path) as conn:
            conn.execute("UPDATE jobs SET created_at = ?", (time.time() - 8 * 86400,))
        assert (settings.data_dir / "jobs" / job_id).is_dir()
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/jobs").json() == []
        assert not (settings.data_dir / "jobs" / job_id).exists(), "files are deleted with the row"


def test_retention_zero_keeps_everything(settings):
    from dataclasses import replace

    settings = replace(settings, retention_days=0)
    app = create_app(settings)
    with TestClient(app) as c:
        job_id = submit(c, text="A short test story about a quiet garden.").json()["id"]
        wait_done(c, job_id)
        with sqlite3.connect(app.state.db.path) as conn:
            conn.execute("UPDATE jobs SET created_at = 1")
        cookie = c.cookies.get(VISITOR_COOKIE)
    with TestClient(create_app(settings)) as c:
        c.cookies.set(VISITOR_COOKIE, cookie)
        assert [j["id"] for j in c.get("/api/jobs").json()] == [job_id]


def test_upgrade_from_the_old_login_schema(settings):
    """A data dir created by the account-based version must still start, and its unreachable data is cleared."""
    settings.data_dir.mkdir(parents=True)
    with sqlite3.connect(settings.data_dir / "sonictales.db") as conn:
        conn.executescript(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, email TEXT, password_hash TEXT,"
            " created_at REAL);"
            "CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER, expires_at REAL);"
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, title TEXT NOT NULL,"
            " status TEXT NOT NULL,"
            " progress REAL, message TEXT, error TEXT, options TEXT, fmt TEXT, engine TEXT, duration_s REAL,"
            " created_at REAL, finished_at REAL);"
            "INSERT INTO users VALUES (1,'old','o@example.com','x',1);"
            "INSERT INTO jobs VALUES ('" + "b" * 32 + "',1,'old book','done',1,'','', '{}','mp3',NULL,1,1,1);"
        )
    (settings.data_dir / "jobs" / ("b" * 32)).mkdir(parents=True)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/jobs").json() == []
        assert not (settings.data_dir / "jobs" / ("b" * 32)).exists(), "orphaned files of removed accounts are deleted"
        assert submit(c, text="A short test story about a quiet garden.").status_code == 202
    with sqlite3.connect(settings.data_dir / "sonictales.db") as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "users" not in tables and "sessions" not in tables


def test_interrupted_jobs_are_failed_on_restart(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        c.get("/api/jobs")
        owner = hashlib.sha256(c.cookies.get(VISITOR_COOKIE).encode()).hexdigest()
        app.state.db.create_job("a" * 32, owner, "stuck", "{}", "mp3")
        cookie = c.cookies.get(VISITOR_COOKIE)
    with TestClient(create_app(settings)) as c:
        c.cookies.set(VISITOR_COOKIE, cookie)
        job = c.get(f"/api/jobs/{'a' * 32}").json()
        assert job["status"] == "failed" and "restart" in job["error"]


# ---------------------------------------------------------------- security headers / origin


def test_cross_origin_posts_rejected(client):
    r = client.post("/api/jobs", data={"text": "hello"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]


# ---------------------------------------------------------------- jobs


def test_full_job_lifecycle_with_text(client):
    r = submit(client, text=STORY, title="My Story")
    assert r.status_code == 202
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "done", job
    assert job["title"] == "My Story" and [c["title"] for c in job["chapters"]] == ["Chapter 1", "Chapter 2"]
    assert job["duration_s"] > 5 and job["engine"] == "synthetic"

    audio = client.get(f"/api/jobs/{job['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"
    assert "attachment" in client.get(f"/api/jobs/{job['id']}/audio?download=true").headers["content-disposition"]
    partial = client.get(f"/api/jobs/{job['id']}/audio", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100, "seeking works"

    script = client.get(f"/api/jobs/{job['id']}/script").json()
    assert script["characters"]["Mary"]["voice"]
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job["id"]]

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert not any(jobs_dir(client).iterdir()), "files removed with the job"


def test_pdf_upload(client, story_pdf: Path):
    r = submit(client, files={"file": ("story.pdf", story_pdf.read_bytes(), "application/pdf")}, chapters="2-3")
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "done" and len(job["chapters"]) == 2


@pytest.mark.parametrize(
    "form, status",
    [
        ({}, 422),
        ({"text": "   "}, 422),
        ({"text": "hello there", "tts": "nope"}, 422),
        ({"text": "hello there", "fmt": "ogg"}, 422),
        ({"text": "hello there", "builder": "llm"}, 422),  # no API key configured
        ({"text": "hello there", "ambience_db": "5"}, 422),
        ({"text": "hello there", "chapters": "abc"}, 422),
        ({"text": "hello there", "chapters": "7"}, 422),
        ({"text": "hello there", "tts": "openai"}, 422),  # engine unavailable
    ],
)
def test_bad_job_requests(client, form, status):
    r = submit(client, **form)
    assert r.status_code == status, r.text
    assert not any(jobs_dir(client).iterdir()), "rejected requests leave nothing behind"


def test_upload_validation(client):
    assert submit(client, files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 422
    assert submit(client, files={"file": ("a.pdf", b"not really a pdf", "application/pdf")}).status_code == 422
    assert submit(client, text="x" * 100_001).status_code == 413


def test_job_ids_are_validated(client):
    assert client.get("/api/jobs/..%2F..%2Fetc%2Fpasswd").status_code in {404, 422}
    assert client.get("/api/jobs/not-a-real-id").status_code == 404


def test_failed_job_reports_error(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("app.server.run_pipeline", boom)
    job = wait_done(client, submit(client, text="A short story that will fail.").json()["id"])
    assert job["status"] == "failed" and "engine exploded" in job["error"]
    assert client.get(f"/api/jobs/{job['id']}/audio").status_code == 409


def test_engines_endpoint(client):
    data = client.get("/api/engines").json()
    assert data["engines"]["synthetic"] is True and data["builders"]["heuristic"] is True


def test_static_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "SonicTales" in r.text
    assert client.get("/scripts.js").status_code == 200 and client.get("/styles.css").status_code == 200


@pytest.mark.parametrize(
    "spec, expected",
    [(None, None), ("", None), (" ", None), ("1", [1]), ("1,3,5-7", [1, 3, 5, 6, 7]), ("2-2", [2]), ("3,1,3", [1, 3])],
)
def test_parse_chapter_spec(spec, expected):
    assert parse_chapter_spec(spec) == expected


@pytest.mark.parametrize("spec", ["a", "1-", "5-2", "1;2", "-3", "1-9999"])
def test_parse_chapter_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_chapter_spec(spec)
