import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import hash_password, verify_password
from app.server import create_app, parse_chapter_spec

from .conftest import STORY

CREDS = {"username": "alice", "email": "alice@example.com", "password": "correct horse battery"}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def alice(client):
    assert client.post("/api/signup", json=CREDS).status_code == 201
    return client


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


# ---------------------------------------------------------------- auth


def test_password_hashing():
    h = hash_password("secret-password")
    assert h.startswith("scrypt$") and "secret-password" not in h
    assert verify_password("secret-password", h) and not verify_password("wrong", h)
    assert hash_password("secret-password") != h, "salted"
    assert not verify_password("x", "garbage")


def test_signup_login_logout_flow(client):
    assert client.get("/api/me").json() == {"user": None}
    r = client.post("/api/signup", json=CREDS)
    assert r.status_code == 201 and "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/me").json()["user"]["username"] == "alice"
    client.post("/api/logout")
    assert client.get("/api/me").json() == {"user": None}
    assert client.post("/api/login", json={"username": "alice", "password": CREDS["password"]}).status_code == 200
    assert client.get("/api/me").status_code == 200


def test_duplicate_username_and_validation(alice):
    assert alice.post("/api/signup", json={**CREDS, "username": "ALICE"}).status_code == 409, "case-insensitive"
    assert alice.post("/api/signup", json={**CREDS, "username": "x y"}).status_code == 422
    assert alice.post("/api/signup", json={**CREDS, "username": "bob", "email": "nope"}).status_code == 422
    assert alice.post("/api/signup", json={**CREDS, "username": "bob", "password": "short"}).status_code == 422


def test_wrong_password_is_generic(alice):
    alice.post("/api/logout")
    a = alice.post("/api/login", json={"username": "alice", "password": "wrong-password"})
    b = alice.post("/api/login", json={"username": "nobody", "password": "wrong-password"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json(), "no user enumeration"


def test_sql_injection_attempts_do_not_log_in(alice):
    alice.post("/api/logout")
    for payload in ["alice' OR '1'='1", "' OR 1=1 --", "alice'; DROP TABLE users; --"]:
        r = alice.post("/api/login", json={"username": payload, "password": "whatever-password"})
        assert r.status_code in {401, 422}
    assert alice.post("/api/login", json={"username": "alice", "password": CREDS["password"]}).status_code == 200


def test_login_is_throttled(alice):
    alice.post("/api/logout")
    codes = [
        alice.post("/api/login", json={"username": "alice", "password": "bad-password-1"}).status_code
        for _ in range(10)
    ]
    assert codes[:8] == [401] * 8 and codes[8:] == [429, 429]
    assert alice.post("/api/login", json={"username": "alice", "password": CREDS["password"]}).status_code == 429


def test_cross_origin_posts_rejected(client):
    r = client.post(
        "/api/login", json={"username": "a", "password": "b" * 8}, headers={"Origin": "https://evil.example"}
    )
    assert r.status_code == 403


def test_security_headers(client):
    r = client.get("/")
    assert (
        r.headers["x-content-type-options"] == "nosniff"
        and "default-src 'self'" in r.headers["content-security-policy"]
    )


# ---------------------------------------------------------------- jobs


def test_jobs_require_login(client):
    assert client.get("/api/jobs").status_code == 401
    assert submit(client, text="hello").status_code == 401


def test_full_job_lifecycle_with_text(alice):
    r = submit(alice, text=STORY, title="My Story")
    assert r.status_code == 202
    job = wait_done(alice, r.json()["id"])
    assert job["status"] == "done", job
    assert job["title"] == "My Story" and [c["title"] for c in job["chapters"]] == ["Chapter 1", "Chapter 2"]
    assert job["duration_s"] > 5 and job["engine"] == "synthetic"

    audio = alice.get(f"/api/jobs/{job['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"
    assert "attachment" in alice.get(f"/api/jobs/{job['id']}/audio?download=true").headers["content-disposition"]
    partial = alice.get(f"/api/jobs/{job['id']}/audio", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100, "seeking works"

    script = alice.get(f"/api/jobs/{job['id']}/script").json()
    assert script["characters"]["Mary"]["voice"]
    assert [j["id"] for j in alice.get("/api/jobs").json()] == [job["id"]]

    assert alice.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert alice.get(f"/api/jobs/{job['id']}").status_code == 404
    assert not any((alice.app.state.settings.data_dir / "jobs").iterdir()), "files removed with the job"


def test_pdf_upload(alice, story_pdf: Path):
    r = submit(alice, files={"file": ("story.pdf", story_pdf.read_bytes(), "application/pdf")}, chapters="2-3")
    job = wait_done(alice, r.json()["id"])
    assert job["status"] == "done" and len(job["chapters"]) == 2


def test_users_cannot_see_each_others_jobs(alice, settings):
    job_id = submit(alice, text="A short test story about a quiet garden.").json()["id"]
    wait_done(alice, job_id)
    with TestClient(alice.app) as bob:
        bob.post("/api/signup", json={"username": "bob", "email": "b@example.com", "password": "another long pass"})
        assert bob.get("/api/jobs").json() == []
        for path in ("", "/audio", "/script"):
            assert bob.get(f"/api/jobs/{job_id}{path}").status_code == 404
        assert bob.delete(f"/api/jobs/{job_id}").status_code == 404


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
def test_bad_job_requests(alice, form, status):
    r = submit(alice, **form)
    assert r.status_code == status, r.text
    assert not any((alice.app.state.settings.data_dir / "jobs").iterdir()), "rejected requests leave nothing behind"


def test_upload_validation(alice):
    assert submit(alice, files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 422
    assert submit(alice, files={"file": ("a.pdf", b"not really a pdf", "application/pdf")}).status_code == 422
    assert submit(alice, text="x" * 100_001).status_code == 413


def test_job_ids_are_validated(alice):
    assert alice.get("/api/jobs/..%2F..%2Fetc%2Fpasswd").status_code in {404, 422}
    assert alice.get("/api/jobs/not-a-real-id").status_code == 404


def test_failed_job_reports_error(alice, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("app.server.run_pipeline", boom)
    job = wait_done(alice, submit(alice, text="A short story that will fail.").json()["id"])
    assert job["status"] == "failed" and "engine exploded" in job["error"]
    assert alice.get(f"/api/jobs/{job['id']}/audio").status_code == 409


def test_interrupted_jobs_are_failed_on_restart(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        c.post("/api/signup", json=CREDS)
        app.state.db.create_job("a" * 32, 1, "stuck", "{}", "mp3")
    with TestClient(create_app(settings)) as c:
        c.post("/api/login", json={"username": "alice", "password": CREDS["password"]})
        job = c.get(f"/api/jobs/{'a' * 32}").json()
        assert job["status"] == "failed" and "restart" in job["error"]


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
