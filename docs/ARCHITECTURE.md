# Architecture and design decisions

## Pipeline

Every stage has a narrow typed interface (`app/models.py`), so each can be tested and replaced alone:

1. `pdf_extract` → `Book` (title, chapters of paragraphs)
2. `script_builder` → `Script` (characters, chapters → scenes → segments with speaker/emotion/pause, scenes with ambience tags)
3. `tts_engine` → mono float32 audio per segment, cached
4. `ambience` → loopable bed per tag set
5. `mixer` → scene audio (speech + ducked bed), stitched with crossfades, mastered
6. `audio_io` / `pipeline` → chapter WAVs on disk, then one ffmpeg concat/encode for the book

Chapters are written to disk as they finish and concatenated by ffmpeg, so memory use stays flat
for long books instead of holding hours of float32 audio.

## Decisions

**Python backend instead of the original PHP/MySQL.** The ML/audio ecosystem is Python; one language
removes a process boundary. The old PHP also concatenated user input into SQL (injection) and ran as MySQL
root with no password; all SQL is now parameterised and secrets come from the environment.

**Rule-based script builder first, LLM optional.** The offline builder means the product works with no
key and is deterministic and free to test. The LLM builder improves accuracy but is guarded: its output is
parsed, schema-checked, tag-whitelisted and verified to cover ≥90% of the source text, otherwise that
chunk silently falls back to the rules. The model can never drop or invent story text unnoticed.

**Procedural ambience instead of bundled recordings.** No licensing risk, no large binaries in git,
deterministic output, and loops are seamless by construction (periodic FFT synthesis). The trade-off is
realism; `assets/ambience/` lets users substitute real recordings per tag.

**Pluggable TTS behind a four-method-sized interface.** Kokoro (free, CPU-friendly), OpenAI (cloud),
espeak-ng (anywhere), and a synthetic non-speech engine so tests and CI never download models.
The cache key includes engine, voice, emotion and text.

**No pydub.** It depends on `audioop`, removed in Python 3.13. numpy for DSP and direct ffmpeg calls for
encoding are fewer moving parts. Moving averages use cumulative sums: `np.convolve` with
second-long kernels was O(n·k) and dominated runtime until profiled.

**Jobs run in one background worker thread.** TTS is CPU-bound and the box is small; a single worker keeps
latency predictable. Jobs interrupted by a restart are marked failed on startup instead of hanging forever.
When this needs to scale, swap the executor for a real queue (RQ/Celery) without touching the pipeline.

**No accounts.** Anyone can use the app without signing up. A random token in an HttpOnly cookie identifies a
browser, and only its sha256 is stored, so a leaked database can't be used to open anyone's audiobooks. The
trade-off is that history lives with the browser: clearing cookies or switching devices loses it. Without
accounts there is nothing to rate-limit by user, so abuse is contained structurally instead: at most 2 active jobs
per browser, 20 server-wide, upload size/type limits, and finished audiobooks are deleted after
`SONICTALES_RETENTION_DAYS` (default 7) so anonymous uploads don't accumulate forever. Data directories from the earlier
account-based version are migrated on startup: the old tables and their now-unreachable files are removed.

**Security posture.** Per-visitor job isolation (404 not 403, so job IDs can't be probed); HttpOnly + SameSite=Lax
cookies; Origin check on unsafe methods; CSP (no inline script/style), nosniff, frame-deny; upload size/type/magic-byte
checks; server-generated job IDs, never user paths; parameterised SQL. For a public deployment also put it behind
HTTPS (`SONICTALES_SECURE_COOKIES=1`) and a reverse proxy with request-rate limiting.

## Known gaps / next steps

- OCR for scanned PDFs (`ocrmypdf`/Tesseract before extraction).
- Voice cloning or user-chosen voices per character in the UI (the script already stores a voice per character).
- Smarter coreference (pronouns across paragraphs) via a small local NLP model.
- Persistent queue + object storage for multi-user deployment; a durable queue would also allow fairer scheduling between visitors.
- Alembic-style migrations once the schema changes.
