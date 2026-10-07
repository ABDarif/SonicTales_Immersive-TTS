# SonicTales: Immersive Text-to-Speech

Turn a book PDF into an **immersive audiobook**: a distinct voice for every character, expressive
delivery, and background ambience that matches each scene (rain on a stormy night, a crackling
fire in the tavern, birdsong at dawn).

```
PDF/TXT ─► pdf_extract ─► script_builder ─► tts_engine ─┐
          clean text,     who speaks, how,  one voice    ├─► mixer ─► MP3 / M4B
          chapters        which ambience    per speaker  │   ducking, crossfades,
                          (rules or Claude) ─────────────┘   loudness
                                         ambience ───────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg            # required for MP3/M4B export (brew install ffmpeg on macOS)

# Real voices (pick one):
pip install -r requirements-tts.txt && sudo apt install espeak-ng   # Kokoro: best free quality, CPU is fine
#   or: sudo apt install espeak-ng                                 # robotic but zero-download
#   or: export OPENAI_API_KEY=...                                  # cloud voices

python main.py build samples/rainy_inn.txt -o output/              # CLI
python main.py serve                                               # web app on http://127.0.0.1:8000
```

With no speech engine installed the app still runs end to end using a **placeholder tone engine**
(clearly labelled in the UI) so you can try the mixing and ambience before installing voices.

### CLI

```bash
python main.py build book.pdf -o out/ --format m4b          # one audiobook file with chapter markers
python main.py build book.pdf --chapters 1 2 --tts kokoro   # just chapters 1-2
python main.py build book.pdf --no-ambience --ambience-db -15
python main.py script book.pdf                              # preview speakers/scenes, no audio
```

### Web app

Sign up, upload a PDF (or paste text), choose options, and watch progress. The result page has a
player with per-chapter playback and download, plus the detected cast and each scene's ambience.
History is kept per user. Interactive API docs live at `/api/docs`.

## What each stage does

| Stage | File | How it works |
|---|---|---|
| Extract | `app/pdf_extract.py` | PyMuPDF text blocks → paragraphs; strips running headers and page numbers, repairs hyphenation, splits chapters by headings (falls back to the PDF outline). Scanned PDFs give a clear error (no OCR yet). |
| Script | `app/script_builder.py` | Splits narration from quoted speech; attributes speakers from "said X", pronouns, role nouns ("said his lady") and back-and-forth alternation; infers gender and emotion; detects scenes and ambience from a lexicon with persistence ("night" lasts until "morning", "rain" until it stops). Optional **Claude** builder does it with an LLM, validates against the source text, and falls back to the rules per chunk if the answer is malformed or incomplete. |
| Voices | `app/tts_engine.py` | One interface, four engines (Kokoro, OpenAI, espeak-ng, placeholder). Each character gets a stable voice from a gendered pool; results are cached on disk so reruns are free; transient failures are retried. |
| Ambience | `app/ambience.py` | 11 procedural soundscapes (rain, thunder, wind, night, fire, ocean, forest, river, city, crowd, birds). Noise is shaped in the frequency domain with whole-cycle modulation so every loop is seamless. Drop real recordings in `assets/ambience/` to override. |
| Mix | `app/mixer.py` | Pauses between lines, ambience bed ducked ~7 dB under speech, crossfades between scenes, then ffmpeg loudness normalisation (-16 LUFS). |
| Serve | `app/server.py`, `app/db.py` | FastAPI + SQLite. Cookie sessions, scrypt password hashes, parameterised SQL, login throttling, per-user jobs, background worker, HTTP range requests so audio seeks. |

## Configuration

Copy `.env.example` to `.env`. All variables are optional: `SONICTALES_DATA`, `SONICTALES_MAX_CHARS`,
`SONICTALES_SECURE_COOKIES` (set to `1` behind HTTPS), `ANTHROPIC_API_KEY` (+ `SONICTALES_LLM_MODEL`)
for the AI script builder, `OPENAI_API_KEY` for cloud voices.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

CI (`.github/workflows/ci.yml`) runs the same on every push. A `Dockerfile` is provided
(`docker build -t sonictales . && docker run -p 8000:8000 sonictales`;
add `--build-arg WITH_KOKORO=1` for neural voices).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions and trade-offs.

## Status and known limitations

Be aware of what is and isn't verified:

- **Tested automatically (128 tests, on Python 3.11 and 3.13):** PDF extraction, the rule-based script builder (including a
  real-prose regression), the LLM builder against a fake client (malformed output, dropped text,
  network errors), all ambience generators (seamless loops, levels), mixing/ducking, MP3/M4B export
  with chapter markers and loudness, the whole pipeline, and the web API (auth, isolation, uploads, jobs).
  The web UI was also driven end to end in headless Chromium.
- **Not built or run anywhere yet:** the `Dockerfile` (no Docker daemon was available when it was written).
- **Not exercised in CI:** the Kokoro and espeak engines and the live Claude/OpenAI calls need
  models, binaries or API keys that CI does not have. They are written against those libraries' documented
  interfaces; run `python main.py build samples/rainy_inn.txt --tts kokoro` once to confirm on your machine.
- **Speaker detection is heuristic.** It handles common dialogue conventions well but will mislabel
  unusual styles (no quotation marks, long unattributed exchanges, single-quote dialogue). The Claude
  builder is more accurate on those. Ambience comes from keywords, so it can be fooled by metaphors.
- **No OCR** for scanned PDFs, and the hosted GitHub Pages site cannot run the backend (it serves
  static files only); deploy with Docker or any Python host.
- Generation time is dominated by TTS: Kokoro on CPU is roughly real-time or a few times faster.
  Use `--chapters` to try a book a chapter at a time. Only use books you have the right to convert
  (Project Gutenberg is a good source of public-domain texts).
