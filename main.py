"""SonicTales command line.

python main.py build book.pdf -o out/            # PDF/TXT -> audiobook
python main.py build book.pdf --chapters 1 2 --tts kokoro --format m4b
python main.py script book.pdf                   # just show the speaker/scene analysis
python main.py serve --port 8000                 # web app
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.pipeline import PipelineOptions, run_pipeline


def _progress(frac: float, msg: str) -> None:
    print(f"\r[{frac * 100:5.1f}%] {msg:<60}", end="", file=sys.stderr, flush=True)


def cmd_build(args: argparse.Namespace) -> int:
    settings = get_settings()
    options = PipelineOptions(
        tts=args.tts,
        builder=args.builder,
        ambience=not args.no_ambience,
        ambience_db=args.ambience_db,
        fmt=args.format,
        chapters=args.chapters,
    )
    result = run_pipeline(Path(args.input), Path(args.output), options, settings, progress=_progress)
    print(file=sys.stderr)
    print(f"Audiobook : {result.audio_path}  ({result.duration_s / 60:.1f} min, engine={result.engine})")
    for p in result.chapter_paths:
        print(f"Chapter   : {p}")
    print(f"Script    : {result.script_path}")
    return 0


def cmd_script(args: argparse.Namespace) -> int:
    from app.pdf_extract import load_book
    from app.pipeline import select_chapters
    from app.script_builder import build_script

    book = select_chapters(load_book(Path(args.input)), args.chapters, None)
    script = build_script(book, args.builder, get_settings())
    for ch in script.chapters:
        print(f"\n# {ch.title}")
        for i, sc in enumerate(ch.scenes, 1):
            print(f"  scene {i}  ambience={sc.ambience or '-'}")
            for seg in sc.segments:
                print(f"    [{seg.speaker} / {seg.emotion}] {seg.text[:90]}")
    print("\nCast:", {n: c.gender for n, c in script.characters.items()})
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sonictales", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", help="PDF or TXT file")
        p.add_argument("--chapters", type=int, nargs="+", help="1-based chapter numbers (default: all)")
        p.add_argument("--builder", choices=["auto", "heuristic", "llm"], default="auto")

    b = sub.add_parser("build", help="generate an audiobook")
    common(b)
    b.add_argument("-o", "--output", default="output")
    b.add_argument("--tts", choices=["auto", "kokoro", "sapi", "openai", "espeak", "synthetic"], default="auto")
    b.add_argument("--format", choices=["mp3", "m4b", "wav"], default="mp3")
    b.add_argument("--no-ambience", action="store_true")
    b.add_argument("--ambience-db", type=float, default=-12.0, help="ambience gain in dB (default -12)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("script", help="print the speaker/scene analysis without generating audio")
    common(s)
    s.set_defaults(func=cmd_script)

    w = sub.add_parser("serve", help="run the web app")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--reload", action="store_true")
    w.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
