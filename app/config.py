"""Runtime configuration read from environment variables (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so we don't need an extra dependency. Never overrides real env vars."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_chars: int
    secure_cookies: bool
    retention_days: int
    llm_model: str
    anthropic_api_key: str | None
    openai_api_key: str | None

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.environ.get("SONICTALES_DATA", ROOT / "data")),
        max_chars=int(os.environ.get("SONICTALES_MAX_CHARS", "300000")),
        secure_cookies=os.environ.get("SONICTALES_SECURE_COOKIES", "0") == "1",
        retention_days=int(os.environ.get("SONICTALES_RETENTION_DAYS", "7")),
        llm_model=os.environ.get("SONICTALES_LLM_MODEL", "claude-sonnet-5-5"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
