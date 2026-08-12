"""Configuration loading compatible with the Java implementation's env names."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


def load_dotenv(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Load a small, dependency-free subset of dotenv syntax.

    Existing process environment values always win. The returned mapping can be
    supplied by tests without mutating ``os.environ``.
    """

    values = dict(os.environ if environ is None else environ)
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key and key not in values:
            values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    project_root: Path
    provider: str | None
    env: Mapping[str, str]
    approval_mode: str = "suggest"
    command_timeout_seconds: int = 60

    @classmethod
    def load(
        cls,
        project_root: str | Path | None = None,
        provider: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        root = Path(project_root or Path.cwd()).expanduser().resolve()
        source = dict(os.environ if environ is None else environ)
        # Priority: process env > project .env > user ~/.env.
        user_values = load_dotenv(Path.home() / ".env", {})
        project_values = load_dotenv(root / ".env", {})
        values = {**user_values, **project_values, **source}
        selected = provider or values.get("PAICLI_PROVIDER") or _detect_provider(values)
        approval = values.get("PAICLI_APPROVAL_MODE", "suggest").strip().lower()
        if approval not in {"suggest", "auto", "never"}:
            approval = "suggest"
        timeout = _positive_int(values.get("PAICLI_COMMAND_TIMEOUT_SECONDS"), 60)
        return cls(root, selected, values, approval, timeout)


def _detect_provider(env: Mapping[str, str]) -> str | None:
    candidates = (
        ("glm", "GLM_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("step", "STEP_API_KEY"),
        ("kimi", "KIMI_API_KEY"),
        ("kimi", "MOONSHOT_API_KEY"),
        ("freellmapi", "FREELLMAPI_API_KEY"),
        ("xfyun", "XFYUN_MAAS_API_KEY"),
        ("agnes", "AGNES_API_KEY"),
    )
    return next((name for name, key in candidates if _configured(env.get(key))), None)


def _configured(value: str | None) -> bool:
    if not value or not value.strip():
        return False
    lowered = value.strip().lower()
    return not lowered.startswith("your_") and lowered not in {"changeme", "test-key"}


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(raw or "")
    except ValueError:
        return default
    return value if value > 0 else default
