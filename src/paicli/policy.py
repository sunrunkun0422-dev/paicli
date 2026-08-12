"""Workspace path and command fast-fail policies."""

from __future__ import annotations

from pathlib import Path
import re


class PolicyError(RuntimeError):
    pass


class PathGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def resolve(self, value: str) -> Path:
        if not value or not value.strip():
            raise PolicyError("路径不能为空")
        raw = Path(value).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PolicyError(f"路径越界: {value} 不在项目根 {self.root} 内") from exc
        return resolved


COMMAND_DENY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("禁止 sudo 提权", re.compile(r"\bsudo\b", re.I)),
    (
        "禁止 rm -rf 删除全盘或用户目录",
        re.compile(r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(/|~|\$home)|\brm\s+-[a-z]*f[a-z]*r[a-z]*\s+(/|~|\$home)", re.I),
    ),
    ("禁止 mkfs 格式化磁盘", re.compile(r"\bmkfs(?:\.|\b)", re.I)),
    ("禁止 dd 写入裸设备", re.compile(r"\bdd\b[^\n]*\bof=/dev/", re.I)),
    ("识别为 fork bomb", re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    ("禁止管道直接执行远端脚本", re.compile(r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sh|bash|zsh|fish|ksh)\b", re.I)),
    ("不允许扫描整个文件系统", re.compile(r"\bfind\s+(?:/|~|\$home)", re.I)),
    ("禁止 chmod 777 全盘", re.compile(r"\bchmod\s+-R\s+777\s+(?:/|~)", re.I)),
    ("禁止关机或重启", re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.I)),
)


def check_command(command: str) -> str | None:
    normalized = re.sub(r"\s+", " ", command or "").strip()
    for reason, pattern in COMMAND_DENY_RULES:
        if pattern.search(normalized):
            return reason
    return None
