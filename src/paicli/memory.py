"""Auditable, Java-file-compatible long-term memory.

Short-term conversation state has deliberately been removed from this module:
the Agent owns one canonical message history and compacts that history directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from .context import estimate_text_tokens


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    content: str
    type: str = "FACT"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, str] = field(default_factory=dict)
    token_count: int = 0

    @classmethod
    def fact(cls, content: str, scope: str, project: str) -> "MemoryEntry":
        metadata = {"source": "fact", "scope": scope}
        if scope == "project":
            metadata["project"] = project
        return cls(
            "fact-" + uuid4().hex[:8],
            content.strip(),
            "FACT",
            metadata=metadata,
            token_count=estimate_text_tokens(content),
        )

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "MemoryEntry":
        content = str(value.get("content") or "")
        metadata = {str(key): str(item) for key, item in (value.get("metadata") or {}).items()}
        return cls(
            str(value.get("id") or "fact-" + uuid4().hex[:8]),
            content,
            str(value.get("type") or "FACT"),
            str(value.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            metadata,
            int(value.get("tokenCount") or value.get("token_count") or estimate_text_tokens(content)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tokenCount": self.token_count,
        }


class LongTermMemory:
    def __init__(self, project_root: Path, storage_dir: Path | None = None) -> None:
        self.project_root = str(project_root.resolve())
        directory = storage_dir or Path.home() / ".paicli" / "memory"
        self.storage_file = directory / "long_term_memory.json"
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = RLock()
        self._load()

    def save(self, content: str, scope: str = "project") -> MemoryEntry:
        with self._lock:
            normalized_scope = "global" if scope.strip().lower() == "global" else "project"
            normalized = content.strip()
            if not normalized:
                raise ValueError("记忆内容不能为空")
            for entry in self._entries.values():
                if entry.content == normalized and self._visible(entry):
                    return entry
            entry = MemoryEntry.fact(normalized, normalized_scope, self.project_root)
            self._entries[entry.id] = entry
            self._persist()
            return entry

    def list(self) -> list[MemoryEntry]:
        with self._lock:
            return sorted(
                (entry for entry in self._entries.values() if self._visible(entry)),
                key=lambda item: item.timestamp,
            )

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        terms = _query_terms(query)
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self.list():
            haystack = (entry.content + " " + " ".join(entry.metadata.values())).lower()
            score = sum(1 for term in terms if term in haystack)
            if query.strip().lower() in haystack:
                score += max(2, len(terms))
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        return [entry for _, entry in scored[: max(1, limit)]]

    def context_for(self, query: str, max_tokens: int) -> str:
        selected = self.search(query, 10)
        if not selected:
            return ""
        lines = ["## 相关长期记忆", ""]
        used = 0
        for entry in selected:
            if used + entry.token_count > max_tokens:
                break
            lines.append(f"- [{entry.type}] {entry.content}")
            used += entry.token_count
        return "\n".join(lines) if len(lines) > 2 else ""

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or not self._visible(entry):
                return False
            del self._entries[entry_id]
            self._persist()
            return True

    def clear_visible(self) -> int:
        with self._lock:
            ids = [entry.id for entry in self.list()]
            for entry_id in ids:
                del self._entries[entry_id]
            self._persist()
            return len(ids)

    def _visible(self, entry: MemoryEntry) -> bool:
        scope = entry.metadata.get("scope", "global").lower()
        return scope == "global" or entry.metadata.get("project") == self.project_root

    def _load(self) -> None:
        if not self.storage_file.is_file():
            return
        try:
            raw = json.loads(self.storage_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for value in raw:
                    if isinstance(value, dict):
                        entry = MemoryEntry.from_json(value)
                        self._entries[entry.id] = entry
        except (OSError, json.JSONDecodeError, ValueError):
            # A broken memory file must not stop the agent from starting.
            self._entries.clear()

    def _persist(self) -> None:
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.storage_file.with_suffix(".json.tmp")
        body = [entry.to_json() for entry in self._entries.values()]
        temp_file.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(self.storage_file)


def _query_terms(query: str) -> set[str]:
    lowered = query.lower()
    words = set(re.findall(r"[a-z0-9_.\-/]{2,}|[\u4e00-\u9fff]{2,}", lowered))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        words.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return words or ({lowered.strip()} if lowered.strip() else set())


def format_entries(entries: Iterable[MemoryEntry]) -> str:
    rows = [f"{entry.id}  [{entry.metadata.get('scope', 'global')}]  {entry.content}" for entry in entries]
    return "\n".join(rows) if rows else "暂无长期记忆"
