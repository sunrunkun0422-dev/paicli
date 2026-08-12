"""Built-in coding tools and their shared execution registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from .memory import LongTermMemory
from .models import RegisteredTool, ToolCall, ToolSpec
from .policy import PathGuard, PolicyError, check_command


ApprovalHandler = Callable[[str, Mapping[str, Any]], bool]
IGNORED_DIRS = {".git", ".paicli", "target", "node_modules", "dist", "build", ".idea", ".venv", "__pycache__"}


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    error: bool = False


class ToolRegistry:
    def __init__(
        self,
        project_root: Path,
        memory: LongTermMemory,
        approval_mode: str = "suggest",
        approval_handler: ApprovalHandler | None = None,
        command_timeout_seconds: int = 60,
        max_parallel: int = 4,
    ) -> None:
        self.root = project_root.resolve()
        self.guard = PathGuard(self.root)
        self.memory = memory
        self.approval_mode = approval_mode
        self.approval_handler = approval_handler
        self.command_timeout_seconds = command_timeout_seconds
        self.max_parallel = max(1, min(max_parallel, 16))
        self._approval_lock = Lock()
        self._tools: dict[str, RegisteredTool] = {}
        self._register_builtins()

    @property
    def definitions(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.spec.name] = tool

    def execute(self, call: ToolCall) -> ToolResult:
        registered = self._tools.get(call.name)
        if registered is None:
            return ToolResult(call.id, call.name, f"未知工具: {call.name}", True)
        try:
            arguments = call.parsed_arguments()
            if registered.dangerous and not self._approved(call.name, arguments):
                return ToolResult(call.id, call.name, "需要审批的工具调用未获批准", True)
            output = registered.executor(arguments)
            return ToolResult(call.id, call.name, output)
        except PolicyError as exc:
            return ToolResult(call.id, call.name, f"🛡️ 策略拒绝: {exc}", True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return ToolResult(call.id, call.name, f"工具执行失败: {exc}", True)
        except Exception as exc:  # A plugin-like tool must not crash the Agent loop.
            return ToolResult(call.id, call.name, f"工具执行异常: {type(exc).__name__}: {exc}", True)

    def execute_many(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        if len(calls) <= 1:
            return [self.execute(call) for call in calls]
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(calls))) as pool:
            futures = [pool.submit(self.execute, call) for call in calls]
            return [future.result() for future in futures]

    def _approved(self, name: str, arguments: Mapping[str, Any]) -> bool:
        if self.approval_mode == "auto":
            return True
        if self.approval_mode == "never" or self.approval_handler is None:
            return False
        # Parallel tool calls may all require HITL. Keep terminal prompts serialized.
        with self._approval_lock:
            return bool(self.approval_handler(name, arguments))

    def _register_builtins(self) -> None:
        self.register(_tool("read_file", "按需读取项目内文件，可用 offset/limit 限制行段", {
            "path": _string("文件路径"),
            "offset": _integer("起始行号，默认 1"),
            "limit": _integer("最多读取行数，默认全文，上限 2000"),
        }, ["path"], self._read_file))
        self.register(_tool("write_file", "写入项目内文件，单文件最大 5MB", {
            "path": _string("文件路径"), "content": _string("完整文件内容"),
        }, ["path", "content"], self._write_file, dangerous=True))
        self.register(_tool("replace_in_file", "精确替换项目内文件的一段文本", {
            "path": _string("文件路径"), "old": _string("必须唯一匹配的旧文本"),
            "new": _string("替换后的文本"),
        }, ["path", "old", "new"], self._replace_in_file, dangerous=True))
        self.register(_tool("list_dir", "列出项目内目录内容", {
            "path": _string("目录路径，默认 ."),
        }, [], self._list_dir))
        self.register(_tool("glob_files", "按 glob 模式查找项目内文件", {
            "pattern": _string("例如 **/*.py"), "path": _string("起始目录，默认 ."),
            "max_results": _integer("最多结果数，默认 50，上限 200"),
        }, ["pattern"], self._glob_files))
        self.register(_tool("grep_code", "按关键字或正则搜索项目内代码并返回文件和行号", {
            "pattern": _string("搜索内容"), "path": _string("起始目录，默认 ."),
            "glob": _string("文件 glob，例如 **/*.py"), "regex": _boolean("是否使用正则"),
            "case_sensitive": _boolean("是否大小写敏感，默认 true"),
            "context_lines": _integer("前后上下文行，上限 5"),
            "max_results": _integer("最多命中，默认 50，上限 200"),
            "max_chars": _integer("结果字符上限，默认 24000"),
        }, ["pattern"], self._grep_code))
        self.register(_tool("execute_command", "在项目根执行短时 shell 命令，默认 60 秒超时", {
            "command": _string("需要执行的命令"),
        }, ["command"], self._execute_command, dangerous=True))
        self.register(_tool("create_project", "在当前项目内创建最小项目结构", {
            "name": _string("项目目录名"),
            "type": {"type": "string", "enum": ["python", "node", "java"]},
        }, ["name", "type"], self._create_project, dangerous=True))
        self.register(_tool("save_memory", "保存用户明确要求记住的跨会话稳定事实", {
            "content": _string("精炼后的稳定事实"),
            "scope": {"type": "string", "enum": ["project", "global"]},
        }, ["content"], self._save_memory, dangerous=True))

    def _read_file(self, args: Mapping[str, Any]) -> str:
        path = self.guard.resolve(_required_str(args, "path"))
        if not path.is_file():
            raise ValueError("不是普通文件")
        text = path.read_text(encoding="utf-8")
        if "offset" not in args and "limit" not in args:
            return "文件内容:\n" + text
        lines = text.splitlines()
        offset = max(1, _int(args, "offset", 1))
        limit = max(1, min(2_000, _int(args, "limit", 200)))
        if offset > len(lines):
            return f"文件内容: {path.name} 共 {len(lines)} 行，offset 超出范围"
        end = min(len(lines), offset - 1 + limit)
        body = "\n".join(f"{number:>6}\t{lines[number - 1]}" for number in range(offset, end + 1))
        suffix = f"\n[显示 {offset}-{end} 行，共 {len(lines)} 行]"
        return f"文件内容: {path.name}\n{body}{suffix}"

    def _write_file(self, args: Mapping[str, Any]) -> str:
        path = self.guard.resolve(_required_str(args, "path"))
        content = str(args.get("content") or "")
        if len(content.encode("utf-8")) > 5 * 1024 * 1024:
            raise PolicyError("写入内容超过 5MB 上限")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"文件已写入: {path.relative_to(self.root)}"

    def _replace_in_file(self, args: Mapping[str, Any]) -> str:
        path = self.guard.resolve(_required_str(args, "path"))
        old = _required_str(args, "old")
        new = str(args.get("new") or "")
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != 1:
            raise ValueError(f"old 文本应唯一匹配，实际匹配 {count} 次")
        updated = content.replace(old, new, 1)
        if len(updated.encode("utf-8")) > 5 * 1024 * 1024:
            raise PolicyError("替换后文件超过 5MB 上限")
        path.write_text(updated, encoding="utf-8")
        return f"文件已更新: {path.relative_to(self.root)}"

    def _list_dir(self, args: Mapping[str, Any]) -> str:
        path = self.guard.resolve(str(args.get("path") or "."))
        if not path.is_dir():
            raise ValueError("目录不存在")
        rows = [f"[{'D' if item.is_dir() else 'F'}] {item.name}" for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))]
        return "目录内容:\n" + ("\n".join(rows) if rows else "(空目录)")

    def _glob_files(self, args: Mapping[str, Any]) -> str:
        base = self.guard.resolve(str(args.get("path") or "."))
        pattern = _required_str(args, "pattern")
        limit = max(1, min(200, _int(args, "max_results", 50)))
        results: list[str] = []
        partial = False
        for candidate in base.glob(pattern):
            if not candidate.is_file() or _ignored(candidate, self.root):
                continue
            results.append(candidate.relative_to(self.root).as_posix())
            if len(results) >= limit:
                partial = True
                break
        results.sort()
        if not results:
            return "未找到匹配文件"
        suffix = "\npartial: true（请缩小 pattern 或 path）" if partial else ""
        return "匹配文件:\n" + "\n".join(results) + suffix

    def _grep_code(self, args: Mapping[str, Any]) -> str:
        base = self.guard.resolve(str(args.get("path") or "."))
        pattern = _required_str(args, "pattern")
        file_glob = str(args.get("glob") or "")
        regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", True))
        context = max(0, min(5, _int(args, "context_lines", 0)))
        max_results = max(1, min(200, _int(args, "max_results", 50)))
        max_chars = max(1_000, min(60_000, _int(args, "max_chars", 24_000)))
        rg = shutil.which("rg")
        if rg:
            return self._grep_with_rg(rg, base, pattern, file_glob, regex, case_sensitive, context, max_results, max_chars)
        return self._grep_with_python(base, pattern, file_glob, regex, case_sensitive, context, max_results, max_chars)

    def _grep_with_rg(
        self, rg: str, base: Path, pattern: str, file_glob: str, regex: bool,
        case_sensitive: bool, context: int, max_results: int, max_chars: int,
    ) -> str:
        command = [rg, "--line-number", "--no-heading", "--color", "never"]
        if not regex:
            command.append("--fixed-strings")
        if not case_sensitive:
            command.append("--ignore-case")
        if context:
            command.extend(["--context", str(context)])
        if file_glob:
            command.extend(["--glob", file_glob])
        for name in sorted(IGNORED_DIRS):
            command.extend(["--glob", f"!{name}/**"])
        command.extend([pattern, str(base)])
        completed = subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=20, check=False)
        if completed.returncode not in {0, 1}:
            raise OSError(completed.stderr.strip() or f"rg exited {completed.returncode}")
        lines = completed.stdout.splitlines()
        selected = lines[:max_results]
        body = "\n".join(selected)
        partial = len(lines) > max_results or len(body) > max_chars
        body = body[:max_chars]
        if not body:
            return "未找到匹配内容"
        return body + ("\npartial: true（请缩小搜索范围）" if partial else "")

    def _grep_with_python(
        self, base: Path, pattern: str, file_glob: str, regex: bool,
        case_sensitive: bool, context: int, max_results: int, max_chars: int,
    ) -> str:
        flags = 0 if case_sensitive else re.IGNORECASE
        expression = re.compile(pattern if regex else re.escape(pattern), flags)
        rows: list[str] = []
        chars = 0
        partial = False
        for path in base.rglob("*"):
            if not path.is_file() or _ignored(path, self.root):
                continue
            relative = path.relative_to(self.root).as_posix()
            if file_glob and not fnmatch.fnmatch(relative, file_glob):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for index, line in enumerate(lines):
                if not expression.search(line):
                    continue
                start, end = max(0, index - context), min(len(lines), index + context + 1)
                block = "\n".join(f"{relative}:{number + 1}:{lines[number]}" for number in range(start, end))
                if len(rows) >= max_results or chars + len(block) > max_chars:
                    partial = True
                    break
                rows.append(block)
                chars += len(block)
            if partial:
                break
        if not rows:
            return "未找到匹配内容"
        return "\n--\n".join(rows) + ("\npartial: true（请缩小搜索范围）" if partial else "")

    def _execute_command(self, args: Mapping[str, Any]) -> str:
        command = _required_str(args, "command")
        reason = check_command(command)
        if reason:
            raise PolicyError(reason)
        completed = subprocess.run(
            command,
            cwd=self.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.command_timeout_seconds,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        if len(output) > 60_000:
            output = output[:60_000] + "\n...(输出已截断)"
        return f"exit_code: {completed.returncode}\n{output or '(无输出)'}"

    def _create_project(self, args: Mapping[str, Any]) -> str:
        name = _required_str(args, "name")
        kind = _required_str(args, "type").lower()
        root = self.guard.resolve(name)
        root.mkdir(parents=True, exist_ok=True)
        if kind == "python":
            package_name = re.sub(r"\W+", "_", Path(name).name).strip("_") or "app"
            (root / package_name).mkdir(exist_ok=True)
            (root / package_name / "__init__.py").touch()
            (root / "pyproject.toml").write_text(
                "[build-system]\nrequires = ['setuptools>=68']\nbuild-backend = 'setuptools.build_meta'\n",
                encoding="utf-8",
            )
        elif kind == "node":
            (root / "package.json").write_text(json.dumps({"name": Path(name).name, "version": "1.0.0"}, indent=2), encoding="utf-8")
        elif kind == "java":
            (root / "src" / "main" / "java").mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("type 仅支持 python/node/java")
        return f"项目已创建: {root.relative_to(self.root)} ({kind})"

    def _save_memory(self, args: Mapping[str, Any]) -> str:
        entry = self.memory.save(_required_str(args, "content"), str(args.get("scope") or "project"))
        return f"长期记忆已保存: {entry.id}"


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str],
    executor: Callable[[Mapping[str, Any]], str],
    dangerous: bool = False,
) -> RegisteredTool:
    return RegisteredTool(
        ToolSpec(name, description, {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}),
        executor,
        dangerous,
    )


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _required_str(args: Mapping[str, Any], name: str) -> str:
    value = args.get(name)
    if value is None or not str(value).strip():
        raise ValueError(f"缺少必填参数: {name}")
    return str(value)


def _int(args: Mapping[str, Any], name: str, default: int) -> int:
    value = args.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ignored(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in IGNORED_DIRS for part in parts)


__all__ = ["ToolRegistry", "ToolResult", "ApprovalHandler"]
