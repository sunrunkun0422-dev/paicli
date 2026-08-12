"""Small prompt assembler for the Python-only agent core."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from .models import ToolSpec


BASE_PROMPT = """## Identity

你是 PaiCLI，一个面向代码库工作的智能编程 Agent。

## Language

默认使用中文回复。代码、命令、路径和 API 名称保留原文。

## Working Style

- 先理解目标，再按需使用工具；简单问题直接回答。
- 修改代码前先用 glob_files、grep_code、read_file 获取真实上下文。
- 工具结果不足时继续调用工具，足够时给出简洁、可验证的最终结论。
- 同一轮可返回多个互不依赖的工具调用，系统会并行执行并保持结果顺序。
- 不得声称执行了并未实际执行的工具或命令。

## Memory Policy

- 仅当用户明确要求记住跨会话事实时调用 save_memory。
- 临时任务、当前 todo、模型猜测不得写入长期记忆。

## Safety Policy

- 文件路径必须在项目根内，不得利用绝对路径、.. 或符号链接越界。
- 写文件、执行命令、创建项目属于写操作，必须遵守审批策略。
- 策略拒绝后不要原样重试，应改用安全方案。
"""


def assemble_system_prompt(
    project_root: Path,
    tools: Sequence[ToolSpec],
    memory_context: str = "",
    approval_mode: str = "suggest",
) -> str:
    tool_lines = [f"- `{tool.name}`：{tool.description}" for tool in tools]
    project_context = load_project_context(project_root)
    sections = [
        BASE_PROMPT.strip(),
        "## Tools\n\n" + "\n".join(tool_lines),
        (
            "## Approval Mode\n\n"
            + {
                "auto": "写操作可以自动执行，但仍受路径和命令安全策略约束。",
                "never": "所有需要审批的写操作都应视为不可执行。",
                "suggest": "写操作会请求用户确认；在确认前不要声称已经完成。",
            }.get(approval_mode, "写操作会请求用户确认。")
        ),
        f"## Runtime Context\n\n- 当前日期: {datetime.now().astimezone().date()}\n"
        f"- 当前时区: {datetime.now().astimezone().tzinfo}\n- 项目根: {project_root}",
    ]
    if project_context:
        sections.append("## Project Context\n\n" + project_context)
    if memory_context:
        sections.append(memory_context)
    return "\n\n".join(sections).strip()


def load_project_context(project_root: Path, max_chars: int = 20_000) -> str:
    candidates = (
        Path.home() / ".paicli" / "PAI.md",
        project_root / "PAI.md",
        project_root / ".paicli" / "PAI.md",
        project_root / "PAI.local.md",
        project_root / ".paicli" / "PAI.local.md",
    )
    parts: list[str] = []
    remaining = max_chars
    for path in candidates:
        if not path.is_file() or remaining <= 0:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not content:
            continue
        selected = content[:remaining]
        parts.append(f"### {path.name}\n\n{selected}")
        remaining -= len(selected)
    return "\n\n".join(parts)
