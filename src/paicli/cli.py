"""Plain terminal entrypoint for the Python-only PaiCLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from . import __version__
from .agent import Agent
from .config import Settings
from .llm import PROVIDERS, create_client
from .memory import format_entries
from .models import StreamListener
from .planning import PlanExecutor


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ConsoleListener(StreamListener):
    def __init__(self, show_reasoning: bool = False) -> None:
        self.show_reasoning = show_reasoning
        self.wrote_content = False
        self.wrote_reasoning = False

    def on_reasoning(self, text: str) -> None:
        if not self.show_reasoning:
            return
        if not self.wrote_reasoning:
            sys.stderr.write("\n[thinking] ")
            self.wrote_reasoning = True
        sys.stderr.write(text)
        sys.stderr.flush()

    def on_content(self, text: str) -> None:
        if not self.wrote_content:
            if self.wrote_reasoning:
                sys.stderr.write("\n")
            self.wrote_content = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def finish(self) -> None:
        if self.wrote_content:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif self.wrote_reasoning:
            sys.stderr.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paicli", description="PaiCLI Python coding agent")
    parser.add_argument("prompt", nargs="*", help="单次执行的任务；省略时进入交互模式")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), help="覆盖自动检测的 LLM provider")
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="项目根目录")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--yes", action="store_true", help="自动批准写入型工具，安全策略拒绝仍然有效")
    approval.add_argument("--never-approve", action="store_true", help="拒绝所有写入型工具")
    parser.add_argument("--show-reasoning", action="store_true", help="在 stderr 显示 reasoning 流")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--version", action="version", version=f"PaiCLI Python {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)
    settings = Settings.load(args.project, args.provider)
    if not settings.project_root.is_dir():
        print(f"项目目录不存在: {settings.project_root}", file=sys.stderr)
        return 2
    approval_mode = "auto" if args.yes else "never" if args.never_approve else settings.approval_mode
    settings = Settings(
        settings.project_root,
        settings.provider,
        settings.env,
        approval_mode,
        settings.command_timeout_seconds,
    )
    try:
        client = create_client(settings)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    agent = Agent(
        client,
        settings.project_root,
        approval_mode=settings.approval_mode,
        approval_handler=_terminal_approval,
        command_timeout_seconds=settings.command_timeout_seconds,
    )

    if args.prompt:
        return _run_once(agent, " ".join(args.prompt), args.show_reasoning)
    return _interactive(agent, settings, args.show_reasoning)


def _run_once(agent: Agent, prompt: str, show_reasoning: bool) -> int:
    listener = ConsoleListener(show_reasoning)
    try:
        answer = agent.run(prompt, listener)
    except (OSError, KeyboardInterrupt) as exc:
        listener.finish()
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1
    listener.finish()
    if not listener.wrote_content and answer:
        print(answer)
    return 0


def _interactive(agent: Agent, settings: Settings, show_reasoning: bool) -> int:
    print(
        f"PaiCLI Python {__version__} | {agent.llm_client.provider_name}/"
        f"{agent.llm_client.model_name} | {settings.project_root}"
    )
    print("输入 /help 查看命令，Ctrl+D 退出。")
    while True:
        try:
            raw = input("* ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n已取消输入；再次按 Ctrl+D 退出。")
            continue
        text = raw.strip()
        if not text:
            continue
        if text in {"/exit", "/quit", "exit", "quit"}:
            return 0
        handled = _handle_command(agent, settings, text)
        if handled is not None:
            if handled:
                print(handled)
            continue
        _run_once(agent, raw, show_reasoning)


def _handle_command(agent: Agent, settings: Settings, text: str) -> str | None:
    if not text.startswith("/"):
        return None
    if text == "/help":
        return (
            "/clear                 清空当前对话\n"
            "/compact               立即压缩旧对话\n"
            "/context               查看下一轮上下文估算\n"
            "/model <provider>       切换 LLM provider\n"
            "/plan <goal>            使用 Plan+DAG 子 Agent 执行复杂任务\n"
            "/memory list            列出当前项目和全局记忆\n"
            "/memory search <query>  搜索长期记忆\n"
            "/memory delete <id>     删除记忆\n"
            "/memory clear           清空当前可见记忆\n"
            "/exit                   退出"
        )
    if text == "/clear":
        agent.clear()
        return "当前对话已清空，长期记忆保持不变。"
    if text == "/compact":
        result = agent.compact_now()
        if result.error:
            return f"压缩失败: {result.error}"
        if not result.compacted:
            return "当前没有可压缩的旧对话。"
        return f"已压缩: {result.before_tokens:,} -> {result.after_tokens:,} tokens"
    if text == "/context":
        return agent.context_status()
    if text.startswith("/model "):
        provider = text[7:].strip().lower()
        try:
            client = create_client(settings, provider)
        except ValueError as exc:
            return f"切换失败: {exc}"
        agent.set_llm_client(client)
        return f"已切换到 {client.provider_name}/{client.model_name}"
    if text.startswith("/plan "):
        goal = text[len("/plan ") :].strip()
        if not goal:
            return "用法: /plan <goal>"
        try:
            return PlanExecutor(agent).run(goal)
        except OSError as exc:
            return f"Plan 执行失败: {exc}"
    if text == "/memory list":
        return format_entries(agent.memory.list())
    if text.startswith("/memory search "):
        return format_entries(agent.memory.search(text[len("/memory search ") :]))
    if text.startswith("/memory delete "):
        entry_id = text[len("/memory delete ") :].strip()
        return "记忆已删除" if agent.memory.delete(entry_id) else "未找到可删除的记忆"
    if text == "/memory clear":
        count = agent.memory.clear_visible()
        return f"已清空 {count} 条当前可见记忆"
    return "未知命令；输入 /help 查看可用命令。"


def _terminal_approval(name: str, arguments: Mapping[str, Any]) -> bool:
    if not sys.stdin.isatty():
        return False
    rendered = " ".join(f"{key}={_short(value)}" for key, value in arguments.items())
    try:
        answer = input(f"允许工具 {name}({rendered})? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes", "是"}


def _short(value: Any, limit: int = 100) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
