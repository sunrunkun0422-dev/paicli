# PaiCLI Project Context

## Positioning

PaiCLI 是纯 Python 的代码仓库 Agent，核心是 ReAct、Plan+DAG、本地工具、上下文压缩、长期记忆和安全策略。

## Commands

```bash
PYTHONPATH=src python3 -m paicli
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
PYTHONPATH=src python3 -m compileall -q src/paicli
```

## Architecture

- `agent.py` 维护唯一的真实对话历史并执行 ReAct 循环。
- `tools.py` 注册工具并保持并行结果顺序。
- `compaction.py` 在 user 边界压缩早期消息。
- `memory.py` 只负责可审计的长期记忆。
- `llm.py` 提供零依赖 OpenAI-compatible SSE 客户端。

## Don'ts

- 不重新引入 Java/Maven 构建。
- 不创建第二套短期消息容器。
- 不绕过路径、命令和审批策略。
- 不自动保存长期记忆。
