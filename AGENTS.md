# AGENTS.md

本仓库是纯 Python 的 PaiCLI Agent。代码实际行为优先于文档。

## 项目快照

- Python 3.10+
- 零第三方运行时依赖
- 包入口：`src/paicli/`
- CLI：`python3 -m paicli`
- 测试：标准库 `unittest`
- 主链路：ReAct、并行工具、Plan+DAG、上下文压缩、长期记忆、安全策略

## 常用命令

```bash
PYTHONPATH=src python3 -m paicli --version
PYTHONPATH=src python3 -m paicli
PYTHONPATH=src python3 -m paicli "任务"
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
PYTHONPATH=src python3 -m compileall -q src/paicli
```

## 结构导航

| 任务 | 文件 |
|---|---|
| ReAct 主循环 | `src/paicli/agent.py` |
| Provider / SSE | `src/paicli/llm.py` |
| 工具 | `src/paicli/tools.py` |
| 安全策略 | `src/paicli/policy.py` |
| 压缩和 token | `src/paicli/compaction.py`, `context.py` |
| 长期记忆 | `src/paicli/memory.py` |
| Plan+DAG | `src/paicli/planning.py` |
| CLI | `src/paicli/cli.py` |

## 必须保持的行为

- `Agent.history` 是唯一真实短期上下文；不要再引入一套并行 shortTermMemory。
- 每次 LLM 请求前按消息和工具 schema 的合计估算触发压缩。
- 压缩切割点必须在 user 边界，不能拆开 assistant tool_call 和 tool result。
- 多工具执行最多 4 并发，返回顺序与模型请求顺序一致。
- 文件工具必须经过 `PathGuard`；路径不可逃出项目根。
- 写文件最大 5MB；命令必须经过 `check_command`。
- 写操作默认需要批准，`auto` 也不能绕过策略拒绝。
- 长期记忆只由用户明确要求保存；必须支持查看和删除。
- Provider 消息必须保留 tool call id；DeepSeek/Kimi 的 reasoning history 必须回传。

## 修改联动

- 改 CLI 命令：同步 `cli.py`、README 和测试。
- 改工具：同步 schema、prompt 和测试。
- 改 provider：同步 `.env.example`、README 和序列化测试。
- 改压缩：同步 `context.py` / `compaction.py` 和边界测试。
- 不提交 `.env`、真实 API Key、`__pycache__`、构建产物或用户记忆。
