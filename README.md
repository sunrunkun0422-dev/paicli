# PaiCLI

PaiCLI 是一个纯 Python 的代码仓库 Agent。它保留项目最核心的能力：多模型调用、ReAct 工具循环、Plan+DAG、上下文自动压缩、长期记忆和本地安全策略。

当前实现不依赖第三方运行时库，Python 3.10+ 即可启动。

## 已实现

- OpenAI-compatible 流式客户端
- GLM、DeepSeek、Step、Kimi、FreeLLMAPI、讯飞 MaaS、Agnes provider
- ReAct 循环与最多 4 个并行工具调用
- Plan+DAG 任务拆分、依赖批次执行和结果汇总
- 对真实消息历史的滚动摘要压缩
- 项目级/全局长期记忆，持久化到 `~/.paicli/memory/long_term_memory.json`
- 项目根路径围栏、命令快速拒绝和写操作审批
- 文件读取/写入/精确替换、目录、glob、grep、命令执行、项目创建和记忆工具
- 交互式 CLI 与单次任务模式
- 纯标准库单元测试

## 快速开始

```bash
cp .env.example .env
# 编辑 .env，至少填写一个 API Key

PYTHONPATH=src python3 -m paicli
```

单次执行：

```bash
PYTHONPATH=src python3 -m paicli "分析这个项目的入口"
```

如果环境有 pip，可以安装成命令：

```bash
python3 -m pip install -e .
paicli
```

写操作默认逐次确认。自动批准或全部拒绝：

```bash
PYTHONPATH=src python3 -m paicli --yes
PYTHONPATH=src python3 -m paicli --never-approve
```

## 配置

配置优先级是：进程环境变量 > 项目 `.env` > 用户 `~/.env`。

| Provider | API Key | 默认模型 | 默认窗口 |
|---|---|---|---:|
| GLM | `GLM_API_KEY` | `glm-5.1` | 200k |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` | 1M |
| Step | `STEP_API_KEY` | `step-3.5-flash` | 256k |
| Kimi | `KIMI_API_KEY` / `MOONSHOT_API_KEY` | `kimi-k2.6` | 256k |
| FreeLLMAPI | `FREELLMAPI_API_KEY` | `auto` | 128k |
| 讯飞 MaaS | `XFYUN_MAAS_API_KEY` | `Qwen3.6-35B-A3B` | 128k |
| Agnes | `AGNES_API_KEY` | `agnes-2.0-flash` | 1M |

可以用 `PAICLI_PROVIDER` 指定默认 provider，也可以在 CLI 中执行 `/model <provider>`。

## CLI 命令

```text
/clear                 清空当前对话，保留长期记忆
/compact               立即压缩旧对话
/context               查看下一轮上下文估算
/model <provider>       切换 provider
/plan <goal>            使用 Plan+DAG 执行复杂任务
/memory list            列出记忆
/memory search <query>  搜索记忆
/memory delete <id>     删除记忆
/memory clear           清空当前可见记忆
/exit                   退出
```

## 上下文压缩

Agent 只维护一份真实的 `history`，不再保留与消息历史重复的短期记忆容器。每次模型调用前会估算：

```text
system prompt + conversation history + tool schemas
```

达到下式阈值时，把早期消息摘要成一个 user/assistant 上下文对，同时保留最近 3 个 user 轮次及完整工具调用边界：

```text
window - min(20k, window/4) - min(13k, window/8)
```

`/compact` 跳过阈值并保留最近 1 个 user 轮次。

## 目录

```text
src/paicli/
├── agent.py        ReAct 主循环
├── cli.py          命令行入口
├── compaction.py   消息历史压缩
├── config.py       .env 与 provider 配置
├── context.py      token 估算和窗口策略
├── llm.py          OpenAI-compatible providers
├── memory.py       长期记忆
├── models.py       消息/工具领域模型
├── planning.py     Plan+DAG 执行
├── policy.py       路径和命令策略
├── prompts.py      system prompt 组装
└── tools.py        内置工具注册与并行执行

tests/              标准库 unittest 测试
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
PYTHONPATH=src python3 -m compileall -q src/paicli
```

旧 Java 阶段文档只作为历史参考；当前行为以 Python 代码、[AGENTS.md](AGENTS.md) 和 [docs/architecture.md](docs/architecture.md) 为准。
