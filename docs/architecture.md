# PaiCLI Python Architecture

## 请求链路

```text
CLI / caller
   → Agent.run(user_input)
   → 更新 system prompt（PAI.md + 相关长期记忆）
   → 估算 messages + tool schemas
   → ConversationCompactor（达到阈值时）
   → LLM chat/completions SSE
      ├─ final content → 返回
      └─ tool_calls → ToolRegistry.execute_many → tool results → 下一轮
```

## 设计决策

### 唯一短期上下文

`Agent.history` 是唯一真实消息历史。长期记忆是独立持久化事实，不承担当前会话的消息副本。这样压缩指标与实际请求不会再次错位。

### 压缩

触发估算包含 system、用户/助手/tool 消息、tool arguments、图片估算和工具 schema。自动压缩保留最近三个 user 边界；手动压缩保留一个。历史只在摘要成功后原子替换。

### 工具

工具以 `ToolSpec + executor` 注册。多个 tool call 使用固定大小线程池并行执行，最终按调用顺序回灌。所有文件路径先经过真实路径解析，避免绝对路径、`..` 和符号链接逃逸。

### Provider

所有 provider 复用标准库 `urllib` 的 OpenAI-compatible SSE 实现，通过 `ProviderSpec` 描述 endpoint、模型、窗口和 reasoning history 差异。

### Plan+DAG

Planner 只生成有限 JSON 任务图；`ExecutionPlan.batches()` 校验未知依赖、自依赖和环。每批独立子 Agent 并行执行，最后由模型汇总真实结果。

## 核心边界

- 没有 MCP、浏览器自动化、RAG 向量库、微信通道或全屏 TUI。
- Token 使用混合字符估算，不等同于 provider tokenizer。
- 命令策略是辅助拒绝层，不是容器/VM 沙箱。
