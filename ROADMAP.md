# PaiCLI Python Roadmap

## 已交付

- 纯 Python 工程和 CLI
- 多 Provider OpenAI-compatible SSE 客户端
- ReAct 工具循环与并行工具
- Plan+DAG 执行
- 统一消息历史和自动/手动压缩
- 长期记忆
- 路径、命令和审批策略
- 标准库测试

## 下一步

- 工具调用审计 JSONL
- 会话导入/导出
- 图片输入预处理
- MCP stdio / Streamable HTTP（作为可选扩展，不进入核心依赖）
- 更精确的 provider tokenizer 适配
- 压缩失败时的确定性紧急裁剪

## 非目标

- 恢复 Java/JLine/Lanterna 代码
- 微信通道和营销演示代码进入 Agent 核心
- 强制依赖大型框架
