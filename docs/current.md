# Current State

**Current Phase:** Phase 1 — MVP（核心链路已闭环 ✅）

**Current Task:** 7. 子 Agent ✅ 全部完成（recipe + chat + media）

**Current SubTask:** 全部子 Agent 已实现（recipe / chat / media），均为对话式

**Current Blocker:** None

**Next Step:** 重新端到端验证（JSON fallback 已修复）：启动 CLI，测试 chat_agent / media_agent 的 RAG 命中 + 对话流程，确认输出纯文本非 JSON。

**Important Decisions:**
1. LLM Client 采用单 provider + 环境变量模式，非多 provider YAML profile
2. `json_mode` 参数由调用方负责 prompt 合规性（含 "json" 字样 + 结构示例）
3. DeepSeek 不支持 `json_schema`，只能通过 `response_format={'type': 'json_object'}` + prompt 示例约束输出
4. 同步 I/O 贯穿 MVP，async 留到 Phase 3
5. 结果整合用模板拼装（确定性），不调 LLM
6. RAG 向量存储后端选定 sqlite-vec（替代原设计 ChromaDB），零依赖、单 .db 文件、~300kB
7. Chat 和 Embedding 使用独立 provider（`LLM_*` vs `EMBED_*` 环境变量），因 DeepSeek 不提供 embedding API
8. 相似度阈值默认 0.55（text-embedding-v4 模型下精确匹配 ~0.99，变体匹配 ~0.55-0.60）
9. Orchestrator 双模式：Hit → 子Agent（1次LLM）+ 模板拼装（确定性）；Miss → Persona + Chat Rules + LLM（1次LLM, json_mode=False）
10. 模板注册机制：per-agent 模板 callable（agent_id → fn），Orchestrator 构造时注入，未注册 fallback 到通用模板
11. Hit 路径异常容错：子Agent 崩溃 / 未注册 → 自动 fallback 到 Miss 聊天模式
12. Debug 日志：`DEBUG` 环境变量（`true`/`1`）控制 INFO 级别日志，输出 RAG 命中/切换/子Agent 切换全链路信息；`settings.yaml` 的 `debug: true` 等效
13. 对话式 Agent（chat/media）override `handle()` 使用 `json_mode=False`，非 BaseAgent 默认的 `json_mode=True`；模板直接透传 LLM 回复，不做军师包装
