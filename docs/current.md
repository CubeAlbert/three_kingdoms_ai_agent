# Current State

**Current Phase:** Phase 1 — MVP

**Current Task:** 7. 子 Agent (`agents/base.py` + `recipe.py` + `chat.py` + `media.py`)

**Current SubTask:** `agents/base.py` — BaseAgent + AgentContext + AgentResult

**Current Blocker:** None

**Next Step:** Implement `agents/base.py` (BaseAgent with three-layer prompt assembly + json_mode support), then `agents/recipe.py`, `agents/chat.py`, `agents/media.py`.

**Important Decisions:**
1. LLM Client 采用单 provider + 环境变量模式，非多 provider YAML profile
2. `json_mode` 参数由调用方负责 prompt 合规性（含 "json" 字样 + 结构示例）
3. DeepSeek 不支持 `json_schema`，只能通过 `response_format={'type': 'json_object'}` + prompt 示例约束输出
4. 同步 I/O 贯穿 MVP，async 留到 Phase 3
5. 结果整合用模板拼装（确定性），不调 LLM
6. RAG 向量存储后端选定 sqlite-vec（替代原设计 ChromaDB），零依赖、单 .db 文件、~300kB
7. Chat 和 Embedding 使用独立 provider（`LLM_*` vs `EMBED_*` 环境变量），因 DeepSeek 不提供 embedding API
8. 相似度阈值默认 0.55（text-embedding-v4 模型下精确匹配 ~0.99，变体匹配 ~0.55-0.60）
