# Current State

**Current Phase:** Phase 1 — MVP

**Current Task:** 6. RAG 系统 (`core/rag/`)

**Current SubTask:** `core/rag/embedder.py` — Embedder 封装

**Current Blocker:** None

**Next Step:** Create `core/rag/embedder.py` (encapsulate `llm.embed()`), then `core/rag/store.py` (ChromaDB VectorStore), `core/rag/router.py` (startup load + runtime route), and `data/memes.yaml`.

**Important Decisions:**
1. LLM Client 采用单 provider + 环境变量模式，非多 provider YAML profile
2. `json_mode` 参数由调用方负责 prompt 合规性（含 "json" 字样 + 结构示例）
3. DeepSeek 不支持 `json_schema`，只能通过 `response_format={'type': 'json_object'}` + prompt 示例约束输出
4. 同步 I/O 贯穿 MVP，async 留到 Phase 3
5. 结果整合用模板拼装（确定性），不调 LLM
