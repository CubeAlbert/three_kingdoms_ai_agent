# Current State

**Current Phase:** Phase 1 — MVP

**Current Task:** 5. Memory 层 (`core/memory/`)

**Current SubTask:** `core/memory/base.py` — MemoryManager 抽象接口

**Current Blocker:** None

**Next Step:** Create `core/memory/base.py` with `MemoryManager` ABC (add / get_context / store_long_term / recall_long_term). Then implement `core/memory/window.py` (deque, N turns) and `core/memory/long_term.py` (no-op stub).

**Important Decisions:**
1. LLM Client 采用单 provider + 环境变量模式，非多 provider YAML profile
2. `json_mode` 参数由调用方负责 prompt 合规性（含 "json" 字样 + 结构示例）
3. DeepSeek 不支持 `json_schema`，只能通过 `response_format={'type': 'json_object'}` + prompt 示例约束输出
4. 同步 I/O 贯穿 MVP，async 留到 Phase 3
5. 结果整合用模板拼装（确定性），不调 LLM
