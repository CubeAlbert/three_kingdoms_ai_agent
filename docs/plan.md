<!--
READING GUIDE: This file may be large. Read the Table of Contents first, then jump to the
relevant section. Read 100 lines at a time until the section is complete.
-->

# Plan

## Table of Contents

- [1. Milestones](#1-milestones)
- [2. Key Dependencies](#2-key-dependencies)

---

## 1. Milestones

### Milestone 1 — MVP（当前）

- **Expected Output:** CLI 对话系统，用户发一句新三国梗 → RAG 命中 → 子 Agent 三层拼装 → 结构化结果整合返回。完整实现 Channel、LLM Client、RAG、Memory、Orchestrator、首批 3 个子 Agent。
- **Acceptance Criteria:**
  - 端到端验证：发一句梗 → RAG 命中 → 子 Agent 三层拼装 → 结果整合返回
  - 136+ 单元测试通过（LLM Client 已验证）
  - CLI 对话循环可运行
- **Dependencies:** Python 3.11+, openai SDK, ChromaDB, PyYAML

#### Phase 1 子任务

- [x] 1. 项目骨架：pyproject.toml、目录结构、依赖声明
- [x] 2. 配置层：`config/llm.yaml` + `config/settings.yaml` + ConfigLoader
- [x] 3. LLM Client：chat + embed + Action + JSON parser（136 tests pass）
- [x] 4. Channel 层：Channel 抽象 + CliChannel
- [x] 5. Memory 层：MemoryManager 接口 + WindowMemory + LongTermMemory（空）
- [x] 6. RAG 系统：Embedder + SqliteVecStore + Router + memes.yaml
- [ ] 7. 子 Agent：BaseAgent + RecipeAgent + ChatAgent + MediaAgent
- [ ] 8. Orchestrator：双模式 prompt 拼装 + 主循环 + template_render
- [ ] 9. CLI 入口：`main.py` 组装所有模块 + 端到端验证

### Milestone 2 — 增强（后续）

- **Expected Output:** 更多子 Agent、共享模块抽取、配置热加载、Agent 热注册
- **Acceptance Criteria:**
  - 新增子 Agent 只需添加文件 + 注册 meme，不改主流程
  - 聊天模式护栏规则已实现
- **Dependencies:** Milestone 1 完成

#### Phase 2 子任务

- [ ] 更多子 Agent 和梗语料
- [ ] 从子 Agent 中抽取共享模式到 `shared/`
- [ ] 配置热加载
- [ ] 子 Agent 热注册（不改主流程加新 Agent）
- [ ] **聊天模式护栏规则设计**（⚠️ 待讨论）

### Milestone 3 — 多通道（远期）

- **Expected Output:** Web Channel + 微信 Channel + 前端 UI + LongTermMemory 真实实现
- **Acceptance Criteria:**
  - Web Channel（FastAPI + WebSocket + SSE）可运行
  - 微信 Channel（iLink API）可运行
  - LongTermMemory 替换为真实向量数据库实现
- **Dependencies:** Milestone 2 完成

## 2. Key Dependencies

### External

- **DeepSeek API** (`api.deepseek.com`) — chat + embedding，当前使用中。需有效的 `DEEPSEEK_API_KEY` 环境变量。
- **ChromaDB** — ~~向量存储。通过 pip 安装，无外部服务依赖（嵌入式模式）。~~ **已替换为 sqlite-vec**（零依赖 SQLite 向量扩展，~300kB）。
- **qwen2.5:7b (Ollama)** — embedding 备用方案。当前 Ollama 上 qwen2.5:7b 不支持 embedding 端点，待集成验证。

### Cross-cutting

- **json_mode + prompt 联动** — `json_mode=True` 要求 System Prompt 包含 "json" 字样 + 目标 JSON 结构。此项约束跨越 LLM Client、BaseAgent、各子 Agent 三层，需在 BaseAgent 模板中统一保证。
- **梗语料管线** — `meme.md` (人工) → `memes.yaml` (LLM 提取) → ChromaDB (embed)。三个环节间的格式约定必须一致（H1/agent_id、H2/sub_type、列表项/text）。
