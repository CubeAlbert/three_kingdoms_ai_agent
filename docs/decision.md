<!--
READING GUIDE: Read the Table of Contents first to find the relevant decision, then jump
directly to that section. No need to load the entire file.
-->

# Decisions

[Chronological log of important project decisions. Each entry gets a sequential number. The most recent decisions should be summarized in `current.md` under Important Decisions.]

## Table of Contents

- [Decision 1 — 技术栈选择](#decision-1--技术栈选择)
- [Decision 2 — LLM Client 架构：单 provider 环境变量 vs 多 provider YAML](#decision-2--llm-client-架构单-provider-环境变量-vs-多-provider-yaml)
- [Decision 3 — 触发匹配：RAG 语义匹配 vs 关键词/规则](#decision-3--触发匹配rag-语义匹配-vs-关键词规则)
- [Decision 4 — 梗语料格式：H1+H2 二级结构](#decision-4--梗语料格式h1h2-二级结构)
- [Decision 5 — 同步 vs 异步 I/O](#decision-5--同步-vs-异步-io)
- [Decision 6 — 流式输出延后](#decision-6--流式输出延后)
- [Decision 7 — json_mode 与 prompt 责任边界](#decision-7--json_mode-与-prompt-责任边界)
- [Decision 8 — RAG 向量存储：ChromaDB → sqlite-vec](#decision-8--rag-向量存储chromadb--sqlite-vec)
- [Decision 9 — Chat 与 Embedding provider 分离](#decision-9--chat-与-embedding-provider-分离)
- [Decision 10 — 结果整合：确定性模板拼装](#decision-10--结果整合确定性模板拼装)
- [Decision 11 — Orchestrator 双模式 + Hit 异常容错](#decision-11--orchestrator-双模式--hit-异常容错)
- [Decision 12 — 对话式 Agent：json_mode=False override](#decision-12--对话式-agentjson_modefalse-override)

---

### Decision 1 — 技术栈选择

**Context:** 项目启动时需要确定编程语言和核心依赖。

**Decision:** Python 3.11+，pyproject.toml 管理项目，openai SDK 调用 LLM，ChromaDB 做向量存储，PyYAML 做配置解析。

**Rationale:** Python 生态在 LLM/向量数据库方面最成熟；openai SDK 的 provider-agnostic 特性支持一键切换后端；ChromaDB 嵌入式模式无需额外服务。

**Alternatives Considered:**
- TypeScript/Node.js — LangChain.js 成熟度不如 Python，且项目无 Web 优先需求
- SQLite + 全文搜索 — 无法做语义匹配，不满足"不同梗表达相同意图"的 RAG 需求

---

### Decision 2 — LLM Client 架构：单 provider 环境变量 vs 多 provider YAML

**Context:** 原始设计 (`config/llm.yaml`) 定义了多 provider profiles（deepseek/ollama/qwen 等），意图通过 YAML 切换。实现阶段发现 MVP 只需一个 provider 切换，多 profile 带来不必要的复杂度。

**Decision:** 采用单 provider + 环境变量模式（`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_EMBEDDING_MODEL`, `LLM_EMBEDDING_API_KEY`），配 YAML 文件仅存操作级配置（timeout, max_retries）。`ConfigLoader` 类负责从环境变量 + YAML 拼装 `LLMConfig`。

**Rationale:** 简化配置路径；环境变量与 12-factor app 兼容；实际使用时切换 provider 只需改 3 个环境变量，不需要编辑 YAML。

**Alternatives Considered:**
- 多 provider YAML profiles — 需要 provider 选择逻辑、profile 热切换、多套凭证管理，MVP 阶段过度设计
- dotenv 文件 — 已有环境变量，不引入额外文件格式

---

### Decision 3 — 触发匹配：RAG 语义匹配 vs 关键词/规则

**Context:** 新三国梗的触发方式。用户可能用不同措辞表达同一个梗（如"军师救我" vs "军师快救我" vs "救我啊军师"）。

**Decision:** RAG 语义匹配（embedding → ChromaDB 相似度搜索 + 阈值），全确定性，不调 LLM 做路由判断。

**Rationale:** 语义匹配覆盖同一意图的多种表达；确定性路由避免 LLM 幻觉导致错误路由；ChromaDB 嵌入式部署零运维。

**Alternatives Considered:**
- 关键词/正则匹配 — 无法覆盖变体表达，维护成本随梗数量线性增长
- LLM 做路由 — 增加一次 LLM 调用的延迟和成本，且可能路由幻觉

---

### Decision 4 — 梗语料格式：H1+H2 二级结构

**Context:** 需要一种人类可维护的梗语料格式，同时机器可解析。

**Decision:** Markdown 二级结构 — H1 = 分类 + Agent ID（1:1 绑定），H2 = 子类型，列表项 = 台词文本。可选 `→ 外部: xxx` 标注外部资源。人工维护 `docs/meme.md`，LLM 提取生成 `data/memes.yaml`，启动时 embed 入 ChromaDB。

**Rationale:** Markdown 人类友好，二级结构提供 Agent 和子类型的天然分组；LLM 提取步骤自动化转换，减少人工维护 YAML 的错误率。

**Alternatives Considered:**
- 纯 YAML 手写 — 结构严格但书写体验差，嵌套层级容易出错
- JSON — 比 YAML 更严格，不适合人类频繁编辑

---

### Decision 5 — 同步 vs 异步 I/O

**Context:** Python async 提供更好的并发性能，但增加代码复杂度和调试难度。

**Decision:** MVP 全程同步 I/O。async 留到 Phase 3（多通道阶段）再评估。

**Rationale:** MVP 是单用户 CLI，同步足够；调试同步代码远简单于 async；`openai` SDK 同步 API 无需 event loop。

**Alternatives Considered:**
- async 全链路 — Phase 3 Web/微信通道确实需要，但 MVP 阶段引入 async 会增加约 30% 的样板代码

---

### Decision 6 — 流式输出延后

**Context:** LLM 流式输出可改善用户体验（首 token 延迟更低），但增加响应解析复杂度。

**Decision:** MVP 不做流式输出，直接输出完整结果，简化调试。`chat_stream` 接口预留但暂不实现。

**Rationale:** CLI 场景下流式输出收益有限；结构化输出（JSON）流式解析需要增量 parser，实现成本高。

**Alternatives Considered:**
- 流式 + 增量 JSON parser — 体验好但实现复杂，MVP 阶段优先级低

---

### Decision 7 — json_mode 与 prompt 责任边界

**Context:** `json_mode=True` 调用 `response_format={'type': 'json_object'}` 要求 prompt 中必须含 "json" 字样，否则 API 返回 400；DeepSeek 不支持 `json_schema`，field 结构只能通过 prompt 示例引导。

**Decision:** 此责任归调用方（`BaseAgent.handle()` → 各子 Agent 的 prompt 模板），不在 `LLMClient.chat()` 内部处理。三层 prompt 拼装时必须保证 System Prompt 包含 "json" 关键字和目标 JSON 结构示例。

**Rationale:** 关注点分离 — LLMClient 是通用 HTTP 客户端，不应包含业务 prompt 逻辑；各 Agent 的 JSON schema 不同，由 Agent 自己负责最合理。

**Alternatives Considered:**
- LLMClient 自动注入 JSON 提示 — 会与 Agent 的 role prompt 冲突，且无法知道目标 schema
- 放弃 json_mode，纯靠 parser 修复 — parser 是兜底，服务端约束能显著提高 JSON 输出率

---

### Decision 8 — RAG 向量存储：ChromaDB → sqlite-vec

**Context:** 原始设计使用 ChromaDB 做向量存储。实际梗语料规模仅 25 条（未来也不会超过 200 条），ChromaDB 的 200MB+ 依赖链（onnxruntime / grpcio / kubernetes / opentelemetry 等）在此规模下完全过度。实现前对替代方案做了系统调研。

**Decision:** 采用 **sqlite-vec**（Alex Garcia 开发，MIT 协议，Mozilla Firefox 已内置）。VectorStore ABC 定义接口，SqliteVecStore 实现，未来换后端只需改 store.py。

**Rationale:**
- 安装体积：~300kB（单一预编译 wheel）vs ChromaDB 200-400MB
- 零传递依赖 — 仅需 Python stdlib `sqlite3`
- 单 `.db` 文件持久化，调试友好（任意 SQLite GUI 可打开）
- 25-200 向量规模下，暴力搜索已亚毫秒级，ChromaDB 的 HNSW 索引无收益
- 可与 SQLite FTS5 组合做混合搜索（向量 + 关键词），未来扩展空间大

**Alternatives Considered:**
- ChromaDB（原设计）— 在 25 条规模下过度设计，已知内存泄漏 bug
- numpy 暴力搜索 — 更简单但需手写持久化，不如 sqlite-vec 开箱即用
- FAISS + SQLite — FAISS 构建依赖复杂，跨平台分发差

---

### Decision 9 — Chat 与 Embedding provider 分离

**Context:** DeepSeek API 不支持 `/v1/embeddings` 端点（经 PowerShell 脚本实测确认 404）。Chat 走 DeepSeek，但 embedding 需要另一个 provider（如阿里云 DashScope text-embedding-v4）。

**Decision:** 新增 `EmbedConfig` 数据类，读取独立的 `EMBED_*` 环境变量（`EMBED_BASE_URL` / `EMBED_API_KEY` / `EMBED_MODEL` / `EMBED_AUTH_ENABLED`），未设时 fallback 到对应的 `LLM_*` 值。`LLMClient.__init__` 创建两个 `OpenAI` 实例：`self._client`（chat）和 `self._embed_client`（embedding）。

**Rationale:**
- 关注点分离 — chat 和 embedding 是不同的后端需求
- fallback 机制保持向后兼容（单 provider 场景无需额外配置）
- `LLMConfig.embed` 属性对上层透明，Router 无需感知

**Alternatives Considered:**
- 单 client + 运行时切换 base_url — Openai SDK 不支持 per-request 切换
- 全部切到支持 embedding 的 provider — 限制 provider 选择自由
- 本地 Ollama 做 embedding — 增加运维复杂度，且用户已有 cloud embedding provider

---

### Decision 10 — 结果整合：确定性模板拼装

**Context:** RAG 命中后，子 Agent 返回结构化结果（JSON）。原始设计中曾考虑由 LLM 将子 Agent 结果整合为自然语言回复。实际实现中发现子 Agent 的输出 schema 是已知的、有限的，LLM 整合引入不必要的延迟和不确定性。

**Decision:** 子 Agent 返回 `AgentResult`（结构化数据），由 Orchestrator 的 per-agent 模板 callable 做确定性字符串拼装。不再调用 LLM 做结果整合。

**Rationale:**
- 确定性输出 — 同一结构化输入始终产生一致的回复，无 LLM 幻觉风险
- 零延迟 — 模板拼装是纯字符串操作，无需额外 API 调用
- 分离关注点 — 子 Agent 负责内容生成（LLM），Orchestrator 负责"怎么说"（模板）
- 可定制 — 模板通过 `templates` 参数注入，调用方可覆盖默认模板

**Alternatives Considered:**
- LLM 整合子 Agent 结果 — 增加一次 LLM 调用（延迟 + 成本），且可能曲解子 Agent 的输出
- 子 Agent 直接返回自然语言 — 失去了结构化数据的灵活性，模板无法按字段组织回复

---

### Decision 11 — Orchestrator 双模式 + Hit 异常容错

**Context:** 用户消息可能命中 RAG（需要子 Agent 处理）也可能不命中（普通闲聊）。同时子 Agent 可能未注册、崩溃或返回解析失败的结果，需要保证对话不中断。

**Decision:** Orchestrator 维护两种处理路径：

- **Hit 路径**：RAG 路由 → 子 Agent.handle() → 模板拼装 → 回复。一次 LLM 调用（子 Agent 内部）。
- **Miss 路径**：Persona + Chat Rules + LLM（json_mode=False）→ 回复。一次 LLM 调用。

**异常容错**：Hit 路径中若 agent_id 未注册、或 `agent.handle()` 抛出异常，自动 fallback 到 Miss 路径（聊天模式），不中断对话循环。

**Rationale:**
- 鲁棒性 — 子 Agent 故障不影响整体对话可用性
- 调试友好 — 异常通过 logger.exception 记录完整 traceback，同时用户看到的是自然降级回复
- 未注册 agent 通常是配置问题（memes.yaml 有 agent_id 但 main.py 未注册），fallback 让问题可在线修复

**Alternatives Considered:**
- Hit 失败直接报错退出 — 用户体验差，一个子 Agent 问题导致整个系统不可用
- Hit 失败返回固定错误消息 — 不如 fallback 到聊天模式自然

---

### Decision 12 — 对话式 Agent：json_mode=False override

**Context:** ChatAgent 和 MediaAgent 是纯对话式 Agent，不需要结构化 JSON 输出。但 BaseAgent 的默认 `handle()` 使用 `json_mode=True`（为 RecipeAgent 等结构化 Agent 设计）。

**Decision:** ChatAgent 和 MediaAgent override `handle()` 方法，使用 `json_mode=False` 调用 LLM。`data` 字段仍保持为 `{"response": raw_content}` 字典，供模板统一处理。模板（`_chat_template` / `_media_template`）直接透传 LLM 回复文本，不做军师角色包装（对话式 Agent 自身已在角色中）。

**Rationale:**
- 对话质量 — `json_mode=False` 允许 LLM 自由输出自然语言，不受 JSON 格式约束，回复更生动
- 接口一致 — 仍返回 `AgentResult`（而非裸字符串），Orchestrator 的模板渲染路径不变
- 扩展性 — 未来 MediaAgent 需要工具调用（打开链接、播放音乐）时，可在 `handle()` 内自由扩展

**Alternatives Considered:**
- 强制 json_mode=True + `{"response": "..."}` — 约束 LLM 输出 JSON 会降低对话自然度，且 prompt 中需要"撒谎"说输出是 JSON
- 为对话式 Agent 新建基类 — 过度设计，仅两个 Agent 且 override 一个方法足够
