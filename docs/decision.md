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
