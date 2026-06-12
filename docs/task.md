# 任务追踪

## 🔥 当前任务

Phase 1.3 — LLM Client (`core/llm/client.py`): chat() + embed()

---

## 状态图例

| 符号 | 含义 |
|------|------|
| ⬜ | pending — 尚未开始 |
| 🔄 | in_progress — 正在做 |
| ✅ | done — 已完成 |
| ⏸️ | blocked — 被阻塞（注明原因） |

---

## Phase 1 — MVP

### 1. 项目骨架
- ✅ 创建 `pyproject.toml` + 依赖声明
- ✅ 创建完整目录结构 (`src/three_kingdoms_ai_agent/`, `config/`, `data/`, `tests/`)
- ✅ 创建 `requirements.txt`
- ✅ `pip install -e .` 可安装验证

### 2. 配置层
- ✅ `config/llm.yaml` — LLM provider 操作配置（timeout, max_retries），凭证从环境变量读取
- ✅ `config/settings.yaml` — 全局 settings（相似度阈值、窗口大小等）
- ✅ `core/config.py` — 配置加载工具（LLMConfig 四环境变量 + Settings + ConfigLoader）

### 3. LLM Client
- ⬜ `core/llm/client.py` — `__init__()` 初始化（读配置、选 provider）
- ⬜ `core/llm/client.py` — `chat(messages, system_prompt?, temperature?) -> str`
- ⬜ `core/llm/client.py` — `embed(text) -> list[float]` / `embed_batch(texts) -> list[list[float]]`
- ⬜ 验证：调通至少一个 provider 的 chat + embed

### 4. Channel 层
- ⬜ `core/channel/base.py` — Channel 抽象基类 + Message / AgentResponse 数据类
- ⬜ `core/channel/cli.py` — CliChannel 实现 (`input()` / `print()`)
- ⬜ 验证：CliChannel 收发消息正常

### 5. Memory 层
- ⬜ `core/memory/base.py` — MemoryManager 抽象接口
- ⬜ `core/memory/window.py` — WindowMemory（`collections.deque`，保留最近 N 轮）
- ⬜ `core/memory/long_term.py` — LongTermMemory（空实现，接口完整）

### 6. RAG 系统
- ⬜ `core/rag/embedder.py` — Embedder（封装 `llm.embed()`）
- ⬜ `core/rag/store.py` — ChromaDB VectorStore（`add`, `search`, `count`）
- ⬜ `core/rag/router.py` — Router（启动加载 memes.yaml → embed → store；运行时 `route(user_text) -> RouteResult | None`）
- ⬜ `data/memes.yaml` — 从 `docs/meme.md` 提取生成（LLM 辅助或手动）
- ⬜ 验证：RAG 搜索能正确匹配梗文本

### 7. 子 Agent
- ⬜ `agents/base.py` — BaseAgent + AgentContext + AgentResult
- ⬜ `agents/recipe.py` — RecipeAgent（"吃什么" / "喝什么"）
- ⬜ `agents/chat.py` — ChatAgent（"废话文学" / "哲理名言" / "与实不符"）
- ⬜ `agents/media.py` — MediaAgent（"关羽之歌" / "折棒吐槽"）

### 8. Orchestrator
- ⬜ `core/orchestrator.py` — Persona 定义 + 双模式 prompt 拼装 + 主循环
- ⬜ Persona / Chat Rules / Integration Rules 的 prompt 内容编写
- ⬜ `template_render()` — 子 Agent 结构化结果 → 自然语言模板拼装（确定性）

### 9. CLI 入口
- ⬜ `main.py` — 组装所有模块、启动 CLI 对话循环
- ⬜ 端到端验证：发一句梗 → RAG 命中 → 子 Agent 三层拼装 → 结果整合返回

---

## 笔记 / 踩坑记录

<!-- 开发过程中发现的问题、备忘、临时决策写在这里 -->
