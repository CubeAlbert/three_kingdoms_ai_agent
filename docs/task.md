# 任务追踪

## 🔥 当前任务

Phase 1 — MVP 核心链路已闭环（CLI → Orchestrator → RAG → SubAgent → 模板渲染 ✅）

剩余子 Agent：`chat.py` + `media.py`

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
- ✅ `core/config.py` — 配置加载工具（LLMConfig 五环境变量 + Settings + ConfigLoader）

### 3. LLM Client
- ✅ `core/llm/action.py` — ActionType 枚举 + Action dataclass + from_dict() 严格校验
- ✅ `core/llm/parser.py` — 5 层 JSON 修复链 + parse_structured()
- ✅ `core/llm/client.py` — LLMClient（chat / embed / embed_batch）+ ChatResult + LLMError
- ✅ `chat()` 新增 `json_mode` 参数 → 透传 `response_format={'type': 'json_object'}` 给 API，从服务端约束 JSON 输出
- ✅ 验证：136 个单元测试通过 + 3 个集成测试通过（真实 DeepSeek API，含 json_mode 端到端验证）
- ✅ 集成测试独立文件 `tests/core/llm/test_client_integration.py`，不 mock 环境变量
- ✅ 集成验证：embedding provider 调通（`EMBED_*` 环境变量，4 个集成测试通过）

### 4. Channel 层
- ✅ `core/channel/base.py` — Channel 抽象基类 + Message / AgentResponse 数据类
- ✅ `core/channel/cli.py` — CliChannel 实现 (`input()` / `print()`)
- ✅ 验证：CliChannel 收发消息正常

### 5. Memory 层
- ✅ `core/memory/base.py` — MemoryManager 抽象接口
- ✅ `core/memory/window.py` — WindowMemory（`collections.deque`，保留最近 N 轮）
- ✅ `core/memory/long_term.py` — LongTermMemory（空实现，接口完整）
- ✅ 验证：38 个单元测试通过

### 6. RAG 系统
- ✅ `core/rag/embedder.py` — Embedder（LLMClient 薄封装）
- ✅ `core/rag/store.py` — VectorStore ABC + **SqliteVecStore**（sqlite-vec，替代原设计 ChromaDB）
- ✅ `core/rag/router.py` — Router + RouteResult（`from_config()` 工厂 + `route()` 运行时）
- ✅ `data/memes.yaml` — 从 `docs/meme.md` 手动提取（25 条梗，3 个 Agent）
- ✅ `core/config.py` — EmbedConfig 独立 embedding provider（`EMBED_*` 环境变量，fallback `LLM_*`）
- ✅ `core/llm/client.py` — 双 OpenAI client（chat + embed 分离）
- ✅ `embed_batch` 分批发送 + `embed_batch_size` 可配置
- ✅ 验证：21 个单元测试 + 10 个集成测试通过（全量 208 tests pass）
- ✅ `similarity_threshold` 调至 0.55（适配 text-embedding-v4 相似度分布）
- 📝 设计变更：ChromaDB → sqlite-vec（零依赖、~300kB、单 .db 文件）
- ⬜ `scripts/extract_memes.py` — 自动从 `docs/meme.md` 提取生成 `data/memes.yaml`（当前手动维护）
- ✅ `.gitignore` 忽略 `data/memes.yaml`（生成文件，源为 `docs/meme.md`）

### 7. 子 Agent
- ✅ `agents/base.py` — BaseAgent + AgentContext + AgentResult
- ✅ `agents/recipe.py` — RecipeAgent（"吃什么" / "喝什么"）
- ⬜ `agents/chat.py` — ChatAgent（"废话文学" / "哲理名言" / "与实不符"）
- ⬜ `agents/media.py` — MediaAgent（"关羽之歌" / "折棒吐槽"）

### 8. Orchestrator
- ✅ `core/orchestrator.py` — Persona + Chat Rules + 双模式（Hit 模板拼装 / Miss LLM 聊天） + 主循环
- ✅ Persona / Chat Rules / Integration Rules prompt 内容编写
- ✅ `template_render()` — 子 Agent 结构化结果 → 自然语言模板拼装（确定性，per-agent 模板注册）
- ✅ 39 个单元测试通过

### 9. CLI 入口
- ✅ `main.py` — 组装所有模块、启动 CLI 对话循环
- ✅ 端到端测试通过（Hit: 用户 → RAG → 子Agent → 模板拼装 / Miss: 用户 → RAG → LLM 聊天）
- ✅ `scripts/run.bat` — Windows 启动脚本（UTF-8 BOM + CRLF 编码修正 + DEBUG 开关说明）
- ✅ Debug logging — `DEBUG` 环境变量控制 INFO 日志输出（RAG 命中/切换/切换子Agent 全链路可见）

---

## 笔记 / 踩坑记录

### ⏸️ 待讨论：ActionType 扩展与重构
- 当前三种（switch/exit/tool），后续需审视频繁新增
- Action 字段是否需要更通用的 payload/meta
- 校验逻辑是否应外置为 registry 模式
- 与 structured output prompt 模板的联动机制
- 详见 plan 文件 `TODO 1`

### ⏸️ 待讨论：JSON fallback parse 逻辑
- `_light_fix` 的侵略性边界 — 需真实 bad case 积累
- `_extract_first_json_block` 的 `\"` 处理 — 是否换 `raw_decode`
- Step 3/4 顺序优化 — 先 extract 再 fix 可能更安全
- 失败静默返回 None — 是否需要日志/计数器
- 详见 plan 文件 `TODO 2`

### ✅ TODO：json_mode 的 prompt 拼接（已完成）

`json_mode=True` 要求 prompt 中必须包含 "json" 字样 + JSON 结构示例，否则 API 返回 400。这项责任在**调用方**——即 `BaseAgent.handle()` 的三层 prompt 拼装。

**已完成：**

- [x] `agents/base.py` — System Prompt 内嵌 "json" 关键字 + `_JSON_FALLBACK_SUFFIX` 兜底
- [x] `agents/base.py` — `handle()` 调用 `llm.chat()` 时传入 `json_mode=True`
- [x] 各子 Agent 的 `system_prompt` 和 `sub_type_prompts` 遵循 JSON 输出约定（RecipeAgent 已验证）
- [x] Orchestrator 的聊天模式（RAG miss）不开 json_mode（自由对话）
- [x] 验证：41 个 agent 单元测试 + 39 个 orchestrator 单元测试全部通过
