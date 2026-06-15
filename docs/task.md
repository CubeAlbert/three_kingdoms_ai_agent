# 任务追踪

## 🔥 当前任务

Phase 1.4 — Channel 层 (`core/channel/base.py` + `cli.py`)

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
- ⏸️ 待集成验证：设好环境变量后调通至少一个 provider 的 embed（当前 qwen2.5:7b 不支持 embedding 端点）

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

### ⬜ TODO：json_mode 的 prompt 拼接

`json_mode=True` 要求 prompt 中必须包含 "json" 字样 + JSON 结构示例，否则 API 返回 400。这项责任在**调用方**——即 `BaseAgent.handle()` 的三层 prompt 拼装。

**待做：**

- [ ] `agents/base.py` — System Prompt 模板需内嵌 "json" 关键字和目标 Action JSON 结构示例
- [ ] `agents/base.py` — `handle()` 调用 `llm.chat()` 时传入 `json_mode=True`
- [ ] 各子 Agent 的 `system_prompt` 和 `sub_type_prompts` 需遵循 JSON 输出约定
- [ ] Orchestrator 的聊天模式（RAG miss）**不**开 json_mode（自由对话）
- [ ] 验证：`LLMClient.chat(json_mode=True)` + 合规 prompt → API 不报 400 → 返回结构化 Action

**为什么需要这个：** DeepSeek 不支持 `json_schema` 类型，只能通过 `response_format={'type': 'json_object'}` + prompt 示例来约束输出结构。没有 prompt 配合，json_mode 形同虚设。
