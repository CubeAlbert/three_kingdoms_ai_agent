<!--
READING GUIDE: This file may be large. Read the Table of Contents first, then jump to the
relevant section. Read 100 lines at a time until the section is complete. Do not load the
entire file at once unless it is short.
-->

# Design

## Table of Contents

- [1. Project Description](#1-project-description)
- [2. Architecture Design](#2-architecture-design)
- [3. Project Structure](#3-project-structure)
- [4. Module Design](#4-module-design)
  - [4.1 Channel 抽象层](#41-channel-抽象层)
  - [4.2 LLM 客户端](#42-llm-客户端)
  - [4.3 RAG 梗匹配系统](#43-rag-梗匹配系统)
  - [4.4 记忆管理](#44-记忆管理)
  - [4.5 编排器](#45-编排器)
  - [4.6 子 Agent 基类](#46-子-agent-基类)
  - [4.7 首批子 Agent](#47-首批子-agent)
- [5. Document References & Conventions](#5-document-references--conventions)

---

## 1. Project Description

构建一个基于「新三国梗」的多 Agent 对话系统。用户通过主 Agent 入口进行聊天，当对话中触发特定的新三国梗/台词时，自动路由到对应的子 Agent 执行具体功能。技术栈 Python，LLM 使用 OpenAI 兼容接口，交互层先 CLI 后扩展 Web/微信。

## 2. Architecture Design

### 设计原则：确定性优先

核心理念：**尽可能用确定性逻辑控制流程，LLM 只做必须由它做的事。**

| 环节 | 方式 | 手段 |
|------|------|------|
| RAG 路由 | ✅ 确定性 | 余弦相似度 + 阈值 |
| Agent + SubType 选择 | ✅ 确定性 | Chroma metadata（agent_id, sub_type） |
| 子 Agent Prompt 选择 | ✅ 确定性 | `sub_type → prompt` 字典映射 |
| 子 Agent 内容生成 | ⚠️ LLM | 不可避免，用结构化输出约束 |
| 结果整合呈现 | ✅ 确定性 | 模板拼接，子 Agent 返回结构化数据 |
| 普通聊天 | ⚠️ LLM | 不可避免，加规则护栏（**待讨论**） |

### 核心架构

```
用户 → Channel(CLI) → Orchestrator(主Agent) ──→ SubAgent
         ▲                │        │                │
         │                ▼        ▼                │
         │           LLM Client  RAG Router         │
         │         (OpenAI兼容) (向量相似度)         │
         │                │        │                │
         │                ▼        ▼                │
         └────────── Memory Manager ◄───────────────┘
                    (窗口+长期接口)
```

### 数据流

1. 用户输入 → Channel → Orchestrator
2. Orchestrator 将用户输入送入 RAG Router 做梗语义匹配（**确定性**）
3. 如果命中梗（相似度 > 阈值）→ 从 metadata 取 agent_id + sub_type → 路由到对应 SubAgent
4. SubAgent 根据 sub_type 确定性选择 Sub-Type Prompt → 拼装 System + Sub-Type + User → 一次 LLM 调用 → 返回**结构化结果**
5. Orchestrator 用模板拼装结构化结果 → Channel.send()（**确定性，不调 LLM**）
6. 如果未命中 → Orchestrator 用聊天模式 prompt + LLM 回复

### Prompt 设计

#### 子 Agent Prompt（三层拼装）

```
┌─────────────────────────────┐
│  Agent System Prompt        │  ← 角色人设，全局不变
├─────────────────────────────┤
│  Sub-Type Prompt            │  ← 子类型处理逻辑，确定性映射
├─────────────────────────────┤
│  User Message + Context     │  ← 用户原话 + 对话历史
└─────────────────────────────┘
```

- System Prompt：Agent 的角色定位、说话风格、底线约束
- Sub-Type Prompt：根据 `ctx.sub_type` 确定性选择，定义具体处理逻辑
- 拼装后一次 LLM 调用

#### Orchestrator Prompt（分场景拼装）

Orchestrator 的 LLM 用于两种互斥场景：

```
场景 A：RAG 未命中 → 聊天模式
  prompt = Persona + Chat Rules + User Message

场景 B：RAG 命中 → 结果整合模式
  # RAG 已确定 agent_id + sub_type，不需要 LLM 做路由判断
  result = sub_agent.handle(ctx)    # 子 Agent 返回结构化数据
  prompt = Persona + Integration Rules + result + User Message
```

- **路由硬逻辑归 RAG**（确定性，余弦相似度 + 阈值）
- **聊天/整合软逻辑归 LLM**
- Persona：三国军师人设，贯穿所有场景

> ⚠️ **待讨论**：聊天模式（场景 A）的具体护栏规则，包括话题边界、功能引导策略等，后续单独设计。

### 梗语料结构 (`docs/meme.md`)

三级结构，H1 绑 Agent，H2 提供子类型区分：

```markdown
# 分类名 — agent_id          ← H1: 大分类 + Agent（1:1）

## 子类型1                    ← H2: 子类型，携带额外上下文给 Agent

- "台词1"
- "台词2" → 外部: API名称 / URL

## 子类型2

- "台词3"
```

- H1 = 分类名 + Agent ID，一一对应
- H2 = 子类型，同一 Agent 下可分多个子类型，Agent 根据 `sub_type` 走不同处理分支
- 列表项 = RAG 语料台词，`→ 外部: xxx` 可选
- 不再保留出处、解释等散文元信息
- 人工维护 `meme.md`，后由 LLM 提取生成 `data/memes.yaml`

### 梗知识管线

```
docs/meme.md  ──(LLM extraction)──▶  data/memes.yaml  ──(embed)──▶  ChromaDB
     (human-authored)                        (machine-readable)              (vector store)
                                              with metadata:
                                              agent_id, sub_type, text
```

### 已确认的设计决策

| 决策点 | 选择 | 说明 |
|--------|------|------|
| 交互形式 | CLI 优先 | 通过 Channel 抽象保证未来迁移 Web/微信成本低 |
| 技术栈 | Python | pyproject.toml 管理项目 |
| LLM 后端 | OpenAI 兼容接口 | deepseek/qwen/minimax/ollama 统一通过 base_url 切换 |
| 流式输出 | MVP 不做 | 直接输出完整结果，简化调试 |
| 子 Agent 结构 | 独立代码模块 | 共同功能渐进抽取到 shared/ |
| 触发匹配 | **RAG 语义匹配** | 不同梗表达相同意图（如多条喝酒梗 → 同一个 Agent）|
| 梗语料格式 | H1 + H2 二级结构 | H1 = 分类 + Agent，H2 = 子类型，列表项 = 台词 |
| 会话记忆 | 窗口记忆 + 长期接口（空实现） | MVP 用最近 N 轮，预留长期记忆扩展点 |

## 3. Project Structure

```
three_kingdoms_ai_agent/
├── src/three_kingdoms_ai_agent/
│   ├── __init__.py
│   ├── main.py                    # CLI 入口
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── channel/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Channel 抽象基类
│   │   │   └── cli.py             # CLI 实现
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── client.py          # OpenAI 兼容 LLM 客户端
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py        # Embedding 生成（调用 embedding 模型）
│   │   │   ├── store.py           # 向量存储（ChromaDB）
│   │   │   └── router.py          # 梗匹配路由：embed → search → 命中Agent
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Memory 抽象接口
│   │   │   ├── window.py          # 窗口记忆实现（最近 N 轮）
│   │   │   └── long_term.py       # 长期记忆接口 + 空实现（预留）
│   │   └── orchestrator.py        # 主 Agent 编排器
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # 子 Agent 基类（三层 prompt + 结构化输出）
│   │   ├── recipe.py              # 饮食 Agent（吃什么 / 喝什么）
│   │   ├── chat.py                # 聊天 Agent（废话文学 / 哲理名言 / 与实不符）
│   │   └── media.py               # 媒体 Agent（关羽之歌 / 折棒吐槽）
│   │
│   └── shared/                    # 渐进抽取的共享模块
│       ├── __init__.py
│       └── (按需添加)
│
├── data/
│   └── memes.yaml                 # 梗语料库：梗文本 + 归属Agent
│
├── config/
│   ├── settings.yaml              # 全局配置
│   └── llm.yaml                   # LLM 后端 + Embedding 模型配置
│
├── docs/                          # 已有文档目录
├── scripts/
│   ├── test_embed.ps1              # Embedding API 端点测试
│   └── run.bat                     # Windows 启动脚本（UTF-8 + CRLF）
├── tests/
├── pyproject.toml
└── requirements.txt
```

## 4. Module Design

### 4.1 Channel 抽象层

**Purpose:** I/O 抽象 — Agent 核心永远不直接触碰 stdin/stdout/HTTP，只与 Channel 交互。

**Key Interfaces:**

```python
class Channel(ABC):
    """所有交互通道的抽象基类"""

    @abstractmethod
    async def receive(self) -> Message:
        """接收用户消息，返回标准化 Message"""
        ...

    @abstractmethod
    async def send(self, response: AgentResponse):
        """发送 Agent 响应"""
        ...
```

**Internal Structure:**
- `CliChannel`：`input()` / `print()`，MVP 直接输出
- 未来：`WebChannel`（WebSocket）、`WeChatChannel`（iLink API）
- Agent 核心只依赖 `Channel` 抽象，不感知具体实现

### 4.2 LLM 客户端

**Purpose:** 封装 `openai` Python SDK，统一 OpenAI 兼容接口。同时支持 chat 和 embedding 两种调用。

**Key Interfaces:**
- `chat(messages, system_prompt?, temperature?, json_mode?) -> ChatResult`
- `chat_stream(messages, ...) -> AsyncIterator[str]`（预留）
- `embed(text) -> list[float]`
- `embed_batch(texts) -> list[list[float]]`

**Internal Structure:**
- `core/llm/client.py` — LLMClient 主类
- `core/llm/action.py` — ActionType 枚举 + Action dataclass + from_dict() 严格校验
- `core/llm/parser.py` — 5 层 JSON 修复链 + parse_structured()
- `ChatResult` — 区分结构化 action vs 非结构化对话

**Design Decisions:**
- 单 provider + 环境变量（`LLM_BASE_URL` 等），非多 provider YAML profile
- 同步模式（非 async），MVP 简化
- `json_mode` 参数：为 `True` 时透传 `response_format={'type': 'json_object'}` 给 API，从服务端约束输出为合法 JSON
- DeepSeek 不支持 `json_schema` 类型（OpenAI structured outputs），只能通过 prompt 示例引导字段结构

> 📝 **已实现**: 实际实现与原始 YAML multi-provider 设计有差异——见上方 Design Decisions。详见 `core/llm/` 源码。

#### json_mode 约束与责任

`json_mode=True` 生效有两个前提，**均由调用方负责**：
1. **Prompt 中必须包含 "json" 字样** — 否则 API 返回 400 错误（OpenAI 和 DeepSeek 共同的硬性要求）
2. **Prompt 中应包含目标 JSON 结构示例** — DeepSeek 不支持 `json_schema`，只能通过 prompt 示例引导字段结构

调用链：`BaseAgent.handle()` → `LLMClient.chat(json_mode=True)`，其中 `BaseAgent` 的三层 prompt 拼装（System + Sub-Type + User）必须保证 System Prompt 中包含 "json" 关键字和 Action 结构示例。详见 `agents/base.py` 的 prompt 模板。

#### ⏸️ 待讨论：ActionType 扩展与重构

当前 `ActionType` 三种（`switch` / `exit` / `tool`），随子 Agent 增加需要审视：
- 新增类型：`reply`（结构化回复模板）、`ask`（反问用户）、`chain`（串联多个 action）
- `Action` 字段：是否需要更通用的 `payload` / `meta`，`params` 是否所有 action 都需要
- 校验逻辑：`from_dict()` 当前内聚在 Action，后续是否外置为 registry 模式
- Prompt 联动：新增 type 需同步更新 `prompts/` 下的 structured output 模板

#### ⏸️ 待讨论：JSON fallback parse 逻辑

当前 5 层修复链（直接解析 → 去 fences → 修复引号/逗号/key → brace 提取 → 放弃），需要审查：
- `_light_fix` 侵略性边界：单引号替换在混用引号风格时可能引入新错误
- Brace 提取的 `\"` 处理：当前用 brace counting，是否换 `json.JSONDecoder.raw_decode`
- Step 3/4 顺序：先 fix 再 extract，若文本有多个 `{...}` 且第一个非 JSON，可能误改
- 失败策略：静默返回 `None`，是否需要日志/计数器让调用方感知"接近 JSON 但解析失败"

### 4.3 RAG 梗匹配系统

**Purpose:** 这是触发匹配的核心——解决"不同梗表达相同意图"的问题。

> 📝 **已实现**: 最终采用 **sqlite-vec** 替代原始设计的 ChromaDB。理由：25-200 条梗的规模下 ChromaDB 的 200MB+ 依赖链（onnxruntime / grpcio / kubernetes / opentelemetry 等）完全浪费；sqlite-vec 零依赖、~300kB 单 wheel、单 `.db` 文件持久化。详见 `docs/decision.md`。

**Sub-modules:**

#### Embedder (`core/rag/embedder.py`)
- LLM Client 的薄封装，解耦 Router 与具体 embedding 后端
- `embed(text: str) -> list[float]`
- `embed_batch(texts: list[str]) -> list[list[float]]`

#### Vector Store (`core/rag/store.py`)
- **VectorStore** ABC 定义接口：`add()` / `search()` / `count()` / `clear()`
- **SqliteVecStore** 实现：基于 `sqlite-vec` 的 `vec0` virtual table
  - 向量存在 `meme_vectors` (vec0)，metadata 在 `meme_metadata`，rowid 关联
  - 向量表惰性创建：首次 `add()` 时根据实际向量维度建表
  - `search()` 返回 `list[Match]`（含 `agent_id` / `sub_type` / `text` / `distance` / `similarity`）
  - 阈值过滤在 Router 层做，Store 只负责检索 top_k

#### Router (`core/rag/router.py`)
- `RouteResult` dataclass: `agent_id` / `sub_type` / `meme_text` / `similarity`
- `from_config()` 工厂方法：加载 `data/memes.yaml` → embed 全部 → 入库（幂等：count > 0 则跳过）
- `route(user_text)` 运行时方法：embed → search → 阈值过滤 → RouteResult 或 None
- 路由完全确定性，不需要 LLM 参与

### 4.4 记忆管理

**Purpose:** 管理对话上下文。MVP 使用窗口记忆，预留长期记忆扩展点。

**Key Interfaces (`core/memory/base.py`):**

```python
class MemoryManager(ABC):
    """记忆管理抽象接口"""

    @abstractmethod
    async def add(self, role: str, content: str, metadata: dict = None):
        """添加一条记忆"""
        ...

    @abstractmethod
    async def get_context(self, limit: int = 10) -> list[Message]:
        """获取当前上下文（窗口记忆）"""
        ...

    @abstractmethod
    async def store_long_term(self, key: str, value: any):
        """存储长期记忆 —— MVP 空实现，预留接口"""
        ...

    @abstractmethod
    async def recall_long_term(self, key: str) -> any:
        """召回长期记忆 —— MVP 空实现，预留接口"""
        ...
```

**Internal Structure:**
- `WindowMemory` — `collections.deque`，保留最近 N 轮（默认 10 轮），提供给 Orchestrator 拼接到 LLM 上下文中
- `LongTermMemory` — MVP：**空实现**（方法体 `pass`，返回 None），子 Agent 可以在 `handle()` 中调用，未来可替换为向量数据库 + 摘要策略

### 4.5 编排器

**Purpose:** 主 Agent 的核心循环。两类 Prompt 按场景拼装。

**Key Components:**

```python
# Persona 层：贯穿所有场景的三国军师人设
PERSONA = "你是新三国世界的军师，善于聊天也善于调度人手..."

# 聊天规则：未命中梗时使用（⚠️ 具体护栏规则待后续设计）
CHAT_RULES = "以三国武将的豪迈风格与用户聊天，适当引导用户发现可用功能..."

# 整合规则：命中梗时使用
INTEGRATION_RULES = "将子 Agent 的结构化结果按以下模板呈现..."
```

**主循环：**

```
loop:
  1. msg = await channel.receive()
  2. memory.add("user", msg.text)
  3. route_result = await router.route(msg.text)    # 确定性路由
  4. if route_result:
       # 命中梗 → 调用子 Agent（一次 LLM）
       ctx = AgentContext(
         user_message=msg.text,
         sub_type=route_result.sub_type,    # 确定性，来自 Chroma metadata
         matched_meme=route_result.meme_text,
         history=await memory.get_context(),
         llm=self.llm,
         memory=self.memory,
       )
       structured_result = await route_result.agent.handle(ctx)
       response = self.template_render(INTEGRATION_RULES, structured_result)  # 确定性模板
     else:
       # 未命中 → 聊天模式（一次 LLM）
       history = await memory.get_context()
       response = await llm.chat(history, system_prompt=PERSONA + CHAT_RULES)
  5. memory.add("assistant", response)
  6. await channel.send(response)
```

- 命中时最多 1 次 LLM 调用（子 Agent），结果整合不用 LLM
- 未命中时 1 次 LLM 调用（聊天）
- 路由判断 RAG 确定性完成，不调 LLM

> 📝 **已实现差异**:
> - **同步 I/O**：原始设计使用 `async/await`，实际实现为同步（与 Phase 1 其余模块一致）。
> - **模板渲染**：非单一 `INTEGRATION_RULES` 字符串，而是 `agent_id → callable` 字典。每个子 Agent 一个模板函数（如 `_recipe_template`），构造时注入自定义覆盖，未注册走 `_fallback_template`。
> - **Hit 容错**：子 Agent 未注册 / `handle()` 抛异常 → 自动 fallback 到 Miss 聊天模式，不中断对话。
> - **Debug 日志**：`DEBUG` 环境变量或 `settings.yaml` 的 `debug: true` 控制 `logging.INFO` 级别输出，RAG 命中/未命中、子 Agent 切换全链路可见。

### 4.6 子 Agent 基类

**Purpose:** 子 Agent 合约。每个子 Agent 持有自己的 System Prompt 和 Sub-Type Prompt 字典。

**Key Interfaces:**

```python
class BaseAgent(ABC):
    """子 Agent 基类。每个子 Agent 持有自己的 System Prompt 和 Sub-Type Prompt 字典。"""

    name: str                          # 唯一标识
    description: str                   # 功能描述
    system_prompt: str                 # Agent 级人设 prompt
    sub_type_prompts: dict[str, str]   # sub_type → prompt 确定性映射

    async def handle(self, ctx: AgentContext) -> AgentResult:
        """默认实现：确定性选择 sub prompt → 三层拼装 → 一次 LLM 调用"""
        sub_prompt = self.sub_type_prompts.get(ctx.sub_type, "")
        full_prompt = self.system_prompt + sub_prompt
        messages = [
            {"role": "system", "content": full_prompt},
            *ctx.history,
            {"role": "user", "content": ctx.user_message},
        ]
        raw = await ctx.llm.chat(messages)
        return self.parse_result(raw)   # 子类可覆写，返回结构化数据

    @abstractmethod
    def parse_result(self, raw: str) -> AgentResult:
        """将 LLM 原始输出解析为结构化结果"""
        ...

class AgentContext:
    user_message: str
    sub_type: str              # 子类型标识，来自 RAG metadata（确定性）
    matched_meme: str          # 命中的具体梗文本
    history: list[Message]     # 窗口记忆
    llm: LLMClient
    memory: MemoryManager
```

**Design Decisions:**
- 继承 `BaseAgent`，定义 `system_prompt` 和 `sub_type_prompts` 字典
- 默认 `handle()` 已实现三层拼装流程，子类通常只需覆写 `parse_result()`
- 在 `data/memes.yaml` 中注册梗语料（H2 子类型名 = yaml 中的 sub_type）
- 返回**结构化数据**（dict/Pydantic），供 Orchestrator 模板拼接

### 4.7 首批子 Agent

| Agent ID | 子类型 | 功能 |
|----------|--------|------|
| `recipe_agent` | `吃什么`、`喝什么` | 菜谱/饮品推荐 |
| `chat_agent` | `废话文学`、`哲理名言`、`与实不符` | 三国风格聊天互动 |
| `media_agent` | `关羽之歌`、`折棒吐槽` | 媒体内容检索/播放 |

## 5. Document References & Conventions

**References:**
- `docs/meme.md` — 梗语料主文档，人工维护。结构：`# 分类 — agent_id` → `## 子类型` → `- "台词"`
- `docs/task.md` — 任务追踪，Phase → Task → SubTask 三级结构
- `docs/current.md` — 当前状态快照，每次会话的入口点
- `docs/decision.md` — 决策日志，记录重要设计决策及理由

**Conventions:**
- **确定性优先**: RAG 路由、Agent 选择、Sub-Type prompt 选择、结果整合全部走确定性逻辑；LLM 只做内容生成和开放聊天
- **同步 I/O**: MVP 全同步，async 在 Phase 3 再议
- **环境变量存凭证**: API key 从环境变量读取，永不硬编码
- **模板拼接 > LLM 整合**: 子 Agent 返回结构化数据后，Orchestrator 用模板拼装，不调 LLM
- **三层 Prompt**: System (角色) + Sub-Type (处理逻辑) + User (输入) — 确定性拼装，一次 LLM 调用
- **Channel 抽象**: 核心代码不依赖任何具体 I/O 实现
