# 三国梗多 Agent 系统 — 架构设计计划

## Context

构建一个基于「新三国梗」的多 Agent 对话系统。用户通过主 Agent 入口进行聊天，当对话中触发特定的新三国梗/台词时，自动路由到对应的子 Agent 执行具体功能。技术栈 Python，LLM 使用 OpenAI 兼容接口，交互层先 CLI 后扩展 Web/微信。

## 已确认的设计决策

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

### Prompt 设计总览：确定性优先

核心理念：**尽可能用确定性逻辑控制流程，LLM 只做必须由它做的事。**

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

#### 确定性控制边界

| 环节 | 方式 | 手段 |
|------|------|------|
| RAG 路由 | ✅ 确定性 | 余弦相似度 + 阈值 |
| Agent + SubType 选择 | ✅ 确定性 | Chroma metadata（agent_id, sub_type） |
| 子 Agent Prompt 选择 | ✅ 确定性 | `sub_type → prompt` 字典映射 |
| 子 Agent 内容生成 | ⚠️ LLM | 不可避免，用结构化输出约束 |
| 结果整合呈现 | ✅ 确定性 | 模板拼接，子 Agent 返回结构化数据 |
| 普通聊天 | ⚠️ LLM | 不可避免，加规则护栏（**待讨论**） |

---

## 核心架构

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

### 数据流（更新）

1. 用户输入 → Channel → Orchestrator
2. Orchestrator 将用户输入送入 RAG Router 做梗语义匹配（**确定性**）
3. 如果命中梗（相似度 > 阈值）→ 从 metadata 取 agent_id + sub_type → 路由到对应 SubAgent
4. SubAgent 根据 sub_type 确定性选择 Sub-Type Prompt → 拼装 System + Sub-Type + User → 一次 LLM 调用 → 返回**结构化结果**
5. Orchestrator 用模板拼装结构化结果 → Channel.send()（**确定性，不调 LLM**）
6. 如果未命中 → Orchestrator 用聊天模式 prompt + LLM 回复

---

## 项目结构

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
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## 模块设计

### 1. Channel 抽象层 (`core/channel/base.py`)

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

- **CliChannel**：`input()` / `print()`，MVP 直接输出
- 未来：`WebChannel`（WebSocket）、`WeChatChannel`（iLink API）
- Agent 核心只依赖 `Channel` 抽象，不感知具体实现

### 2. LLM 客户端 (`core/llm/client.py`)

- 封装 `openai` Python SDK，统一 OpenAI 兼容接口
- 核心方法：
  - `chat(messages, system_prompt?, temperature?) -> str`
  - `chat_stream(messages, ...) -> AsyncIterator[str]`（预留）
- 配置驱动切换后端：

```yaml
# config/llm.yaml
default: deepseek

providers:
  deepseek:
    base_url: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}
    chat_model: deepseek-chat
    embedding_model: deepseek-embedding

  ollama:
    base_url: http://localhost:11434/v1
    api_key: ollama
    chat_model: qwen2.5:7b
    embedding_model: nomic-embed-text
```

- 同时支持 chat 和 embedding 两种调用

### 3. RAG 梗匹配系统 (`core/rag/`)

这是触发匹配的核心——解决"不同梗表达相同意图"的问题。

**梗语料工作流**：
1. 人工在 `docs/meme.md` 中按结构手写梗
2. LLM 读取 `meme.md`，提取台词列表和触发 Agent 映射，生成 `data/memes.yaml`
3. 系统启动时从 `memes.yaml` 加载 → embed → 存入 ChromaDB

#### Embedder (`core/rag/embedder.py`)

- 调用 LLM Client 的 embedding 接口
- `embed(text: str) -> list[float]`
- `embed_batch(texts: list[str]) -> list[list[float]]`

#### Vector Store (`core/rag/store.py`)

- 使用 **ChromaDB**，从 MVP 开始，支持持久化
- 封装 Chroma collection，提供接口：
  - `add(id, vector, metadata)` — 注册一条梗，metadata 含 `agent_id`、`sub_type`、原始台词
  - `search(vector, top_k, threshold) -> list[Match]` — 相似度搜索
  - `count() -> int` — 已注册梗数量

#### Router (`core/rag/router.py`)

- 启动时：加载 `data/memes.yaml` → embed 所有梗 → 存入 ChromaDB Store
- 运行时：`route(user_text: str) -> Optional[RouteResult]`
  1. embed 用户输入
  2. 在 ChromaDB 中搜索最相似的梗
  3. 如果相似度 > 阈值 → 从 metadata 取出 `agent_id` + `sub_type` + 原始台词，返回 `RouteResult`
  4. 否则返回 None（没有命中梗，普通聊天）
- 路由完全确定性，不需要 LLM 参与

### 4. 记忆管理 (`core/memory/`)

#### 接口 (`core/memory/base.py`)

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

#### WindowMemory (`core/memory/window.py`)

- `collections.deque`，保留最近 N 轮（默认 10 轮）
- 提供给 Orchestrator 拼接到 LLM 上下文中

#### LongTermMemory (`core/memory/long_term.py`)

- MVP：**空实现**（方法体 `pass`，返回 None）
- 子 Agent 可以在 `handle()` 中调用 `store_long_term` / `recall_long_term`
- 未来可替换为向量数据库 + 摘要策略

### 5. 编排器 (`core/orchestrator.py`)

主 Agent 的核心循环。两类 Prompt 按场景拼装：

```python
# Persona 层：贯穿所有场景的三国军师人设
PERSONA = "你是新三国世界的军师，善于聊天也善于调度人手..."

# 聊天规则：未命中梗时使用（⚠️ 具体护栏规则待后续设计）
CHAT_RULES = "以三国武将的豪迈风格与用户聊天，适当引导用户发现可用功能..."

# 整合规则：命中梗时使用
INTEGRATION_RULES = "将子 Agent 的结构化结果按以下模板呈现..."
```

主循环：

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

### 6. 子 Agent 基类 (`agents/base.py`)

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

每个子 Agent：
- 继承 `BaseAgent`，定义 `system_prompt` 和 `sub_type_prompts` 字典
- 默认 `handle()` 已实现三层拼装流程，子类通常只需覆写 `parse_result()`
- 在 `data/memes.yaml` 中注册梗语料（H2 子类型名 = yaml 中的 sub_type）
- 返回**结构化数据**（dict/Pydantic），供 Orchestrator 模板拼接

### 7. 首批子 Agent

| Agent ID | 子类型 | 功能 |
|----------|--------|------|
| `recipe_agent` | `吃什么`、`喝什么` | 菜谱/饮品推荐 |
| `chat_agent` | `废话文学`、`哲理名言`、`与实不符` | 三国风格聊天互动 |
| `media_agent` | `关羽之歌`、`折棒吐槽` | 媒体内容检索/播放 |

---

## 开发阶段

### Phase 1 — MVP（当前）

- [ ] 项目骨架：pyproject.toml、目录结构、依赖声明
- [ ] `config/llm.yaml` + LLM Client（chat + embedding）
- [ ] `core/channel/`：Channel 抽象 + CliChannel
- [ ] `core/rag/`：Embedder + Vector Store（ChromaDB）+ Router
- [ ] `docs/meme.md`：首批梗语料（饮食、聊天、媒体 三个 Agent）
- [ ] `data/memes.yaml`：由 LLM 从 meme.md 提取生成
- [ ] `core/memory/`：MemoryManager 接口 + WindowMemory + LongTermMemory（空）
- [ ] `core/orchestrator.py`：主 Agent 编排逻辑（双模式 prompt 拼装）
- [ ] `agents/base.py`：子 Agent 基类（三层 prompt + 结构化输出）
- [ ] `agents/recipe.py`：RecipeAgent（吃什么 / 喝什么）
- [ ] `agents/chat.py`：ChatAgent（废话文学 / 哲理名言 / 与实不符）
- [ ] `agents/media.py`：MediaAgent（关羽之歌 / 折棒吐槽）
- [ ] `main.py`：CLI 入口，组装所有模块
- [ ] 端到端验证：CLI 对话 → RAG 识别梗 → 子 Agent 三层拼装 → 结构化结果整合

### Phase 2 — 增强（后续）

- [ ] 更多子 Agent 和梗语料
- [ ] 从子 Agent 中抽取共享模式到 `shared/`
- [ ] 配置热加载
- [ ] 子 Agent 热注册（不改主流程加新 Agent）
- [ ] **聊天模式护栏规则设计**（⚠️ 待讨论）

### Phase 3 — 多通道（远期）

- [ ] Web Channel（FastAPI + WebSocket + SSE）
- [ ] 微信 Channel（iLink API）
- [ ] 前端 UI
- [ ] LongTermMemory 真实实现

---

## 关键依赖

```
openai           # LLM 客户端（兼容接口）
pyyaml           # 配置文件解析
chromadb         # 向量存储（ChromaDB）
rich             # CLI 美化（可选）
fastapi + uvicorn # Phase 3 Web 服务（暂不引入）
```
