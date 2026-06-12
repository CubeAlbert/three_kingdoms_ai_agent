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
| 会话记忆 | 窗口记忆 + 长期接口（空实现） | MVP 用最近 N 轮，预留长期记忆扩展点 |

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

### 数据流

1. 用户输入 → Channel → Orchestrator
2. Orchestrator 调用 LLM 理解意图（同时做一般对话）
3. Orchestrator 将用户输入送入 RAG Router 做梗语义匹配
4. 如果命中梗（相似度 > 阈值）→ 路由到对应 SubAgent.handle()
5. SubAgent 可调用 LLM Client（自己的 prompt）、可读写 Memory
6. 结果返回 Orchestrator → 融合到对话 → Channel.send()

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
│   │   ├── base.py                # 子 Agent 基类
│   │   ├── recipe.py              # "是啊，吃什么" → 菜谱推荐
│   │   └── medical.py             # "医死的人越多..." → 疾病建议
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

#### 梗语料库 (`data/memes.yaml`)

```yaml
# 每个 agent 下挂多条梗文本，语义相近的梗指向同一个 agent
agents:
  drinking:
    agent: drinking_agent
    memes:
      - "当浮一大白"
      - "这美酒包治百病"
      - "四五十杯怎么够呢？"
      - "来来来，满饮此杯"
      - "今天不醉不归"

  eating:
    agent: recipe_agent
    memes:
      - "是啊，吃什么"
      - "今晚吃什么"
      - "有什么好吃的推荐"
      - "肚子饿了，弄点吃的"

  medical:
    agent: medical_agent
    memes:
      - "医死的人越多，医术越高明"
      - "我这头疼脑热的"
      - "有点不舒服"
      - "帮我看看这是什么病"
```

#### Embedder (`core/rag/embedder.py`)

- 调用 LLM Client 的 embedding 接口
- `embed(text: str) -> list[float]`
- `embed_batch(texts: list[str]) -> list[list[float]]`

#### Vector Store (`core/rag/store.py`)

- 使用 **ChromaDB** 作为向量存储（从 MVP 阶段开始）
- 封装 Chroma collection，提供简洁接口：
  - `add(id, vector, metadata)` — 注册一条梗
  - `search(vector, top_k, threshold) -> list[Match]` — 相似度搜索
  - `count() -> int` — 已注册梗数量
- Chroma 支持持久化，免去后续迁移成本
- metadata 中存储 `agent_id` 和原始梗文本，命中后直接路由

#### Router (`core/rag/router.py`)

- 启动时：加载 `data/memes.yaml` → embed 所有梗 → 存入 Chroma Vector Store
- 运行时：`route(user_text: str) -> Optional[RouteResult]`
  1. embed 用户输入
  2. 在 Chroma Store 中搜索最相似的梗
  3. 如果相似度 > 阈值 → 从 metadata 取出 agent_id，返回对应的 Agent
  4. 否则返回 None（没有命中梗，普通聊天）

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

主 Agent 的核心循环：

```
loop:
  1. msg = await channel.receive()
  2. memory.add("user", msg.text)
  3. route_result = await router.route(msg.text)
  4. if route_result:
       # 命中梗 → 调用子 Agent
       context = build_context(msg, memory, route_result)
       result = await route_result.agent.handle(context)
       response = format_agent_result(result)
     else:
       # 普通聊天 → 主 Agent 用 LLM 回复
       history = await memory.get_context()
       response = await llm.chat(history, system_prompt=MAIN_SYSTEM_PROMPT)
  5. memory.add("assistant", response)
  6. await channel.send(response)
```

### 6. 子 Agent 基类 (`agents/base.py`)

```python
class BaseAgent(ABC):
    """子 Agent 基类"""

    name: str              # 唯一标识，如 "recipe_agent"
    description: str       # Agent 功能描述（给主 Agent 的 LLM 参考）

    @abstractmethod
    async def handle(self, ctx: AgentContext) -> AgentResult:
        """
        处理被触发后的业务逻辑。
        ctx 包含：用户消息、对话历史、LLM Client、Memory Manager
        """
        ...

class AgentContext:
    user_message: str
    matched_meme: str          # 命中的具体梗文本
    history: list[Message]     # 窗口记忆
    llm: LLMClient             # 子 Agent 可自行调用 LLM
    memory: MemoryManager      # 可读写记忆
```

每个子 Agent：
- 继承 `BaseAgent`，实现 `handle()`
- 在 `data/memes.yaml` 中注册自己的梗语料
- 可调用 `ctx.llm.chat()` 做自己的推理
- 可调用 `ctx.memory.store_long_term()` 存长期信息（MVP 是空操作）

### 7. 首批子 Agent

| Agent ID | 触发梗示例 | 功能 | handle 逻辑 |
|----------|-----------|------|------------|
| `recipe_agent` | "是啊，吃什么"、"今晚吃什么" | 推荐菜谱 | LLM + 菜谱知识生成推荐 |
| `medical_agent` | "医死的人越多，医术越高明"、"头疼脑热" | 简单疾病居家护理 | LLM + 疾病知识，限制仅常见轻症 |

---

## 开发阶段

### Phase 1 — MVP（当前）

- [ ] 项目骨架：pyproject.toml、目录结构、依赖声明
- [ ] `config/llm.yaml` + LLM Client（chat + embedding）
- [ ] `core/channel/`：Channel 抽象 + CliChannel
- [ ] `core/rag/`：Embedder + Vector Store（内存）+ Router
- [ ] `data/memes.yaml`：梗语料库（首批：eating + medical 两个 agent 的梗）
- [ ] `core/memory/`：MemoryManager 接口 + WindowMemory + LongTermMemory（空）
- [ ] `core/orchestrator.py`：主 Agent 编排逻辑
- [ ] `agents/base.py`：子 Agent 基类 + AgentContext
- [ ] `agents/recipe.py`：RecipeAgent
- [ ] `agents/medical.py`：MedicalAgent
- [ ] `main.py`：CLI 入口，组装所有模块
- [ ] 端到端验证：CLI 对话 → 识别梗 → 路由到子 Agent → 返回结果

### Phase 2 — 增强（后续）

- [ ] 更多子 Agent 和梗语料
- [ ] 从子 Agent 中抽取共享模式到 `shared/`
- [ ] 配置热加载
- [ ] 子 Agent 热注册（不改主流程加新 Agent）

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
chromadb         # 向量存储
rich             # CLI 美化（可选）
fastapi + uvicorn # Phase 3 Web 服务（暂不引入）
```
