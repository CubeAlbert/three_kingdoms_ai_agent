# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent chatbot system triggered by **New Three Kingdoms (新三国) memes**. Users chat with a main orchestrator agent; when a meme is detected via RAG semantic matching, the conversation is routed to a specialized sub-agent (e.g., recipe recommendations, chat interactions, media retrieval).

## Architecture

```
Channel (CLI) → Orchestrator → RAG Router (ChromaDB) → SubAgent
                    ↓                ↓
               LLM Client      Memory Manager
            (OpenAI-compat)   (window + long-term stub)
```

**Key abstractions:**

- **`Channel`** (`core/channel/base.py`) — I/O abstraction. Agent core never touches stdin/stdout/HTTP directly; it only talks to a `Channel`. Current: `CliChannel`. Future: `WebChannel`, `WeChatChannel`.
- **`LLMClient`** (`core/llm/client.py`) — Unified OpenAI-compatible client. Switches between providers (deepseek, qwen, minimax, ollama) via `config/llm.yaml`. Supports both `chat()` and `embed()`.
- **`Router`** (`core/rag/router.py`) — On startup, loads `data/memes.yaml`, embeds all meme phrases, stores in ChromaDB with metadata (`agent_id`, `sub_type`, original text). On each user message, embeds it and does similarity search. If similarity > threshold, returns `RouteResult` with `agent_id` + `sub_type`. **Deterministic — no LLM involved.**
- **`Orchestrator`** (`core/orchestrator.py`) — Main loop with two prompt modes:
  - **Hit**: RAG routes deterministically → sub-agent generates structured result → Orchestrator template-renders (no LLM for integration)
  - **Miss**: Persona + Chat Rules + LLM for general conversation
  - ⚠️ Chat guardrails TBD — see `docs/plan.md`
- **`MemoryManager`** (`core/memory/`) — `WindowMemory` (deque, N recent turns) for MVP; `LongTermMemory` is a no-op stub with the full interface, called by sub-agents so they're ready when it's implemented.
- **`BaseAgent`** (`agents/base.py`) — Sub-agent contract. Each agent holds:
  - `system_prompt`: role/language style (invariant)
  - `sub_type_prompts`: `dict[str, str]` — deterministic mapping from `sub_type` to processing prompt
  - `handle()`: default implementation does three-layer assembly (System + Sub-Type + User) → one LLM call → `parse_result()` for structured output

**Data flow:**

```
User text → embed → ChromaDB search (deterministic)
  ├── Hit:  agent_id + sub_type from metadata
  │         → sub-agent.handle() → structured result
  │         → Orchestrator template-renders response (deterministic)
  └── Miss: Orchestrator persona + chat rules + LLM → response
```

## Design Principle: Deterministic-First

Prefer deterministic control wherever possible. LLM is only used when necessary:
- ✅ Deterministic: RAG routing, agent selection, sub-prompt selection, result integration
- ⚠️ LLM: content generation inside sub-agents, open-ended chat

## Tech Stack

- **Python 3.11+**, managed via `pyproject.toml`
- **`openai`** SDK for LLM calls (chat + embedding), provider-agnostic via config
- **ChromaDB** for vector storage (meme embeddings), from MVP onwards
- **PyYAML** for config (`config/llm.yaml`, `config/settings.yaml`, `data/memes.yaml`)
- **`rich`** optional for CLI formatting
- **Async throughout** (`async/await`)

## Adding a New Sub-Agent

1. Create `src/three_kingdoms_ai_agent/agents/<name>.py`, subclass `BaseAgent`
2. Define `system_prompt` (role) and `sub_type_prompts` dict (sub_type → processing prompt)
3. Implement `parse_result(raw)` to convert LLM output to structured data
4. Add meme entries to `docs/meme.md`:
   ```markdown
   # 分类名 — agent_id
   
   ## 子类型
   
   - "台词1"
   - "台词2" → 外部: xxx
   ```
5. Run LLM extraction on `meme.md` to regenerate `data/memes.yaml`
6. Register the agent instance in the orchestrator's agent registry

## Meme Knowledge Pipeline

```
docs/meme.md  ──(LLM extraction)──▶  data/memes.yaml  ──(embed)──▶  ChromaDB
     (human-authored)                        (machine-readable)              (vector store)
                                              with metadata:
                                              agent_id, sub_type, text
```

- `meme.md` is the **primary authoring source** — humans write memes here
- Structure: `# 分类 — agent_id` → `## 子类型` → `- "台词"` → `- "台词" → 外部: resource`
- Each meme line is embedded and stored in ChromaDB with its `agent_id` and `sub_type` in metadata
- `memes.yaml` is **auto-generated** — LLM reads meme.md and extracts structured data
- ChromaDB is populated from memes.yaml at startup

## Configuration

- `config/llm.yaml` — LLM provider selection (default, base_url, api_key via env var, model names for chat and embedding)
- `config/settings.yaml` — App-level settings (similarity threshold, window size, etc.)
- `data/memes.yaml` — Auto-generated meme corpus with agent_id, sub_type, and trigger phrases

API keys are read from environment variables (e.g., `${DEEPSEEK_API_KEY}`), never hardcoded.

## Documentation

- `docs/plan.md` — Full architecture plan, module design, development phases. **Source of truth for design decisions.**
- `docs/meme.md` — **Primary meme knowledge base.** Human-authored, structured as `# 分类 — agent_id` → `## 子类型` → flat list of meme lines with optional `→ 外部:` resource annotations.
- `docs/source_description.md` — Data source documentation (stub)

## Current State

Project is in **Phase 1 (MVP)** — planning complete, no code written yet. See `docs/plan.md` for the implementation checklist.
