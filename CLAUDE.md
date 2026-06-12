# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent chatbot system triggered by **New Three Kingdoms (新三国) memes**. Users chat with a main orchestrator agent; when a meme is detected via RAG semantic matching, the conversation is routed to a specialized sub-agent (e.g., recipe recommendations, medical advice).

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
- **`Router`** (`core/rag/router.py`) — Loads memes from `data/memes.yaml`, embeds them, stores in ChromaDB. On each user message, embeds it and does similarity search. If similarity > threshold, routes to the matching sub-agent.
- **`Orchestrator`** (`core/orchestrator.py`) — Main loop: receive → check meme match → if hit, delegate to sub-agent; if miss, reply with general LLM chat.
- **`MemoryManager`** (`core/memory/`) — `WindowMemory` (deque, N recent turns) for MVP; `LongTermMemory` is a no-op stub with the full interface, called by sub-agents so they're ready when it's implemented.
- **`BaseAgent`** (`agents/base.py`) — Sub-agent contract: `name`, `description`, `handle(ctx: AgentContext) -> AgentResult`. Each sub-agent is an independent module.

**Data flow:** User text → embed → ChromaDB similarity search → matched meme → agent_id → sub-agent.handle() → response merged into chat.

## Tech Stack

- **Python 3.11+**, managed via `pyproject.toml`
- **`openai`** SDK for LLM calls (chat + embedding), provider-agnostic via config
- **ChromaDB** for vector storage (meme embeddings)
- **PyYAML** for config (`config/llm.yaml`, `config/settings.yaml`, `data/memes.yaml`)
- **`rich`** optional for CLI formatting
- **Async throughout** (`async/await`)

## Adding a New Sub-Agent

1. Create `src/three_kingdoms_ai_agent/agents/<name>.py`, subclass `BaseAgent`, implement `handle(ctx)`
2. Add meme phrases to `data/memes.yaml` under the agent's ID
3. Register the agent instance in the orchestrator's agent registry
4. The RAG router automatically picks up new memes on next startup (embeds and stores in ChromaDB)

## Configuration

- `config/llm.yaml` — LLM provider selection (default, base_url, api_key via env var, model names for chat and embedding)
- `config/settings.yaml` — App-level settings (similarity threshold, window size, etc.)
- `data/memes.yaml` — Meme corpus: `agents.<group>.agent` → agent ID, `agents.<group>.memes` → list of trigger phrases

API keys are read from environment variables (e.g., `${DEEPSEEK_API_KEY}`), never hardcoded.

## Documentation

- `docs/plan.md` — Full architecture plan, module design, development phases. **Source of truth for design decisions.**
- `docs/source_description.md` — Data source documentation (stub)
- `docs/terminology.md` — Domain terminology (stub)

## Current State

Project is in **Phase 1 (MVP)** — planning complete, no code written yet. See `docs/plan.md` for the implementation checklist.
