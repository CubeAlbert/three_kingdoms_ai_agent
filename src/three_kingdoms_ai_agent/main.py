"""CLI entry point — bootstraps all modules and starts the orchestrator loop.

Usage::

    python -m three_kingdoms_ai_agent.main
"""

from __future__ import annotations

import logging
import sys

from .agents.recipe import RecipeAgent
from .core.channel.cli import CliChannel
from .core.config import ConfigLoader, LLMConfig
from .core.llm.client import LLMClient
from .core.memory.window import WindowMemory
from .core.orchestrator import Orchestrator
from .core.rag.router import Router


def main() -> None:
    """Bootstrap the application and enter the conversation loop."""

    # -- config ----------------------------------------------------------------
    loader = ConfigLoader()
    settings = loader.load_settings()
    llm_opts = loader.load_llm_options()

    # -- logging ---------------------------------------------------------------
    log_level = logging.INFO if settings.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    if settings.debug:
        logger.info("Debug mode ON (set DEBUG=false to disable)")

    # -- LLM config (env vars) --------------------------------------------------
    print("⚙️  加载配置...")
    try:
        llm_cfg = LLMConfig.from_env()
    except Exception as exc:
        print(f"❌ LLM 配置加载失败: {exc}")
        sys.exit(1)

    issues = llm_cfg.validate()
    if issues:
        print("❌ LLM 配置有误，请检查环境变量:")
        for issue in issues:
            print(f"   - {issue}")
        sys.exit(1)

    # Warn about embedding config (non-fatal — chat still works without RAG)
    embed_issues = llm_cfg.embed.validate()
    if embed_issues:
        print("⚠️  Embedding 配置有误（RAG 路由将不可用）:")
        for issue in embed_issues:
            print(f"   - {issue}")

    # -- LLM client ------------------------------------------------------------
    print("🔌 连接 LLM...")
    llm_client = LLMClient(
        config=llm_cfg,
        timeout=int(llm_opts.get("timeout", 60)),
        max_retries=int(llm_opts.get("max_retries", 3)),
    )

    # -- RAG router ------------------------------------------------------------
    print("🧠 加载梗知识库...")
    try:
        router = Router.from_config(llm_client, settings)
        if settings.debug:
            logger.info(
                "RAG initialized: threshold=%.2f top_k=%d db=%s",
                settings.rag.similarity_threshold,
                settings.rag.top_k,
                settings.rag.db_path,
            )
    except Exception as exc:
        print(f"⚠️  RAG 路由初始化失败（将继续以纯聊天模式运行）: {exc}")
        # Create a dummy router that always returns None
        from .core.rag.embedder import Embedder
        from .core.rag.store import SqliteVecStore

        router = Router(
            embedder=Embedder(llm_client),
            store=SqliteVecStore(":memory:"),
            threshold=1.0,  # impossibly high → always miss
        )

    # -- memory ----------------------------------------------------------------
    memory = WindowMemory(window_size=settings.memory.window_size)

    # -- agents ----------------------------------------------------------------
    agents = {
        "recipe_agent": RecipeAgent(),
    }
    if settings.debug:
        logger.info(
            "Registered %d agent(s): %s",
            len(agents),
            ", ".join(f"{a.name} ({a.description})" for a in agents.values()),
        )

    # -- channel + orchestrator ------------------------------------------------
    channel = CliChannel(prompt="🗣️  ")
    orchestrator = Orchestrator(
        channel=channel,
        llm=llm_client,
        router=router,
        memory=memory,
        agents=agents,
    )

    # -- go! -------------------------------------------------------------------
    print("✅ 军师就位，开始对话吧！")
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n👋 军师告退，后会有期！")
    except Exception:
        logging.exception("Unhandled exception in orchestrator loop.")
        print("💥 军师帐中起火……请检查日志后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
