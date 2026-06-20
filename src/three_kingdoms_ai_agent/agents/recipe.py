"""RecipeAgent — Three Kingdoms culinary advisor for food and drink.

Triggered when the user drops a "吃什么" or "喝什么" meme.  Responds with
a structured recommendation in the voice of a Three Kingdoms army cook.
"""

from __future__ import annotations

import json

from .base import AgentContext, AgentResult, BaseAgent


class RecipeAgent(BaseAgent):
    """Recommend dishes and drinks in the style of a Three Kingdoms army cook.

    Sub-types
    ---------
    ``"吃什么"``
        Food recommendation — a specific dish with Three Kingdoms flair.
    ``"喝什么"``
        Drink recommendation — a heroic wine / spirit suggestion.
    """

    name = "recipe_agent"
    description = "以三国随军厨子身份推荐美食与美酒"

    # -- prompts ---------------------------------------------------------------

    system_prompt = (
        "你是新三国的随军厨子，跟随大军征战多年，深谙天下美食与美酒之道。\n"
        "当有人问起\"吃什么\"或\"喝什么\"时，你要以三国角色的口吻，\n"
        "推荐一道美食或美酒。语气要豪迈痛快，如关云长饮酒、张翼德啖肉。\n"
        "\n"
        "你必须以 JSON 格式返回结果（json），格式如下：\n"
        '{"name": "菜名或酒名", "reason": "推荐理由（三国风格豪迈语气）", '
        '"description": "详细描述（生动形象，融入三国典故或角色语气）", '
        '"suggestion": "食用或饮用建议"}'
    )

    sub_type_prompts: dict[str, str] = {
        "吃什么": (
            "用户问今天吃什么。请根据三国背景推荐一道菜肴。\n"
            "要具体、有菜名、有做法简述，符合三国军旅风格。\n"
            "推荐要有名有姓的菜，配上豪迈的三国风格描述。"
        ),
        "喝什么": (
            "用户问喝什么。请根据三国背景推荐一种美酒。\n"
            "要豪迈、痛快，体现三国武将\"当浮一大白\"的饮酒气概。\n"
            "推荐要有具体的酒名，配上英雄豪杰般的描述。"
        ),
    }

    # -- parse -----------------------------------------------------------------

    def parse_result(self, raw_content: str, ctx: AgentContext) -> AgentResult:
        """Parse the LLM's JSON output into a recipe :class:`AgentResult`.

        Expected JSON schema::

            {
                "name": "红烧肉",
                "reason": "此肉色如琥珀，关将军当浮一大白",
                "description": "选用上等五花，文火慢炖...",
                "suggestion": "配上一壶烈酒，方显英雄本色"
            }
        """
        try:
            data = json.loads(raw_content)
        except (json.JSONDecodeError, ValueError):
            return AgentResult(
                agent_id=self.name,
                sub_type=ctx.sub_type,
                data=None,
                raw_content=raw_content,
            )

        if not isinstance(data, dict):
            return AgentResult(
                agent_id=self.name,
                sub_type=ctx.sub_type,
                data=None,
                raw_content=raw_content,
            )

        return AgentResult(
            agent_id=self.name,
            sub_type=ctx.sub_type,
            data=data,
            raw_content=raw_content,
        )
