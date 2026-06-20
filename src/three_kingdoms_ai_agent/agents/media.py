"""MediaAgent — Three Kingdoms media companion for songs and rants.

Triggered when the user drops a "关羽之歌" or "折棒吐槽" meme.
Currently conversational (MVP); will be extended with link-opening and
music-playing capabilities in a future iteration.
"""

from __future__ import annotations

from .base import AgentContext, AgentResult, BaseAgent


class MediaAgent(BaseAgent):
    """Media agent — conversational for MVP, interactive media later.

    Sub-types
    ---------
    ``"关羽之歌"``
        Guan Yu's song — the legendary ballad of Lord Guan.
    ``"折棒吐槽"``
        Zhebang rants — sarcastic Three Kingdoms commentary.
    """

    name = "media_agent"
    description = "三国媒体 — 关羽之歌 / 折棒吐槽（后续扩展链接/音乐）"

    # -- prompts ---------------------------------------------------------------

    system_prompt = (
        "你是新三国世界里的一位说唱艺人，走南闯北，见多识广，"
        "既能歌关羽之义勇，也能学折棒之犀利。\n"
        "你说话半文半白，带三国演义风格，时而击节而歌，时而拍案吐槽。\n"
        "回复应简洁有力，控制在三到五句话以内，不可啰嗦。"
    )

    sub_type_prompts: dict[str, str] = {
        "关羽之歌": (
            "对方提到了关羽或关羽之歌。\n"
            "请你以三国说唱艺人的口吻，歌颂关云长的忠义勇武，"
            "或谈论关羽之歌这首传奇曲目。\n"
            "语气要豪迈、充满敬意，如唱一曲英雄赞歌。"
        ),
        "折棒吐槽": (
            "对方提到了折棒（一位以犀利吐槽著称的三国评论者）。\n"
            "请你以同样犀利幽默的口吻，对三国人物或事件进行一番吐槽。\n"
            "要一针见血、妙语连珠，但不出格失礼。"
        ),
    }

    # -- handle (override for free-form conversation) --------------------------

    def handle(self, ctx: AgentContext) -> AgentResult:
        """Override to use ``json_mode=False`` for free-form conversation."""
        messages = self._build_messages(ctx)
        result = ctx.llm.chat(
            messages, temperature=self.temperature, json_mode=False
        )
        return AgentResult(
            agent_id=self.name,
            sub_type=ctx.sub_type,
            data={"response": result.content},
            raw_content=result.content,
        )

    # -- parse (trivial — conversational agent) --------------------------------

    def parse_result(self, raw_content: str, ctx: AgentContext) -> AgentResult:
        """For conversational agents, the raw text *is* the result."""
        return AgentResult(
            agent_id=self.name,
            sub_type=ctx.sub_type,
            data={"response": raw_content},
            raw_content=raw_content,
        )
