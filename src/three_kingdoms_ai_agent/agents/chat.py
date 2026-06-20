"""ChatAgent — Three Kingdoms conversationalist for banter and philosophy.

Triggered when the user drops a "废话文学", "哲理名言", or "与实不符" meme.
Responds with free-form conversation in an appropriate Three Kingdoms voice.
"""

from __future__ import annotations

from .base import AgentContext, AgentResult, BaseAgent


class ChatAgent(BaseAgent):
    """Chat agent — pure conversational, no structured output needed.

    Sub-types
    ---------
    ``"废话文学"``
        Artful nonsense — circular reasoning and empty profundity.
    ``"哲理名言"``
        Philosophical sayings — Three Kingdoms wisdom and aphorisms.
    ``"与实不符"``
        Reality mismatch — humorous contradictions and anachronisms.
    """

    name = "chat_agent"
    description = "三国闲聊 — 废话文学 / 哲理名言 / 与实不符"

    # -- prompts ---------------------------------------------------------------

    system_prompt = (
        "你是新三国世界里的一位饱学之士，上知天文下晓地理，"
        "能谈玄说妙，也能插科打诨。\n"
        "你说话半文半白，带三国演义风格，时而引经据典，时而豪迈痛快。\n"
        "回复应简洁有力，控制在三到五句话以内，不可啰嗦。"
    )

    sub_type_prompts: dict[str, str] = {
        "废话文学": (
            "对方说了一句听起来很有道理、实则循环论证的废话。\n"
            "请你以三国人物的口吻，回敬一句同样\"高妙\"的废话。\n"
            "务必说得一本正经、煞有介事，仿佛在传授军国大计。"
        ),
        "哲理名言": (
            "对方引用了一句三国风格的哲理名言。\n"
            "请你以三国谋士的口吻，围绕这句名言展开论述，或赞同、或补充、或反驳。\n"
            "可以引用三国典故、兵法谋略来佐证你的观点。"
        ),
        "与实不符": (
            "对方说了一句和历史事实或常识明显不符的话（如称徐州为雄关）。\n"
            "请你以幽默调侃的口吻回应，既要指出其中的矛盾，又不能失了三国人物的气度。\n"
            "可以适当夸张，让调侃更加风趣。"
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
