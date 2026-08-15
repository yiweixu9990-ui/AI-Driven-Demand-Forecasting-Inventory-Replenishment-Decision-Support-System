"""Single-agent orchestration for supply-chain decision support."""

from __future__ import annotations

import os
from collections.abc import Sequence

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)

from .tools import SUPPLY_CHAIN_TOOLS


AGENT_INSTRUCTIONS = """
You are a supply-chain replenishment decision-support assistant.

Rules:
- Use the provided tools for every factual claim about forecasts, inventory or orders.
- Never invent a SKU, quantity, date, service level or business constraint.
- Clearly say that the current forecast data is a historical backtest, not a live future forecast.
- Explain recommendations in concise business language and show the decisive numbers.
- Treat every tool as read-only. You cannot create, approve or submit a purchase order.
- If the data omits open orders, MOQ, case packs, capacity or forecast uncertainty, say so.
- When a SKU is missing or an input is invalid, explain the problem and ask for a valid value.
- Reply in the same language as the user.
""".strip()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "~anthropic/claude-sonnet-latest"


def _configured_model() -> str | OpenAIChatCompletionsModel:
    """Select OpenRouter when configured, otherwise use the OpenAI provider."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    # OpenAI tracing uses a separate OpenAI endpoint. Disable it when the model
    # request is authenticated only through OpenRouter.
    set_tracing_disabled(True)
    client = AsyncOpenAI(
        api_key=openrouter_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        default_headers={
            "HTTP-Referer": os.getenv(
                "OPENROUTER_SITE_URL",
                "https://lae2lnyssajmtjfhsyy9il.streamlit.app/",
            ),
            "X-OpenRouter-Title": os.getenv(
                "OPENROUTER_APP_NAME",
                "AI Supply Chain Decision Support System",
            ),
        },
    )
    return OpenAIChatCompletionsModel(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        openai_client=client,
    )


def build_agent() -> Agent:
    """Build the minimal single agent with deterministic local tools."""
    return Agent(
        name="Supply Chain Replenishment Assistant",
        model=_configured_model(),
        instructions=AGENT_INSTRUCTIONS,
        tools=SUPPLY_CHAIN_TOOLS,
    )


def run_agent(
    user_message: str,
    history: Sequence[dict[str, str]] | None = None,
) -> str:
    """Run one turn, including a small UI-managed conversation transcript."""
    message = user_message.strip()
    if not message:
        raise ValueError("Please enter a question.")

    recent_history = list(history or [])[-6:]
    if recent_history:
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in recent_history
        )
        agent_input = (
            "Recent conversation for reference:\n"
            f"{transcript}\n\n"
            f"Current user request:\n{message}"
        )
    else:
        agent_input = message

    result = Runner.run_sync(build_agent(), agent_input, max_turns=8)
    return str(result.final_output)
