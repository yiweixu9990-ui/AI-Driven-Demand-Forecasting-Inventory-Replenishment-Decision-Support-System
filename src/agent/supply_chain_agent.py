"""Single-agent orchestration for supply-chain decision support."""

from __future__ import annotations

import os
from collections.abc import Sequence

from agents import Agent, Runner

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


def build_agent() -> Agent:
    """Build the minimal single agent with deterministic local tools."""
    return Agent(
        name="Supply Chain Replenishment Assistant",
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
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
