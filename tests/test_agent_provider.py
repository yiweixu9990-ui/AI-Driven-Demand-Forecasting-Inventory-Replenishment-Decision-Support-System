"""Tests for model-provider selection without making external API calls."""

import os
import unittest
from unittest.mock import patch

from agents import OpenAIChatCompletionsModel

from src.agent.supply_chain_agent import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    build_agent,
)


class AgentProviderTests(unittest.TestCase):
    def test_openrouter_key_selects_openrouter_chat_model(self) -> None:
        environment = {
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENROUTER_MODEL": DEFAULT_OPENROUTER_MODEL,
        }
        with patch.dict(os.environ, environment, clear=True):
            agent = build_agent()

        self.assertIsInstance(agent.model, OpenAIChatCompletionsModel)
        self.assertEqual(agent.model.model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(
            str(agent.model._client.base_url).rstrip("/"),
            OPENROUTER_BASE_URL,
        )

    def test_openai_remains_available_as_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-openai-key", "OPENAI_MODEL": "gpt-test"},
            clear=True,
        ):
            agent = build_agent()

        self.assertEqual(agent.model, "gpt-test")


if __name__ == "__main__":
    unittest.main()
