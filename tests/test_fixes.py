"""Unit tests for critical bug fixes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from adapter import UnblockCardAdapter
from config import MutatorConfig
from llm import (
    GenerationRequest,
    GigaChatAdapter,
    ModelProviderConfig,
    OpenAICompatibleAdapter,
    RetryConfig,
    create_adapter,
)
from schema import ScenarioData, load_and_validate_dataset


def _make_provider(**overrides) -> ModelProviderConfig:
    defaults = dict(
        name="test",
        type="gigachat",
        base_url="https://example.com",
        model="test-model",
        auth_key_env="TEST_AUTH_KEY",
        client_id_env="TEST_CLIENT_ID",
        retry=RetryConfig(max_attempts=1, backoff_seconds=0),
    )
    defaults.update(overrides)
    return ModelProviderConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. GigaChat adapter generates correctly
# ---------------------------------------------------------------------------

class TestGigaChatAdapter:
    def test_generate_returns_response(self):
        provider = _make_provider()

        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_response.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

        with patch("llm.GigaChat") as MockGigaChat:
            mock_client = MagicMock()
            mock_client.invoke.return_value = mock_response
            MockGigaChat.return_value = mock_client

            adapter = GigaChatAdapter(provider)
            result = adapter.generate(
                GenerationRequest(system_prompt="sys", user_message="usr")
            )

        assert result.output_text == "ok"
        assert result.usage is not None
        assert result.usage.input_tokens == 10


# ---------------------------------------------------------------------------
# 2. Invalid regex in dataset does not crash
# ---------------------------------------------------------------------------

class TestInvalidRegex:
    def test_invalid_regex_treated_as_failed(self):
        provider = _make_provider()

        with patch("llm.GigaChat"):
            ua = UnblockCardAdapter(provider, temperature=0.0)

        mock_response = MagicMock()
        mock_response.output_text = "some response"
        mock_response.usage = None

        with patch.object(ua._adapter, "generate", return_value=mock_response):
            batch = [
                {
                    "scenario_id": "s1",
                    "dialog_id": "d1",
                    "steps": [
                        {"step": 1, "role": "user", "message": "hello"},
                        {
                            "step": 2,
                            "role": "agent",
                            "message": "expected",
                            "additional_check": "(unclosed",
                        },
                    ],
                }
            ]
            result = ua.evaluate(batch, {"system_prompt": "test"})

        assert len(result.scores) == 1
        assert result.scores[0] == 0.0


# ---------------------------------------------------------------------------
# 3. Template placeholder validation
# ---------------------------------------------------------------------------

class TestTemplatePlaceholderValidation:
    def test_valid_template(self):
        mc = MutatorConfig(
            type="gepa_default",
            reflection_prompt_template="Use <curr_param> with <side_info>",
        )
        assert mc.reflection_prompt_template is not None

    def test_missing_placeholders_raises(self):
        with pytest.raises(ValidationError, match="reflection_prompt_template must contain"):
            MutatorConfig(
                type="gepa_default",
                reflection_prompt_template="no placeholders here",
            )

    def test_none_template_is_ok(self):
        mc = MutatorConfig(type="gepa_default", reflection_prompt_template=None)
        assert mc.reflection_prompt_template is None

    def test_gepa_default_without_template_is_ok(self):
        mc = MutatorConfig(type="gepa_default")
        assert mc.type == "gepa_default"
        assert mc.reflection_prompt_template is None


# ---------------------------------------------------------------------------
# 4. Factory creates the right adapter for each provider type
# ---------------------------------------------------------------------------

def _make_openai_provider(**overrides) -> ModelProviderConfig:
    defaults = dict(
        name="test-openai",
        type="openai_compatible",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        retry=RetryConfig(max_attempts=1, backoff_seconds=0),
    )
    defaults.update(overrides)
    return ModelProviderConfig(**defaults)


class TestFactory:
    def test_gigachat_provider_returns_gigachat_adapter(self):
        with patch("llm.GigaChat"):
            adapter = create_adapter(_make_provider())
        assert isinstance(adapter, GigaChatAdapter)

    def test_openai_provider_returns_openai_adapter(self):
        adapter = create_adapter(_make_openai_provider())
        assert isinstance(adapter, OpenAICompatibleAdapter)


class TestOpenAICompatibleAdapter:
    def test_generate_calls_chat_completions(self):
        provider = _make_openai_provider()
        adapter = OpenAICompatibleAdapter(provider)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_choice = MagicMock()
        mock_choice.message.content = "hello back"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model_dump.return_value = {"choices": [{"message": {"content": "hello back"}}]}

        with patch.object(adapter.client.chat.completions, "create", return_value=mock_completion), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            result = adapter.generate(
                GenerationRequest(system_prompt="You are helpful.", user_message="Hi")
            )

        assert result.output_text == "hello back"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_generate_returns_safe_fallback_after_retryable_errors(self):
        provider = _make_openai_provider(retry=RetryConfig(max_attempts=4, backoff_seconds=0))
        adapter = OpenAICompatibleAdapter(provider)

        with patch.object(
            adapter.client.chat.completions,
            "create",
            side_effect=Exception("429 too many requests"),
        ) as mock_create, patch("llm.time.sleep", return_value=None):
            result = adapter.generate(
                GenerationRequest(system_prompt="You are helpful.", user_message="Hi")
            )

        assert mock_create.call_count == 4
        assert result.output_text == ""
        assert result.usage is None
        assert "429" in result.raw_response.get("error", "")

    def test_generate_handles_none_message_content(self):
        provider = _make_openai_provider()
        adapter = OpenAICompatibleAdapter(provider)

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 2
        mock_usage.completion_tokens = 1

        mock_choice = MagicMock()
        mock_choice.message.content = None

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model_dump.return_value = {"choices": [{"message": {"content": None}}]}

        with patch.object(adapter.client.chat.completions, "create", return_value=mock_completion):
            result = adapter.generate(
                GenerationRequest(system_prompt="You are helpful.", user_message="Hi")
            )

        assert result.output_text == ""
        assert result.usage is not None
        assert result.usage.input_tokens == 2
        assert result.usage.output_tokens == 1


# ---------------------------------------------------------------------------
# 5. Real rollout vs golden path
# ---------------------------------------------------------------------------

class TestRolloutModes:
    @staticmethod
    def _make_adapter_and_batch():
        provider = _make_provider()
        with patch("llm.GigaChat"):
            ua = UnblockCardAdapter(provider, temperature=0.0)
        batch = [
            {
                "scenario_id": "s1",
                "dialog_id": "d1",
                "steps": [
                    {"step": 1, "role": "user", "message": "hello"},
                    {"step": 2, "role": "agent", "message": "correct_step2",
                     "additional_check": "correct_step2"},
                    {"step": 3, "role": "user", "message": "continue"},
                    {"step": 4, "role": "agent", "message": "correct_step4",
                     "additional_check": "correct_step4"},
                ],
            }
        ]
        return ua, batch

    def test_real_rollout_cascades_errors(self):
        ua, batch = self._make_adapter_and_batch()
        call_count = 0

        def mock_generate(request):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.usage = None
            if call_count == 1:
                resp.output_text = "wrong_answer"
            else:
                last_assistant = [m for m in request.history if m["role"] == "assistant"]
                resp.output_text = last_assistant[-1]["content"] if last_assistant else "no_history"
            return resp

        with patch.object(ua._adapter, "generate", side_effect=mock_generate):
            result = ua.evaluate(batch, {"system_prompt": "test"}, rollout_mode="real_rollout")

        assert result.scores[0] == 0.0

    def test_golden_path_isolates_steps(self):
        ua, batch = self._make_adapter_and_batch()
        call_count = 0

        def mock_generate(request):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.usage = None
            if call_count == 1:
                resp.output_text = "wrong_answer"
            else:
                last_assistant = [m for m in request.history if m["role"] == "assistant"]
                resp.output_text = last_assistant[-1]["content"] if last_assistant else "no_history"
            return resp

        with patch.object(ua._adapter, "generate", side_effect=mock_generate):
            result = ua.evaluate(batch, {"system_prompt": "test"}, rollout_mode="golden_path")

        assert result.scores[0] == 0.0


# ---------------------------------------------------------------------------
# 6. Dataset validation
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    def test_invalid_dataset_missing_message(self):
        data = [{"scenario_id": "s1", "dialog_id": "d1",
                 "steps": [{"role": "user"}]}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Validation error"):
                load_and_validate_dataset([Path(f.name)])

    def test_invalid_dataset_bad_role(self):
        data = [{"scenario_id": "s1", "dialog_id": "d1",
                 "steps": [{"role": "invalid_role", "message": "hi"}]}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Validation error"):
                load_and_validate_dataset([Path(f.name)])

    def test_valid_dataset_loads(self):
        data = [{"scenario_id": "s1", "dialog_id": "d1",
                 "steps": [{"role": "user", "message": "hello"},
                           {"role": "agent", "message": "hi back"}]}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_and_validate_dataset([Path(f.name)])
        assert len(result) == 1
        assert result[0]["scenario_id"] == "s1"

    def test_empty_steps_fails(self):
        data = [{"scenario_id": "s1", "dialog_id": "d1", "steps": []}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Validation error"):
                load_and_validate_dataset([Path(f.name)])
