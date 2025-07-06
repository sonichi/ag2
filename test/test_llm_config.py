# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from _collections_abc import dict_items, dict_keys, dict_values
from copy import copy, deepcopy
from typing import Any

import pytest

from autogen.llm_config import LLMConfig
from autogen.oai.anthropic import AnthropicLLMConfigEntry
from autogen.oai.bedrock import BedrockLLMConfigEntry
from autogen.oai.cerebras import CerebrasLLMConfigEntry
from autogen.oai.client import AzureOpenAILLMConfigEntry, DeepSeekLLMConfigEntry, OpenAILLMConfigEntry
from autogen.oai.cohere import CohereLLMConfigEntry
from autogen.oai.gemini import GeminiLLMConfigEntry
from autogen.oai.groq import GroqLLMConfigEntry
from autogen.oai.mistral import MistralLLMConfigEntry
from autogen.oai.ollama import OllamaLLMConfigEntry
from autogen.oai.together import TogetherLLMConfigEntry
from autogen.oai.client import OpenAIWrapper, ModelClient # Added OpenAIWrapper and ModelClient

from unittest.mock import MagicMock, patch

JSON_SAMPLE = """
[
    {
        "model": "gpt-3.5-turbo",
        "api_type": "openai",
        "tags": ["gpt35"]
    },
    {
        "model": "gpt-4",
        "api_type": "openai",
        "tags": ["gpt4"]
    },
    {
        "model": "gpt-35-turbo-v0301",
        "tags": ["gpt-3.5-turbo", "gpt35_turbo"],
        "api_key": "Your Azure OAI API Key",
        "base_url": "https://deployment_name.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2024-02-01"
    },
    {
        "model": "gpt",
        "api_key": "not-needed",
        "base_url": "http://localhost:1234/v1",
        "tags": []
    }
]
"""

JSON_SAMPLE_DICT = json.loads(JSON_SAMPLE)


@pytest.fixture
def openai_llm_config_entry() -> OpenAILLMConfigEntry:
    return OpenAILLMConfigEntry(model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly")


class TestLLMConfigEntry:
    def test_extra_fields(self) -> None:
        with pytest.raises(ValueError) as e:
            # Intentionally passing extra field to raise an error
            OpenAILLMConfigEntry(  # type: ignore [call-arg]
                model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly", extra="extra"
            )
        assert "Extra inputs are not permitted [type=extra_forbidden, input_value='extra', input_type=str]" in str(
            e.value
        )

    def test_serialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = openai_llm_config_entry.model_dump()
        expected = {
            "api_type": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
            "tags": [],
        }
        assert actual == expected

    def test_deserialization(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        actual = OpenAILLMConfigEntry(**openai_llm_config_entry.model_dump())
        assert actual == openai_llm_config_entry

    def test_get(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        assert openai_llm_config_entry.get("api_type") == "openai"
        assert openai_llm_config_entry.get("model") == "gpt-4o-mini"
        assert openai_llm_config_entry.get("api_key") == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
        assert openai_llm_config_entry.get("doesnt_exists") is None
        assert openai_llm_config_entry.get("doesnt_exists", "default") == "default"

    def test_get_item_and_set_item(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        # Test __getitem__
        assert openai_llm_config_entry["api_type"] == "openai"
        assert openai_llm_config_entry["model"] == "gpt-4o-mini"
        assert openai_llm_config_entry["api_key"] == "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
        assert openai_llm_config_entry["tags"] == []
        with pytest.raises(KeyError) as e:
            openai_llm_config_entry["wrong_key"]
        assert str(e.value) == "\"Key 'wrong_key' not found in OpenAILLMConfigEntry\""

        # Test __setitem__
        assert openai_llm_config_entry["base_url"] is None
        openai_llm_config_entry["base_url"] = "https://api.openai.com"
        assert openai_llm_config_entry["base_url"] == "https://api.openai.com"
        openai_llm_config_entry["base_url"] = None
        assert openai_llm_config_entry["base_url"] is None


class TestLLMConfig:
    @pytest.fixture
    def openai_llm_config(self, openai_llm_config_entry: OpenAILLMConfigEntry) -> LLMConfig:
        return LLMConfig(config_list=[openai_llm_config_entry], temperature=0.5, check_every_ms=1000, cache_seed=42)

    @pytest.mark.parametrize(
        "llm_config, expected",
        [
            (
                # todo add more test cases
                {
                    "config_list": [
                        {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"}
                    ]
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ]
                ),
            ),
            (
                {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"},
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ]
                ),
            ),
            (
                {
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "cache_seed": 42,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ],
                    cache_seed=42,
                ),
            ),
            (
                {
                    "config_list": [
                        {"model": "gpt-4o-mini", "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"}
                    ],
                    "max_tokens": 1024,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            max_tokens=1024,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "o3",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "max_completion_tokens": 1024,
                            "reasoning_effort": "low",
                        }
                    ],
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="o3",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            max_completion_tokens=1024,
                            reasoning_effort="low",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "gpt-4o-mini",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "api_type": "openai",
                        }
                    ],
                    "temperature": 0.5,
                    "check_every_ms": 1000,
                    "cache_seed": 42,
                },
                LLMConfig(
                    config_list=[
                        OpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                        )
                    ],
                    temperature=0.5,
                    check_every_ms=1000,
                    cache_seed=42,
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "gpt-4o-mini",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "api_type": "azure",
                            "base_url": "https://api.openai.com",
                        }
                    ],
                },
                LLMConfig(
                    config_list=[
                        AzureOpenAILLMConfigEntry(
                            model="gpt-4o-mini",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            base_url="https://api.openai.com",
                        )
                    ],
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "model": "o3",
                            "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            "api_type": "azure",
                            "base_url": "https://api.openai.com",
                            "max_completion_tokens": 1024,
                            "reasoning_effort": "low",
                        }
                    ],
                },
                LLMConfig(
                    config_list=[
                        AzureOpenAILLMConfigEntry(
                            model="o3",
                            api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                            base_url="https://api.openai.com",
                            max_completion_tokens=1024,
                            reasoning_effort="low",
                        )
                    ],
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "anthropic",
                            "model": "claude-3-5-sonnet-latest",
                            "api_key": "dummy_api_key",
                            "stream": False,
                            "temperature": 1.0,
                            "top_p": 0.8,
                            "max_tokens": 100,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        AnthropicLLMConfigEntry(
                            model="claude-3-5-sonnet-latest",
                            api_key="dummy_api_key",
                            stream=False,
                            temperature=1.0,
                            top_p=0.8,
                            max_tokens=100,
                        )
                    ],
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "bedrock",
                            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                            "aws_region": "us-east-1",
                            "aws_access_key": "test_access_key_id",
                            "aws_secret_key": "test_secret_access_key",
                            "aws_session_token": "test_session_token",
                            "temperature": 0.8,
                            "topP": 0.6,
                            "stream": False,
                            "tags": [],
                            "supports_system_prompts": True,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        BedrockLLMConfigEntry(
                            model="anthropic.claude-3-sonnet-20240229-v1:0",
                            aws_region="us-east-1",
                            aws_access_key="test_access_key_id",
                            aws_secret_key="test_secret_access_key",
                            aws_session_token="test_session_token",
                            temperature=0.8,
                            topP=0.6,
                            stream=False,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "cerebras",
                            "api_key": "fake_api_key",
                            "model": "llama3.1-8b",
                            "max_tokens": 1000,
                            "seed": 42,
                            "stream": False,
                            "temperature": 1.0,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        CerebrasLLMConfigEntry(
                            api_key="fake_api_key",
                            model="llama3.1-8b",
                            max_tokens=1000,
                            seed=42,
                            stream=False,
                            temperature=1,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "cohere",
                            "model": "command-r-plus",
                            "api_key": "dummy_api_key",
                            "frequency_penalty": 0,
                            "k": 0,
                            "p": 0.75,
                            "presence_penalty": 0,
                            "strict_tools": False,
                            "tags": [],
                            "temperature": 0.3,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        CohereLLMConfigEntry(
                            model="command-r-plus",
                            api_key="dummy_api_key",
                            stream=False,
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "deepseek",
                            "api_key": "fake_api_key",
                            "model": "deepseek-chat",
                            "base_url": "https://api.deepseek.com/v1",
                            "max_tokens": 8192,
                            "temperature": 0.5,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        DeepSeekLLMConfigEntry(
                            api_key="fake_api_key",
                            model="deepseek-chat",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "google",
                            "model": "gemini-2.0-flash-lite",
                            "api_key": "dummy_api_key",
                            "project_id": "fake-project-id",
                            "location": "us-west1",
                            "stream": False,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        GeminiLLMConfigEntry(
                            model="gemini-2.0-flash-lite",
                            api_key="dummy_api_key",
                            project_id="fake-project-id",
                            location="us-west1",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "groq",
                            "model": "llama3-8b-8192",
                            "api_key": "fake_api_key",
                            "temperature": 1,
                            "stream": False,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(config_list=[GroqLLMConfigEntry(api_key="fake_api_key", model="llama3-8b-8192")]),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "mistral",
                            "model": "mistral-small-latest",
                            "api_key": "fake_api_key",
                            "safe_prompt": False,
                            "stream": False,
                            "temperature": 0.7,
                            "tags": [],
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        MistralLLMConfigEntry(
                            model="mistral-small-latest",
                            api_key="fake_api_key",
                        )
                    ]
                ),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "ollama",
                            "model": "llama3.1:8b",
                            "num_ctx": 2048,
                            "num_predict": -1,
                            "repeat_penalty": 1.1,
                            "seed": 0,
                            "stream": False,
                            "tags": [],
                            "temperature": 0.8,
                            "top_k": 40,
                            "top_p": 0.9,
                        }
                    ]
                },
                LLMConfig(config_list=[OllamaLLMConfigEntry(model="llama3.1:8b")]),
            ),
            (
                {
                    "config_list": [
                        {
                            "api_type": "together",
                            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                            "api_key": "fake_api_key",
                            "safety_model": "Meta-Llama/Llama-Guard-7b",
                            "tags": [],
                            "max_tokens": 512,
                            "stream": False,
                        }
                    ]
                },
                LLMConfig(
                    config_list=[
                        TogetherLLMConfigEntry(
                            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            api_key="fake_api_key",
                            safety_model="Meta-Llama/Llama-Guard-7b",
                        )
                    ]
                ),
            ),
            (
                {
                    "model": "gpt-4o-realtime-preview",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "voice": "alloy",
                    "tags": ["gpt-4o-realtime", "realtime"],
                },
                LLMConfig(
                    model="gpt-4o-realtime-preview",
                    api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    voice="alloy",
                    tags=["gpt-4o-realtime", "realtime"],
                ),
            ),
        ],
    )
    def test_init(self, llm_config: dict[str, Any], expected: LLMConfig) -> None:
        actual = LLMConfig(**llm_config)
        assert actual == expected, actual

    def test_extra_fields(self) -> None:
        with pytest.raises(ValueError) as e:
            LLMConfig(
                config_list=[
                    OpenAILLMConfigEntry(
                        model="gpt-4o-mini", api_key="sk-mockopenaiAPIkeysinexpectedformatsfortestingonly"
                    )
                ],
                extra="extra",
            )
        assert "Extra inputs are not permitted [type=extra_forbidden, input_value='extra', input_type=str]" in str(
            e.value
        )

    def test_serialization(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.model_dump()
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }
        assert actual == expected

    def test_deserialization(self, openai_llm_config: LLMConfig) -> None:
        actual = LLMConfig(**openai_llm_config.model_dump())
        assert actual.model_dump() == openai_llm_config.model_dump()
        assert type(actual._model) == type(openai_llm_config._model)
        assert actual._model == openai_llm_config._model
        assert actual == openai_llm_config
        assert isinstance(actual.config_list[0], OpenAILLMConfigEntry)

    def test_get(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.get("temperature") == 0.5
        assert openai_llm_config.get("check_every_ms") == 1000
        assert openai_llm_config.get("cache_seed") == 42
        assert openai_llm_config.get("doesnt_exists") is None
        assert openai_llm_config.get("doesnt_exists", "default") == "default"

    def test_getattr(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.temperature == 0.5
        assert openai_llm_config.check_every_ms == 1000
        assert openai_llm_config.cache_seed == 42
        assert openai_llm_config.config_list == [openai_llm_config.config_list[0]]
        with pytest.raises(AttributeError) as e:
            openai_llm_config.wrong_key
        assert str(e.value) == "'LLMConfig' object has no attribute 'wrong_key'"

    def test_setattr(self, openai_llm_config: LLMConfig) -> None:
        assert openai_llm_config.temperature == 0.5
        openai_llm_config.temperature = 0.8
        assert openai_llm_config.temperature == 0.8

    def test_get_item_and_set_item(self, openai_llm_config: LLMConfig) -> None:
        # Test __getitem__
        assert openai_llm_config["temperature"] == 0.5
        assert openai_llm_config["check_every_ms"] == 1000
        assert openai_llm_config["cache_seed"] == 42
        assert openai_llm_config["config_list"] == [openai_llm_config.config_list[0]]
        with pytest.raises(KeyError) as e:
            openai_llm_config["wrong_key"]
        assert str(e.value) == "\"Key 'wrong_key' not found in LLMConfig\""

        # Test __setitem__
        assert openai_llm_config["timeout"] is None
        openai_llm_config["timeout"] = 60
        assert openai_llm_config["timeout"] == 60
        openai_llm_config["timeout"] = None
        assert openai_llm_config["timeout"] is None

    def test_items(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.items()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_items)
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }
        assert dict(actual) == expected, dict(actual)

    def test_keys(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.keys()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_keys)
        expected = ["temperature", "check_every_ms", "cache_seed", "config_list"]
        assert list(actual) == expected, list(actual)

    def test_values(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.values()  # type: ignore[var-annotated]
        assert isinstance(actual, dict_values)
        expected = [
            0.5,
            1000,
            42,
            [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "tags": [],
                }
            ],
        ]
        assert list(actual) == expected, list(actual)

    def test_unpack(self, openai_llm_config: LLMConfig, openai_llm_config_entry: OpenAILLMConfigEntry) -> None:
        openai_llm_config_entry.base_url = "localhost:8080"  # type: ignore[assignment]
        openai_llm_config.config_list = [  # type: ignore[attr-defined]
            openai_llm_config_entry,
        ]
        expected = {
            "config_list": [
                {
                    "api_type": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-mockopenaiAPIkeysinexpectedformatsfortestingonly",
                    "base_url": "localhost:8080",
                    "tags": [],
                }
            ],
            "temperature": 0.5,
            "check_every_ms": 1000,
            "cache_seed": 42,
        }

        def test_unpacking(**kwargs: Any) -> None:
            for k, v in expected.items():
                assert k in kwargs
                if k == "config_list":
                    assert kwargs[k][0].model_dump() == v[0]  # type: ignore[index]
                else:
                    assert kwargs[k] == v
            # assert kwargs == expected, kwargs

        test_unpacking(**openai_llm_config)

    def test_contains(self, openai_llm_config: LLMConfig) -> None:
        assert "temperature" in openai_llm_config
        assert "check_every_ms" in openai_llm_config
        assert "cache_seed" in openai_llm_config
        assert "config_list" in openai_llm_config
        assert "doesnt_exists" not in openai_llm_config
        assert "config_list" in openai_llm_config
        assert not "config_list" not in openai_llm_config

    def test_with_context(self, openai_llm_config: LLMConfig) -> None:
        # Test with dummy agent
        class DummyAgent:
            def __init__(self) -> None:
                self.llm_config = LLMConfig.get_current_llm_config()

        with openai_llm_config:
            agent = DummyAgent()
        assert agent.llm_config == openai_llm_config
        assert agent.llm_config.temperature == 0.5
        assert agent.llm_config.config_list[0]["model"] == "gpt-4o-mini"

        # Test passing LLMConfig object as parameter
        assert LLMConfig.get_current_llm_config(openai_llm_config) == openai_llm_config

        # Test accessing current_llm_config outside the context
        assert LLMConfig.get_current_llm_config() is None
        with openai_llm_config:
            actual = LLMConfig.get_current_llm_config()
            assert actual == openai_llm_config

        assert LLMConfig.get_current_llm_config() is None

    @pytest.mark.parametrize(
        "filter_dict, exclude, expected",
        [
            (
                {"tags": ["gpt35", "gpt4"]},
                False,
                JSON_SAMPLE_DICT[0:2],
            ),
            (
                {"tags": ["gpt35", "gpt4"]},
                True,
                JSON_SAMPLE_DICT[2:4],
            ),
            (
                {"api_type": "azure", "api_version": "2024-02-01"},
                False,
                [JSON_SAMPLE_DICT[2]],
            ),
            (
                {"api_type": ["azure"]},
                False,
                [JSON_SAMPLE_DICT[2]],
            ),
            (
                {},
                False,
                JSON_SAMPLE_DICT,
            ),
        ],
    )
    def test_where(self, filter_dict: dict[str, Any], exclude: bool, expected: list[dict[str, Any]]) -> None:
        openai_llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT, temperature=0.1)

        actual = openai_llm_config.where(**filter_dict, exclude=exclude)
        assert isinstance(actual, LLMConfig)
        assert actual.config_list == LLMConfig(config_list=expected).config_list
        assert actual.temperature == 0.1

    def test_where_invalid_filter(self) -> None:
        openai_llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT)

        with pytest.raises(ValueError) as e:
            openai_llm_config.where(api_type="invalid")
        assert str(e.value) == "No config found that satisfies the filter criteria: {'api_type': 'invalid'}"

    def test_repr(self, openai_llm_config: LLMConfig) -> None:
        actual = repr(openai_llm_config)
        expected = "LLMConfig(temperature=0.5, check_every_ms=1000, cache_seed=42, config_list=[{'api_type': 'openai', 'model': 'gpt-4o-mini', 'api_key': '**********', 'tags': []}])"
        assert actual == expected, actual

    def test_str(self, openai_llm_config: LLMConfig) -> None:
        actual = str(openai_llm_config)
        expected = "LLMConfig(temperature=0.5, check_every_ms=1000, cache_seed=42, config_list=[{'api_type': 'openai', 'model': 'gpt-4o-mini', 'api_key': '**********', 'tags': []}])"
        assert actual == expected, actual

    def test_from_json_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONFIG", JSON_SAMPLE)
        expected = LLMConfig(config_list=JSON_SAMPLE_DICT)
        actual = LLMConfig.from_json(env="LLM_CONFIG")
        assert isinstance(actual, LLMConfig)
        assert actual == expected, actual

    @pytest.mark.xfail(reason="Currently raises FileNotFoundError")
    def test_from_json_env_not_found(self) -> None:
        with pytest.raises(ValueError) as e:
            LLMConfig.from_json(env="INVALID_ENV")
        assert str(e.value) == "Environment variable 'INVALID_ENV' not found"

    def test_from_json_env_with_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CONFIG", JSON_SAMPLE)
        expected = LLMConfig(config_list=JSON_SAMPLE_DICT, temperature=0.5, check_every_ms=1000, cache_seed=42)
        actual = LLMConfig.from_json(env="LLM_CONFIG", temperature=0.5, check_every_ms=1000, cache_seed=42)
        assert isinstance(actual, LLMConfig)
        assert actual == expected, actual

    def test_from_json_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/llm_config.json"
            with open(file_path, "w") as f:
                f.write(JSON_SAMPLE)

            expected = LLMConfig(config_list=JSON_SAMPLE_DICT)
            actual = LLMConfig.from_json(path=file_path)
            assert isinstance(actual, LLMConfig)
            assert actual == expected, actual

        with pytest.raises(FileNotFoundError) as e:
            LLMConfig.from_json(path="invalid_path")
        assert "No such file or directory: 'invalid_path'" in str(e.value)

    def test_copy(self, openai_llm_config: LLMConfig) -> None:
        actual = openai_llm_config.copy()
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = openai_llm_config.deepcopy()
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = copy(openai_llm_config)
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

        actual = deepcopy(openai_llm_config)
        assert actual == openai_llm_config
        assert actual is not openai_llm_config

    def test_current(self) -> None:
        llm_config = LLMConfig(config_list=JSON_SAMPLE_DICT)

        # Test without context. Should raise an error
        expected_error = "No current LLMConfig set. Are you inside a context block?"
        with pytest.raises(ValueError) as e:
            LLMConfig.current
        assert str(e.value) == expected_error
        with pytest.raises(ValueError) as e:
            LLMConfig.default
        assert str(e.value) == expected_error

        with llm_config:
            assert LLMConfig.get_current_llm_config() == llm_config
            assert LLMConfig.current == llm_config
            assert LLMConfig.default == llm_config

            with LLMConfig.current.where(api_type="openai"):
                assert LLMConfig.get_current_llm_config() == llm_config.where(api_type="openai")
                assert LLMConfig.current == llm_config.where(api_type="openai")
                assert LLMConfig.default == llm_config.where(api_type="openai")

                with LLMConfig.default.where(model="gpt-4"):
                    assert LLMConfig.get_current_llm_config() == llm_config.where(api_type="openai", model="gpt-4")
                    assert LLMConfig.current == llm_config.where(api_type="openai", model="gpt-4")
                    assert LLMConfig.default == llm_config.where(api_type="openai", model="gpt-4")

    def test_routing_method_initialization(self):
        config_list_data = [{"model": "gpt-4", "api_key": "sk-test"}]
        llm_config_default = LLMConfig(config_list=config_list_data)
        assert llm_config_default.routing_method == "fixed_order"
        assert llm_config_default._config_list_index == 0

        llm_config_round_robin = LLMConfig(config_list=config_list_data, routing_method="round_robin")
        assert llm_config_round_robin.routing_method == "round_robin"
        assert llm_config_round_robin._config_list_index == 0

        llm_config_explicit_fixed = LLMConfig(config_list=config_list_data, routing_method="fixed_order")
        assert llm_config_explicit_fixed.routing_method == "fixed_order"
        assert llm_config_explicit_fixed._config_list_index == 0

    def test_get_configs_to_try_fixed_order(self):
        entry1 = OpenAILLMConfigEntry(model="gpt-4", api_key="sk-test1")
        entry2 = OpenAILLMConfigEntry(model="gpt-3.5", api_key="sk-test2")
        llm_config = LLMConfig(config_list=[entry1, entry2], routing_method="fixed_order")

        configs_to_try = llm_config.get_configs_to_try()
        assert configs_to_try == [entry1, entry2]
        # Ensure index is not advanced for fixed_order
        assert llm_config._config_list_index == 0

        # Try again, should be the same
        configs_to_try_again = llm_config.get_configs_to_try()
        assert configs_to_try_again == [entry1, entry2]
        assert llm_config._config_list_index == 0

    def test_get_configs_to_try_round_robin(self):
        entry1 = OpenAILLMConfigEntry(model="gpt-4", api_key="sk-test1")
        entry2 = OpenAILLMConfigEntry(model="gpt-3.5", api_key="sk-test2")
        entry3 = OpenAILLMConfigEntry(model="gpt-4o", api_key="sk-test3")
        llm_config = LLMConfig(config_list=[entry1, entry2, entry3], routing_method="round_robin")

        # First call
        configs_to_try1 = llm_config.get_configs_to_try()
        assert configs_to_try1 == [entry1]
        assert llm_config._config_list_index == 1

        # Second call
        configs_to_try2 = llm_config.get_configs_to_try()
        assert configs_to_try2 == [entry2]
        assert llm_config._config_list_index == 2

        # Third call
        configs_to_try3 = llm_config.get_configs_to_try()
        assert configs_to_try3 == [entry3]
        assert llm_config._config_list_index == 0 # Should wrap around

        # Fourth call (back to first)
        configs_to_try4 = llm_config.get_configs_to_try()
        assert configs_to_try4 == [entry1]
        assert llm_config._config_list_index == 1

    def test_get_configs_to_try_round_robin_single_config(self):
        entry1 = OpenAILLMConfigEntry(model="gpt-4", api_key="sk-test1")
        llm_config = LLMConfig(config_list=[entry1], routing_method="round_robin")

        configs_to_try1 = llm_config.get_configs_to_try()
        assert configs_to_try1 == [entry1]
        assert llm_config._config_list_index == 0

        configs_to_try2 = llm_config.get_configs_to_try()
        assert configs_to_try2 == [entry1]
        assert llm_config._config_list_index == 0

    def test_get_configs_to_try_empty_list(self):
        llm_config_fixed = LLMConfig(config_list=[], routing_method="fixed_order")
        assert llm_config_fixed.get_configs_to_try() == []

        llm_config_round_robin = LLMConfig(config_list=[], routing_method="round_robin")
        assert llm_config_round_robin.get_configs_to_try() == []

    def test_get_configs_to_try_unknown_method(self):
        entry1 = OpenAILLMConfigEntry(model="gpt-4", api_key="sk-test1")
        llm_config = LLMConfig(config_list=[entry1], routing_method="unknown_method")
        # Should default to fixed_order behavior
        configs_to_try = llm_config.get_configs_to_try()
        assert configs_to_try == [entry1]
        assert llm_config._config_list_index == 0 # Index should not advance

    def test_where_resets_round_robin_index(self):
        entry1 = OpenAILLMConfigEntry(model="gpt-4", api_key="sk-key1", tags=["a"])
        entry2 = OpenAILLMConfigEntry(model="gpt-3.5", api_key="sk-key2", tags=["b"])
        entry3 = OpenAILLMConfigEntry(model="gpt-4o", api_key="sk-key3", tags=["a"])
        llm_config_orig = LLMConfig(config_list=[entry1, entry2, entry3], routing_method="round_robin")

        # Advance index on original
        llm_config_orig.get_configs_to_try() # entry1, index becomes 1
        llm_config_orig.get_configs_to_try() # entry2, index becomes 2
        assert llm_config_orig._config_list_index == 2

        # Filter
        llm_config_filtered = llm_config_orig.where(tags=["a"])
        assert len(llm_config_filtered.config_list) == 2
        assert llm_config_filtered.config_list[0].model == "gpt-4"
        assert llm_config_filtered.config_list[1].model == "gpt-4o"

        # Check that the new LLMConfig from where() has its index reset
        assert llm_config_filtered.routing_method == "round_robin" # routing_method should be preserved
        assert llm_config_filtered._config_list_index == 0 # Index should be reset for the new object

        # Test round robin on the filtered list
        configs1 = llm_config_filtered.get_configs_to_try()
        assert len(configs1) == 1
        assert configs1[0].model == "gpt-4"
        assert llm_config_filtered._config_list_index == 1

        configs2 = llm_config_filtered.get_configs_to_try()
        assert len(configs2) == 1
        assert configs2[0].model == "gpt-4o"
        assert llm_config_filtered._config_list_index == 0 # Wraps around in the filtered list

        # Original config's index should be unchanged by operations on the filtered one
        assert llm_config_orig._config_list_index == 2


class TestOpenAIWrapperRouting:
    def _create_mock_client(self, name: str, succeed: bool = True):
        client = MagicMock(spec=ModelClient)
        client.name = name
        if succeed:
            client.create.return_value = MagicMock(spec=ModelClient.ModelClientResponseProtocol)
            client.create.return_value.choices = [MagicMock()]
            client.create.return_value.choices[0].message = MagicMock(content="success from " + name)
            client.create.return_value.model = name
            client.cost.return_value = 0.01
            client.get_usage.return_value = {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
                "cost": 0.01,
                "model": name,
            }
            # Mock message_retrieval_function
            client.create.return_value.message_retrieval_function = lambda x: [choice.message for choice in x.choices]

        else:
            client.create.side_effect = Exception(f"Failed to create from {name}")
        return client

    def test_openai_wrapper_routing_method_init(self):
        config_list = [{"model": "gpt-4", "api_key": "sk-1"}]
        wrapper_default = OpenAIWrapper(config_list=config_list)
        assert wrapper_default.routing_method == "fixed_order"
        assert wrapper_default._current_client_index == 0

        wrapper_round_robin = OpenAIWrapper(config_list=config_list, routing_method="round_robin")
        assert wrapper_round_robin.routing_method == "round_robin"
        assert wrapper_round_robin._current_client_index == 0

        wrapper_fixed_explicit = OpenAIWrapper(config_list=config_list, routing_method="fixed_order")
        assert wrapper_fixed_explicit.routing_method == "fixed_order"
        assert wrapper_fixed_explicit._current_client_index == 0

    @patch("autogen.oai.client.OpenAIClient") # Mock the actual client creation
    def test_openai_wrapper_fixed_order_routing(self, MockOpenAIClient):
        # This test checks that fixed_order tries clients sequentially on failure.
        mock_oai_client_instance_good = self._create_mock_client("good_client", succeed=True)
        mock_oai_client_instance_bad = self._create_mock_client("bad_client", succeed=False)

        # Order of mock return values matters for sequential client registration
        MockOpenAIClient.side_effect = [
            mock_oai_client_instance_bad,  # First client will fail
            mock_oai_client_instance_good  # Second client will succeed
        ]

        config_list = [
            {"model": "bad_model", "api_key": "sk-bad"}, # Will be associated with bad_client
            {"model": "good_model", "api_key": "sk-good"} # Will be associated with good_client
        ]

        wrapper = OpenAIWrapper(config_list=config_list, routing_method="fixed_order")

        # First client is bad_client, second is good_client due to side_effect order
        assert wrapper._clients[0] == mock_oai_client_instance_bad
        assert wrapper._clients[1] == mock_oai_client_instance_good

        response = wrapper.create(messages=[{"role": "user", "content": "hello"}])

        assert response.model == "good_client" # Should succeed with the second client
        mock_oai_client_instance_bad.create.assert_called_once() # First client was called
        mock_oai_client_instance_good.create.assert_called_once() # Second client was called
        assert wrapper._current_client_index == 0 # Index not used/advanced in fixed_order like this

    @patch("autogen.oai.client.OpenAIClient")
    def test_openai_wrapper_round_robin_routing_success_first_attempt(self, MockOpenAIClient):
        # This test checks round_robin cycles the starting client on successive calls,
        # and the first attempt (which is the round-robin selected client) succeeds.
        mock_client1 = self._create_mock_client("client1", succeed=True)
        mock_client2 = self._create_mock_client("client2", succeed=True)
        MockOpenAIClient.side_effect = [mock_client1, mock_client2] # For initializing OpenAIWrapper

        config_list = [
            {"model": "model1", "api_key": "sk-1"}, # Associated with client1
            {"model": "model2", "api_key": "sk-2"}  # Associated with client2
        ]
        wrapper = OpenAIWrapper(config_list=config_list, routing_method="round_robin")
        # wrapper._clients will be [mock_client1, mock_client2]

        # Call 1 - starting index 0 (client1). client1 succeeds.
        response1 = wrapper.create(messages=[{"role": "user", "content": "hello"}])
        assert response1.model == "client1"
        mock_client1.create.assert_called_once()
        mock_client2.create.assert_not_called() # client2 not called as client1 succeeded
        assert wrapper._current_client_index == 1 # Next call will start with client2

        # Call 2 - starting index 1 (client2). client2 succeeds.
        mock_client1.reset_mock()
        response2 = wrapper.create(messages=[{"role": "user", "content": "world"}])
        assert response2.model == "client2"
        mock_client1.create.assert_not_called() # client1 not called this round
        mock_client2.create.assert_called_once()
        assert wrapper._current_client_index == 0 # Next call will start with client1 (wraps around)

        # Call 3 - starting index 0 (client1). client1 succeeds.
        mock_client2.reset_mock()
        response3 = wrapper.create(messages=[{"role": "user", "content": "again"}])
        assert response3.model == "client1"
        mock_client1.create.assert_called_once()
        mock_client2.create.assert_not_called()
        assert wrapper._current_client_index == 1

    @patch("autogen.oai.client.OpenAIClient")
    def test_openai_wrapper_round_robin_failover(self, MockOpenAIClient):
        # This test checks that round_robin attempts failover starting from the round-robin index.
        mock_client1_fails = self._create_mock_client("client1_fails", succeed=False)
        mock_client2_succeeds = self._create_mock_client("client2_succeeds", succeed=True)
        mock_client3_succeeds = self._create_mock_client("client3_succeeds", succeed=True)

        # This side_effect is for when OpenAIWrapper initializes its internal _clients list
        MockOpenAIClient.side_effect = [mock_client1_fails, mock_client2_succeeds, mock_client3_succeeds]

        config_list = [
            {"model": "model1", "api_key": "sk-1"}, # client1_fails
            {"model": "model2", "api_key": "sk-2"}, # client2_succeeds
            {"model": "model3", "api_key": "sk-3"}  # client3_succeeds
        ]
        wrapper = OpenAIWrapper(config_list=config_list, routing_method="round_robin")
        assert wrapper._clients == [mock_client1_fails, mock_client2_succeeds, mock_client3_succeeds]

        # --- First Create Call ---
        # Starts at index 0 (client1_fails). Expected order of attempts: client1_fails -> client2_succeeds
        wrapper._current_client_index = 0
        response1 = wrapper.create(messages=[{"role": "user", "content": "try1"}])
        assert response1.model == "client2_succeeds" # Succeeds with client2
        mock_client1_fails.create.assert_called_once() # client1 attempted and failed
        mock_client2_succeeds.create.assert_called_once() # client2 attempted and succeeded
        mock_client3_succeeds.create.assert_not_called() # client3 not needed
        assert wrapper._current_client_index == 1 # Next call will start at index 1

        # Reset mocks for next call
        mock_client1_fails.reset_mock()
        mock_client2_succeeds.reset_mock()
        mock_client3_succeeds.reset_mock()

        # --- Second Create Call ---
        # Starts at index 1 (client2_succeeds). Expected order of attempts: client2_succeeds
        # wrapper._current_client_index is already 1
        response2 = wrapper.create(messages=[{"role": "user", "content": "try2"}])
        assert response2.model == "client2_succeeds" # Succeeds with client2 immediately
        mock_client1_fails.create.assert_not_called()
        mock_client2_succeeds.create.assert_called_once()
        mock_client3_succeeds.create.assert_not_called()
        assert wrapper._current_client_index == 2 # Next call will start at index 2

        # Reset mocks
        mock_client1_fails.reset_mock()
        mock_client2_succeeds.reset_mock()
        mock_client3_succeeds.reset_mock()

        # --- Third Create Call ---
        # Starts at index 2 (client3_succeeds). Expected order of attempts: client3_succeeds -> client1_fails -> client2_succeeds
        # For this, let's make client3 also fail to test full wrap-around logic
        mock_client3_succeeds.create.side_effect = Exception("Failed from client3_succeeds temporarily")
        mock_client3_succeeds.succeed = False # for clarity in debugging if needed, though side_effect is key

        # client1_fails is already set to fail.
        # client2_succeeds is set to succeed.

        # wrapper._current_client_index is already 2
        response3 = wrapper.create(messages=[{"role": "user", "content": "try3"}])
        assert response3.model == "client2_succeeds" # Should eventually succeed with client2

        mock_client3_succeeds.create.assert_called_once() # client3 (start) attempted and failed
        mock_client1_fails.create.assert_called_once()    # client1 (wrap around) attempted and failed
        mock_client2_succeeds.create.assert_called_once() # client2 (wrap around) attempted and succeeded
        assert wrapper._current_client_index == 0 # Next call will start at index 0

    @patch("autogen.oai.client.OpenAIClient")
    def test_openai_wrapper_round_robin_all_fail(self, MockOpenAIClient):
        # This test checks that if all clients fail in round_robin, the error from the last-tried client propagates.
        mock_client1_fails = self._create_mock_client("client1_fails", succeed=False)
        mock_client2_fails = self._create_mock_client("client2_fails", succeed=False)
        MockOpenAIClient.side_effect = [mock_client1_fails, mock_client2_fails]

        config_list = [
            {"model": "model1", "api_key": "sk-1"},
            {"model": "model2", "api_key": "sk-2"}
        ]
        wrapper = OpenAIWrapper(config_list=config_list, routing_method="round_robin")

        wrapper._current_client_index = 0 # Start with client1_fails
        with pytest.raises(Exception, match="Failed to create from client2_fails"): # Error from last client in sequence client1->client2
            wrapper.create(messages=[{"role": "user", "content": "hello"}])

        mock_client1_fails.create.assert_called_once()
        mock_client2_fails.create.assert_called_once()
        assert wrapper._current_client_index == 1 # Index advanced for next create call.
