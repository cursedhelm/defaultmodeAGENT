import asyncio
import copy
import sys
from pathlib import Path
from types import SimpleNamespace


AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import api_client
from api_schema import ProviderConfig, ProviderOptions, ProviderResponse, ReasoningConfig, SamplingConfig, ToolCall, ToolSpec


def run(coro):
    return asyncio.run(coro)


def openai_response(*, content="answer", reasoning=None, tool_calls=None, model="model"):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        reasoning=None,
        reasoning_details=None,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        id="response-1",
        model=model,
        system_fingerprint=None,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


class FakeOpenAIClient:
    def __init__(self, response, capture, **client_kwargs):
        self._response = response
        capture["client"] = client_kwargs

        async def create(**kwargs):
            capture["request"] = kwargs
            return self._response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def test_unsloth_uses_openai_transport_with_server_extensions(monkeypatch):
    capture = {}
    monkeypatch.setattr(
        api_client.openai,
        "AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(openai_response(reasoning="working"), capture, **kwargs),
    )
    sampling = SamplingConfig(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.1,
        max_output_tokens=1024,
    )
    options = ProviderOptions(
        reasoning=ReasoningConfig(enabled=True),
        server_tools={"enabled": True, "enabled_tools": ["web_search", "python"]},
    )

    result, _ = run(api_client._openai_compat_chat(
        provider="unsloth",
        base_url="http://127.0.0.1:8888/v1",
        api_key="not-needed",
        model="unsloth/Qwen3.8-27B-GGUF",
        msgs=[{"role": "user", "content": "hello"}],
        sampling=sampling,
        options=options,
        tools=None,
        tool_choice=None,
    ))

    request = capture["request"]
    assert request["max_completion_tokens"] == 1024
    assert "tools" not in request
    assert "tool_choice" not in request
    assert request["extra_body"] == {
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "enable_thinking": True,
        "enable_tools": True,
        "enabled_tools": ["web_search", "python"],
    }
    assert result.reasoning == ["working"]
    assert result.as_agent_text() == "<think>\nworking\n</think>\nanswer"


def test_openai_reasoning_models_omit_unsupported_sampling_fields(monkeypatch):
    capture = {}
    monkeypatch.setattr(
        api_client.openai,
        "AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(openai_response(model="gpt-5.1"), capture, **kwargs),
    )

    run(api_client._openai_compat_chat(
        provider="openai",
        base_url=None,
        api_key="key",
        model="gpt-5.1",
        msgs=[{"role": "user", "content": "hello"}],
        sampling=SamplingConfig(max_output_tokens=321),
        options=ProviderOptions(reasoning={"effort": "medium"}),
        tools=None,
        tool_choice=None,
    ))

    request = capture["request"]
    assert request["max_completion_tokens"] == 321
    assert request["reasoning_effort"] == "medium"
    for unsupported in ("max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"):
        assert unsupported not in request


def test_openai_tool_loop_handles_async_runtime_and_preserves_protocol(monkeypatch):
    seen_messages = []
    calls = 0

    async def fake_chat(**kwargs):
        nonlocal calls
        calls += 1
        seen_messages.append(list(kwargs["msgs"]))
        if calls == 1:
            tool_call = ToolCall(id="call-1", name="lookup", arguments={"q": "x"}, raw_arguments='{"q":"x"}')
            return (
                ProviderResponse(provider="openai", model="m", tool_calls=[tool_call], finish_reason="tool_calls"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }],
                },
            )
        return ProviderResponse(provider="openai", model="m", content="done"), {"role": "assistant", "content": "done"}

    async def lookup(args):
        return {"found": args["q"]}

    monkeypatch.setattr(api_client, "_openai_compat_chat", fake_chat)
    result = run(api_client._openai_compat_call_with_auto_tools(
        provider="openai",
        cfg=ProviderConfig(model_name="m", api_key="key"),
        system_prompt="system",
        supplemental_system_context="",
        user_content="hello",
        sampling=SamplingConfig(),
        options=ProviderOptions(),
        tools_payload={"tools": [{"type": "function"}], "tool_choice": "auto"},
        tool_runtime={"lookup": lookup},
    ))

    assert result.content == "done"
    assert seen_messages[1][-2]["role"] == "assistant"
    assert seen_messages[1][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"found": "x"}',
    }


def test_openai_reasoning_tools_use_responses_protocol(monkeypatch):
    capture = {"requests": []}
    responses = [
        SimpleNamespace(
            id="resp-1",
            model="gpt-5.6-sol",
            status="completed",
            incomplete_details=None,
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id="rs-1",
                    summary=[SimpleNamespace(type="summary_text", text="I should update the list.")],
                ),
                SimpleNamespace(
                    type="function_call",
                    id="fc-1",
                    call_id="call-1",
                    name="todo_update",
                    arguments='{"text":"my intention"}',
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=SimpleNamespace(cached_tokens=1),
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        ),
        SimpleNamespace(
            id="resp-2",
            model="gpt-5.6-sol",
            status="completed",
            incomplete_details=None,
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Updated my todo list.")],
            )],
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=4,
                total_tokens=24,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        ),
    ]

    class FakeResponsesClient:
        def __init__(self, **client_kwargs):
            capture["client"] = client_kwargs

            async def create(**kwargs):
                capture["requests"].append(copy.deepcopy(kwargs))
                return responses[len(capture["requests"]) - 1]

            self.responses = SimpleNamespace(create=create)

    async def todo_update(args):
        return {"updated": args["text"]}

    old_state = api_client.api.model_copy(deep=True)
    api_client.api.api_type = "openai"
    api_client.api.api_base = None
    api_client.api.api_key = "key"
    api_client.api.model_name = "gpt-5.6-sol"
    monkeypatch.setattr(api_client.openai, "AsyncOpenAI", FakeResponsesClient)
    monkeypatch.setattr(api_client, "log_to_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_client, "_console_preview", lambda *args, **kwargs: None)
    try:
        result = run(api_client.call_api_detailed(
            "update your own todo list",
            system_prompt="You are Loop.",
            max_tokens=321,
            reasoning_effort="medium",
            tools=[ToolSpec(
                name="todo_update",
                description="Update a todo list",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )],
            tool_runtime={"todo_update": todo_update},
            auto_execute_tools=True,
        ))
    finally:
        for name in api_client.APIState.model_fields:
            setattr(api_client.api, name, getattr(old_state, name))

    first, second = capture["requests"]
    assert capture["client"] == {"api_key": "key"}
    assert first["instructions"] == "You are Loop."
    assert first["max_output_tokens"] == 321
    assert first["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert first["store"] is False
    assert first["include"] == ["reasoning.encrypted_content"]
    assert first["tools"] == [{
        "type": "function",
        "name": "todo_update",
        "description": "Update a todo list",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "strict": False,
    }]
    assert "function" not in first["tools"][0]
    for chat_only in ("messages", "max_completion_tokens", "reasoning_effort", "temperature", "top_p"):
        assert chat_only not in first

    assert second["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"updated": "my intention"}',
    }
    assert second["input"][-2].type == "function_call"
    assert result.content == "Updated my todo list."
    assert result.reasoning == ["I should update the list."]
    assert result.usage.model_dump() == {
        "input_tokens": 30,
        "output_tokens": 9,
        "total_tokens": 39,
        "reasoning_tokens": 3,
        "cached_tokens": 1,
    }
    assert result.provider_metadata["endpoint"] == "responses"


def test_openai_responses_converts_text_and_image_blocks():
    converted = api_client._openai_responses_input([
        {"type": "text", "text": "inspect this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
    ])

    assert converted == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "inspect this"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"},
        ],
    }]

    assert api_client._openai_responses_input([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
    ])[0]["content"][0]["detail"] == "auto"


def test_openai_responses_keeps_dynamic_sampling_at_none_effort(monkeypatch):
    capture = {}
    raw = SimpleNamespace(
        id="resp-1",
        model="gpt-5.6-sol",
        status="completed",
        incomplete_details=None,
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="done")],
        )],
        usage=None,
    )

    class FakeResponsesClient:
        def __init__(self, **_kwargs):
            async def create(**kwargs):
                capture.update(kwargs)
                return raw

            self.responses = SimpleNamespace(create=create)

    monkeypatch.setattr(api_client.openai, "AsyncOpenAI", FakeResponsesClient)
    result = run(api_client._openai_responses_call(
        cfg=ProviderConfig(model_name="gpt-5.6-sol", api_key="key"),
        system_prompt="",
        supplemental_system_context="",
        user_content="hello",
        sampling=SamplingConfig(temperature=1.2, top_p=0.72),
        options=ProviderOptions(reasoning={"effort": "none"}),
        tools=[{
            "type": "function", "name": "noop", "description": "No-op",
            "parameters": {"type": "object", "properties": {}}, "strict": False,
        }],
        tool_runtime=None,
        auto_execute_tools=False,
    ))

    assert result.content == "done"
    assert capture["reasoning"] == {"effort": "none", "summary": "auto"}
    assert capture["temperature"] == 1.2
    assert capture["top_p"] == 0.72


def test_anthropic_collects_all_text_and_thinking_blocks(monkeypatch):
    capture = {}
    response = SimpleNamespace(
        id="msg-1",
        model="claude-test",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=8, cache_read_input_tokens=3),
        content=[
            SimpleNamespace(type="thinking", thinking="reason one"),
            SimpleNamespace(type="text", text="first"),
            SimpleNamespace(type="text", text="second"),
        ],
    )

    async def create(**kwargs):
        capture.update(kwargs)
        return response

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(api_client.anthropic, "AsyncAnthropic", lambda **kwargs: fake)
    result = run(api_client._call_anthropic(
        "hello",
        system_prompt="system",
        supplemental_system_context="context",
        sampling=SamplingConfig(top_k=12, max_output_tokens=4096),
        options=ProviderOptions(reasoning={"enabled": True, "budget_tokens": 2048}),
        config=ProviderConfig(model_name="claude-test", api_key="key"),
        tools_payload={},
        tool_runtime=None,
        auto_execute_tools=False,
    ))

    assert capture["system"] == "system\n\ncontext"
    assert capture["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
        "display": "summarized",
    }
    assert capture["temperature"] == 1.0
    assert capture["top_k"] == 12
    assert result.content == "first\nsecond"
    assert result.reasoning == ["reason one"]
    assert result.usage.total_tokens == 18


def test_current_anthropic_models_use_adaptive_thinking_and_effort(monkeypatch):
    capture = {}
    response = SimpleNamespace(
        id="msg-2",
        model="claude-opus-4-8",
        stop_reason="end_turn",
        usage=None,
        content=[SimpleNamespace(type="text", text="answer")],
    )

    async def create(**kwargs):
        capture.update(kwargs)
        return response

    monkeypatch.setattr(
        api_client.anthropic,
        "AsyncAnthropic",
        lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create)),
    )
    run(api_client._call_anthropic(
        "hello",
        system_prompt="",
        supplemental_system_context="",
        sampling=SamplingConfig(temperature=0.4, top_p=0.7, top_k=10),
        options=ProviderOptions(reasoning={"enabled": True, "effort": "medium", "capture": False}),
        config=ProviderConfig(model_name="claude-opus-4-8", api_key="key"),
        tools_payload={},
        tool_runtime=None,
        auto_execute_tools=False,
    ))

    assert capture["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert capture["output_config"] == {"effort": "medium"}
    assert "temperature" not in capture
    assert "top_p" not in capture
    assert "top_k" not in capture


def test_explicit_reasoning_overrides_nested_options_and_state(monkeypatch):
    captured = {}
    old_state = api_client.api.model_copy(deep=True)
    api_client.api.api_type = "openai"
    api_client.api.api_base = None
    api_client.api.api_key = "key"
    api_client.api.model_name = "gpt-4.1-mini"
    api_client.api.reasoning_enabled = False
    api_client.api.capture_reasoning = True

    async def fake_call(provider, user_content, **kwargs):
        captured["sampling"] = kwargs["sampling"]
        captured["options"] = kwargs["options"]
        return ProviderResponse(provider="openai", model="gpt-4.1-mini", content="ok")

    monkeypatch.setattr(api_client, "_call_openai_compat", fake_call)
    monkeypatch.setattr(api_client, "log_to_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_client, "_console_preview", lambda *args, **kwargs: None)
    try:
        result = run(api_client.call_api_detailed(
            "hello",
            temperature=1.2,
            reasoning_enabled=True,
            capture_reasoning=False,
            provider_options={"reasoning": {"enabled": False, "capture": True}},
        ))
    finally:
        for name, value in old_state.model_dump().items():
            setattr(api_client.api, name, value)

    assert result.content == "ok"
    assert captured["sampling"].temperature == 1.2
    assert captured["options"].reasoning.enabled is True
    assert captured["options"].reasoning.capture is False


def test_gemini_preserves_thoughts_and_builds_native_thinking_config(monkeypatch):
    capture = {}
    candidate_content = SimpleNamespace(parts=[
        SimpleNamespace(text="considering", thought=True, thought_signature=b"sig", function_call=None),
        SimpleNamespace(text="answer", thought=False, thought_signature=None, function_call=None),
    ])
    response = SimpleNamespace(
        model_version="gemini-3-test",
        candidates=[SimpleNamespace(content=candidate_content, finish_reason="STOP")],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5,
            candidates_token_count=4,
            total_token_count=9,
            thoughts_token_count=2,
            cached_content_token_count=0,
        ),
    )

    async def generate_content(**kwargs):
        capture.update(kwargs)
        return response

    async def aclose():
        capture["closed"] = True

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content),
            aclose=aclose,
        )
    )
    monkeypatch.setattr(api_client.genai, "Client", lambda **kwargs: fake_client)
    result = run(api_client._call_gemini(
        "hello",
        system_prompt="system",
        supplemental_system_context="",
        sampling=SamplingConfig(top_k=30, max_output_tokens=2048),
        options=ProviderOptions(reasoning={"enabled": True, "effort": "medium", "capture": True}),
        config=ProviderConfig(model_name="gemini-3-test", api_key="key"),
        tools_payload={},
        tool_runtime=None,
        auto_execute_tools=False,
    ))

    assert capture["config"].thinking_config.include_thoughts is True
    assert capture["config"].thinking_config.thinking_level.value == "MEDIUM"
    assert capture["config"].top_k == 30
    assert capture["closed"] is True
    assert result.content == "answer"
    assert result.reasoning == ["considering"]
    assert result.provider_metadata["thought_signatures"] == ["c2ln"]
    assert result.usage.reasoning_tokens == 2


def test_local_and_router_reasoning_extensions_match_their_openai_routes(monkeypatch):
    capture = {}
    monkeypatch.setattr(
        api_client.openai,
        "AsyncOpenAI",
        lambda **kwargs: FakeOpenAIClient(openai_response(), capture, **kwargs),
    )

    run(api_client._openai_compat_chat(
        provider="ollama",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="gpt-oss:20b",
        msgs=[{"role": "user", "content": "hello"}],
        sampling=SamplingConfig(),
        options=ProviderOptions(reasoning={"enabled": True, "effort": "high"}),
        tools=None,
        tool_choice=None,
    ))
    assert capture["request"]["reasoning_effort"] == "high"
    assert "extra_body" not in capture["request"]

    router = api_client._openai_extra_body(
        "openrouter",
        SamplingConfig(),
        ProviderOptions(reasoning={"enabled": True, "effort": "high", "budget_tokens": 2048, "capture": False}),
    )
    assert router == {"reasoning": {"max_tokens": 2048, "exclude": True}}

    llama = api_client._openai_extra_body(
        "llama-server",
        SamplingConfig(top_k=20),
        ProviderOptions(reasoning={"enabled": False}),
    )
    assert llama == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_vllm_embeddings_use_configured_openai_compatible_endpoint(monkeypatch):
    capture = {}
    old_state = api_client.api.model_copy(deep=True)
    api_client.api.api_type = "vllm"
    api_client.api.api_base = "http://localhost:4000"
    api_client.api.api_key = "key"
    api_client.api.model_name = "embed-model"

    class FakeEmbeddings:
        async def create(self, **kwargs):
            capture["request"] = kwargs
            return SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[1.0, 2.0]),
            ])

    class FakeClient:
        def __init__(self, **kwargs):
            capture["client"] = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(api_client.openai, "AsyncOpenAI", FakeClient)
    try:
        vectors = run(api_client.get_embeddings(["first", "second"], provider="vllm"))
    finally:
        for name, value in old_state.model_dump().items():
            setattr(api_client.api, name, value)

    assert capture["client"]["base_url"] == "http://localhost:4000/v1"
    assert capture["request"] == {"model": "embed-model", "input": ["first", "second"]}
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


def test_video_content_uses_each_server_extension(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    media = [{"type": "video", "path": str(video)}]

    openrouter, _ = api_client.prepare_multimodal_content(
        "describe", [], [], "openrouter", media_parts=media
    )
    assert openrouter[1] == {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,dmlkZW8="},
    }

    llama, _ = api_client.prepare_multimodal_content(
        "describe", [], [], "llama-server", media_parts=media
    )
    assert llama[1] == {
        "type": "input_video",
        "input_video": {"data": "dmlkZW8="},
    }


def test_local_unsloth_uses_no_auth_sentinel_and_request_level_media(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_BASE", "http://127.0.0.1:8888/v1")
    monkeypatch.setenv("UNSLOTH_API_KEY", "stale-key")
    monkeypatch.delenv("UNSLOTH_REQUIRE_AUTH", raising=False)
    config = api_client.get_api_config("unsloth", "model")
    assert config.api_key == "not-needed"

    audio = tmp_path / "sample.wav"
    video = tmp_path / "sample.mp4"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    options = ProviderOptions()
    api_client._apply_unsloth_media_extensions(
        options,
        audio_paths=[str(audio)],
        media_parts=[{"type": "video", "path": str(video)}],
    )
    assert options.extra_body == {
        "audio_base64": "YXVkaW8=",
        "video_base64": "dmlkZW8=",
    }
