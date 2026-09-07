"""Provider-neutral, persistable schemas for LLM requests and responses.

The public agent contract remains ``call_api(...) -> str``.  These models are
the ground-truth representation used inside the provider adapters and for
additive structured logging/testing.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ProviderName = Literal[
    "openai",
    "openrouter",
    "ollama",
    "llama-server",
    "vllm",
    "unsloth",
    "anthropic",
    "gemini",
]


class PersistableModel(BaseModel):
    """Strict-enough persistence base while allowing provider metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProviderConfig(PersistableModel):
    model_name: str
    api_key: str | None = Field(default=None, exclude=True)
    api_base: str | None = None


class SamplingConfig(PersistableModel):
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    frequency_penalty: float = Field(0.8, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.5, ge=-2.0, le=2.0)
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_output_tokens: int = Field(default=12_000, gt=0)


class ReasoningConfig(PersistableModel):
    enabled: bool | None = None
    effort: str | None = None
    budget_tokens: int | None = Field(default=None, gt=0)
    capture: bool = True


class ProviderToolOptions(PersistableModel):
    enabled: bool = False
    enabled_tools: list[str] = Field(default_factory=list)


class ProviderOptions(PersistableModel):
    """Typed common options plus a bounded provider-specific extension map."""

    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    server_tools: ProviderToolOptions = Field(default_factory=ProviderToolOptions)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class MediaPart(PersistableModel):
    type: Literal["text", "image", "audio", "video"]
    path: str | None = None
    text: str | None = None
    mime_type: str | None = None

    @field_validator("path")
    @classmethod
    def _non_empty_path(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("media path cannot be empty")
        return value

    @model_validator(mode="after")
    def _required_payload(self):
        if self.type == "text" and self.text is None:
            raise ValueError("text media requires text")
        if self.type != "text" and self.path is None:
            raise ValueError(f"{self.type} media requires a path")
        return self


class ToolSpec(PersistableModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(PersistableModel):
    id: str
    provider_call_id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None


class ToolResult(PersistableModel):
    tool_call_id: str
    provider_call_id: str | None = None
    name: str
    content: str
    is_error: bool = False


class TokenUsage(PersistableModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)


class ProviderResponse(PersistableModel):
    provider: ProviderName
    model: str
    content: str = ""
    reasoning: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    def as_agent_text(self, *, capture_reasoning: bool = True) -> str:
        """Render the structured result through the legacy agent string ABI."""
        content = (self.content or "").strip()
        if not capture_reasoning or not self.reasoning:
            return content

        traces = []
        for trace in self.reasoning:
            clean = (trace or "").strip()
            if clean and clean not in content:
                traces.append(f"<think>\n{clean}\n</think>")
        if not traces:
            return content
        return "\n".join([*traces, content] if content else traces).strip()


def json_content(value: Any) -> str:
    """Convert tool output to stable text without requiring JSON-only tools."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)
