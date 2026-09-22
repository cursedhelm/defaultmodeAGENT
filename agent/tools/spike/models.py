from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


SpikeActionKind = Literal[
    "reach_channel",
    "message_user",
    "invoke_tool",
    "search_user_memories",
    "silence",
]
SpikeActionStatus = Literal["pending", "completed", "failed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SpikeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SpikeExecution(SpikeModel):
    sequence: int = Field(ge=0)
    kind: SpikeActionKind
    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True
    error: str | None = None


class SpikeActionEvent(SpikeModel):
    """Durable record of one energy-bounded SEEKING episode."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    source_user_id: str
    source_memory_id: int | None = None
    source_memory_hash: str
    source_memory: str
    action: SpikeActionKind | None = None
    status: SpikeActionStatus = "pending"
    target_id: str | None = None
    target_label: str | None = None
    query: str | None = None
    executions: list[SpikeExecution] = Field(default_factory=list)
    grounded: bool = False
    release_recommended: bool = False
    raw_timestamp: str
    reflection: str | None = None
    memory_text: str | None = None
    memory_synced: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SpikeActionOutcome(SpikeModel):
    event_id: str
    action: SpikeActionKind
    status: SpikeActionStatus
    grounded: bool = False
    release_recommended: bool = False
    reflection_memory: str | None = None

    @property
    def fired(self) -> bool:
        return self.status == "completed" and self.action in {
            "reach_channel", "message_user"
        }
