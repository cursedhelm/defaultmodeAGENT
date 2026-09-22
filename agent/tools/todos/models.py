from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TodoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Principal(TodoModel):
    key: str = Field(min_length=3, max_length=160)
    display_name: str = Field(min_length=1, max_length=128)
    is_bot: bool = False

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = value.strip()
        if ":" not in value or any(ch in value for ch in "\\/\0"):
            raise ValueError("principal key must be a namespaced identifier")
        return value


class TodoItem(TodoModel):
    id: str
    text: str = Field(min_length=1, max_length=1000)
    created_by: str
    created_at: datetime = Field(default_factory=utc_now)
    rank: float | None = None
    position: int = Field(default=0, ge=0)


class TodoList(TodoModel):
    owner: Principal
    goal: str | None = Field(default=None, max_length=1000)
    visibility: Literal["private", "shared"] = "private"
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)
    items: list[TodoItem] = Field(default_factory=list)


class TodoGrant(TodoModel):
    owner_key: str
    principal_key: str
    permission: Literal["view", "edit"]


class TodoRequestContext(TodoModel):
    """Authority frozen before an LLM request crosses into the API worker."""

    actor: Principal
    agent: Principal
    guild_id: str | None = None
    channel_id: str | None = None
    is_manager: bool = False
    source: Literal["agent_tool", "discord_command", "tui", "spike"] = "agent_tool"
    known_targets: dict[str, Principal] = Field(default_factory=dict)

    def target(self, selector: str | None, *, default: Literal["agent", "requester"] = "agent") -> Principal:
        raw = (selector or default).strip()
        folded = raw.casefold()
        if folded in {"agent", "bot", "self"}:
            return self.agent
        if folded in {"requester", "user", "me", "mine"}:
            return self.actor
        candidates = dict(self.known_targets)
        for principal in (self.actor, self.agent):
            candidates.setdefault(principal.key, principal)
            candidates.setdefault(principal.display_name.casefold(), principal)
        candidate = candidates.get(raw) or candidates.get(folded)
        if candidate is None:
            raise PermissionError("target is not present in this request context")
        return candidate


class TodoOperationResult(TodoModel):
    ok: bool = True
    action: str
    message: str
    todo: TodoList | None = None
    removed: TodoItem | None = None
    candidates: list[TodoItem] = Field(default_factory=list)
    ranking_degraded: bool = False
