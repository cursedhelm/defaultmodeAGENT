from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api_schema import ToolSpec

from .models import TodoRequestContext
from .service import TodoService


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OwnerInput(ToolInput):
    owner: Literal["agent", "requester", "target"] = Field(
        default="agent",
        description="Whose list: agent for the bot's list, requester for the current user's list, or target.",
    )
    target: str | None = Field(
        default=None, description="Mentioned user's name or Discord ID; required when owner is target."
    )


class GoalInput(OwnerInput):
    goal: str | None = Field(default=None, description="New goal; null or blank clears it.")


class AddInput(OwnerInput):
    text: str = Field(min_length=1, max_length=1000)


class RemoveInput(OwnerInput):
    selector: str = Field(min_length=1, description="Item number, exact text, stable ID, or semantic description.")


class ClearInput(OwnerInput):
    confirm: bool = Field(default=False, description="Must be true to clear the entire list and goal.")


class ShareInput(OwnerInput):
    grantee: str = Field(description="Mentioned user's name or Discord ID.")
    permission: Literal["view", "edit", "revoke"]


@dataclass(frozen=True)
class TodoToolBundle:
    specs: list[ToolSpec]
    runtime: dict[str, Any]


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def build_todo_tool_bundle(service: TodoService, context: TodoRequestContext) -> TodoToolBundle:
    def owner_from(data: OwnerInput):
        if data.owner == "agent":
            return context.agent
        if data.owner == "requester":
            return context.actor
        if not data.target:
            raise ValueError("target is required when owner is target")
        return context.target(data.target)

    async def todo_list(arguments: dict):
        data = OwnerInput.model_validate(arguments)
        return (await service.list_todos(context, owner_from(data))).model_dump(mode="json")

    async def todo_set_goal(arguments: dict):
        data = GoalInput.model_validate(arguments)
        return (await service.set_goal(context, owner_from(data), data.goal)).model_dump(mode="json")

    async def todo_add(arguments: dict):
        data = AddInput.model_validate(arguments)
        return (await service.add(context, owner_from(data), data.text)).model_dump(mode="json")

    async def todo_remove(arguments: dict):
        data = RemoveInput.model_validate(arguments)
        return (await service.remove(context, owner_from(data), data.selector)).model_dump(mode="json")

    async def todo_clear(arguments: dict):
        data = ClearInput.model_validate(arguments)
        return (await service.clear(context, owner_from(data), confirm=data.confirm)).model_dump(mode="json")

    async def todo_share(arguments: dict):
        data = ShareInput.model_validate(arguments)
        return (await service.share(
            context, owner_from(data), context.target(data.grantee), data.permission
        )).model_dump(mode="json")

    specs = [
        ToolSpec(name="todo_list", description="View the agent's, requester's, or an authorized mentioned user's todo list.", parameters=_schema(OwnerInput)),
        ToolSpec(name="todo_set_goal", description="Set or clear the goal used to rank an authorized todo list.", parameters=_schema(GoalInput)),
        ToolSpec(name="todo_add", description="Add an item to an authorized todo list and rerank it against the goal.", parameters=_schema(AddInput)),
        ToolSpec(name="todo_remove", description="Remove one item by number, text, ID, or unambiguous semantic match.", parameters=_schema(RemoveInput)),
        ToolSpec(name="todo_clear", description="Clear an authorized todo list only with explicit confirmation.", parameters=_schema(ClearInput)),
        ToolSpec(name="todo_share", description="Grant, change, or revoke another mentioned user's access to a todo list.", parameters=_schema(ShareInput)),
    ]
    return TodoToolBundle(
        specs=specs,
        runtime={
            "todo_list": todo_list,
            "todo_set_goal": todo_set_goal,
            "todo_add": todo_add,
            "todo_remove": todo_remove,
            "todo_clear": todo_clear,
            "todo_share": todo_share,
        },
    )
