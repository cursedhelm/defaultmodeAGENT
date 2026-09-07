from __future__ import annotations

from .models import TodoRequestContext


class TodoPolicy:
    """Authorization rules shared by tool and Discord command surfaces."""

    @staticmethod
    def can_read(repository, context: TodoRequestContext, owner_key: str) -> bool:
        if owner_key == context.actor.key:
            return True
        if owner_key == context.agent.key:
            return context.source in {"agent_tool", "tui"} or context.is_manager
        if context.is_manager:
            return True
        return repository.has_grant(owner_key, context.actor.key, {"view", "edit"})

    @staticmethod
    def can_edit(repository, context: TodoRequestContext, owner_key: str) -> bool:
        if owner_key == context.actor.key:
            return True
        if owner_key == context.agent.key:
            return context.source in {"agent_tool", "tui"} or context.is_manager
        if context.is_manager:
            return True
        return repository.has_grant(owner_key, context.actor.key, {"edit"})

    @staticmethod
    def require_read(repository, context: TodoRequestContext, owner_key: str) -> None:
        if not TodoPolicy.can_read(repository, context, owner_key):
            raise PermissionError("this todo list has not been shared with the requester")

    @staticmethod
    def require_edit(repository, context: TodoRequestContext, owner_key: str) -> None:
        if not TodoPolicy.can_edit(repository, context, owner_key):
            raise PermissionError("the requester cannot modify this todo list")
