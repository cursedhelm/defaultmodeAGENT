from __future__ import annotations

from .embeddings import TodoEmbeddings
from .models import Principal, TodoItem, TodoOperationResult, TodoRequestContext
from .policy import TodoPolicy
from .repository import TodoRepository, normalize_text


class TodoService:
    def __init__(
        self,
        repository: TodoRepository,
        embeddings: TodoEmbeddings,
        *,
        max_items: int = 5,
        remove_threshold: float = 0.55,
        ambiguity_margin: float = 0.05,
    ):
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.repository = repository
        self.embeddings = embeddings
        self.max_items = max_items
        self.remove_threshold = remove_threshold
        self.ambiguity_margin = ambiguity_margin

    def _audit(self, context: TodoRequestContext, owner: Principal, action: str, detail: dict) -> None:
        enriched = {
            "guild_id": context.guild_id,
            "channel_id": context.channel_id,
            **detail,
        }
        self.repository.audit(
            context.actor.key,
            context.agent.key,
            owner.key,
            action,
            context.source,
            enriched,
        )

    async def list_todos(self, context: TodoRequestContext, owner: Principal) -> TodoOperationResult:
        TodoPolicy.require_read(self.repository, context, owner.key)
        todo = self.repository.get_list(owner)
        self._audit(context, owner, "list", {})
        return TodoOperationResult(
            action="list", message=f"Todo list for {todo.owner.display_name}", todo=todo
        )

    async def set_goal(
        self, context: TodoRequestContext, owner: Principal, goal: str | None
    ) -> TodoOperationResult:
        TodoPolicy.require_edit(self.repository, context, owner.key)
        clean = " ".join((goal or "").split()) or None
        if clean and len(clean) > 1000:
            raise ValueError("goal is longer than 1000 characters")
        self.repository.set_goal(owner, clean)
        degraded = await self._rerank(owner)
        todo = self.repository.get_list(owner)
        self._audit(context, owner, "set_goal", {"goal": clean})
        return TodoOperationResult(
            action="set_goal", message="Goal updated." if clean else "Goal cleared.",
            todo=todo, ranking_degraded=degraded,
        )

    async def add(
        self, context: TodoRequestContext, owner: Principal, text: str
    ) -> TodoOperationResult:
        TodoPolicy.require_edit(self.repository, context, owner.key)
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("todo item cannot be empty")
        if len(clean) > 1000:
            raise ValueError("todo item is longer than 1000 characters")
        item = self.repository.add_item(owner, clean, context.actor.key)
        degraded = await self._rerank(owner)
        todo = self.repository.get_list(owner)
        retained = any(candidate.id == item.id for candidate in todo.items)
        message = "Todo item added." if retained else "Item was below the current relevance cutoff."
        self._audit(context, owner, "add", {"item_id": item.id, "text": item.text, "retained": retained})
        return TodoOperationResult(
            action="add", message=message, todo=todo, ranking_degraded=degraded
        )

    async def remove(
        self, context: TodoRequestContext, owner: Principal, selector: str
    ) -> TodoOperationResult:
        TodoPolicy.require_edit(self.repository, context, owner.key)
        todo = self.repository.get_list(owner)
        if not todo.items:
            return TodoOperationResult(ok=False, action="remove", message="The todo list is empty.", todo=todo)

        clean = selector.strip()
        selected: TodoItem | None = None
        if clean.isdigit() and 1 <= int(clean) <= len(todo.items):
            selected = todo.items[int(clean) - 1]
        else:
            by_id = [item for item in todo.items if item.id == clean]
            if by_id:
                selected = by_id[0]
            exact = [item for item in todo.items if normalize_text(item.text) == normalize_text(clean)]
            if selected is None and len(exact) == 1:
                selected = exact[0]
            if selected is None:
                substring = [item for item in todo.items if normalize_text(clean) in normalize_text(item.text)]
                if len(substring) == 1:
                    selected = substring[0]
                elif len(substring) > 1:
                    return TodoOperationResult(
                        ok=False, action="remove", message="More than one item matches; use its number or ID.",
                        todo=todo, candidates=substring,
                    )

        if selected is None:
            try:
                vectors = await self.embeddings.get_many([clean, *[item.text for item in todo.items]])
                scored = sorted(
                    zip(todo.items, (self.embeddings.cosine(vectors[0], vector) for vector in vectors[1:])),
                    key=lambda pair: pair[1], reverse=True,
                )
            except Exception:
                return TodoOperationResult(
                    ok=False, action="remove",
                    message="No exact match and semantic matching is unavailable; use an item number.", todo=todo,
                )
            best_item, best_score = scored[0]
            ambiguous = len(scored) > 1 and best_score - scored[1][1] < self.ambiguity_margin
            if best_score < self.remove_threshold or ambiguous:
                return TodoOperationResult(
                    ok=False, action="remove",
                    message="Semantic match was too weak or ambiguous; choose a candidate explicitly.",
                    todo=todo, candidates=[item for item, _ in scored[:3]],
                )
            selected = best_item

        self.repository.delete_items(owner.key, [selected.id])
        todo = self.repository.get_list(owner)
        self._audit(context, owner, "remove", {"item_id": selected.id, "text": selected.text})
        return TodoOperationResult(
            action="remove", message=f"Removed: {selected.text}", todo=todo, removed=selected
        )

    async def clear(
        self, context: TodoRequestContext, owner: Principal, *, confirm: bool
    ) -> TodoOperationResult:
        TodoPolicy.require_edit(self.repository, context, owner.key)
        if not confirm:
            return TodoOperationResult(
                ok=False, action="clear", message="Confirmation is required to clear the goal and every item.",
                todo=self.repository.get_list(owner),
            )
        count = self.repository.clear(owner.key)
        todo = self.repository.get_list(owner)
        self._audit(context, owner, "clear", {"removed_count": count})
        return TodoOperationResult(action="clear", message=f"Cleared {count} todo item(s) and the goal.", todo=todo)

    async def share(
        self,
        context: TodoRequestContext,
        owner: Principal,
        grantee: Principal,
        permission: str,
    ) -> TodoOperationResult:
        TodoPolicy.require_edit(self.repository, context, owner.key)
        if owner.key != context.actor.key and owner.key != context.agent.key and not context.is_manager:
            raise PermissionError("only an owner or manager can change sharing")
        if grantee.key == owner.key:
            raise ValueError("an owner does not need a grant on their own list")
        mapped = None if permission == "revoke" else permission
        if mapped not in {None, "view", "edit"}:
            raise ValueError("permission must be view, edit, or revoke")
        self.repository.ensure_list(owner)
        self.repository.set_grant(owner.key, grantee.key, mapped)
        self._audit(context, owner, "share", {"grantee": grantee.key, "permission": permission})
        return TodoOperationResult(
            action="share",
            message=(f"Revoked access for {grantee.display_name}." if mapped is None
                     else f"Granted {mapped} access to {grantee.display_name}."),
            todo=self.repository.get_list(owner),
        )

    async def _rerank(self, owner: Principal) -> bool:
        for _ in range(3):
            todo = self.repository.get_list(owner)
            degraded = False
            if not todo.goal:
                retained = sorted(todo.items, key=lambda item: item.created_at)[-self.max_items:]
                ordered = [(item.id, None) for item in retained]
            else:
                try:
                    vectors = await self.embeddings.get_many([todo.goal, *[item.text for item in todo.items]])
                    scored = [
                        (item.id, self.embeddings.cosine(vectors[0], vector), item.created_at)
                        for item, vector in zip(todo.items, vectors[1:])
                    ]
                    scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
                    ordered = [(item_id, score) for item_id, score, _ in scored]
                except Exception:
                    retained = sorted(todo.items, key=lambda item: item.created_at)[-self.max_items:]
                    ordered = [(item.id, None) for item in retained]
                    degraded = True
            if self.repository.apply_ranking(
                owner.key,
                ordered,
                self.max_items,
                expected_revision=todo.revision,
            ):
                return degraded
        raise RuntimeError("todo list changed concurrently; please retry")
