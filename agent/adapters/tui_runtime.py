"""TUI implementation of AgentRuntime.

Satisfies the AgentRuntime protocol by wrapping a fresh api_client module
instance (same isolation pattern as discord_bot.load_private_api_client) so
multiple TUI chat connections don't clobber each other's API state.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Any, Optional


class TUIRuntime:
    """
    Minimal AgentRuntime for in-process TUI chat.

    - No Discord dependency.
    - No SpikeProcessor (spike requires live Discord channels).
    - resolve_user returns the uid as-is (no network lookup).
    """

    def __init__(self, bot_name: str, api_type: str, model: Optional[str] = None):
        self._agent_id = "tui"
        self._agent_name = bot_name
        self._amygdala = 50
        self._processing = True

        # Isolated api_client module instance keyed by (bot, api_type) so two
        # simultaneous connections can't overwrite each other's api.api_type.
        module_key = f"api_client_tui_{bot_name}_{api_type}"
        if module_key not in sys.modules:
            spec = importlib.util.find_spec("api_client")
            if spec is None:
                raise ImportError("api_client module not found — is agent/ in sys.path?")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[module_key] = mod
        mod = sys.modules[module_key]
        mod.initialize_api_client(argparse.Namespace(api=api_type, model=model))
        self._api = mod

        from logger import BotLogger
        self._log = BotLogger(bot_name)

        from bot_config import config
        from tools.todos.factory import create_todo_service
        self.todo_service = (
            create_todo_service(config.todo, mod.get_embeddings, self._log)
            if config.todo.enabled else None
        )

    # ── AgentRuntime identity ──────────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    # ── State flags ────────────────────────────────────────────────────────────

    @property
    def processing_enabled(self) -> bool:
        return self._processing

    # ── Emotional state ────────────────────────────────────────────────────────

    @property
    def amygdala_response(self) -> int:
        return self._amygdala

    @amygdala_response.setter
    def amygdala_response(self, value: int) -> None:
        self._amygdala = value

    # ── Subsystems ─────────────────────────────────────────────────────────────

    @property
    def spike_processor(self) -> None:
        return None  # spike requires live Discord channels

    # ── Logging ────────────────────────────────────────────────────────────────

    @property
    def logger(self) -> Any:
        return self._log

    # ── User resolution ────────────────────────────────────────────────────────

    async def resolve_user(self, uid: str) -> str:
        return uid  # no Discord lookup available

    # ── LLM access ────────────────────────────────────────────────────────────

    async def call_api(self, **kwargs) -> str:
        return await self._api.call_api(**kwargs)

    def build_tools_for_message(self, msg):
        if self.todo_service is None:
            return None
        from tools.todos.models import Principal, TodoRequestContext
        from tools.todos.toolset import build_todo_tool_bundle

        raw_actor_id = str(msg.author_id)
        actor_key = (
            f"discord:{raw_actor_id}"
            if raw_actor_id.isdigit()
            else f"tui-user:{raw_actor_id.replace(':', '_')}"
        )
        actor = Principal(
            key=actor_key,
            display_name=msg.author_name or str(msg.author_id),
        )
        agent = Principal(key=f"tui:{self._agent_name}", display_name=self._agent_name, is_bot=True)
        context = TodoRequestContext(
            actor=actor,
            agent=agent,
            channel_id=str(msg.channel_id),
            source="tui",
        )
        return build_todo_tool_bundle(self.todo_service, context)

    def update_api_temperature(self, temperature: float) -> None:
        self._api.update_api_temperature(temperature)

    def update_api_top_p(self, top_p: float) -> None:
        self._api.update_api_top_p(top_p)
