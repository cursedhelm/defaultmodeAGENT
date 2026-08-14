"""
AgentRuntime protocol — the narrow interface that DMNProcessor and SpikeProcessor
need from the host bot, without depending on Discord.

Any object that satisfies this structural protocol can host the agent subsystems.
The Discord bot satisfies it after setup_bot() wires in resolve_user and agent_id.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class AgentRuntime(Protocol):
    """
    Minimal interface required by DMNProcessor and SpikeProcessor.

    The Discord ``commands.Bot`` instance satisfies this protocol once
    ``discord_bot.setup_bot()`` attaches the extra attributes defined below.
    Non-Discord hosts provide their own implementation.
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    def agent_id(self) -> str:
        """
        Stable string identifier for this agent instance.
        On Discord: ``str(bot.user.id)``
        On other platforms: any unique string.
        """
        ...

    @property
    def agent_name(self) -> str:
        """
        Human-readable display name for this agent.
        On Discord: ``bot.user.name``
        Used in memory strings: ``@{agent_name}: {response}``
        """
        ...

    # ------------------------------------------------------------------ #
    # State flags                                                          #
    # ------------------------------------------------------------------ #

    @property
    def processing_enabled(self) -> bool:
        """When False the agent should skip all LLM calls (kill switch)."""
        ...

    # ------------------------------------------------------------------ #
    # Emotional state                                                      #
    # ------------------------------------------------------------------ #

    @property
    def amygdala_response(self) -> int:
        """Current arousal level 0–100.  DMN writes this back after each cycle."""
        ...

    @amygdala_response.setter
    def amygdala_response(self, value: int) -> None: ...

    # ------------------------------------------------------------------ #
    # User resolution                                                      #
    # ------------------------------------------------------------------ #

    async def resolve_user(self, user_id: str) -> str:
        """
        Resolve a stored user_id string to a human-readable display name.

        On Discord: fetches the user object and returns ``user.name``.
        Falls back to ``f"User({user_id})"`` if lookup fails.
        """
        ...

    # ------------------------------------------------------------------ #
    # LLM access                                                           #
    # ------------------------------------------------------------------ #

    async def call_api(self, **kwargs) -> str:
        """
        Call the configured LLM.  Accepts the same kwargs as
        ``api_client.call_api``: user_content, supplemental_system_context,
        system_prompt, temperature, api_type_override, model_override,
        image_paths, audio_paths, etc.

        Normal agent turns render ``assembled_context`` into ``user_content``.
        Use ``supplemental_system_context`` only for deliberately system-role
        content, not for the assembled conversation context.
        """
        ...

    def update_api_temperature(self, temperature: float) -> None: ...

    def update_api_top_p(self, top_p: float) -> None: ...

    # ------------------------------------------------------------------ #
    # Subsystems                                                           #
    # ------------------------------------------------------------------ #

    @property
    def spike_processor(self) -> Optional[Any]:
        """The SpikeProcessor instance, or None if not initialised."""
        ...

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    @property
    def logger(self) -> Any:
        """A BotLogger (or stdlib logging.Logger) instance."""
        ...
