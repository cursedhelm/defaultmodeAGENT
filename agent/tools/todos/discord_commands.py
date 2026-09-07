from __future__ import annotations

from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands

from .models import Principal, TodoOperationResult, TodoRequestContext


def _principal(user) -> Principal:
    return Principal(
        key=f"discord:{user.id}",
        display_name=getattr(user, "display_name", None) or user.name,
        is_bot=bool(getattr(user, "bot", False)),
    )


def _is_manager(user, manager_role: str) -> bool:
    permissions = getattr(user, "guild_permissions", None)
    if permissions and (permissions.administrator or permissions.manage_guild):
        return True
    return any(role.name == manager_role for role in getattr(user, "roles", []))


def _context(bot, user, channel, manager_role: str, targets=()) -> TodoRequestContext:
    known = {}
    for target in targets:
        if target is None:
            continue
        principal = _principal(target)
        known[str(target.id)] = principal
        known[principal.key] = principal
        known[principal.display_name.casefold()] = principal
        known[target.name.casefold()] = principal
    return TodoRequestContext(
        actor=_principal(user),
        agent=_principal(bot.user),
        guild_id=str(channel.guild.id) if getattr(channel, "guild", None) else None,
        channel_id=str(channel.id),
        is_manager=_is_manager(user, manager_role),
        source="discord_command",
        known_targets=known,
    )


def format_result(result: TodoOperationResult) -> str:
    lines = [result.message]
    todo = result.todo
    if todo is None:
        return lines[0]
    lines.append(f"**{todo.owner.display_name}**")
    lines.append(f"Goal: {todo.goal or '—'}")
    if todo.items:
        for index, item in enumerate(todo.items, 1):
            score = f" ({item.rank:.2f})" if item.rank is not None else ""
            lines.append(f"{index}. {item.text}{score}")
    else:
        lines.append("No todo items.")
    if result.ranking_degraded:
        lines.append("_Embedding ranking was unavailable; newest items were retained._")
    if result.candidates:
        lines.append("Candidates: " + "; ".join(item.text for item in result.candidates))
    return "\n".join(lines)


def register_todo_commands(bot, config) -> None:
    def service():
        current = getattr(bot, "todo_service", None)
        if current is None:
            raise RuntimeError("todo service is disabled or has not initialized")
        return current

    async def run(ctx, operation):
        try:
            result = await operation
            await ctx.send(format_result(result))
        except (PermissionError, ValueError, RuntimeError) as exc:
            await ctx.send(f"Todo error: {exc}")

    @bot.group(name="todo", invoke_without_command=True)
    async def todo_group(ctx, *, text: str | None = None):
        """View your todo list, or add text with `!todo <text>`."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        owner = context.actor
        if text:
            await run(ctx, service().add(context, owner, text))
        else:
            await run(ctx, service().list_todos(context, owner))

    @todo_group.command(name="list")
    async def todo_list_prefix(ctx):
        """View your todo list."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        await run(ctx, service().list_todos(context, context.actor))

    @todo_group.command(name="add")
    async def todo_add_prefix(ctx, *, text: str):
        """Add an item to your todo list."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        await run(ctx, service().add(context, context.actor, text))

    @bot.command(name="goal")
    async def todo_goal_prefix(ctx, *, goal: str | None = None):
        """View your goal, or set it with `!goal <text>`."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        if goal is None:
            await run(ctx, service().list_todos(context, context.actor))
        else:
            await run(ctx, service().set_goal(context, context.actor, goal))

    @bot.command(name="todont")
    async def todo_remove_prefix(ctx, *, selector: str):
        """Remove an item by number, exact text, ID, or semantic description."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        await run(ctx, service().remove(context, context.actor, selector))

    @bot.command(name="clear_todos")
    async def todo_clear_prefix(ctx, confirmation: str | None = None):
        """Clear your goal and items with `!clear_todos confirm`."""
        context = _context(bot, ctx.author, ctx.channel, config.discord.bot_manager_role)
        await run(
            ctx,
            service().clear(
                context, context.actor, confirm=(confirmation or "").casefold() == "confirm"
            ),
        )

    slash = app_commands.Group(name="todo", description="Manage user and bot todo lists")

    async def slash_send(interaction: discord.Interaction, operation) -> None:
        try:
            result = await operation
            content = format_result(result)
        except (PermissionError, ValueError, RuntimeError) as exc:
            content = f"Todo error: {exc}"
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True)
        else:
            await interaction.response.send_message(content, ephemeral=True)

    @slash.command(name="list", description="View a todo list you are allowed to see")
    async def slash_list(interaction: discord.Interaction, user: discord.User | None = None):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        owner = _principal(user) if user else context.actor
        await slash_send(interaction, service().list_todos(context, owner))

    @slash.command(name="add", description="Add an item to an authorized todo list")
    async def slash_add(interaction: discord.Interaction, text: str, user: discord.User | None = None):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        owner = _principal(user) if user else context.actor
        await slash_send(interaction, service().add(context, owner, text))

    @slash.command(name="remove", description="Remove one item from an authorized todo list")
    async def slash_remove(interaction: discord.Interaction, selector: str, user: discord.User | None = None):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        owner = _principal(user) if user else context.actor
        await slash_send(interaction, service().remove(context, owner, selector))

    @slash.command(name="goal", description="Set, view, or clear a todo goal")
    async def slash_goal(interaction: discord.Interaction, goal: str | None = None, user: discord.User | None = None):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        owner = _principal(user) if user else context.actor
        operation = service().set_goal(context, owner, goal) if goal is not None else service().list_todos(context, owner)
        await slash_send(interaction, operation)

    @slash.command(name="clear", description="Clear a todo list and goal with confirmation")
    async def slash_clear(interaction: discord.Interaction, confirm: bool, user: discord.User | None = None):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        owner = _principal(user) if user else context.actor
        await slash_send(interaction, service().clear(context, owner, confirm=confirm))

    @slash.command(name="share", description="Grant or revoke access to your todo list")
    @app_commands.choices(permission=[
        app_commands.Choice(name="view", value="view"),
        app_commands.Choice(name="edit", value="edit"),
        app_commands.Choice(name="revoke", value="revoke"),
    ])
    async def slash_share(
        interaction: discord.Interaction,
        user: discord.User,
        permission: app_commands.Choice[str],
    ):
        context = _context(bot, interaction.user, interaction.channel, config.discord.bot_manager_role, [user])
        await slash_send(
            interaction,
            service().share(context, context.actor, _principal(user), permission.value),
        )

    bot.tree.add_command(slash)
