# Discord Slash Commands Integration Plan

## Goal

Add Discord slash commands without breaking the existing prefix command behavior. The first pass should make command routing and permissions easier to reason about, then migrate individual commands in small groups.

## Current Command Flow

The Discord entry point is `agent/discord_bot.py`.

Current prefix commands are registered inside `setup_bot()` with `@bot.command(...)`. The active inbound flow is:

1. `on_message(message)` ignores the bot's own messages.
2. `ctx = await bot.get_context(message)` checks whether the message is a recognized command.
3. If `ctx.command is not None`, it calls `await bot.invoke(ctx)` and returns.
4. Non-command messages continue into attention checks and then the normalized agent flow through `DiscordAdapter.normalize(...)` and `agent_core.process_message(...)`.

This means command dispatch is intentionally separated from normal agent chat.

## Current Prefix Commands

Registered command handlers:

- `persona`
- `attention`
- `spike`
- `add_memory`
- `clear_memories`
- `summarize`
- `index_repo`
- `repo_file_chat`
- `ask_repo`
- `search_memories`
- `dmn`
- `kill`
- `resume`
- `mentions`
- `get_logs`
- `reranking`

There is also a custom help command and embedded command invocation support used by agent-generated responses.

## Current Prefix Behavior

`dynamic_prefix()` allows:

- In DMs: bare `!`
- In guilds: bot mention or bot role mention followed by `!`

Guild prefix examples:

- `@Bot!persona`
- `@Bot !persona`
- `@BotRole!persona`
- `@BotRole !persona`

Plain `!persona` in a server is not currently accepted by the active prefix function.

## Current Permission Model

Permissions are centralized in `agent/bot_config.py` under `DiscordConfig`.

Command groups:

- `system_commands`: admin-level controls such as `kill`, `resume`, `get_logs`, `dmn`, `mentions`, `persona`, `search_memories`, `spike`.
- `management_commands`: role/admin controls such as `add_memory`, `index_repo`, `reranking`, `clear_memories`, `attention`, `spike`.
- `general_commands`: broadly available commands such as `summarize`, `ask_repo`, `repo_file_chat`, `analyze_file`.
- `bot_action_commands`: commands that may be invoked by bot-authored embedded command flows.

`has_command_permission(command_name, ctx)` currently handles:

- Rejecting unknown commands.
- Restricting bot self-invocation to `bot_action_commands`.
- Allowing `general_commands`.
- In DMs, checking whether the user is an admin or has the configured bot manager role in any shared guild.
- In guilds, allowing admins/manage-guild users.
- In guilds, allowing the configured bot manager role for `management_commands`.

Slash command checks should reuse this policy or a shared adapter around it, rather than reimplementing different permission rules.

## Integration Strategy

### 1. Keep Prefix Commands During Migration

Do not remove `@bot.command` handlers initially. Slash commands should wrap the same underlying actions so existing workflows keep working while slash behavior is tested.

Recommended shape:

- Extract command bodies into service functions where useful.
- Keep prefix handlers as thin wrappers.
- Add slash handlers as thin wrappers.
- Put permission checks at the wrapper boundary.

Example structure:

```python
async def handle_persona(runtime, actor, channel, intensity):
    ...

@bot.command(name="persona")
async def persona_prefix(ctx, intensity: int = None):
    ...
    await handle_persona(bot, ctx.author, ctx.channel, intensity)

@bot.tree.command(name="persona")
async def persona_slash(interaction, intensity: int | None = None):
    ...
    await handle_persona(bot, interaction.user, interaction.channel, intensity)
```

### 2. Add a Slash Permission Adapter

Prefix commands receive `commands.Context`; slash commands receive `discord.Interaction`. The current permission helper expects `ctx`.

Add a small compatibility layer, not a second permission model:

```python
def has_interaction_permission(command_name: str, interaction: discord.Interaction) -> bool:
    ...
```

It should match `has_command_permission()` behavior:

- General commands are allowed.
- Guild admin/manage-guild is allowed.
- Bot manager role can use management commands.
- DM checks inspect shared guild membership for admin/manager status.
- Bot/app-originated invocations remain restricted.

Longer term, both `ctx` and `interaction` checks can call one shared policy function that accepts normalized fields:

- `command_name`
- `is_dm`
- `is_bot_author`
- `is_admin_or_manage_guild`
- `has_bot_manager_role`

### 3. Add Command Sync Lifecycle

Slash commands must be registered to Discord's application command tree. Add an explicit sync strategy:

- During development, sync to a configured guild for fast updates.
- In production, sync globally or to approved guild IDs.
- Log command sync results in `on_ready()`.

Recommended config additions:

- `DISCORD_SYNC_SLASH_COMMANDS=true|false`
- `DISCORD_SLASH_GUILD_ID=<guild id>` for development guild sync
- Optional `DISCORD_GLOBAL_SLASH_COMMANDS=true|false`

Avoid silently syncing every startup until the registration behavior is well understood.

### 4. Start With Low-Risk Commands

Migrate commands in this order:

1. `persona`, `attention`, `mentions`, `reranking`
2. `search_memories`, `add_memory`, `clear_memories`
3. `dmn`, `kill`, `resume`, `spike`
4. `summarize`
5. `index_repo`, `ask_repo`, `repo_file_chat`
6. `get_logs`

The first group is mostly state toggles and simple argument parsing. The later groups touch background jobs, GitHub integration, file sending, DMs, channel permissions, or LLM calls.

### 5. Preserve Agent Embedded Commands Separately

The agent can invoke embedded commands from generated responses. That is separate from user-facing slash commands.

Do not route embedded commands through `interaction` objects. Keep that path on prefix-style command invocation or move it to explicit internal handler functions after command bodies are extracted.

## Command-by-Command Notes

### `persona`

Good first slash command. Single optional integer. Needs the same permission rule as prefix `persona`.

Slash options:

- `intensity`: optional integer, min `0`, max `100`

### `attention`

Good first slash command. Replace free-form `state` with choices.

Choices:

- `status`
- `on`
- `off`

### `mentions`

Good first slash command. Replace free-form `state` with choices.

Choices:

- `status`
- `on`
- `off`

### `reranking`

Good first slash command. Replace free-form setting with choices.

Choices:

- `status`
- `on`
- `off`

### `spike`

Moderate risk because it depends on `bot.spike_processor` lifecycle. Choices should be explicit.

Choices:

- `status`
- `on`
- `off`

### `add_memory`

Simple behavior but has user-data implications. Use a required string option.

Slash options:

- `memory_text`: required string

### `clear_memories`

Destructive user-memory action. Consider confirmation UX before exposing as slash.

Possible approaches:

- Keep as slash command but respond with a confirmation button.
- Require an explicit `confirm: true` boolean.
- Keep prefix-only until confirmation components are added.

### `search_memories`

Useful slash candidate. Query is a required string. Responses may need ephemeral mode depending on privacy expectations.

Slash options:

- `query`: required string
- `private`: optional boolean, default `true`

### `dmn`

Moderate risk because it starts/stops background processing. Use choices.

Choices:

- `status`
- `start`
- `stop`

### `kill` and `resume`

High-impact controls. Keep admin/system permission. Slash responses should be ephemeral by default unless server-visible status is desired.

### `summarize`

Needs channel option and message count.

Slash options:

- `channel`: required text channel
- `count`: optional integer

It already checks user read permissions and bot history permissions. Keep those checks.

### `index_repo`

Needs careful cleanup before slash migration.

Known issue to resolve first:

- The `list` branch references `bot.cache_managers['file']`, while setup currently defines `bot.cache`.

Slash options:

- `action`: choices `start`, `status`, `list`
- `branch`: optional string, default `main`

### `ask_repo`

LLM/GitHub command. Requires repo index availability. Good later-stage slash command.

Slash options:

- `question`: required string

### `repo_file_chat`

LLM/GitHub command. Current prefix parser splits a single free-form string into file path and task. Slash should model these separately.

Slash options:

- `file_path`: required string
- `task`: required string

### `get_logs`

Sensitive output. Keep admin/system permission. Prefer ephemeral acknowledgement plus DM delivery, matching current behavior.

## Discord Interaction Constraints

Slash commands must respond quickly. For commands that may take longer than a few seconds:

- Call `await interaction.response.defer(...)`.
- Use `interaction.followup.send(...)` for the eventual result.

Commands likely needing defer:

- `summarize`
- `index_repo start`
- `ask_repo`
- `repo_file_chat`
- `search_memories` if the index is slow
- `get_logs` if file work is slow

Ephemeral responses should be considered for:

- Permission failures
- `get_logs`
- `search_memories`
- `kill` / `resume` status acknowledgements
- Memory management commands

## Suggested Implementation Phases

### Phase 1: Infrastructure

- Add an `app_commands` import.
- Add slash command sync config.
- Add interaction permission helper.
- Add helper functions for responding to interactions consistently.
- Add tests or dry-run checks for permission mapping.

### Phase 2: Simple Toggles

Implement slash wrappers for:

- `persona`
- `attention`
- `mentions`
- `reranking`
- `spike status/on/off`

Keep prefix handlers intact.

### Phase 3: Memory Commands

Implement slash wrappers for:

- `add_memory`
- `search_memories`
- `clear_memories` with confirmation strategy

### Phase 4: Operational Commands

Implement slash wrappers for:

- `dmn`
- `kill`
- `resume`
- `get_logs`

### Phase 5: Context and Repo Commands

Fix the `index_repo list` cache reference, then implement slash wrappers for:

- `summarize`
- `index_repo`
- `ask_repo`
- `repo_file_chat`

## Testing Checklist

- Bot starts with slash command registration disabled.
- Bot starts with guild slash sync enabled.
- Permission denial is ephemeral and does not throw.
- Admin can run system commands in guild.
- Bot manager role can run management commands in guild.
- General commands remain available.
- DM permission behavior matches prefix commands.
- Prefix commands still work.
- Slash commands do not fall through into normal agent message processing.
- Long-running slash commands defer before doing work.
- `index_repo list` uses the active cache object.
- Help output is updated or a slash `/help` equivalent is added.

## Open Decisions

- Whether slash command responses should default to ephemeral for memory and admin commands.
- Whether global command sync should be automatic or manually triggered.
- Whether to keep mention-based guild prefixes after slash commands are stable.
- Whether `clear_memories` should require button confirmation.
- Whether slash commands should be grouped, for example `/memory add`, `/memory search`, `/repo ask`.

## Recommended First Patch

Start with infrastructure and one low-risk command:

1. Add `discord.app_commands` usage.
2. Add slash sync config with guild-only development sync.
3. Add `has_interaction_permission()`.
4. Add `/persona`.
5. Verify prefix `persona` still works.

This gives the project a minimal slash-command path while keeping the current command system untouched.
