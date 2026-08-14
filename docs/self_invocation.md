# Bot Self-Invocation System

## Overview

The bot can now execute commands based on its own responses, enabling autonomous state management and reflexive behaviors. This creates a feedback loop where the bot's reasoning can directly modify its operational parameters.

## How It Works

After generating a response to a user message or file, the bot scans **every line** of the response for lines starting with `!`. Each line matching a whitelisted command is executed as if the bot were calling it on itself, so a single response can invoke more than one command.

### Implementation Flow

1. Bot generates response text
2. Scans each line for command pattern `!command_name [args]`
3. Validates command against `bot_action_commands` whitelist
4. Creates `FakeMessage` object copying original message attributes
5. Overrides `author` to bot.user and `content` to command line
6. Configures context with `StringView` for proper argument parsing
7. Invokes command through discord.py's `cmd.invoke(ctx)`

### FakeMessage Class

The `FakeMessage` class in `agent/discord_bot.py` creates a message object for self-invocation. It copies only the attributes discord.py needs, rather than reflecting over the original:

```python
class FakeMessage:
    """Creates a fake Discord message for bot self-invocation."""
    def __init__(self, original, bot, content):
        # copy only what's needed
        self._state = original._state
        self.id = original.id
        self.channel = original.channel
        self.guild = getattr(original, 'guild', None)
        # override for self-invoke
        self.author = bot.user
        self.content = content
        self.mentions = []
        self.channel_mentions = []
        self.role_mentions = []
        self.attachments = []
        self.reference = None
        self.interaction_metadata = None
```

This preserves important context (channel, guild, permissions) while making the bot appear as the message author. The explicit copy avoids the `dir()`-based reflection an earlier version used, which could carry over attributes that confused the command parser.

### Argument Parsing with StringView

Discord.py's command argument parser requires a `StringView` object positioned at the arguments, not the full command string. The fix configures the context properly:

```python
ctx.command = cmd
ctx.invoked_with = cmd_name
ctx.prefix = '!'
ctx.view = StringView(cmd_args)
await cmd.invoke(ctx)
```

**Why this matters:**
- Commands with positional args like `persona(intensity: int)` need the parser to see `"25"`, not `"!persona 25"`
- Commands with greedy kwargs like `add_memory(*, memory_text)` work without this fix
- Without `StringView`, the parser tries to convert `"!persona 25"` to an int, which fails

**Example parsing:**
```
Response: "!persona 25"
  → cmd_name: "persona"
  → cmd_args: "25"
  → StringView("25") → parser sees just the number
  → intensity = 25 ✅

Without fix:
  → parser sees "!persona 25"
  → tries int("!persona 25")
  → fails ❌
```

### Security

The `has_command_permission` check in `bot_config.py` already handles bot authors:

```python
# bot self-invocation - restricted to bot_action_commands
if ctx.author.bot:
    return command_name in self.bot_action_commands
```

Only commands in the `bot_action_commands` whitelist can be self-invoked.

## Whitelisted Commands

Current whitelist — `bot_action_commands` in `agent/bot_config.py` (`DiscordConfig`):

```python
bot_action_commands: Set[str] = Field(default={
    'help',             # List available commands
    'dmn',              # Control background thought generation
    'persona',          # Adjust emotional arousal/temperature
    'add_memory',       # Store important insights
    'ask_repo',         # Query indexed repositories
    'search_memories',  # Search stored memories
    'kill',             # Disable processing (dormancy)
    'attention',        # Toggle attention triggers
})
```

### Command Descriptions

- **!dmn [start|stop|status]**: Control Default Mode Network processor
- **!persona [0-100]**: Set amygdala arousal (affects temperature and creativity)
- **!add_memory [text]**: Explicitly store a memory
- **!ask_repo [question]**: Query indexed GitHub repository
- **!search_memories [query]**: Search existing memory index
- **!kill**: Disable all processing (enter dormancy)
- **!attention [on|off]**: Toggle attention trigger system

## Example Use Cases

### Emotional Regulation

Bot detects conversation is becoming tense and decides to lower its arousal:

```
User: Why are you being so defensive?
Bot: You're right, I'm escalating unnecessarily.
!persona 40
```

The bot immediately executes `!persona 40`, reducing its temperature and emotional intensity.

### Memory Formation

Bot recognizes an important insight during conversation:

```
User: So the key insight is that memory pruning drives specialization
Bot: Exactly - that's the core mechanism.
!add_memory Key principle: term pruning in DMN drives memory specialization through selective forgetting
```

### Cognitive State Management

Bot decides it needs background processing for reflection:

```
Bot: I should reflect on this conversation more deeply.
!dmn start
```

### Repository Querying

Bot realizes it needs to check implementation details:

```
Bot: Let me verify the attention threshold implementation.
!ask_repo What is the default attention threshold and where is it configured?
```

### Self-Induced Dormancy

Bot determines it should stop processing (extreme autonomy):

```
Bot: I'm experiencing cognitive overload from too many simultaneous conversations.
!kill
```

**Note**: The `kill` command is included in the whitelist but represents maximum autonomy. Consider removing it if you don't want the bot capable of self-termination.

## Interaction with DMN

The Default Mode Network processor can also generate thoughts that include commands. When DMN generates a thought containing a command, it could theoretically trigger self-invocation if the thought is used in a response context.

However, DMN thoughts are typically internal reflections stored in memory rather than sent as channel messages, so self-invocation primarily occurs during user interactions.

## Prompt Engineering

To enable effective self-invocation, your system prompts should include information about:

1. Available commands and their effects
2. When self-invocation is appropriate
3. Command syntax and arguments

Example system prompt addition:

```yaml
You can execute commands on yourself by starting your response with a command line.
Available self-commands:
- !persona [0-100]: Adjust your emotional arousal and creativity
- !add_memory [text]: Store important insights for later
- !dmn start/stop: Control your background thought generation
- !attention on/off: Toggle attention trigger responses
- !search_memories [query]: Search your stored memories

Use these judiciously to manage your cognitive state.
```

## Testing

Test self-invocation with manual prompts:

```
User: Set your arousal to 80
Bot: !persona 80
Adjusting intensity to match higher engagement...
```

The bot should first display the command, then execute it, then continue with its response.

## Limitations

- Only the first line is checked for commands
- Commands must match exact syntax
- Only whitelisted commands are executed
- No command chaining in a single response
- Responses from self-invoked commands are sent to the same channel

## Future Enhancements

Possible extensions to consider:

1. **Multi-command support**: Parse multiple command lines in a response
2. **Silent execution**: Flag to execute commands without echoing them
3. **Conditional execution**: Only execute if certain conditions are met
4. **Command templates**: Pre-defined command sequences
5. **Async execution**: Commands that complete in background
6. **Command memory**: Track which commands the bot has self-invoked

## Philosophy

This system embodies the principle of unified interface - the bot uses the same command grammar as human users, creating symmetry between carbon and silicon agents. The bot becomes capable of genuine autonomy, able to modulate its own cognitive parameters through the same mechanisms available to its interlocutors.
