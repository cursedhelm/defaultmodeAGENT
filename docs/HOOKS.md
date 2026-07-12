# defaultMODE Platform Hooks

This document describes the two interfaces you must implement to run the
defaultMODE agent on a new platform (Telegram, CLI, web, etc.).

---

## Overview

```
New Platform
    └── MyAdapter(PlatformAdapter)   ← I/O, formatting, history
    └── MyRuntime(AgentRuntime)      ← LLM access, state, identity

agent_core.process_message(msg, adapter, runtime, ...)
    ├── adapter.fetch_history(...)
    ├── adapter.thinking(...)        ← "bot is typing"
    ├── adapter.format_context_header(msg)
    ├── adapter.sanitize_content(...)
    ├── adapter.format_response(...)
    ├── adapter.send(channel_id, text)
    └── runtime.call_api(...)
```

---

## NormalizedMessage

Produced by your adapter's `normalize()` method and passed everywhere.

> `normalize()` is a convention, not part of the `PlatformAdapter` ABC — its
> signature differs per platform (Discord's takes a `discord.Message`; the TUI
> constructs `NormalizedMessage` directly and defines no `normalize()` at all).
> Build the `NormalizedMessage` however suits your platform.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Platform message ID |
| `content` | `str` | User text, bot-mention-stripped |
| `author_id` | `str` | Platform user identifier |
| `author_name` | `str` | Display name |
| `channel_id` | `str` | Where to send replies |
| `channel_name` | `str` | Human-readable channel name |
| `guild_name` | `str \| None` | Server/workspace name; `None` for DMs |
| `is_dm` | `bool` | True if private conversation |
| `mentions_bot` | `bool` | True if the agent should respond |
| `attachments` | `List[NormalizedAttachment]` | Files attached to the message |
| `reply_to` | `NormalizedMessage \| None` | Replied-to message, if any |
| `raw` | `Any` | Original platform object (escape hatch) |

### NormalizedAttachment

| Field | Type | Description |
|---|---|---|
| `filename` | `str` | Original filename |
| `content_type` | `str` | MIME type e.g. `"image/png"` |
| `size` | `int` | Bytes |
| `_fetch` | `async () -> bytes` | Platform-specific download function |

Call `await attachment.read()` to download content.

---

## PlatformAdapter (`agent/adapters/base.py`)

### Required methods

```python
async def send(self, channel_id: str, text: str) -> None
```
Send text to the channel.  Split long messages as needed for your platform.

```python
async def thinking(self, channel_id: str)  # async context manager
```
Signal that the agent is working.  Use `async with adapter.thinking(channel_id):` around API calls.

```python
async def fetch_history(
    self, channel_id: str, limit: int, skip_id: str | None = None
) -> tuple[list, dict]
```
Return `(messages, reactions_map)` in the format consumed by `context.process_history_dual`.
`skip_id` is the triggering message's ID and should be excluded.
For platforms without reactions, return an empty dict as `reactions_map`.

```python
def format_context_header(self, msg: NormalizedMessage) -> str
```
Return the platform context string injected into every LLM prompt.
Example: `"Current Discord server: X, channel: #general\n"`

```python
def format_response(self, response: str, msg: NormalizedMessage) -> str
```
Post-process the LLM response before delivery.  On Discord this converts
`@username` strings to snowflake mentions.  Return the string unchanged if
no conversion is needed.

```python
def sanitize_content(self, content: str, msg: NormalizedMessage) -> str
```
Strip platform-specific markup from user content before it reaches the LLM.

### Optional methods

```python
async def invoke_embedded_commands(self, response: str, msg: NormalizedMessage) -> None
```
Default is a no-op.  Override to scan the LLM response for whitelisted
commands and execute them (used by the Discord adapter for `!commands`).

---

## AgentRuntime (`agent/runtime.py`)

The runtime is the host object the DMN and spike processor call into.  On
Discord the `commands.Bot` instance satisfies this protocol after `setup_bot()`
attaches a few extra attributes.

### Required properties / methods

```python
@property
def agent_id(self) -> str
```
Stable identity string for this agent.  Used as a fallback `user_id` when
storing spike-originated memories.

```python
@property
def agent_name(self) -> str
```
Human-readable display name (e.g. bot username).  Used in memory strings:
`@{agent_name}: {response}`.

```python
@property / @setter
def amygdala_response(self) -> int   # 0–100
```
Current arousal level.  The DMN writes this back after each thought cycle.

```python
@property
def processing_enabled(self) -> bool
```
Kill switch.  When `False`, `agent_core.process_message` returns immediately.

```python
async def resolve_user(self, user_id: str) -> str
```
Resolve a stored `user_id` string to a display name.  Fall back to
`f"User({user_id})"` on lookup failure.

```python
async def call_api(self, **kwargs) -> str
```
Call the LLM.  Accepts the same kwargs as `api_client.call_api`:
`prompt`, `system_prompt`, `temperature`, `api_type_override`,
`model_override`, `image_paths`, `audio_paths`, etc.  Forward `**kwargs`
verbatim so new media kwargs keep working as the core evolves.

```python
def update_api_temperature(self, temperature: float) -> None
def update_api_top_p(self, top_p: float) -> None
```
Adjust sampling parameters on the API client.

```python
@property
def spike_processor(self) -> SpikeProcessor | None
```
The spike processor instance, or `None`.

```python
@property
def logger(self) -> BotLogger
```
Logging target.  Must implement `.info()`, `.debug()`, `.warning()`,
`.error()`, and optionally `.log(dict)` for structured JSONL events.

---

## Minimal skeleton: CLI adapter

```python
# agent/adapters/cli_adapter.py
import asyncio
from contextlib import asynccontextmanager
from .base import NormalizedAttachment, NormalizedMessage, PlatformAdapter

class CLIAdapter(PlatformAdapter):

    async def send(self, channel_id: str, text: str) -> None:
        print(text)

    @asynccontextmanager
    async def thinking(self, channel_id: str):
        print("...")
        yield

    async def fetch_history(self, channel_id, limit, skip_id=None):
        return [], {}  # no history on CLI

    def format_context_header(self, msg: NormalizedMessage) -> str:
        return "Current channel: CLI\n"

    def format_response(self, response: str, msg: NormalizedMessage) -> str:
        return response  # no mention conversion needed

    def sanitize_content(self, content: str, msg: NormalizedMessage) -> str:
        return content  # no markup to strip

    def normalize(self, raw_input: str, author_id="cli", author_name="user") -> NormalizedMessage:
        return NormalizedMessage(
            id="0",
            content=raw_input,
            author_id=author_id,
            author_name=author_name,
            channel_id="cli",
            channel_name="cli",
            guild_name=None,
            is_dm=True,
            mentions_bot=True,
        )
```

### Minimal runtime

```python
# agent/adapters/cli_runtime.py
import api_client   # load_private_api_client equivalent

class CLIRuntime:
    agent_id = "cli-agent"
    agent_name = "agent"
    amygdala_response = 50
    processing_enabled = True
    spike_processor = None

    def __init__(self, logger):
        self.logger = logger

    async def resolve_user(self, user_id: str) -> str:
        return user_id

    async def call_api(self, **kwargs) -> str:
        return await api_client.call_api(**kwargs)

    def update_api_temperature(self, t):
        api_client.update_api_temperature(t)

    def update_api_top_p(self, p):
        api_client.update_api_top_p(p)
```

### Wiring it up

```python
from agent_core import process_message
from adapters.cli_adapter import CLIAdapter
from adapters.cli_runtime import CLIRuntime
from memory import UserMemoryIndex

adapter = CLIAdapter()
runtime = CLIRuntime(logger=my_logger)
memory = UserMemoryIndex("cache/cli/memory_index")

while True:
    raw = input("> ")
    msg = adapter.normalize(raw)
    asyncio.run(process_message(msg, adapter, runtime, memory, prompt_formats, system_prompts))
```

---

## What does NOT need to change

- `memory.py` — `UserMemoryIndex` is fully platform-neutral
- `defaultmode.py` — uses `AgentRuntime` only
- `attention.py`, `temporality.py`, `chunker.py` — pure utilities
- `api_client.py` — LLM providers, no platform coupling
- Discord `!commands` — remain Discord-only; no abstraction needed
