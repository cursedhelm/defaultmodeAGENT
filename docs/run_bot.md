# run_bot.py — DefaultMODE Agent Manager TUI

## Overview

`run_tui.py` is the entry point for a [Textual](https://textual.textualize.io/)-based terminal UI that manages defaultMODE Discord bot agents. The application is split across `run_tui.py` (app shell) and the `tui/` package (page modules and shared utilities).

Features:
- **Multi-instance bot launching** with per-instance log streams and stop controls
- **Log viewer** with JSONL parsing, search, and auto-refresh
- **YAML prompt editor** with live validation against schema requirements
- **Memory inspector** with search, inline editing, find/replace, and user cascade delete
- **Memory latent-space visualizer** with PCA/UMAP projection, ASCII canvas, zoom/pan, and connection graph

Keyboard shortcuts `1`–`5` navigate tabs; `q` quits.

---

## Package Structure

```
run_tui.py               ← App entry point and AgentManagerApp shell
agent/viz_live.py        ← localhost live snapshot protocol used by Discord
tui/
├── __init__.py          ← Exports all page classes
├── shared.py            ← Models, globals, utility functions, memory ops, viz helpers
├── launch_page.py       ← LaunchPage + BotInstanceCard
├── logs_page.py         ← LogsPage
├── prompts_page.py      ← PromptsPage
├── memory_page.py       ← MemoryPage
├── viz_page.py          ← VizPage
└── viz_graph.py         ← read-only sparse graph model and derived cache
tui/run_bot.css          ← Pink-accented stylesheet (#ffb6c1 / #ff69b4)
```

---

## `run_bot.py` — App Shell

### `_TextRedirector`

Captures `stdout`/`stderr` writes and routes them to the status bar `#console-bar` label. Thread-safe: uses `call_from_thread` when writing from non-main threads.

### `AgentManagerApp(App)`

| Method | Purpose |
|--------|---------|
| `compose()` | Builds status bar, `TabbedContent` with 5 panes, console bar, footer |
| `on_mount()` | Redirects `sys.stdout` and `sys.stderr` to `_TextRedirector` |
| `push_console(text)` | Updates console bar with last message (truncated to 200 chars) |
| `update_global_status()` | Syncs status bar with `STATE.selected_bot/api/model` and running count |
| `action_quit()` | Restores streams, sends interrupt to all running instances, exits |
| `action_tab_*()` | Tab switching for keys 1–5 |

**Bindings:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `1`–`5` | Switch to Launch / Logs / Prompts / Memory / Viz tab |

---

## `tui/shared.py` — Shared Layer

All pages import from here. Contains models, globals, utility functions, memory operations, and visualization helpers.

### Models

#### `PathConfig(BaseModel)`

Pydantic model for centralized path management.

| Property/Method | Returns | Purpose |
|---|---|---|
| `root` | `Path` | Project root (directory of `tui/`) |
| `prompts_dir` | `Path` | `agent/prompts/` |
| `cache_dir` | `Path` | `cache/` |
| `discord_bot` | `Path` | `agent/discord_bot.py` |
| `bot_prompts(name)` | `Path` | `agent/prompts/<name>/` |
| `bot_memory(name)` | `Path` | `cache/<name>/memory_index/memory_cache.pkl` |
| `bot_log(name)` | `Path` | `cache/<name>/logs/bot_log_<name>.jsonl` |
| `bot_system_prompts(name)` | `Path` | `agent/prompts/<name>/system_prompts.yaml` |
| `bot_prompt_formats(name)` | `Path` | `agent/prompts/<name>/prompt_formats.yaml` |

#### `BotInstance(dataclass)`

Represents one running bot process.

| Field | Type | Purpose |
|-------|------|---------|
| `bot_name` | `str` | Bot identifier |
| `api` | `str` | API provider |
| `model` | `str` | Model string |
| `dmn_api` | `Optional[str]` | Separate API for DMN (background thought) processing |
| `dmn_model` | `Optional[str]` | Separate model for DMN |
| `process` | `Optional[Any]` | `asyncio.subprocess.Process` handle |
| `running` | `bool` | Whether the process is alive |
| `worker` | `Optional[Any]` | Textual worker reference |
| `instance_id` | `str` (property) | Lowercase, hyphenated bot name (used for widget IDs) |

#### `AppState`

Global mutable state singleton (`STATE`).

| Field/Property | Type | Purpose |
|---|---|---|
| `selected_bot` | `str` | Currently selected bot in the launch panel |
| `selected_api` | `str` | Selected API provider |
| `selected_model` | `str` | Selected model string |
| `dmn_api` | `str` | DMN API selection |
| `dmn_model` | `str` | DMN model selection |
| `instances` | `Dict[str, BotInstance]` | All launched instances keyed by bot name |
| live contexts | `Dict[str, LiveBotContext]` | In-process Chat memory/runtime references published for read-only consumers |
| `running_count` | `int` (property) | Count of instances where `running=True` |
| `is_bot_running(name)` | `bool` | Check if a specific bot is currently running |

**Note:** `AppState` supports multiple simultaneous bot instances. The old single `process`/`running` fields are replaced by the `instances` dict.

---

### Globals

```python
PATHS = PathConfig()   # Singleton path config
STATE = AppState()     # Singleton app state
```

---

### Bot Discovery

| Function | Purpose |
|----------|---------|
| `discover_bots()` | Lists subdirs in `agent/prompts/` (excluding `.`, `__`, `archive` prefixes) |
| `get_bot_caches()` | Lists bot names that have a `memory_cache.pkl` file |

---

### API & Model Discovery

| Function | Purpose |
|----------|---------|
| `get_default_model(api)` | Returns default model for a provider via `get_api_config()` |
| `get_api_env_key(api)` | Maps provider name → environment variable key |
| `check_api_available(api)` | Returns `True` if the required env var is set (Ollama always true) |
| `get_models_for_api(api)` | Dispatches to provider-specific lister; suppresses stdout/stderr |

**Model listers:**

| Function | Provider | Notes |
|----------|----------|-------|
| `list_ollama_models()` | Ollama | Runs `ollama list` CLI |
| `list_openai_models()` | OpenAI | Filters for gpt/o1/o3/o4 |
| `list_anthropic_models()` | Anthropic | Falls back to hardcoded list on error |
| `list_vllm_models()` | vLLM | Queries `VLLM_API_BASE/v1/models` |
| `list_openrouter_models()` | OpenRouter | Returns up to 10 free + 10 paid |
| `list_gemini_models()` | Google Gemini | Filters for `generateContent` capability |

---

### Prompt Handling

Prompt schema requirements (`REQ_SYS`, `REQ_FMT`) are imported from `agent/bot_config.PromptSchema`.

| Function | Purpose |
|----------|---------|
| `extract_tokens(s)` | Extracts `{placeholder}` tokens from a template string |
| `load_yaml_file(path)` | Safe YAML load with error fallback to `{}` |
| `save_yaml_file(path, data)` | Writes dict as YAML with unicode support |
| `validate_prompts(sys, fmt)` | Checks required tokens per key, returns dict with `missing` sets and `valid` flag |
| `create_bot_stub(name)` | Creates `agent/prompts/<name>/` with template YAML stubs |

---

### Text Processing

| Function | Purpose |
|----------|---------|
| `tokenize(text)` | Normalizes for search: strips special tokens, punctuation, numbers, stopwords; keeps words ≥5 chars |

---

### Memory Operations

The memory system uses pickle-serialized files with three structures:
- `memories`: `List[Optional[str]]` — `None` marks deleted entries
- `user_memories`: `Dict[user_id, List[int]]` — maps users to memory indices
- `inverted_index`: `Dict[token, List[int]]` — BM25-style inverted index

| Function | Purpose |
|----------|---------|
| `load_memory_cache(bot_name)` | Loads pickle file; returns dict with `memories`, `user_memories`, `inverted_index`, `path` |
| `save_memory_cache(cache)` | Atomic save via temp file + `os.replace` |
| `search_memories(cache, query, user_id, page, per_page)` | BM25 TF-IDF search when query tokenizes; substring fallback for short queries; returns all if no query |
| `delete_memory(cache, mid)` | Nullifies entry, removes from inverted index and user maps |
| `update_memory(cache, mid, new_text)` | Replaces memory text and rebuilds its index entries |
| `_rebuild_index(cache)` | Full inverted index reconstruction from scratch |
| `find_replace_memories(cache, find, replace, ...)` | Regex find/replace with case/whole-word options; rebuilds index on change |
| `delete_user_cascade(cache, user_id)` | Nullifies all of a user's memories, removes user entry, rebuilds index |

---

### Visualization Helpers

Used by `VizPage` for latent-space rendering.

#### `VizNode(dataclass)`

| Field | Purpose |
|-------|---------|
| `mid` | Memory index |
| `x`, `y` | 2D projection coordinates |
| `text` | Memory text |
| `user_id` | Owning user |
| `score` | Combined TF-IDF magnitude + distance-from-centroid score (0–1) |
| `grid_x`, `grid_y` | Rendered grid position |

#### Viz Functions

| Function | Purpose |
|----------|---------|
| `build_tfidf_vectors(cache, memory_ids)` | Builds normalized TF-IDF from surviving inverted-index postings, excluding singleton and very common terms; returns `(vectors, terms, raw_magnitudes)` |
| `reduce_dimensions(vectors, method)` | Reduces to two dimensions with LSA, PCA, or cosine UMAP |
| `find_connections(cache, mid, top_k)` | Finds top-K related memories by Jaccard similarity over shared index terms |
| `render_ascii_viz(nodes, width, height, ...)` | Renders nodes as ASCII canvas with box border, connection lines, and legend |
| `_draw_line(grid, x1, y1, x2, y2, ...)` | Bresenham-style line draw using `─│╲╱┼` box-drawing characters |

### Read-only graph model

`tui/viz_graph.py` separates the memory graph from its Textual representation.
`VizGraph` owns the sparse topology, user ownership, read-only theme snapshot,
cosine-neighbor navigation, search, and serializable node payloads. It resolves
sources in this order: an in-process Chat `UserMemoryIndex`, a running Discord
bot's localhost live hook, a current/last-known derived graph cache, then the
offline `memory_cache.pkl` checkpoint. Both live transports expose the same
versioned `memory` / `runtime` / `themes` schema. The graph cache is disposable
and never writes or reconstructs the canonical memory pickle.

#### `SelectableItem(ListItem)`

Shared list item widget showing `✓/✗` availability indicator, bold label, and optional dim subtitle.

---

## `tui/launch_page.py` — LaunchPage

### `BotInstanceCard(Vertical)`

A card widget created per running bot instance. Contains:
- Header row: bot name/api/model label, status indicator, Stop button, Log toggle button
- `RichLog` widget (collapsible via `toggle-log-*` button or `.expanded` CSS class)

| Method | Purpose |
|--------|---------|
| `get_log()` | Returns the `RichLog` widget |
| `update_status(running)` | Switches status label between `● RUNNING` and `● STOPPED` |
| `toggle_log_visibility()` | Toggles `.expanded` class to show/hide log |

### `LaunchPage(Vertical)`

Three-column selection panel + scrollable instances panel.

**Layout:**
```
┌──────────────────────────────────┬─────────────────────────┐
│ [Bot list] [API list] [DMN list] │ Running Instances        │
│            [Model]  [DMN model]  │ ┌─ BotInstanceCard ─┐   │
│ config summary                   │ │ name / api / model│   │
│ [Launch Bot]  [Stop All]         │ │ [RichLog output]  │   │
└──────────────────────────────────┴─────────────────────────┘
```

**Key behaviors:**
- Selecting an API triggers async `_fetch_models()` (threaded worker) to populate the model list
- DMN column is optional; selecting `(same)` leaves `dmn_api`/`dmn_model` as `None`
- Launch spawns `discord_bot.py` as a subprocess; a `BotInstanceCard` appears in the instances panel
- Output lines are streamed to the card's `RichLog` with ERROR/WARNING colorization
- Per-instance Stop sends `CTRL_BREAK_EVENT` (Windows) or `SIGINT` (Unix); falls back to `taskkill`/`SIGKILL` after 10s
- Stop All iterates `STATE.instances` and kills each

**Process lifecycle:**
1. `on_launch()` → creates `BotInstance`, mounts `BotInstanceCard`, calls `_run_bot()`
2. `_run_bot()` (async worker) → `asyncio.create_subprocess_exec`, reads stdout line-by-line
3. `_kill_instance()` → graceful interrupt → force kill after timeout
4. On exit: updates card status, calls `app.update_global_status()`

---

## `tui/logs_page.py` — LogsPage

JSONL log viewer.

**Layout:** bot dropdown + search input + refresh interval selector + Refresh button → status bar → `TextArea` (read-only)

**Key behaviors:**
- Bot list is populated from any `cache/<name>/logs/` directories
- `_load_logs(full=True)` reads and parses entire JSONL file; `full=False` skips if file size unchanged
- Auto-refresh uses Textual's `set_interval()` (Off / 5s / 10s / 30s options)
- `_render_logs()` formats each entry as a text block:
  - Header: `[timestamp] EVENT (level)`
  - Fields: `Key: value` with special prefixes (`>>> ` for user messages, `<<< ` for AI, `!!! ` for errors)
  - Values truncated to 1000 chars; displays last 300 entries
- Search filters blocks by keyword (case-insensitive substring match)

---

## `tui/prompts_page.py` — PromptsPage

Dual-pane YAML editor for bot personality files.

**Layout:** sidebar (bot dropdown + list + create/save/validate buttons) + main area (side-by-side `TextArea` editors + validation output)

**Key behaviors:**
- Bot list shows `✓/✗` based on whether `system_prompts.yaml` exists
- `_load_bot()` reads both YAML files into the two `TextArea` editors as raw text
- Save: parses both editors with `yaml.safe_load` and writes via `save_yaml_file`
- Validate: calls `validate_prompts()`, displays missing tokens per key or "all tokens valid"
- Create: validates name regex (`[A-Za-z0-9_\-]+`), calls `create_bot_stub()`, auto-selects new bot
- Refresh: re-discovers bots from filesystem

---

## `tui/memory_page.py` — MemoryPage

Memory inspection and manipulation interface.

**Layout:**
```
[bot select] [Load] [Save] [⚠ warning]
[search query] [user filter] [Search] [Delete User]
[find] [replace] [case] [whole] [Find/Replace] [status]
stats
ScrollableContainer (memory items)
[< prev] [page N/M] [next >]
```

**Key behaviors:**
- Load reads pickle via `load_memory_cache()`; displays memory count, user count, term count
- Save is blocked (button disabled + warning) when the bot is currently running
- Search uses `search_memories()`:
  - Tokenizable query → BM25 TF-IDF ranking
  - Short/symbol query → substring match
  - Empty query → shows all memories
- Each memory item renders as: header label (id, user, score) + editable `TextArea` + Update/Delete buttons
- Update: reads edited `TextArea` text, calls `update_memory()` to replace text and rebuild index entries
- Delete: calls `delete_memory()`, removes from local results list
- Find/Replace: regex substitution with case-sensitive and whole-word options, optional user scope
- Delete User: calls `delete_user_cascade()` for the selected user
- Pagination: 30 items per page; `<`/`>` buttons, page info label

---

## `tui/viz_page.py` — VizPage

Memory latent-space visualizer.

**Layout:**
```
[bot select] [user filter] [method: LSA/PCA/UMAP]
[Context ☐] [Extended ☐] [Themes ☑] [Refresh State] [Rebuild Map]
[graph search or #memory-id] [Find] [Next]
status
┌──────────────────────────────┬──────────────────────┐
│  ASCII canvas                │ Memory Details       │
│  (nodes, connections, density)│ (text + connections) │
└──────────────────────────────┴──────────────────────┘
```

**Visualization pipeline:**
1. Snapshot live Chat or Discord memory read-only, load a derived graph, or read the offline checkpoint
2. Build one global sparse space from the agent's surviving memory/term associations
3. Cache that secondary graph independently from `memory_cache.pkl`
4. Load or reduce to 2D via LSA, PCA, or cosine UMAP
5. Score nodes: `0.5 × distance_from_centroid + 0.5 × TF-IDF magnitude`
6. Apply user and theme scopes without changing coordinates
7. Render a collision-aware terminal canvas with density and connection lines

Selecting **all users** displays the complete global projection. Selecting a user
shows only that user's memories at their global coordinates. **Context** retains
other users as dim landmarks. **Refresh State** takes a new live RAM snapshot
when Chat or a running Discord process hosts that bot, otherwise rereads the
offline checkpoint. **Rebuild Map** bypasses only the derived coordinates and
recomputes the projection.
Normal opening may show the last derived graph immediately after a newer bot
checkpoint; the status labels it `stale` until **Refresh State** is requested.

Running Discord agents advertise an ephemeral localhost endpoint at
`cache/<bot>/viz_live.json`. Requests are authenticated by a per-process token
and stream a consistent memory/runtime/theme snapshot from the warm process.
The advertisement is removed during graceful shutdown; stale advertisements
simply fail over to the derived/offline sources. Concurrent snapshot requests
are serialized to avoid multiplying peak RAM use on large indexes.

**Canvas rendering:**
- Single-node characters by score: `◆ ● ○ ∘` (high → low)
- Theme-associated nodes/stacks: `◇ ◈`
- Selected node: `◉`; connected nodes: `◎`
- Braille cells retain 2×4 sub-cell structure; `▓` and `█` mark denser stacks
- Connection lines between selected and related nodes: `─ │ ╲ ╱ ┼`
- Percentile bounds prevent isolated projection outliers from collapsing the map
- Every occupied cell retains its complete memory stack for navigation

**Interaction:**

| Input | Action |
|-------|--------|
| `W/A/S/D` | Navigate to nearest occupied cell in direction |
| `↑↓←→` | Pan viewport 15% of current range |
| `+` / `-` | Zoom in / out |
| `f` | Focus viewport on selected node |
| `[` / `]` | Cycle memories occupying the selected cell |
| `n` / `p` | Follow the first/last displayed connection |
| `Enter` | Show full memory text in detail panel |
| Mouse click | Select nearest occupied cell; repeated clicks cycle its stack |
| Scroll up/down | Zoom in/out over canvas |
| Search + `Find` | Search graph terms/text or jump directly with `#<memory-id>` |
| `Next` | Cycle through the current search result set |

**Detail panel:**
- Shows selected memory's text, user, score, and matching cached themes
- Lists top 6 (or 16 in Extended mode) connected memories with similarity score and shared terms
- Connections are sparse cosine neighbors from the same surviving-term graph used by the projection

---

## Required Prompt Tokens

Source of truth is `agent/bot_config.PromptSchema`.

### System Prompts (`REQ_SYS`)
Key prompts require `{amygdala_response}` for emotional intensity injection.

### Prompt Formats (`REQ_FMT`)
Each format key requires specific `{placeholder}` tokens. Partial list:

| Format | Required Tokens |
|--------|----------------|
| `chat_with_memory` | context, user_name, user_message |
| `introduction` | context, user_name, user_message |
| `analyze_code` | context, code_content, user_name, user_message |
| `summarize_channel` | channel_name, channel_history |
| `generate_thought` | user_name, memory_text, timestamp, conversation_context |

---

## Entry Point

```python
def main():
    AgentManagerApp().run()

if __name__ == "__main__":
    main()
```

The TUI runs synchronously, blocking until quit. Async operations (model fetching, process management) use Textual's `@work` decorator. On Windows, UTF-8 encoding is forced on `sys.stdout`/`sys.stderr` before the app starts.
