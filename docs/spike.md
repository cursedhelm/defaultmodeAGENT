# SPIKE / SEEKING

Spike is the DMN's bounded SEEKING path. It runs when pruning leaves a selected
memory unable to resolve another association above the active BM25 threshold.
This is intentionally an energy-bound retrieval condition, not an importance
score and not a claim that the memory has no indexed terms.

## Lifecycle

```text
DMN generation and pruning
  -> unresolved seed
  -> one SEEKING episode
       -> reach an eligible recent channel
       -> message an eligible related user
       -> invoke todo/bookshelf tools
       -> search a related user's memories with a generated query
       -> choose silence
  -> durable action record
  -> private action reflection stored under the agent's id
  -> grounded successor is tested by the next DMN pass
       -> resolves independently: survives normal generation/pruning
       -> remains unresolved: dissolves
```

Every completed, failed, or silent action is reflected into the bot-owned
memory index. The reflection does not by itself count as grounding. A completed
channel message, related-user DM, authorized agent tool, or non-empty scoped
memory search supplies grounding evidence. Silence and an empty memory search
recommend release of the unresolved source.

On grounding, the old source is retired and the reflection becomes its
reconsolidated successor. The successor is queued under the bot's identity. If
it cannot independently resolve on the next DMN pass it is disconnected and
cleaned up, preventing spike from proving relevance with its own bookkeeping.

## Actions

### Recent channel

Successful foreground responses update a per-agent engagement map. Spike keeps
recent Discord text channels and DMs, fetches user-authored text, compresses it,
and ranks it with inline BM25-style term coverage plus current theme resonance.
Only surfaces at or above `match_threshold` are exposed to the decision model.
The model must select an enumerated channel id and provide the final message via
`spike_reach_channel`.

### Related-user DM

`spike_message_user` can target the owner of the unresolved memory or a Discord
member whose exact name is mentioned in it. Bot users and invented ids are not
eligible. Outward channel and DM actions share one cooldown and only one
outward message may execute in an episode.

### Agent tools

Spike receives the same provider-neutral todo and bookshelf tools as the
foreground agent, without attachment-only tools. The bot is the action actor.
It can always manage its own todo list; another user's todo list still requires
an existing grant to the bot. Bookshelf tools operate on that agent's cached
library and can inspect status, list books, or begin a ready book.

### Memory search

`spike_search_user_memories` requires a query authored by the model. The search
is scoped to the unresolved memory owner or an eligible related user and
excludes the unresolved memory itself. A non-empty result grounds the action;
an empty result permits dissolution.

### Silence

`spike_choose_silence` is a successful terminal action. It always produces a
private reflection, but the source memory is released when
`release_on_silence` is enabled.

## Persistence and recovery

Each agent stores:

```text
cache/<agent>/spike/engagement_log.pkl
cache/<agent>/spike/spike.sqlite3
```

SQLite stores the source provenance, typed executions, results, grounding
decision, reflection, and outbox state. On boot, interrupted episodes are
marked failed and reflected rather than repeated. Completed reflections not yet
present in the memory pickle are replayed exactly once by text deduplication.

Raw framework timestamps are stored in SQLite and memory strings. Historical
timestamps and the current moment are rendered through `TemporalParser` before
the model sees them.

## Models and prompts

Ground-truth models live in `agent/tools/spike/models.py`; durable storage lives
in `agent/tools/spike/repository.py`. Shared prompt defaults are:

- `agent/prompts/spike_action_prompt_formats.yaml`
- `agent/prompts/spike_action_system_prompts.yaml`

A persona may override `spike_action_selection` and
`spike_action_reflection`. Existing `spike_engagement` prompts remain available
to the legacy direct outreach method but are not used by the SEEKING dispatcher.

## Configuration

All fields can also be set per bot through `config_overrides.json`.

| Environment variable | Default | Meaning |
|---|---:|---|
| `SPIKE_ENABLED` | `true` | Enable SEEKING |
| `SPIKE_DATABASE` | `spike.sqlite3` | Per-agent action/outbox database |
| `SPIKE_CONTEXT_MESSAGES` | `50` | Initial surface history window |
| `SPIKE_MAX_EXPANSION` | `150` | Maximum prefetched messages |
| `SPIKE_EXPANSION_STEP` | `25` | Context expansion step |
| `SPIKE_MATCH_THRESHOLD` | `0.35` | Minimum eligible surface score |
| `SPIKE_COMPRESSION_RATIO` | `0.6` | Chronpression ratio |
| `SPIKE_COOLDOWN_SECONDS` | `120` | Shared outward-message cooldown |
| `SPIKE_MAX_SURFACES` | `8` | Recent surfaces considered |
| `SPIKE_RECENCY_HOURS` | `512` | Engagement eligibility window |
| `SPIKE_MEMORY_K` | `12` | Scoped memory-search candidates |
| `SPIKE_MEMORY_TRUNCATION` | `512` | Tokens shown per memory result |
| `SPIKE_THEME_WEIGHT` | `0.3` | Theme component of surface score |
| `SPIKE_MAX_TOOL_ACTIONS` | `3` | Tool calls allowed per episode |
| `SPIKE_MAX_ATTEMPTS_PER_MEMORY` | `1` | Completed SEEKING episodes allowed for one source trace |
| `SPIKE_DECISION_TEMPERATURE` | `0.5` | Action-selection temperature |
| `SPIKE_ALLOW_CHANNELS` | `true` | Expose eligible recent channels |
| `SPIKE_ALLOW_DMS` | `true` | Expose related-user DM action |
| `SPIKE_ALLOW_TOOLS` | `true` | Expose agent todo/bookshelf tools |
| `SPIKE_ALLOW_MEMORY_SEARCH` | `true` | Expose scoped generated-query search |
| `SPIKE_RELEASE_ON_SILENCE` | `true` | Retire source after chosen silence |

## Runtime controls

`!spike status`, `/spike status`, `!spike on`, and `!spike off` inspect or
toggle the processor. Status includes recent surfaces, outward cooldown, and
the most recently persisted action.
