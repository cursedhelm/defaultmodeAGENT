# Bookshelf and READER

Each agent owns an isolated library at `cache/<agent>/bookshelf`. Drop PDF or EPUB files anywhere beneath that directory (the `inbox/` folder is provided for clarity), or ask the agent to add an attached book through its foreground tools.

The managed layout is:

```text
cache/<agent>/bookshelf/
  bookshelf.sqlite3
  inbox/
  books/<title>--<hash>/
    source.pdf|epub
    book.md
    media/
```

On boot, ingestion converts the source to Markdown, extracts available images, creates stable reading chunks, embeds them, and commits metadata plus vectors to SQLite. EPUB structure and assets use anydoc; PDF text, page locators, and images use PyMuPDF. The source and generated Markdown remain available for inspection.

EPUB package anchors and internal XHTML navigation are removed from the reading copy; headings, emphasis, lists, tables, external links, notes, and images remain Markdown. Libraries created before this cleanup are upgraded on boot, with their existing chunk IDs, reading cursors, and reflection events preserved while the cleaned chunks are re-embedded.

The READER resumes its SQLite cursor at the global tick rate. If no book is active, it takes the densest memory-graph representation as a curiosity seed, hybrid-ranks unread books with BM25 and embeddings, and chooses through a constrained `bookshelf_begin` tool call. Every section recalls normal agent memories and already-reached bookshelf chunks before the configured READER model reflects on it. It never uses future chunks as priors.

Prompts use natural temporal expressions from `TemporalParser`. The authoritative reading event and autobiographical memory retain `(HH:MM [DD/MM/YY])`, matching the rest of the memory index.

Foreground model tools are `bookshelf_status`, `bookshelf_list`, `bookshelf_begin`, and, when a PDF/EPUB is attached, `bookshelf_add`. They are available in server turns, Discord DMs, and TUI chat. Libraries are not shared between bot profiles; two agents using this tooling operate on different cache roots.

Set `READER_API_TYPE` and `READER_MODEL`, or launch Discord with `--reader-api` and `--reader-model`, to route background reading independently from chat and DMN. The TUI launcher exposes the same choices. See `.env.example` for indexing, retrieval, media, interval, and prior-scope settings. `READER_PRIOR_SCOPE=global` matches normal public-channel recall; use `agent` to restrict memory priors to the bot's own memory owner ID.

The YAML interfaces are `agent/prompts/reading_prompt_formats.yaml` and `agent/prompts/reading_system_prompts.yaml`. A persona can override their named keys in its existing prompt files.
