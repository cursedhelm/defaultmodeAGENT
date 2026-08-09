# TOOLS.md — Instruments for the Animated Skeleton

> razor parts for the mind’s darkroom. minimal, modular, mean.

---

## Chronpression — Chronomic Text Filter

**Role:** lexical gate. binarizes signal vs chatter; dedups echoes. 

### What it does

* Bimodal significance filter: keeps **novel** (rare) and **significant** (very frequent) words; erases the mushy middle.
* Preserves punctuation/spacing; removes **consecutive duplicate words**.
* Ships as a function and a CLI.

### API (import)

```python
from chronpression import chronomic_filter
t = chronomic_filter(text, alpha=0.3, beta=3.0)
```

### CLI (quick)

```
# light  : -a 0.3 -b 3.0
# medium : -a 0.2 -b 4.0
# heavy  : -a 0.1 -b 6.0
python chronpression.py -i in.txt -o out.txt -a 0.3 -b 3.0
```

### Knobs

* `alpha` → novelty threshold (lower = more things count as novel)
* `beta`  → significance threshold (higher = only the truly frequent survive)

### Use it when

* Pre-conditioning long scrapes/transcripts.
* Purging filler before embedding/rerank.

---

## discordGITHUB — Repo Index & Retrieval

**Role:** pull code in, crack it open, make it searchable. Asynchronous, cached, extension-aware. 

### What it does

* Fetches repo structure/files via GitHub API (1 MB guard on direct file fetch).
* Cleans → tokenizes → builds a lightweight **inverted index** (stopworded).
* Caches index via `CacheManager` (`repo_index.pkl`) per-bot.
* Async background indexing with branch + depth controls.
* Honors `config.files.allowed_extensions`.

### Primitives

```python
from discordGITHUB import Github, GitHubRepo, RepoIndex, start_background_processing

repo = GitHubRepo(token, "owner/name")
idx  = RepoIndex(bot_name="agent", max_depth=3)
await start_background_processing(repo.repo, idx, max_depth=3, branch="main")  # builds cache

hits = await idx.search_repo_async("vector rerank amygdala", k=5)  # [(path, score), ...]
```

### Knobs

* **Traversal:** `max_depth`, `branch`
* **Filtering:** `config.files.allowed_extensions`
* **Cache ops:** `idx.save_cache()`, `idx.clear_cache()`

### Guarantees

* Non-blocking background crawl; sets a readiness event when done.
* Tolerates binary/decoding errors; logs, skips, continues.

---

## discordSUMMARISER — Channel & Thread Digest

**Role:** compress a room into a paragraph; respect names, count heat, keep vibe. 

### What it does

* Walks **main channel** + **threads** (up to `max_entries`).
* Tracks **participants**, **file types**, and content shards.
* Sanitizes Discord mentions → readable names.
* Builds a summary prompt from your `prompt_formats['summarize_channel']` and `system_prompts['channel_summarization']`.
* **Temperature** auto-scaled from **amygdala_response/100**.

### Primitive

```python
from discordSUMMARISER import ChannelSummarizer

summ = ChannelSummarizer(bot, prompt_formats, system_prompts, max_entries=100)
txt  = await summ.summarize_channel(channel_id)  # returns formatted summary string
```

### Inputs/Outputs

* **In:** live Discord history objects; your prompt templates.
* **Out:** single formatted summary: participants, files, content synthesis.

### Notes

* Reverses chunks before summarizing (recent last → model sees latest last).
* Fails soft: returns inline error text if LLM call blows up.

---

## webSCRAPE — Single-Entry Async Scraper

**Role:** one door, six keys. You pass URLs; it returns clean, uniform artifacts. YouTube-aware. Compression-savvy. 

### Contract (always)

Returns a dict:

```python
{ "url", "title", "description", "content", "content_type", "error_info" }
```

* `content_type ∈ {"text/html","youtube","html_preview","none"}`
* `error_info` is **always {}** (silent-error policy; details live in logs)

### Branches

* **YouTube**: detects via multiple patterns; uses `yt_dlp`; fetches subs (VTT or m3u8), **dedups lines**, then **chronomic_filter**; emits Markdown: title, channel, duration, views, description, transcript.
* **HTML**: fetches with UA + 15s timeout, strips UI junk; prefers `<article/main>` or significant text blocks; **chronomic_filter** on body; falls back to **raw HTML preview** safely.

### Token Discipline

* Title ≤ 50 tokens, description ≤ 125, content ≤ 12.5k, transcripts ≤ 15k, preview ≤ 2.5k.
* Middle-truncate with **head/tail preservation** (~40 % head + 40 % tail).

### Primitive

```python
from webSCRAPE import scrape_webpage
res = await scrape_webpage(url)  # awaitable; returns the dict above
```

### Guarantees

* Never returns an “error” content type.
* Normalizes Unicode, unescapes HTML, de-noises punctuation soup.
* Plays nice with downstream token budgets (`tokenizer.count/encode/decode`).

---

## anyDOCER — Single-Entry Document Decoder

**Role:** binary documents in, Markdown out. Wraps `anydoc` (firecrawl/anydoc) so a shared `.pdf`/`.docx`/`.xlsx` enters context the same way a `.txt` attachment does.

### Contract (always)

Returns a dict:

```python
{ "filename", "content", "content_type", "doc_format", "error_info" }
```

* `content_type ∈ {"document","none"}`
* `error_info` is **always {}** (silent-error policy; details live in logs)
* `doc_format` is the detected format (`"pdf"`, `"docx"`, …) or `""` when nothing decoded

### Detection

* Format comes from the **bytes**, not the extension — a mislabeled upload still converts.
* Signature-less formats (CSV) have no magic bytes; the extension is the fallback hint.
* 17 extensions across 8 families: Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, CSV, PDF.

### Token Discipline

* Decoding is **not** budgeting. `decode_bytes` returns the whole document.
* `decode_and_compress` chronpresses only above `chronpress_threshold`, targeting `chronpress_target_chars`.
* Compression is **chunked** (split on blank lines so Markdown tables/lists stay intact) — large PDFs never hit `chronomic_filter` in a single shot.

### Primitives

```python
from tools.anyDOCER import decode_and_compress, is_document

if is_document(att.filename, att.content_type):      # routing gate
    doc = await decode_and_compress(data, att.filename,
                                    threshold_chars=16000, target_chars=8000)
```

CLI: `python -m tools.anyDOCER -i report.pdf -c -v`

### Guarantees

* Never raises — every failure path returns `content_type="none"`.
* Encrypted, malformed, and oversized inputs are logged and skipped, not fatal.
* Degrades gracefully when `anydoc` is not installed (`ANYDOC_AVAILABLE=False`).

---

## How they interlock (fast sketch)

* **webSCRAPE → chronpression:** transcripts/articles get bimodal filtered before downstream LLMs touch them.  
* **anyDOCER → chronpression:** anyDOCER is the *decoder* (bytes → Markdown), chronpression the *budgeter* (long Markdown → context-sized). They compose; they don't compete.  
* **discordGITHUB → cache/index → query:** background job builds a lightweight IR; queries are cheap and immediate. 
* **discordSUMMARISER → prompts + amygdala:** summarizer temperature flows from state; templates stay external. 

---

## Minimal Recipes

### Scrape & Summarize a Link Drop (DM or Channel)

```python
urls=[...]
blobs=[await scrape_webpage(u) for u in urls]
text="\n\n".join(b["content"] for b in blobs if b["content"])
out = chronomic_filter(text,0.3,3.0)
```

 

### Index a Repo, Answer a Code Question

```python
repo=GitHubRepo(token,"o/n")
idx=RepoIndex("agent",3)
await start_background_processing(repo.repo,idx,3,"main")
hits=await idx.search_repo_async("memory rerank prune",5)
```



### Summarize a Busy Channel

```python
txt=await ChannelSummarizer(bot,prompt_formats,system_prompts,200).summarize_channel(cid)
```



---