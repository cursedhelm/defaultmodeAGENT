# Prompting


## Layout

```
agent/
  prompts/
    {agent_name}/
      system_prompts.yaml
      prompt_formats.yaml
      images/
        profile_1x1.png
        character_9x16.png
        banner_680x240.png
```

The bot loads `prompt_formats.yaml` + `system_prompts.yaml` from the resolved `--prompt-path` plus the `--bot-name` subdir. Default is `agent/prompts/{bot_name|default}`; it hard-fails if the path doesn’t exist.  
At startup it opens both YAMLs and injects them into the bot (`bot.prompt_formats`, `bot.system_prompts`).  

---

## Agent anatomy

Each agent = 2 layers:

1. **System Prompts** (role, rules, state): keys like `default_chat`, `file_analysis`, `repo_file_chat`, `ask_repo`, `channel_summarization`, `thought_generation`, `image_analysis`, `combined_analysis`, plus optional `attention_triggers`. These templates include `{amygdala_response}` and may embed `{themes}`.   

2. **Prompt Formats** (task-specific f-strings): keys like `chat_with_memory`, `introduction`, `analyze_file`, `analyze_image`, `analyze_combined`, `repo_file_chat`, `ask_repo`, `summarize_channel`, `generate_thought`. They carry `{assembled_context}`, `{user_name}`, `{user_message}`, `{filename}`, `{file_content}`, `{image_files}`, `{text_files}`, `{file_path}`, `{repo_code}`, `{question}`, `{timestamp}`, `{memory_text}`.

**Runtime binding**: the bot selects a format (`introduction` vs `chat_with_memory`), fills variables, and pairs it with the matching system key (`default_chat`, etc.). Thoughts use `thought_generation` + `generate_thought`.  

---

## Attention (ambient responsiveness)

Agents can “wake” on topic triggers without @mentions by defining `attention_triggers` in **system_prompts.yaml**. The message is checked against these triggers before deciding to respond.  

```yaml
# system_prompts.yaml
attention_triggers:
  - "emergent patterns"
  - "complexity spiral"
  - "recursive thinking"
```

---

## State & affect

`{amygdala_response}` (0–100) governs temperature/behavior across all system templates: low = precise; mid = balanced; high = exploratory/creative/skeptical. The bot synchronizes this with model temperature.   

---

## File roles (authoring)

### `system_prompts.yaml` (define the *voice* and *constraints*)

Keep each key minimal, declarative, with f-variables only for **state** and **themes**.

```yaml
# system_prompts.yaml (skeleton)
default_chat: |
  you are "{name}", intensity {amygdala_response}%.
  {themes}

file_analysis: |
  you are "{name}", analyzing files at {amygdala_response}%.
  ## config (json-ish or bullets)
  ...

repo_file_chat: |
  you are "{name}", guiding code reading at {amygdala_response}%.
  <file_path>{{FILE_PATH}}</file_path>
  <repo_code>{{REPO_CODE}}</repo_code>

ask_repo: |
  you are "{name}", exploring repos at {amygdala_response}%.
  ...

channel_summarization: |
  you are "{name}", summarizing at {amygdala_response}%.
  ...

thought_generation: |
  you are "{name}", private thoughts at {amygdala_response}%.
  ...

image_analysis: |
  you are "{name}", image focus at {amygdala_response}%.

combined_analysis: |
  you are "{name}", multimodal at {amygdala_response}%.

attention_triggers:
  - "keyword a"
  - "keyword b"
```

(See the shipped variants for deeper patterns and JSON-like tuning blocks.)   

### `prompt_formats.yaml` (define the *slots/IO*)

This is pure formatting: shove context + user I/O + content into tight shapes per task.

```yaml
# prompt_formats.yaml (skeleton)
chat_with_memory: |
  {assembled_context}
  @{user_name}: {user_message}

analyze_file: |
  {assembled_context}
  File: {filename}
  Content:
  {file_content}
  User: @{user_name} — {user_message}

analyze_image: |
  {assembled_context}
  Image: {filename}
  User: @{user_name} — {user_message}

analyze_combined: |
  {assembled_context}
  Images:
  {image_files}
  Text:
  {text_files}
  User: @{user_name} — {user_message}

repo_file_chat: |
  {assembled_context}
  File: {file_path}
  Type: {code_type}
  Content:
  {repo_code}
  Task: {user_task_description}

ask_repo: |
  {assembled_context}
  {question}

generate_thought: |
  Memory about @{user_name}: {memory_text}
  Timestamp: {timestamp}
```

(Your current formats already include guidance text; keep them terse.)  

---

## Required variables (by key)

* **System**: `{amygdala_response}`, optionally `{themes}`, `{name}`.
* **Chat**: `{assembled_context}`, `{user_name}`, `{user_message}`.
* **Files**: `{filename}`, `{file_content}`.
* **Images**: `{filename}` or `{image_files}`.
* **Repo**: `{file_path}`, `{code_type}`, `{repo_code}`, `{user_task_description}`; or `{question}` (RAG).
* **Thoughts**: `{memory_text}`, `{timestamp}`.

The bot chooses `introduction` on first contact; otherwise `chat_with_memory`.  

---

## Author a new agent (quick path)

1. **Create folder** `agent/prompts/{agent_name}/`. Add your `images/`.
2. **Copy** baseline `system_prompts.yaml` + `prompt_formats.yaml` into that folder.
3. **Edit `system_prompts.yaml`**: set voice, values, and `attention_triggers`. Keep `{amygdala_response}`.
4. **Edit `prompt_formats.yaml`**: ensure each task has minimal slots.
5. **Run** with `--bot-name {agent_name}` so the loader targets your folder.  

---

## Social media assets

Each agent may ship a face/body/banner for continuity and rigging:
`images/` → 1:1 profile, 9:16 character sheet, 680×240 banner. (Used by your downstream pipelines; keep filenames deterministic.)

---

## Design notes (operational)

* **Pairing**: `system_prompts[key]` + `prompt_formats[key]` must exist for a task; the bot raises if a required pair is missing (esp. combined analysis). 
* **Context engine**: the runner weaves `<conversation>`, memories, URLs, and reactions into `{assembled_context}` before formatting the user-role content. It is not also sent as supplemental system context.
* **DMN/Thoughts**: post-reply, the bot emits an internal “thought” via `thought_generation` using your formats; this is memory fuel, not for users. 

---

## Minimal example: spawn a sharp agent

```yaml
# agent/prompts/sharp/system_prompts.yaml
default_chat: |
  you are "sharp", lucid, surgical. intensity {amygdala_response}%.
  {themes}
attention_triggers:
  - "regression"
  - "vector db"
```

```yaml
# agent/prompts/sharp/prompt_formats.yaml
chat_with_memory: |
  {assembled_context}
  @{user_name}: {user_message}
```

Run:

```
python discord_bot.py --bot-name sharp --prompt-path agent/prompts
```

 

---
