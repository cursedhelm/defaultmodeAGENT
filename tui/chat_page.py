"""Chat page — talk to a bot in-process via the PlatformAdapter/AgentRuntime hooks."""

import io, contextlib, os, time, uuid, yaml
from typing import Optional

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, ListView, RichLog

from tui.shared import (
    PATHS, STATE, SUPPORTED_APIS,
    SelectableItem,
    check_api_available,
    discover_bots,
    get_default_model,
    get_models_for_api,
    load_memory_cache,
    load_tui_names,
    save_tui_names,
)

_NEW_USER = "__new__"
_CHANNEL_ID = "tui-chat"


def _discord_lookup(user_id: str, token: str) -> Optional[str]:
    """
    Fetch the display name for a single Discord user ID via the REST API.
    Returns global_name → username → None in preference order.
    Called from a thread; uses httpx sync client.
    """
    try:
        import httpx
        url = f"https://discord.com/api/v10/users/{user_id}"
        with httpx.Client(timeout=5) as client:
            resp = client.get(url, headers={"Authorization": f"Bot {token}"})
        if resp.status_code == 200:
            data = resp.json()
            return data.get("global_name") or data.get("username")
    except Exception:
        pass
    return None


class ChatPage(Vertical):
    """
    In-process chat interface.

    Selects a bot, API, model, and user-id then routes messages through
    agent_core.process_message using TUIAdapter + TUIRuntime.

    User IDs in the memory index are Discord snowflakes.  "Resolve Names"
    looks them up once via the Discord REST API (using the bot token from .env)
    and caches the results in tui_names.json.  Subsequent runs are fully
    offline.  The "Use display names" toggle controls whether the agent sees
    the resolved name or the raw ID in prompts and memories.
    """

    def compose(self) -> ComposeResult:
        yield Horizontal(
            # ── Left sidebar ────────────────────────────────────────────────
            Vertical(
                Label("[bold]Bot[/bold]"),
                ListView(id="chat-bot-list"),
                Label("[bold]API[/bold]"),
                ListView(id="chat-api-list"),
                Label("[bold]Model[/bold]"),
                ListView(id="chat-model-list"),
                Input(placeholder="or type model...", id="chat-model-input"),
                Label("[bold]User[/bold]"),
                ListView(id="chat-user-list"),
                Input(placeholder="new user id...", id="chat-user-input"),
                Button("Resolve Names", id="chat-resolve-btn", disabled=True),
                Checkbox("Use display names", value=True, id="chat-use-names"),
                Button("Connect", variant="success", id="chat-connect-btn", disabled=True),
                id="chat-sidebar",
            ),
            # ── Right panel ─────────────────────────────────────────────────
            Vertical(
                Label(
                    "[dim]select bot + api + user, then connect[/dim]",
                    id="chat-status",
                ),
                RichLog(id="chat-log", highlight=True, markup=True, wrap=True),
                Horizontal(
                    Input(
                        placeholder="type a message...",
                        id="chat-input",
                        disabled=True,
                    ),
                    Button("Send", id="chat-send-btn", disabled=True),
                    id="chat-input-row",
                ),
                id="chat-panel",
            ),
            id="chat-root",
        )

    def on_mount(self) -> None:
        self._selected_bot: Optional[str] = None
        self._selected_api: Optional[str] = None
        self._selected_model: Optional[str] = None
        self._selected_user: Optional[str] = None
        self._tui_names: dict = {}      # user_id → display name (cached from Discord)
        self._cached_users: list = []   # last-loaded user list, for list refresh
        self._cached_counts: dict = {}  # user_id → memory count

        self._runtime = None
        self._adapter = None
        self._memory_index = None
        self._prompt_formats: Optional[dict] = None
        self._system_prompts: Optional[dict] = None
        self._connected = False
        self._connected_bot: Optional[str] = None

        self.query_one("#chat-user-input", Input).display = False

        self._populate_bots()
        self._populate_apis()

    # ── Sidebar population ─────────────────────────────────────────────────────

    def _populate_bots(self) -> None:
        lv = self.query_one("#chat-bot-list", ListView)
        lv.clear()
        for bot in discover_bots():
            has_cfg = PATHS.bot_system_prompts(bot).exists()
            lv.append(SelectableItem(bot, bot, "ready" if has_cfg else "no config", has_cfg))

    def _populate_apis(self) -> None:
        lv = self.query_one("#chat-api-list", ListView)
        lv.clear()
        for api in SUPPORTED_APIS:
            avail = check_api_available(api)
            lv.append(SelectableItem(api.upper(), api, get_default_model(api), avail))

    @work(thread=True)
    def _fetch_models(self, api: str) -> list:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            result = get_models_for_api(api)
        captured = buf.getvalue().strip()
        if captured and hasattr(self.app, "push_console"):
            try:
                self.app.call_from_thread(self.app.push_console, captured)
            except Exception:
                pass
        return result

    async def _populate_models(self, api: str) -> None:
        lv = self.query_one("#chat-model-list", ListView)
        lv.clear()
        lv.loading = True
        models = await self._fetch_models(api).wait()
        lv.loading = False
        d = get_default_model(api)
        for m in models or ([d] if d else []):
            lv.append(SelectableItem(m, m, "(default)" if m == d else "", True))
        self.query_one("#chat-model-input", Input).value = d or ""

    @work(thread=True)
    def _populate_users(self, bot_name: str) -> None:
        """Load memory cache + name cache in one background thread."""
        try:
            cache = load_memory_cache(bot_name)
            users = sorted(cache.get("user_memories", {}).keys())
            counts = {uid: len(cache["user_memories"][uid]) for uid in users}
        except Exception:
            users = []
            counts = {}
        try:
            tui_names = load_tui_names(bot_name)
        except Exception:
            tui_names = {}
        self.app.call_from_thread(self._apply_users, users, counts, tui_names)

    def _apply_users(self, users: list, counts: dict, tui_names: dict) -> None:
        self._tui_names = tui_names
        self._cached_users = users
        self._cached_counts = counts
        lv = self.query_one("#chat-user-list", ListView)
        lv.clear()
        lv.append(SelectableItem("(+ new)", _NEW_USER, "enter id below", True))
        for uid in users:
            display = tui_names.get(uid, "")
            subtitle = f"{display} • {counts.get(uid, 0)} mem" if display else f"{counts.get(uid, 0)} mem"
            lv.append(SelectableItem(uid, uid, subtitle, True))
        inp = self.query_one("#chat-user-input", Input)
        if not users:
            inp.display = True
            self._selected_user = None
        else:
            inp.display = False
        # Enable resolve button when a bot with a token is selected
        self._refresh_resolve_btn()

    # ── Name resolution via Discord REST API ──────────────────────────────────

    def _refresh_resolve_btn(self) -> None:
        """Enable "Resolve Names" only when a bot token exists in env."""
        btn = self.query_one("#chat-resolve-btn", Button)
        if not self._selected_bot:
            btn.disabled = True
            return
        token_key = f"DISCORD_TOKEN_{self._selected_bot.upper()}"
        btn.disabled = not bool(os.environ.get(token_key))

    @on(Button.Pressed, "#chat-resolve-btn")
    def on_resolve_pressed(self) -> None:
        if not self._selected_bot:
            return
        self._set_status("[dim]resolving names from Discord...[/dim]")
        self.query_one("#chat-resolve-btn", Button).disabled = True
        self._resolve_discord_names(self._selected_bot)

    @work(thread=True)
    def _resolve_discord_names(self, bot_name: str) -> None:
        """
        For every user_id in the memory index that has no cached name,
        call the Discord REST API with the bot token to get the display name.
        Saves results to tui_names.json; no network needed after this.
        """
        token_key = f"DISCORD_TOKEN_{bot_name.upper()}"
        token = os.environ.get(token_key)
        if not token:
            self.app.call_from_thread(
                self._set_status,
                f"[bold]no token:[/bold] {token_key} not found in .env",
            )
            self.app.call_from_thread(self._refresh_resolve_btn)
            return

        uncached = [uid for uid in self._cached_users if uid not in self._tui_names]
        if not uncached:
            self.app.call_from_thread(self._set_status, "all names already cached")
            self.app.call_from_thread(self._refresh_resolve_btn)
            return

        updated = dict(self._tui_names)
        resolved = 0
        for uid in uncached:
            name = _discord_lookup(uid, token)
            if name:
                updated[uid] = name
                resolved += 1
            time.sleep(0.1)  # respect Discord rate limits

        save_tui_names(bot_name, updated)
        self.app.call_from_thread(self._on_names_resolved, updated, resolved, len(uncached))

    def _on_names_resolved(self, names: dict, resolved: int, total: int) -> None:
        self._apply_users(self._cached_users, self._cached_counts, names)
        self._set_status(
            f"resolved [bold]{resolved}/{total}[/bold] names from Discord"
            + (" (non-Discord IDs skipped)" if resolved < total else "")
        )
        self._refresh_resolve_btn()

    # ── Selection handlers ─────────────────────────────────────────────────────

    @on(ListView.Selected, "#chat-bot-list")
    def on_bot_selected(self, event: ListView.Selected) -> None:
        if not isinstance(event.item, SelectableItem):
            return
        self._selected_bot = event.item.value
        self._selected_user = None
        self._populate_users(self._selected_bot)
        self._refresh_resolve_btn()
        self._refresh_connect()

    @on(ListView.Selected, "#chat-api-list")
    async def on_api_selected(self, event: ListView.Selected) -> None:
        if not isinstance(event.item, SelectableItem):
            return
        self._selected_api = event.item.value
        self._selected_model = get_default_model(self._selected_api)
        await self._populate_models(self._selected_api)
        self._refresh_connect()

    @on(ListView.Selected, "#chat-model-list")
    def on_model_selected(self, event: ListView.Selected) -> None:
        if not isinstance(event.item, SelectableItem):
            return
        self._selected_model = event.item.value
        self.query_one("#chat-model-input", Input).value = event.item.value
        self._refresh_connect()

    @on(Input.Changed, "#chat-model-input")
    def on_model_input_changed(self, event: Input.Changed) -> None:
        v = event.value.strip()
        self._selected_model = v if v else (
            get_default_model(self._selected_api) if self._selected_api else None
        )
        self._refresh_connect()

    @on(Input.Submitted, "#chat-model-input")
    def on_model_input_submitted(self, event: Input.Submitted) -> None:
        v = event.value.strip()
        if v:
            self._selected_model = v
        self._refresh_connect()

    @on(ListView.Selected, "#chat-user-list")
    def on_user_selected(self, event: ListView.Selected) -> None:
        if not isinstance(event.item, SelectableItem):
            return
        val = event.item.value
        inp = self.query_one("#chat-user-input", Input)
        if val == _NEW_USER:
            self._selected_user = inp.value.strip() or None
            inp.display = True
            inp.focus()
        else:
            self._selected_user = val
            inp.display = False
        if self._connected_bot and self._selected_bot == self._connected_bot:
            STATE.update_live_user(self._connected_bot, self._selected_user)
        self._refresh_connect()

    @on(Input.Changed, "#chat-user-input")
    def on_user_input_changed(self, event: Input.Changed) -> None:
        v = event.value.strip()
        self._selected_user = v if v else None
        if self._connected_bot and self._selected_bot == self._connected_bot:
            STATE.update_live_user(self._connected_bot, self._selected_user)
        self._refresh_connect()

    @on(Input.Submitted, "#chat-user-input")
    def on_user_input_submitted(self, event: Input.Submitted) -> None:
        v = event.value.strip()
        self._selected_user = v if v else None
        self._refresh_connect()
        if self._can_connect():
            self.query_one("#chat-connect-btn", Button).focus()

    # ── Connect readiness ──────────────────────────────────────────────────────

    def _can_connect(self) -> bool:
        return bool(self._selected_bot and self._selected_api and self._selected_user)

    def _refresh_connect(self) -> None:
        self.query_one("#chat-connect-btn", Button).disabled = not self._can_connect()

    def _resolved_name(self) -> str:
        """Return the name the agent will see: cached display name or raw user_id."""
        use = self.query_one("#chat-use-names", Checkbox).value
        if use and self._selected_user:
            return self._tui_names.get(self._selected_user, self._selected_user)
        return self._selected_user or ""

    # ── Connect flow ───────────────────────────────────────────────────────────

    @on(Button.Pressed, "#chat-connect-btn")
    def on_connect_pressed(self) -> None:
        if not self._can_connect():
            return
        if self._connected_bot and self._memory_index is not None:
            STATE.unregister_live_context(self._connected_bot, self._memory_index)
            self._connected_bot = None
        self._connected = False
        self._set_status("[dim]connecting...[/dim]")
        self.query_one("#chat-connect-btn", Button).disabled = True
        model = self._selected_model or get_default_model(self._selected_api)
        self._load_runtime(self._selected_bot, self._selected_api, model, self._selected_user)

    @work(thread=True)
    def _load_runtime(self, bot_name: str, api_type: str, model: str, user_id: str) -> None:
        from adapters.tui_runtime import TUIRuntime
        from logger import BotLogger
        from memory import UserMemoryIndex

        try:
            runtime = TUIRuntime(bot_name, api_type, model)
            memory_index = UserMemoryIndex(
                f"{bot_name}/memory_index",
                logger=BotLogger(bot_name),
            )
            prompt_path = PATHS.bot_prompts(bot_name)
            with open(prompt_path / "prompt_formats.yaml", "r", encoding="utf-8") as fh:
                prompt_formats = yaml.safe_load(fh)
            with open(prompt_path / "system_prompts.yaml", "r", encoding="utf-8") as fh:
                system_prompts = yaml.safe_load(fh)
        except Exception as exc:
            self.app.call_from_thread(self._on_connect_failed, str(exc))
            return

        self.app.call_from_thread(
            self._on_connect_ready,
            runtime,
            memory_index,
            prompt_formats,
            system_prompts,
        )

    def _on_connect_failed(self, err: str) -> None:
        self._set_status(f"[bold]connect failed:[/bold] {err}")
        self.query_one("#chat-connect-btn", Button).disabled = False

    def _on_connect_ready(self, runtime, memory_index, prompt_formats, system_prompts) -> None:
        from adapters.tui_adapter import TUIAdapter
        from attention import snapshot_theme_cache

        log = self.query_one("#chat-log", RichLog)
        bot_label = runtime.agent_name

        def _on_send(channel_id: str, text: str) -> None:
            log.write(f"[bold]{bot_label}:[/bold] {text}\n")

        if self._connected_bot and self._memory_index is not None:
            STATE.unregister_live_context(self._connected_bot, self._memory_index)

        self._adapter = TUIAdapter(_on_send)
        self._runtime = runtime
        self._memory_index = memory_index
        self._prompt_formats = prompt_formats
        self._system_prompts = system_prompts
        self._connected = True
        self._connected_bot = runtime.agent_name
        STATE.register_live_context(
            self._connected_bot,
            memory_index,
            runtime=runtime,
            user_id=self._selected_user,
            theme_provider=snapshot_theme_cache,
        )

        model = self._selected_model or get_default_model(self._selected_api)
        display = self._resolved_name()
        self._set_status(
            f"[bold]connected[/bold]  "
            f"bot=[bold]{runtime.agent_name}[/bold]  "
            f"api={self._selected_api}  "
            f"model={model}  "
            f"user=[bold]{display}[/bold]"
        )

        self.query_one("#chat-input", Input).disabled = False
        self.query_one("#chat-send-btn", Button).disabled = False
        self.query_one("#chat-connect-btn", Button).disabled = False
        self.query_one("#chat-input", Input).focus()

    def on_unmount(self) -> None:
        if self._connected_bot and self._memory_index is not None:
            STATE.unregister_live_context(self._connected_bot, self._memory_index)

    # ── Send flow ──────────────────────────────────────────────────────────────

    @on(Button.Pressed, "#chat-send-btn")
    def on_send_pressed(self) -> None:
        self._do_send()

    @on(Input.Submitted, "#chat-input")
    def on_chat_input_submitted(self, event: Input.Submitted) -> None:
        self._do_send()

    def _do_send(self) -> None:
        if not self._connected:
            return
        inp = self.query_one("#chat-input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.value = ""
        inp.disabled = True
        self.query_one("#chat-send-btn", Button).disabled = True

        display = self._resolved_name()
        self.query_one("#chat-log", RichLog).write(f"[bold]{display}:[/bold] {text}\n")
        self._dispatch(text)

    @work()
    async def _dispatch(self, text: str) -> None:
        from adapters.base import NormalizedMessage
        from agent_core import process_message

        msg = NormalizedMessage(
            id=str(uuid.uuid4()),
            content=text,
            author_id=self._selected_user,
            author_name=self._resolved_name(),  # what the agent sees in prompts + memories
            channel_id=_CHANNEL_ID,
            channel_name="tui",
            guild_name=None,
            is_dm=True,
            mentions_bot=True,
        )

        try:
            await process_message(
                msg=msg,
                adapter=self._adapter,
                runtime=self._runtime,
                memory_index=self._memory_index,
                prompt_formats=self._prompt_formats,
                system_prompts=self._system_prompts,
                github_repo=None,
            )
        except Exception as exc:
            self.query_one("#chat-log", RichLog).write(f"[bold]error:[/bold] {exc}\n")
        finally:
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()
            self.query_one("#chat-send-btn", Button).disabled = False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self.query_one("#chat-status", Label).update(text)
