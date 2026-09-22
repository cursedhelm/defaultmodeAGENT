"""Launch page for the Agent Manager TUI."""

import io, json, os, sys, asyncio, subprocess, contextlib
from typing import Optional, Any

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Label, ListView, Input, Button, RichLog

from tui.shared import (
    SCRIPT_DIR, STATE, PATHS, SUPPORTED_APIS, BotInstance,
    discover_bots, check_api_available, get_default_model,
    get_api_env_key, get_models_for_api, SelectableItem,
    check_pid_running,
)

LOG_READ_BYTES = 4096
LOG_LINE_LIMIT = 256000


def _trim_log_line(text: str, limit: int = LOG_LINE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    keep_start = limit // 2
    keep_end = limit - keep_start
    return f"{text[:keep_start]}\n[dim]... omitted {omitted} chars from long log line ...[/dim]\n{text[-keep_end:]}"


def _write_process_log(log: RichLog, text: str) -> None:
    text = _trim_log_line(text.rstrip())
    s = "bold" if "ERROR" in text else "bold" if "WARNING" in text else "" if "INFO" in text else "dim"
    log.write(f"[{s}]{text}[/{s}]" if s else text)


class BotInstanceCard(Vertical):
    """Widget representing a running bot instance with controls and log output."""

    def __init__(self, instance: BotInstance):
        super().__init__(id=f"card-{instance.instance_id}")
        self.instance = instance

    def compose(self) -> ComposeResult:
        inst = self.instance
        yield Horizontal(
            Label(f"[bold]{inst.bot_name}[/bold] [dim]{inst.api}/{inst.model}[/dim]", classes="card-title"),
            Label("[bold]● RUNNING[/bold]", id=f"status-{inst.instance_id}", classes="card-status"),
            Button("Stop", variant="error", id=f"stop-{inst.instance_id}", classes="card-btn"),
            Button("Log", id=f"toggle-log-{inst.instance_id}", classes="card-btn"),
            classes="card-header",
        )
        yield RichLog(id=f"log-{inst.instance_id}", highlight=True, markup=True, wrap=True, classes="card-log")

    def get_log(self) -> RichLog:
        return self.query_one(f"#log-{self.instance.instance_id}", RichLog)

    def update_status(self, running: bool):
        status = self.query_one(f"#status-{self.instance.instance_id}", Label)
        status.update("[bold]● RUNNING[/bold]" if running else "[dim]● STOPPED[/dim]")

    def toggle_log_visibility(self):
        self.toggle_class("expanded")


class LaunchPage(Vertical):
    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Horizontal(
                    Vertical(
                        Label("[bold]Bot[/bold]"),
                        ListView(id="bot-list"),
                        classes="launch-col",
                    ),
                    Vertical(
                        Label("[bold]API[/bold]"),
                        ListView(id="api-list"),
                        #Label("[bold]Model[/bold]", classes="section-label"),
                        ListView(id="model-list"),
                        Input(placeholder="or type model", id="model-input"),
                        classes="launch-col",
                    ),
                    Vertical(
                        Label("[bold]DMN[/bold] [dim](optional)[/dim]"),
                        ListView(id="dmn-api-list"),
                        ListView(id="dmn-model-list"),
                        Input(placeholder="or type dmn model", id="dmn-model-input"),
                        classes="launch-col",
                    ),
                    Vertical(
                        Label("[bold]READER[/bold] [dim](optional)[/dim]"),
                        ListView(id="reader-api-list"),
                        ListView(id="reader-model-list"),
                        Input(placeholder="or type reader model", id="reader-model-input"),
                        classes="launch-col",
                    ),
                    id="launch-selectors",
                ),
                Label("", id="config-summary"),
                Horizontal(
                    Button("Launch Bot", variant="success", id="launch-btn"),
                    Button("Stop All", variant="error", id="stop-all-btn"),
                    id="launch-buttons",
                ),
                id="config-panel",
            ),
            Vertical(
                Label("[bold]Running Instances[/bold]"),
                Label("[dim]No bots running[/dim]", id="no-instances-label"),
                ScrollableContainer(id="instances-container"),
                id="instances-panel",
            ),
            id="launch-root",
        )

    def on_mount(self):
        self._populate_bots()
        self._populate_apis()
        self._populate_dmn_apis()
        self._populate_reader_apis()
        self.query_one("#dmn-model-list", ListView).append(
            SelectableItem("chronpression", "chronpression", "no LLM required", True)
        )
        self._scan_existing_pids()

    def _populate_bots(self):
        lv = self.query_one("#bot-list", ListView)
        lv.clear()
        for bot in discover_bots():
            has_cfg = PATHS.bot_system_prompts(bot).exists()
            lv.append(SelectableItem(bot, bot, "ready" if has_cfg else "no config", has_cfg))

    def _populate_apis(self):
        lv = self.query_one("#api-list", ListView)
        lv.clear()
        for api in SUPPORTED_APIS:
            avail = check_api_available(api)
            lv.append(SelectableItem(api.upper(), api, get_default_model(api), avail))

    def _populate_dmn_apis(self):
        lv = self.query_one("#dmn-api-list", ListView)
        lv.clear()
        lv.append(SelectableItem("(same)", "", "use main api", True))
        for api in SUPPORTED_APIS:
            avail = check_api_available(api)
            lv.append(SelectableItem(api.upper(), api, "", avail))

    def _populate_reader_apis(self):
        lv = self.query_one("#reader-api-list", ListView)
        lv.clear()
        lv.append(SelectableItem("(same)", "", "use main api", True))
        for api in SUPPORTED_APIS:
            lv.append(SelectableItem(api.upper(), api, "", check_api_available(api)))

    @work(thread=True)
    def _fetch_models(self, api: str) -> list[str]:
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

    async def _populate_models(self, api: str, target: str = "#model-list"):
        lv = self.query_one(target, ListView)
        lv.clear()
        lv.loading = True
        models = await self._fetch_models(api).wait()
        lv.loading = False
        d = get_default_model(api)
        if target in {"#dmn-model-list", "#reader-model-list"}:
            lv.append(SelectableItem("(same)", "", "use main model", True))
        if target == "#dmn-model-list":
            lv.append(SelectableItem("chronpression", "chronpression", "no LLM required", True))
        for m in models or ([d] if d else []):
            lv.append(SelectableItem(m, m, "(default)" if m == d else "", True))
        if target == "#model-list":
            self.query_one("#model-input", Input).value = d or ""

    @on(ListView.Selected, "#bot-list")
    async def on_bot_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            STATE.selected_bot = event.item.value
            self._update_summary()
            self.app.update_global_status()

    @on(ListView.Selected, "#api-list")
    async def on_api_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            STATE.selected_api = event.item.value
            STATE.selected_model = get_default_model(STATE.selected_api)
            await self._populate_models(STATE.selected_api)
            self._update_summary()
            self.app.update_global_status()

    @on(ListView.Selected, "#model-list")
    def on_model_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            STATE.selected_model = event.item.value
            self.query_one("#model-input", Input).value = event.item.value
            self._update_summary()
            self.app.update_global_status()

    @on(Input.Changed, "#model-input")
    def on_model_input(self, event: Input.Changed):
        v = event.value.strip()
        if v:
            STATE.selected_model = v
            self._update_summary()
            self.app.update_global_status()

    @on(ListView.Selected, "#dmn-api-list")
    async def on_dmn_api_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            val = event.item.value
            if val:
                STATE.dmn_api = val
                await self._populate_models(val, "#dmn-model-list")
            else:
                STATE.dmn_api = None
                self.query_one("#dmn-model-list", ListView).clear()
            self._update_summary()

    @on(ListView.Selected, "#dmn-model-list")
    def on_dmn_model_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            val = event.item.value
            STATE.dmn_model = val if val else None
            if val:
                self.query_one("#dmn-model-input", Input).value = val
            self._update_summary()

    @on(Input.Changed, "#dmn-model-input")
    def on_dmn_model_input(self, event: Input.Changed):
        v = event.value.strip()
        STATE.dmn_model = v if v else None
        self._update_summary()

    @on(ListView.Selected, "#reader-api-list")
    async def on_reader_api_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            value = event.item.value
            STATE.reader_api = value or None
            if value:
                await self._populate_models(value, "#reader-model-list")
            else:
                STATE.reader_model = None
                self.query_one("#reader-model-list", ListView).clear()
            self._update_summary()

    @on(ListView.Selected, "#reader-model-list")
    def on_reader_model_selected(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            STATE.reader_model = event.item.value or None
            if event.item.value:
                self.query_one("#reader-model-input", Input).value = event.item.value
            self._update_summary()

    @on(Input.Changed, "#reader-model-input")
    def on_reader_model_input(self, event: Input.Changed):
        STATE.reader_model = event.value.strip() or None
        self._update_summary()

    def _update_summary(self):
        parts = []
        if STATE.selected_bot:
            parts.append(STATE.selected_bot)
        if STATE.selected_api:
            parts.append(STATE.selected_api)
        if STATE.selected_model:
            parts.append(STATE.selected_model)
        if STATE.dmn_api or STATE.dmn_model:
            dmn_parts = []
            if STATE.dmn_api:
                dmn_parts.append(STATE.dmn_api)
            if STATE.dmn_model:
                dmn_parts.append(STATE.dmn_model)
            parts.append(f"DMN:{'/'.join(dmn_parts)}")
        if STATE.reader_api or STATE.reader_model:
            reader_parts = [value for value in (STATE.reader_api, STATE.reader_model) if value]
            parts.append(f"READER:{'/'.join(reader_parts)}")
        self.query_one("#config-summary", Label).update(" / ".join(parts))

    @on(Button.Pressed, "#launch-btn")
    async def on_launch(self):
        if not STATE.selected_bot or not STATE.selected_api:
            return
        if STATE.is_bot_running(STATE.selected_bot):
            return
        instance = BotInstance(
            bot_name=STATE.selected_bot,
            api=STATE.selected_api,
            model=STATE.selected_model or get_default_model(STATE.selected_api),
            dmn_api=STATE.dmn_api,
            dmn_model=STATE.dmn_model,
            reader_api=STATE.reader_api,
            reader_model=STATE.reader_model,
        )
        STATE.instances[instance.bot_name] = instance
        await self._add_instance_card(instance)
        self.app.update_global_status()
        self._run_bot(instance)

    @work()
    async def _run_bot(self, instance: BotInstance):
        try:
            card = self.query_one(f"#card-{instance.instance_id}", BotInstanceCard)
            log = card.get_log()
        except Exception:
            return

        cmd = [sys.executable, str(PATHS.discord_bot), "--api", instance.api, "--bot-name", instance.bot_name]
        if instance.model and instance.model != get_default_model(instance.api):
            cmd.extend(["--model", instance.model])
        if instance.dmn_model == "chronpression":
            cmd.append("--use-chronpression")
        else:
            if instance.dmn_api:
                cmd.extend(["--dmn-api", instance.dmn_api])
            if instance.dmn_model:
                cmd.extend(["--dmn-model", instance.dmn_model])
        if instance.reader_api:
            cmd.extend(["--reader-api", instance.reader_api])
        if instance.reader_model:
            cmd.extend(["--reader-model", instance.reader_model])

        log.write(f"[bold]$ {' '.join(cmd)}[/bold]\n")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "cwd": str(SCRIPT_DIR),
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        instance.process = await asyncio.create_subprocess_exec(*cmd, **kwargs)
        instance.running = True
        log.write(f"[dim]pid={instance.process.pid}[/dim]\n")

        # Write PID file so a future TUI session can reattach to this process
        try:
            pid_file = PATHS.bot_pid(instance.bot_name)
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(json.dumps({
                "pid": instance.process.pid,
                "bot_name": instance.bot_name,
                "api": instance.api,
                "model": instance.model,
                "reader_api": instance.reader_api,
                "reader_model": instance.reader_model,
            }), encoding="utf-8")
        except Exception:
            pass

        pending_log = ""
        while instance.running and instance.process.returncode is None:
            try:
                chunk = await asyncio.wait_for(instance.process.stdout.read(LOG_READ_BYTES), timeout=0.1)
                if chunk:
                    pending_log += chunk.decode("utf-8", errors="replace")
                    while "\n" in pending_log:
                        line, pending_log = pending_log.split("\n", 1)
                        _write_process_log(log, line)
                    if len(pending_log) > LOG_LINE_LIMIT:
                        _write_process_log(log, pending_log)
                        pending_log = ""
                elif pending_log:
                    _write_process_log(log, pending_log)
                    pending_log = ""
            except asyncio.TimeoutError:
                continue

        if pending_log:
            _write_process_log(log, pending_log)

        if instance.process.returncode is None:
            await self._kill_instance(instance, log)

        instance.running = False
        PATHS.bot_pid(instance.bot_name).unlink(missing_ok=True)
        log.write(f"\n[bold]exited code={instance.process.returncode}[/bold]")
        card.update_status(False)
        self.app.update_global_status()

    async def _kill_instance(self, instance: BotInstance, log: RichLog = None):
        """Gracefully terminate a bot instance via CTRL+C / SIGINT."""
        # Determine active PID — launched instances use process.pid, detected
        # instances use external_pid (process is None in that case).
        if instance.process is not None and instance.process.returncode is not None:
            return  # subprocess already finished
        pid = instance.pid
        if pid is None:
            return

        if log:
            log.write(f"[bold]sending interrupt to pid={pid}...[/bold]\n")

        import signal
        try:
            if sys.platform == "win32":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(pid), signal.SIGINT)
        except Exception as e:
            if log:
                log.write(f"[bold]signal error: {e}[/bold]\n")

        if instance.process is not None:
            # Launched this session — we have an asyncio subprocess to wait on
            try:
                await asyncio.wait_for(instance.process.wait(), timeout=10)
                if log:
                    log.write("graceful shutdown complete\n")
                PATHS.bot_pid(instance.bot_name).unlink(missing_ok=True)
                return
            except asyncio.TimeoutError:
                if log:
                    log.write("[bold]graceful shutdown timeout, forcing...[/bold]\n")
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=5)
                else:
                    instance.process.kill()
                await asyncio.wait_for(instance.process.wait(), timeout=3)
            except Exception:
                pass
        else:
            # Detected pre-existing process — poll until it dies
            for _ in range(10):
                await asyncio.sleep(1)
                alive = await asyncio.to_thread(check_pid_running, pid)
                if not alive:
                    if log:
                        log.write("graceful shutdown complete\n")
                    PATHS.bot_pid(instance.bot_name).unlink(missing_ok=True)
                    return
            if log:
                log.write("[bold]graceful shutdown timeout, forcing...[/bold]\n")
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=5)
                else:
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

        PATHS.bot_pid(instance.bot_name).unlink(missing_ok=True)
        if log:
            log.write("process terminated\n")

    @work()
    async def _scan_existing_pids(self):
        """Detect bot.pid files left by processes that outlived a previous TUI session."""
        if not PATHS.cache_dir.exists():
            return
        for pid_file in PATHS.cache_dir.glob("*/bot.pid"):
            try:
                data = json.loads(pid_file.read_text(encoding="utf-8"))
                pid = int(data["pid"])
                bot_name = data.get("bot_name", pid_file.parent.name)
                api = data.get("api", "?")
                model = data.get("model", "?")
                reader_api = data.get("reader_api")
                reader_model = data.get("reader_model")
            except Exception:
                continue

            if STATE.is_bot_running(bot_name):
                continue  # already tracked this session

            alive = await asyncio.to_thread(check_pid_running, pid)
            if not alive:
                pid_file.unlink(missing_ok=True)
                continue

            instance = BotInstance(
                bot_name=bot_name, api=api, model=model,
                reader_api=reader_api, reader_model=reader_model,
                running=True, external_pid=pid,
            )
            STATE.instances[bot_name] = instance
            await self._add_instance_card(instance)

            try:
                card = self.query_one(f"#card-{instance.instance_id}", BotInstanceCard)
                log = card.get_log()
                log.write(f"[dim]detected existing process pid={pid}[/dim]\n")
                log.write(f"[dim]log streaming unavailable for pre-existing processes[/dim]\n")
            except Exception:
                pass

            self._watch_external_pid(instance)
            self.app.update_global_status()

    @work()
    async def _watch_external_pid(self, instance: BotInstance):
        """Poll a detected-but-not-launched process until it exits."""
        pid = instance.external_pid
        while instance.running:
            await asyncio.sleep(3)
            alive = await asyncio.to_thread(check_pid_running, pid)
            if not alive:
                instance.running = False
                PATHS.bot_pid(instance.bot_name).unlink(missing_ok=True)
                try:
                    card = self.query_one(f"#card-{instance.instance_id}", BotInstanceCard)
                    card.get_log().write(f"\n[bold]process pid={pid} exited[/bold]")
                    card.update_status(False)
                except Exception:
                    pass
                self.app.update_global_status()
                break

    async def _add_instance_card(self, instance: BotInstance):
        """Add a new instance card to the instances panel."""
        container = self.query_one("#instances-container", ScrollableContainer)
        # Evict any stale card from a previous run of the same bot
        try:
            await self.query_one(f"#card-{instance.instance_id}", BotInstanceCard).remove()
        except Exception:
            pass
        self.query_one("#no-instances-label", Label).display = False
        await container.mount(BotInstanceCard(instance))

    def _remove_instance_card(self, bot_name: str):
        """Remove an instance card from the instances panel."""
        instance = STATE.instances.get(bot_name)
        if not instance:
            return
        inst_id = instance.instance_id
        try:
            card = self.query_one(f"#card-{inst_id}", BotInstanceCard)
            card.remove()
        except Exception:
            pass
        del STATE.instances[bot_name]
        if not STATE.instances:
            self.query_one("#no-instances-label", Label).display = True

    @on(Button.Pressed, "#stop-all-btn")
    async def on_stop_all(self):
        """Stop all running bot instances."""
        for bot_name, instance in list(STATE.instances.items()):
            if instance.running:
                instance.running = False
                try:
                    card = self.query_one(f"#card-{instance.instance_id}", BotInstanceCard)
                    log = card.get_log()
                    await self._kill_instance(instance, log)
                    card.update_status(False)
                except Exception:
                    pass
        self.app.update_global_status()

    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses for instance cards."""
        bid = event.button.id or ""

        if bid.startswith("stop-"):
            inst_id = bid[5:]
            for bot_name, instance in STATE.instances.items():
                if instance.instance_id == inst_id and instance.running:
                    instance.running = False
                    try:
                        card = self.query_one(f"#card-{inst_id}", BotInstanceCard)
                        log = card.get_log()
                        await self._kill_instance(instance, log)
                        card.update_status(False)
                    except Exception:
                        pass
                    self.app.update_global_status()
                    break

        elif bid.startswith("toggle-log-"):
            inst_id = bid[11:]
            try:
                card = self.query_one(f"#card-{inst_id}", BotInstanceCard)
                card.toggle_log_visibility()
            except Exception:
                pass
