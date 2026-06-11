#!/usr/bin/env python3
"DefaultMODE Agent Manager TUI"

import io, logging, threading
import os, sys, asyncio, subprocess

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Silence HTTP client loggers that corrupt the TUI via StreamHandler
for _logger_name in (
    "httpx", "httpcore", "openai", "anthropic", "urllib3",
    "google", "google.auth", "google.genai",
    "requests", "urllib3.connectionpool",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)
# Prevent any future basicConfig from adding a StreamHandler
logging.getLogger().addHandler(logging.NullHandler())

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Label, TabbedContent, TabPane

from tui import LaunchPage, LogsPage, PromptsPage, MemoryPage, VizPage, ConfigPage, ChatPage
from tui.shared import STATE


class _TextRedirector(io.TextIOBase):
    """Captures writes to stdout/stderr and routes them to a TUI label."""

    def __init__(self, app: "AgentManagerApp", original: io.TextIOBase):
        self._app = app
        self._original = original
        self._main_thread = threading.current_thread()

    def write(self, text: str) -> int:
        if text and text.strip():
            try:
                if threading.current_thread() is self._main_thread:
                    self._app.push_console(text.strip())
                else:
                    self._app.call_from_thread(self._app.push_console, text.strip())
            except Exception:
                pass
        return len(text)

    def flush(self):
        pass

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")


class AgentManagerApp(App):
    TITLE = "defaultMODE Manager"
    CSS_PATH = "tui/run_bot.css"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "tab_launch", "1:Launch"),
        Binding("2", "tab_logs", "2:Logs"),
        Binding("3", "tab_prompts", "3:Prompts"),
        Binding("4", "tab_memory", "4:Memory"),
        Binding("5", "tab_viz", "5:Viz"),
        Binding("6", "tab_config", "6:Config"),
        Binding("7", "tab_chat", "7:Chat"),
    ]

    def __init__(self):
        super().__init__()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label("[dim]no synth selected[/dim]", id="status-config"),
            Label("", id="status-indicator"),
            id="status-bar",
        )
        with TabbedContent():
            with TabPane("Launch", id="tab-launch"):
                yield LaunchPage()
            with TabPane("Logs", id="tab-logs"):
                yield LogsPage()
            with TabPane("Prompts", id="tab-prompts"):
                yield PromptsPage()
            with TabPane("Memory", id="tab-memory"):
                yield MemoryPage()
            with TabPane("Viz", id="tab-viz"):
                yield VizPage()
            with TabPane("Config", id="tab-config"):
                yield ConfigPage()
            with TabPane("Chat", id="tab-chat"):
                yield ChatPage()
        yield Label("", id="console-bar")
        yield Footer()

    def on_mount(self):
        sys.stdout = _TextRedirector(self, self._original_stdout)
        sys.stderr = _TextRedirector(self, self._original_stderr)

    def push_console(self, text: str):
        """Push a message to the console bar."""
        label = self.query_one("#console-bar", Label)
        # Show last message, truncated to one line
        clean = text.replace("\n", " ").strip()
        if len(clean) > 200:
            clean = clean[:200] + "..."
        label.update(f"[dim]{clean}[/dim]")

    def update_global_status(self):
        config = self.query_one("#status-config", Label)
        indicator = self.query_one("#status-indicator", Label)
        if STATE.selected_bot and STATE.selected_api:
            parts = [f"[bold]{STATE.selected_bot}[/bold]", STATE.selected_api]
            if STATE.selected_model:
                parts.append(STATE.selected_model)
            config.update(" / ".join(parts))
        else:
            config.update("[dim]no synth selected[/dim]")
        count = STATE.running_count
        if count > 0:
            indicator.update(f"[bold]● {count} BOT{'S' if count > 1 else ''} RUNNING[/bold]")
        else:
            indicator.update("")

    async def action_quit(self):
        """Override quit to stop all running instances before exiting."""
        # Restore streams before exit
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        for bot_name, instance in list(STATE.instances.items()):
            if instance.running:
                instance.running = False
                try:
                    if instance.process and instance.process.returncode is None:
                        pid = instance.process.pid
                        if sys.platform == "win32":
                            import signal
                            os.kill(pid, signal.CTRL_BREAK_EVENT)
                        else:
                            import signal
                            os.killpg(os.getpgid(pid), signal.SIGINT)
                        try:
                            await asyncio.wait_for(instance.process.wait(), timeout=2)
                        except asyncio.TimeoutError:
                            if sys.platform == "win32":
                                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=2)
                            else:
                                instance.process.kill()
                except Exception:
                    pass
        self.exit()

    def action_tab_launch(self):
        self.query_one(TabbedContent).active = "tab-launch"

    def action_tab_logs(self):
        self.query_one(TabbedContent).active = "tab-logs"

    def action_tab_prompts(self):
        self.query_one(TabbedContent).active = "tab-prompts"

    def action_tab_memory(self):
        self.query_one(TabbedContent).active = "tab-memory"

    def action_tab_viz(self):
        self.query_one(TabbedContent).active = "tab-viz"

    def action_tab_config(self):
        self.query_one(TabbedContent).active = "tab-config"

    def action_tab_chat(self):
        self.query_one(TabbedContent).active = "tab-chat"


def main():
    AgentManagerApp().run()


if __name__ == "__main__":
    main()
