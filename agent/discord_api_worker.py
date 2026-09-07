"""Discord-side isolation for long-running model requests.

The Discord gateway event loop should only submit and await work.  A single
worker thread owns the API coroutine's event loop, keeping synchronous setup,
local inference clients, response parsing, and retries away from Discord's
heartbeat thread while preserving the API client's one-request-at-a-time
behavior.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging
import queue
import threading
from typing import Any, Callable


_STOP = object()
_STATEFUL_DEFAULTS = (
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("frequency_penalty", "frequency_penalty"),
    ("presence_penalty", "presence_penalty"),
    ("top_k", "top_k"),
    ("min_p", "min_p"),
    ("repetition_penalty", "repetition_penalty"),
    ("max_tokens", "max_output_tokens"),
    ("reasoning_enabled", "reasoning_enabled"),
    ("reasoning_effort", "reasoning_effort"),
    ("reasoning_budget_tokens", "reasoning_budget_tokens"),
    ("capture_reasoning", "capture_reasoning"),
)


@dataclass(slots=True)
class _APIJob:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: concurrent.futures.Future


class DiscordAPIWorker:
    """Run one API coroutine at a time outside Discord's asyncio thread."""

    def __init__(
        self,
        call_api: Callable[..., Any],
        api_state: Any,
        *,
        name: str = "default",
        queue_size: int = 128,
    ) -> None:
        self._call_api = call_api
        self._api_state = api_state
        self._jobs: queue.Queue[_APIJob | object] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._state_lock = threading.Lock()
        self._closed = False
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"discord.api.{name}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("Discord API worker did not start")
        if self._startup_error is not None:
            raise RuntimeError("Discord API worker failed to start") from self._startup_error

    @property
    def thread_id(self) -> int | None:
        return self._thread.ident

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _snapshot_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Freeze mutable API defaults when Discord submits the request."""
        snapshot = dict(kwargs)
        for request_name, state_name in _STATEFUL_DEFAULTS:
            if snapshot.get(request_name) is None:
                value = getattr(self._api_state, state_name, None)
                if value is not None:
                    snapshot[request_name] = value
        return snapshot

    def submit(self, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        job = _APIJob(args, self._snapshot_kwargs(kwargs), future)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Discord API worker is shut down")
            try:
                self._jobs.put_nowait(job)
            except queue.Full as exc:
                raise RuntimeError("Discord API worker queue is full") from exc
        return future

    async def call_api(self, *args: Any, **kwargs: Any) -> Any:
        """Submit without blocking Discord, then await the thread-safe future."""
        future = self.submit(*args, **kwargs)
        return await asyncio.wrap_future(future)

    def _run(self) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        try:
            while True:
                job = self._jobs.get()
                try:
                    if job is _STOP:
                        return
                    assert isinstance(job, _APIJob)
                    if not job.future.set_running_or_notify_cancel():
                        continue
                    try:
                        result = loop.run_until_complete(
                            self._call_api(*job.args, **job.kwargs)
                        )
                    except BaseException as exc:
                        job.future.set_exception(exc)
                    else:
                        job.future.set_result(result)
                finally:
                    self._jobs.task_done()
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                logging.exception("Discord API worker async-generator shutdown failed")
            asyncio.set_event_loop(None)
            loop.close()

    def shutdown(self, *, timeout: float = 5.0, cancel_pending: bool = True) -> None:
        """Stop accepting jobs and give the active request a bounded grace period."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

            if cancel_pending:
                while True:
                    try:
                        pending = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if isinstance(pending, _APIJob):
                            pending.future.cancel()
                    finally:
                        self._jobs.task_done()

            # The queue has at least one free slot after pending jobs are
            # cancelled. If a request is active, it will observe this next.
            self._jobs.put_nowait(_STOP)

        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            logging.warning(
                "Discord API worker did not stop within %.1fs; active inference "
                "will finish on the daemon thread",
                timeout,
            )

    def __enter__(self) -> "DiscordAPIWorker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()
