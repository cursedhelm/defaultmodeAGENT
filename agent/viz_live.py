"""Versioned, localhost-only live state hook for secondary visualizations.

Both in-process hosts and Discord expose the same payload schema::

    {"protocol": 1, "memory": {...}, "runtime": {...}, "themes": {...}}

The memory mapping is a consistent read-only copy of ``UserMemoryIndex``.  The
server never writes the canonical memory pickle; its only file is an ephemeral
endpoint advertisement in the bot cache directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import hmac
import json
import os
import pickle
import secrets
import socket
import socketserver
import struct
import threading
import time


VIZ_LIVE_PROTOCOL = 1
_MAX_REQUEST_BYTES = 64 * 1024
_OK = b"OKAY"
_FAIL = b"FAIL"


def make_live_payload(
    memory: dict,
    *,
    runtime: Optional[dict] = None,
    themes: Optional[dict] = None,
) -> dict:
    """Construct the transport-neutral snapshot consumed by Viz."""
    return {
        "protocol": VIZ_LIVE_PROTOCOL,
        "memory": memory,
        "runtime": dict(runtime or {}),
        "themes": dict(themes or {}),
        "captured_at": time.time(),
    }


def validate_live_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("live Viz payload is not a mapping")
    if payload.get("protocol") != VIZ_LIVE_PROTOCOL:
        raise ValueError("unsupported live Viz protocol")
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("live Viz payload has no memory mapping")
    for key in ("memories", "user_memories", "inverted_index"):
        if key not in memory:
            raise ValueError(f"live Viz memory schema is missing {key}")
    payload.setdefault("runtime", {})
    payload.setdefault("themes", {})
    return payload


def endpoint_path(cache_root: os.PathLike[str] | str, bot_name: str) -> Path:
    return Path(cache_root) / bot_name / "viz_live.json"


def discover_live_bot_names(cache_root: os.PathLike[str] | str) -> list[str]:
    root = Path(cache_root)
    if not root.exists():
        return []
    names = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        if child.is_dir() and (child / "viz_live.json").exists():
            names.append(child.name)
    return sorted(names)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _SnapshotHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        owner: "VizLiveServer" = self.server.owner  # type: ignore[attr-defined]
        self.request.settimeout(owner.request_timeout)
        try:
            header = _recv_exact(self.request, 4)
            size = struct.unpack("!I", header)[0]
            if not 0 < size <= _MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            request = json.loads(_recv_exact(self.request, size).decode("utf-8"))
            token = str(request.get("token", ""))
            if not hmac.compare_digest(token, owner.token):
                raise PermissionError("invalid token")
            if request.get("command") != "snapshot":
                raise ValueError("unknown command")

            # A 40k-node snapshot is intentionally expensive. Serialize live
            # requests so repeated refreshes cannot multiply peak RAM usage.
            with owner._snapshot_lock:
                payload = validate_live_payload(owner.snapshot_provider())
                self.request.sendall(_OK)
                stream = self.request.makefile("wb")
                pickle.Pickler(stream, protocol=5).dump(payload)
                stream.flush()
        except Exception as exc:
            try:
                self.request.sendall(_FAIL)
                stream = self.request.makefile("wb")
                pickle.Pickler(stream, protocol=5).dump({
                    "error": f"{type(exc).__name__}: {exc}"
                })
                stream.flush()
            except Exception:
                pass


@dataclass
class VizLiveDescriptor:
    bot_name: str
    host: str
    port: int
    token: str
    pid: int
    protocol: int


class VizLiveServer:
    """Expose snapshots from a running agent without sharing mutable objects."""

    def __init__(
        self,
        bot_name: str,
        cache_root: os.PathLike[str] | str,
        snapshot_provider: Callable[[], dict],
        *,
        request_timeout: float = 60.0,
        logger: Any = None,
    ):
        self.bot_name = bot_name
        self.cache_root = Path(cache_root)
        self.snapshot_provider = snapshot_provider
        self.request_timeout = request_timeout
        self.logger = logger
        self.token = secrets.token_urlsafe(32)
        self._snapshot_lock = threading.Lock()
        self._server: Optional[_ThreadingServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def advertisement(self) -> Path:
        return endpoint_path(self.cache_root, self.bot_name)

    def start(self) -> VizLiveDescriptor:
        if self._server is not None:
            return self.descriptor()
        server = _ThreadingServer(("127.0.0.1", 0), _SnapshotHandler)
        server.owner = self  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name=f"viz-live-{self.bot_name}",
            daemon=True,
        )
        self._thread.start()
        descriptor = self.descriptor()
        try:
            self._write_advertisement(descriptor)
        except Exception:
            self.stop()
            raise
        self._log("info", f"viz.live.start port={descriptor.port}")
        return descriptor

    def descriptor(self) -> VizLiveDescriptor:
        if self._server is None:
            raise RuntimeError("live Viz server is not running")
        host, port = self._server.server_address
        return VizLiveDescriptor(
            self.bot_name, str(host), int(port), self.token,
            os.getpid(), VIZ_LIVE_PROTOCOL,
        )

    def _write_advertisement(self, descriptor: VizLiveDescriptor) -> None:
        path = self.advertisement
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(
            f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp"
        )
        temp.write_text(json.dumps({
            **descriptor.__dict__,
            "started_at": time.time(),
        }), encoding="utf-8")
        os.replace(temp, path)

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        try:
            current = json.loads(self.advertisement.read_text(encoding="utf-8"))
            if hmac.compare_digest(str(current.get("token", "")), self.token):
                self.advertisement.unlink(missing_ok=True)
        except Exception:
            pass
        self._log("info", "viz.live.stop")

    def _log(self, level: str, message: str) -> None:
        fn = getattr(self.logger, level, None)
        if callable(fn):
            try:
                fn(message)
            except Exception:
                pass


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("live Viz connection closed early")
        chunks.extend(chunk)
    return bytes(chunks)


def load_descriptor(
    cache_root: os.PathLike[str] | str,
    bot_name: str,
) -> Optional[VizLiveDescriptor]:
    path = endpoint_path(cache_root, bot_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        descriptor = VizLiveDescriptor(
            bot_name=str(data["bot_name"]),
            host=str(data["host"]),
            port=int(data["port"]),
            token=str(data["token"]),
            pid=int(data["pid"]),
            protocol=int(data["protocol"]),
        )
    except Exception:
        return None
    if (
        descriptor.bot_name != bot_name
        or descriptor.protocol != VIZ_LIVE_PROTOCOL
        or descriptor.host not in {"127.0.0.1", "localhost", "::1"}
        or not (0 < descriptor.port < 65536)
    ):
        return None
    return descriptor


def fetch_live_snapshot(
    cache_root: os.PathLike[str] | str,
    bot_name: str,
    *,
    connect_timeout: float = 1.5,
    response_timeout: float = 60.0,
) -> Optional[dict]:
    """Fetch one streamed live snapshot, returning None when no hook is live."""
    descriptor = load_descriptor(cache_root, bot_name)
    if descriptor is None:
        return None
    request = json.dumps({
        "protocol": VIZ_LIVE_PROTOCOL,
        "command": "snapshot",
        "token": descriptor.token,
    }).encode("utf-8")
    try:
        with socket.create_connection(
            (descriptor.host, descriptor.port), timeout=connect_timeout
        ) as sock:
            sock.settimeout(response_timeout)
            sock.sendall(struct.pack("!I", len(request)) + request)
            status = _recv_exact(sock, 4)
            stream = sock.makefile("rb")
            payload = pickle.Unpickler(stream).load()
            if status != _OK:
                raise RuntimeError(str(payload.get("error", "live Viz request failed")))
            return validate_live_payload(payload)
    except (OSError, EOFError, pickle.PickleError, ValueError, RuntimeError):
        return None
