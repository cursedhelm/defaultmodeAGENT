import os, json, base64, asyncio, logging, mimetypes, atexit, queue, threading, time, inspect
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import urlparse

import openai
import anthropic
from google import genai
from google.genai import types

from PIL import Image
from dotenv import load_dotenv
from colorama import Fore, init as color_init
from pydantic import BaseModel, Field

from tokenizer import count_tokens, calculate_image_tokens
from api_schema import (
    MediaPart,
    ProviderConfig,
    ProviderOptions,
    ProviderResponse,
    ReasoningConfig,
    SamplingConfig,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
    json_content,
)

# ───────────────────────────  constants & init  ────────────────────────────
MAX_IMAGE_DIM = 640
CONSOLE_PREVIEW_CHARS = 256000
API_LOG_QUEUE_SIZE = 2048
API_LOG_BATCH_SIZE = 64
API_LOG_FLUSH_INTERVAL = 0.5
color_init(autoreset=True)
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _console_preview(label: str, text: str | None, color: str) -> None:
    text = text or ""
    suffix = ""
    if len(text) > CONSOLE_PREVIEW_CHARS:
        suffix = f"\n... omitted {len(text) - CONSOLE_PREVIEW_CHARS} chars ..."
        text = text[:CONSOLE_PREVIEW_CHARS]
    print(color + f"{label} ({len(text)} chars shown){suffix}\n{text}")

class APIState(BaseModel):
    api_type:   str | None = None
    api_base:   str | None = None
    api_key:    str | None = None
    model_name: str | None = None
    api_log_path: str = "api_calls.jsonl"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.9, gt=0.0, le=1.0)
    frequency_penalty: float = Field(0.8, ge=-2.0, le=2.0)
    presence_penalty:  float = Field(0.5, ge=-2.0, le=2.0)
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_output_tokens: int = Field(default=12_000, gt=0)
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = None
    reasoning_budget_tokens: int | None = Field(default=None, gt=0)
    capture_reasoning: bool = True

api = APIState()

# ───────────────────────────  tools wiring  ────────────────────────────────
PROVIDER_TOOL_STYLE = {
    "openai": "openai",
    "openrouter": "openai",
    "ollama": "openai",
    "llama-server": "openai",
    "vllm": "openai",
    "unsloth": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
}

def adapt_tools(tools: Optional[List[ToolSpec]], provider: str) -> dict:
    if not tools: return {}
    style = PROVIDER_TOOL_STYLE.get(provider)
    if style == "openai":
        return {
            "tools": [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            } for t in tools],
            "tool_choice": "auto"
        }
    if style == "anthropic":
        return {
            "tools": [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters
            } for t in tools]
        }
    if style == "gemini":
        return {
            "tools": [{
                "function_declarations": [{
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                } for t in tools]
            }]
        }
    return {}


def adapt_openai_responses_tools(tools: Optional[List[ToolSpec]]) -> list[dict[str, Any]]:
    """Render function tools in the flat shape required by ``/v1/responses``."""
    return [{
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        # The harness accepts ordinary JSON Schema. Strict mode additionally
        # requires every object property to be required, which is not true for
        # all imported/user tools.
        "strict": False,
    } for tool in (tools or [])]

# ───────────────────────────  helpers  ─────────────────────────────────────
def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val: raise EnvironmentError(f"Environment variable {var} is required")
    return val


def _env_bool(var: str, default: bool | None = None) -> bool | None:
    value = os.getenv(var)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(var: str, default: int | None = None) -> int | None:
    value = os.getenv(var)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(var: str, default: float | None = None) -> float | None:
    value = os.getenv(var)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_list(var: str) -> list[str]:
    value = os.getenv(var, "")
    return [item.strip() for item in value.split(",") if item.strip()]

def build_chat_messages(
    system_prompt: str,
    supplemental_system_context: str,
    user_content,
):
    """Build one provider-neutral turn from already-rendered user content.

    The agent's assembled conversation context normally lives inside
    ``user_content``. ``supplemental_system_context`` exists only for callers
    that deliberately need an additional system-role message.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if supplemental_system_context:
        msgs.append({"role": "system", "content": supplemental_system_context})
    msgs.append({"role": "user", "content": user_content})
    return msgs

def _is_gpt5(name: str | None) -> bool:
    return (name or "").lower().startswith("gpt-5")


def _is_openai_reasoning_model(name: str | None) -> bool:
    value = (name or "").lower()
    return value.startswith(("gpt-5", "o1", "o3", "o4"))


def _anthropic_prefers_adaptive(name: str | None) -> bool:
    value = (name or "").lower()
    return any(marker in value for marker in (
        "opus-4-6", "sonnet-4-6", "opus-4-7", "sonnet-4-7",
        "opus-4-8", "sonnet-4-8", "claude-opus-5", "claude-sonnet-5",
        "claude-fable-5", "claude-mythos",
    ))


def _anthropic_rejects_sampling(name: str | None) -> bool:
    value = (name or "").lower()
    return any(marker in value for marker in (
        "opus-4-7", "sonnet-4-7", "opus-4-8", "sonnet-4-8",
        "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos",
    ))

def _with_v1_base(api_base: str | None) -> str:
    base = (api_base or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _is_loopback_url(api_base: str | None) -> bool:
    hostname = (urlparse(api_base or "").hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}

def _openai_compat_base(provider: str, api_base: str | None) -> str | None:
    if provider in ("ollama", "llama-server", "vllm", "unsloth"):
        return _with_v1_base(api_base)
    if provider == "openrouter":
        return api_base
    return None

def _openai_compat_key(provider: str, api_key: str | None) -> str:
    if provider in ("ollama", "llama-server"):
        return api_key or provider
    return api_key or ""

def _is_gemma4(name: str | None) -> bool:
    n = (name or "").lower()
    return "gemma-4" in n or "gemma4" in n

def _audio_format(path: str) -> str:
    ext = os.path.splitext(path.lower())[1].lstrip(".")
    return ext if ext in ("wav", "mp3") else "wav"


def _normalise_media_parts(
    user_content: str,
    image_paths: List[str],
    audio_paths: List[str],
    media_parts: Optional[List[Dict[str, Any] | MediaPart]],
) -> list[MediaPart]:
    parts: list[MediaPart] = []
    for item in media_parts or []:
        parts.append(item if isinstance(item, MediaPart) else MediaPart.model_validate(item))
    parts.extend(MediaPart(type="image", path=p) for p in image_paths)
    parts.extend(MediaPart(type="audio", path=p) for p in audio_paths)
    if not any(part.type == "text" and part.text == user_content for part in parts):
        parts.insert(0, MediaPart(type="text", text=user_content))
    return parts


def _apply_unsloth_media_extensions(
    options: ProviderOptions,
    *,
    audio_paths: list[str],
    media_parts: Optional[List[Dict[str, Any] | MediaPart]],
) -> None:
    """Map Studio-only audio/video inputs to its current request envelope."""
    typed = [
        item if isinstance(item, MediaPart) else MediaPart.model_validate(item)
        for item in (media_parts or [])
    ]
    audio = [*audio_paths, *(item.path for item in typed if item.type == "audio" and item.path)]
    video = [item.path for item in typed if item.type == "video" and item.path]
    for field, paths in (("audio_base64", audio), ("video_base64", video)):
        if len(paths) > 1:
            raise ValueError(f"Unsloth Studio accepts one {field.removesuffix('_base64')} input per request")
        if paths:
            options.extra_body.setdefault(field, _read_b64(paths[0]))

def _read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def encode_image(path: str) -> Tuple[str, Tuple[int, int]]:
    with Image.open(path) as img:
        if max(img.size) > MAX_IMAGE_DIM:
            ratio  = MAX_IMAGE_DIM / max(img.size)
            new_sz = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_sz, Image.Resampling.LANCZOS)
        if img.mode != "RGB": img = img.convert("RGB")
        buf = BytesIO(); img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode(), img.size

def prepare_multimodal_content(user_content: str, image_paths: List[str], audio_paths: List[str],
                               api_type: str, model_name: str | None = None,
                               media_parts: Optional[List[Dict[str, Any] | MediaPart]] = None) -> Tuple[object, list]:
    if not image_paths and not audio_paths and not media_parts:
        return user_content, []

    items = _normalise_media_parts(user_content, image_paths, audio_paths, media_parts)
    media_first = api_type == "gemini" or (api_type in ("ollama", "llama-server", "unsloth") and _is_gemma4(model_name))
    if media_first:
        ordered = [x for x in items if x.type in ("image", "video")]
        ordered.extend(x for x in items if x.type == "text")
        ordered.extend(x for x in items if x.type == "audio")
    else:
        ordered = items

    if api_type == "gemini":
        content, dims = [], []
        for item in ordered:
            typ = item.type
            if typ == "text":
                content.append(item.text or "")
            elif typ == "image":
                if not item.path:
                    raise ValueError("Gemini image media requires a path")
                with Image.open(item.path) as source:
                    source.load()
                    img = source.copy()
                if max(img.size) > MAX_IMAGE_DIM:
                    ratio = MAX_IMAGE_DIM / max(img.size)
                    img = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=75)
                content.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
                dims.append(img.size)
            elif typ == "audio":
                if not item.path:
                    raise ValueError("Gemini audio media requires a path")
                content.append(types.Part.from_bytes(
                    data=_read_bytes(item.path),
                    mime_type=item.mime_type or _mime(item.path),
                ))
            elif typ == "video":
                if not item.path:
                    raise ValueError("Gemini video media requires a path")
                content.append(types.Part.from_bytes(
                    data=_read_bytes(item.path),
                    mime_type=item.mime_type or _mime(item.path),
                ))
        return content, dims

    dims = []
    if api_type == "anthropic":
        parts = []
        for item in ordered:
            typ = item.type
            if typ == "text":
                parts.append({"type": "text", "text": item.text or ""})
            elif typ == "image":
                if not item.path:
                    raise ValueError("Anthropic image media requires a path")
                b, dim = encode_image(item.path); dims.append(dim)
                parts.append({"type": "image","source":{"type":"base64","media_type":"image/jpeg","data":b}})
            elif typ == "audio":
                raise ValueError("Audio input is not supported for anthropic")
            elif typ == "video":
                raise ValueError("Video input is not supported for anthropic; pass extracted frames as images")
        return parts, dims
    if api_type in ("openai", "ollama", "llama-server", "openrouter", "vllm", "unsloth"):
        parts = []
        for item in ordered:
            typ = item.type
            if typ == "text":
                parts.append({"type": "text", "text": item.text or ""})
            elif typ == "image":
                if not item.path:
                    raise ValueError("Image media requires a path")
                b, dim = encode_image(item.path); dims.append(dim)
                parts.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}})
            elif typ == "audio":
                if not item.path:
                    raise ValueError("Audio media requires a path")
                if api_type == "unsloth":
                    continue  # Unsloth Studio receives audio_base64 at request level.
                parts.append({"type":"input_audio","input_audio":{"data":_read_b64(item.path),"format":_audio_format(item.path)}})
            elif typ == "video":
                if api_type == "unsloth":
                    continue  # Unsloth Studio receives video_base64 at request level.
                if api_type in ("openrouter", "vllm"):
                    if not item.path:
                        raise ValueError("Video media requires a path")
                    media_type = item.mime_type or _mime(item.path)
                    parts.append({
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{media_type};base64,{_read_b64(item.path)}"
                        },
                    })
                    continue
                if api_type != "llama-server":
                    raise ValueError(f"Video input is not supported by the {api_type} chat adapter")
                if not item.path:
                    raise ValueError("Video media requires a path")
                parts.append({
                    "type": "input_video",
                    "input_video": {"data": _read_b64(item.path)},
                })
        return parts, dims
    raise ValueError(f"Unsupported multimodal provider: {api_type}")

def prepare_image_content(user_content: str, image_paths: List[str], api_type: str) -> Tuple[object, list]:
    return prepare_multimodal_content(user_content, image_paths, [], api_type)

class _BatchedJsonlWriter:
    """Bounded off-thread JSONL writer used only by this API client module."""

    def __init__(self, *, queue_size: int = API_LOG_QUEUE_SIZE,
                 batch_size: int = API_LOG_BATCH_SIZE,
                 flush_interval: float = API_LOG_FLUSH_INTERVAL):
        self._queue = queue.Queue(maxsize=queue_size)
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.01, flush_interval)
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._closed = False
        self._dropped = 0
        self._reported_drops = 0
        self._thread = threading.Thread(
            target=self._run,
            name="api.log.writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_records(self) -> int:
        with self._state_lock:
            return self._dropped

    def submit(self, data: dict, path: str) -> bool:
        """Enqueue one shallow-copied record without waiting for disk I/O."""
        with self._state_lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait((os.fspath(path), dict(data)))
                return True
            except queue.Full:
                self._dropped += 1
                return False

    def _write_batch(self, batch) -> None:
        lines_by_path = defaultdict(list)
        for path, data in batch:
            lines_by_path[path].append(json.dumps(data, ensure_ascii=False))
        for path, lines in lines_by_path.items():
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
                f.write("\n")

    def _report_drops(self) -> None:
        dropped = self.dropped_records
        if dropped != self._reported_drops:
            logging.warning("API log queue full; dropped %d record(s)", dropped)
            self._reported_drops = dropped

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                first = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                self._report_drops()
                continue

            batch = [first]
            deadline = time.monotonic() + self._flush_interval
            while len(batch) < self._batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    if self._stop.is_set():
                        batch.append(self._queue.get_nowait())
                    else:
                        batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                self._write_batch(batch)
            except Exception:
                logging.exception("Failed to write API log batch")
            finally:
                for _ in batch:
                    self._queue.task_done()
            self._report_drops()

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=timeout)


_api_log_writer = None
_api_log_writer_lock = threading.Lock()


def _get_api_log_writer() -> _BatchedJsonlWriter:
    global _api_log_writer
    if _api_log_writer is None:
        with _api_log_writer_lock:
            if _api_log_writer is None:
                _api_log_writer = _BatchedJsonlWriter()
                atexit.register(_api_log_writer.shutdown)
    return _api_log_writer


def shutdown_api_logger(timeout: float = 5.0) -> None:
    """Drain queued API logs and stop this module's private writer."""
    if _api_log_writer is not None:
        _api_log_writer.shutdown(timeout=timeout)


def log_to_jsonl(data: dict, path: str | None = None) -> None:
    """Queue an API log record; JSON serialization and file I/O stay off-thread."""
    _get_api_log_writer().submit(data, path or api.api_log_path)

def get_api_config(api_type: str, model_override: str | None = None) -> ProviderConfig:
    if api_type == "ollama":
        return ProviderConfig(api_base=os.getenv("OLLAMA_API_BASE","http://localhost:11434"),
                              api_key="ollama",
                              model_name=model_override or os.getenv("OLLAMA_MODEL_NAME","gemma4:12b"))
    if api_type == "llama-server":
        return ProviderConfig(api_base=os.getenv("LLAMA_SERVER_API_BASE","http://127.0.0.1:8888"),
                              api_key=os.getenv("LLAMA_SERVER_API_KEY","llama-server"),
                              model_name=model_override or os.getenv("LLAMA_SERVER_MODEL_NAME","unsloth/gemma-4-E2B-it-GGUF:Q4_K_XL"))
    if api_type == "openai":
        return ProviderConfig(api_key=_require_env("OPENAI_API_KEY"),
                              model_name=model_override or os.getenv("OPENAI_MODEL_NAME","gpt-4.1-mini"))
    if api_type == "anthropic":
        return ProviderConfig(api_key=_require_env("ANTHROPIC_API_KEY"),
                              model_name=model_override or os.getenv("ANTHROPIC_MODEL_NAME","claude-haiku-4-5"))
    if api_type == "vllm":
        return ProviderConfig(api_base=os.getenv("VLLM_API_BASE","http://localhost:4000"),
                              api_key=_require_env("VLLM_API_KEY"),
                              model_name=model_override or os.getenv("VLLM_MODEL_NAME","google/gemma-3-4b-it"))
    if api_type == "unsloth":
        unsloth_base = os.getenv("UNSLOTH_API_BASE","http://localhost:8888")
        configured_key = os.getenv("UNSLOTH_API_KEY")
        require_local_auth = bool(_env_bool("UNSLOTH_REQUIRE_AUTH", False))
        unsloth_key = (
            configured_key if (not _is_loopback_url(unsloth_base) or require_local_auth)
            else "not-needed"
        )
        if not unsloth_key:
            raise EnvironmentError("UNSLOTH_API_KEY is required for remote or authenticated Unsloth servers")
        return ProviderConfig(api_base=unsloth_base,
                              api_key=unsloth_key,
                              model_name=model_override or os.getenv("UNSLOTH_MODEL_NAME","gemma-4-26B-A4B-it-GGUF"))
    if api_type == "gemini":
        return ProviderConfig(api_key=_require_env("GEMINI_API_KEY"),
                              model_name=model_override or os.getenv("GEMINI_MODEL_NAME","gemini-3-flash-preview"))
    if api_type == "openrouter":
        return ProviderConfig(api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                              api_key=_require_env("OPENROUTER_API_KEY"),
                              model_name=model_override or os.getenv("OPENROUTER_MODEL_NAME","moonshotai/kimi-k2:free"))
    raise ValueError(f"Unsupported API type: {api_type}")


def _load_stateful_provider_defaults(provider: str) -> None:
    prefix = provider.upper().replace("-", "_")
    api.top_k = _env_int(f"{prefix}_TOP_K")
    api.min_p = _env_float(f"{prefix}_MIN_P")
    api.repetition_penalty = _env_float(f"{prefix}_REPETITION_PENALTY")
    api.max_output_tokens = _env_int(f"{prefix}_MAX_OUTPUT_TOKENS", 12_000) or 12_000
    api.reasoning_enabled = _env_bool(f"{prefix}_ENABLE_THINKING")
    api.reasoning_effort = os.getenv(f"{prefix}_REASONING_EFFORT") or None
    api.reasoning_budget_tokens = _env_int(f"{prefix}_REASONING_BUDGET_TOKENS")
    capture = _env_bool(f"{prefix}_CAPTURE_REASONING", True)
    api.capture_reasoning = True if capture is None else capture

def initialize_api_client(args):
    api.api_type  = args.api
    cfg = get_api_config(api.api_type, args.model)
    api.model_name = cfg.model_name
    api.api_base   = cfg.api_base
    api.api_key    = cfg.api_key
    api.api_log_path = os.fspath(
        getattr(args, "api_log_path", None)
        or os.getenv("API_LOG_PATH", "api_calls.jsonl")
    )
    _load_stateful_provider_defaults(api.api_type)
    if api.api_type == "openai": openai.api_key = api.api_key
    logging.info("Initialized API client (%s, model=%s)", api.api_type, api.model_name)

def update_api_temperature(temperature: float) -> None:
    if not 0.0 <= temperature <= 2.0: raise ValueError("temperature must be between 0.0 and 2.0")
    api.temperature = temperature; logging.info("temperature set to %.2f", api.temperature)

def update_api_top_p(p: float) -> None:
    if not 0.0 < p <= 1.0: raise ValueError("top_p must be 0-1")
    api.top_p = p; logging.info("top_p set to %.2f", api.top_p)

def update_api_frequency_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0: raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    api.frequency_penalty = penalty; logging.info("frequency_penalty set to %.2f", api.frequency_penalty)

def update_api_presence_penalty(penalty: float) -> None:
    if not -2.0 <= penalty <= 2.0: raise ValueError("presence_penalty must be between -2.0 and 2.0")
    api.presence_penalty = penalty; logging.info("presence_penalty set to %.2f", api.presence_penalty)


def update_api_top_k(value: int | None) -> None:
    api.top_k = value


def update_api_min_p(value: float | None) -> None:
    api.min_p = value


def update_api_repetition_penalty(value: float | None) -> None:
    api.repetition_penalty = value


def update_api_reasoning(*, enabled: bool | None = None, effort: str | None = None,
                         budget_tokens: int | None = None, capture: bool | None = None) -> None:
    api.reasoning_enabled = enabled
    api.reasoning_effort = effort
    api.reasoning_budget_tokens = budget_tokens
    if capture is not None:
        api.capture_reasoning = capture

async def retry_api_call(func, *a, max_retries=3, retry_delay=1, **kw):
    for attempt in range(max_retries):
        try:
            return await func(*a, **kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = status == 429 or (status and 500 <= status < 600)
            if retryable and attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logging.warning("HTTP %s from provider, retry %d/%d in %.1fs",
                                status, attempt + 1, max_retries, delay)
                await asyncio.sleep(delay); continue
            raise

# ─────────────────────────── provider response helpers ─────────────────────
def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True, by_alias=True)
    return value


def _normalise_arguments(value: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, dict):
        return value, json.dumps(value, ensure_ascii=False)
    raw = value if isinstance(value, str) else json_content(value)
    try:
        parsed = json.loads(raw or "{}")
        return (parsed if isinstance(parsed, dict) else {"value": parsed}), raw
    except Exception:
        return {}, raw


def _extract_openai_tool_calls(msg) -> List[ToolCall]:
    tcs = getattr(msg, "tool_calls", None) or []
    out: list[ToolCall] = []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        if not name:
            continue
        arguments, raw = _normalise_arguments(getattr(fn, "arguments", None) or "{}")
        out.append(ToolCall(
            id=getattr(tc, "id", None) or f"call_{len(out)}",
            provider_call_id=getattr(tc, "id", None),
            name=name,
            arguments=arguments,
            raw_arguments=raw,
        ))
    return out


def _usage_from_openai(value: Any) -> TokenUsage | None:
    if value is None:
        return None
    prompt = int(getattr(value, "prompt_tokens", 0) or 0)
    completion = int(getattr(value, "completion_tokens", 0) or 0)
    total = int(getattr(value, "total_tokens", prompt + completion) or 0)
    prompt_details = getattr(value, "prompt_tokens_details", None)
    completion_details = getattr(value, "completion_tokens_details", None)
    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
        reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_from_openai_responses(value: Any) -> TokenUsage | None:
    if value is None:
        return None
    input_tokens = int(_field(value, "input_tokens", 0) or 0)
    output_tokens = int(_field(value, "output_tokens", 0) or 0)
    total_tokens = int(_field(value, "total_tokens", input_tokens + output_tokens) or 0)
    input_details = _field(value, "input_tokens_details")
    output_details = _field(value, "output_tokens_details")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=int(_field(input_details, "cached_tokens", 0) or 0),
        reasoning_tokens=int(_field(output_details, "reasoning_tokens", 0) or 0),
    )


def _add_usage(total: TokenUsage | None, current: TokenUsage | None) -> TokenUsage | None:
    if current is None:
        return total
    if total is None:
        return current.model_copy(deep=True)
    for field in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
        setattr(total, field, getattr(total, field) + getattr(current, field))
    return total


def _openai_responses_input(user_content: Any) -> list[dict[str, Any]]:
    """Translate the harness' OpenAI-compatible media blocks to Responses input."""
    if isinstance(user_content, str):
        return [{"role": "user", "content": user_content}]
    if not isinstance(user_content, list):
        return [{"role": "user", "content": str(user_content)}]

    content: list[dict[str, Any]] = []
    for part in user_content:
        if not isinstance(part, dict):
            raise ValueError("OpenAI Responses input parts must be objects")
        part_type = part.get("type")
        if part_type == "text":
            content.append({"type": "input_text", "text": part.get("text", "")})
        elif part_type == "image_url":
            image = part.get("image_url") or {}
            image_url = image.get("url") if isinstance(image, dict) else image
            item = {
                "type": "input_image",
                "image_url": image_url,
                "detail": image.get("detail", "auto") if isinstance(image, dict) else "auto",
            }
            content.append(item)
        elif part_type == "input_audio":
            raise ValueError(
                "OpenAI Responses does not accept the Chat Completions input_audio block. "
                "Use an audio-capable Chat Completions model, or transcribe the audio before "
                "calling a reasoning model with function tools."
            )
        else:
            raise ValueError(f"Unsupported OpenAI Responses input part: {part_type!r}")
    return [{"role": "user", "content": content}]


def _parse_openai_responses(value: Any, model: str) -> ProviderResponse:
    text_parts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in (_field(value, "output", []) or []):
        item_type = _field(item, "type")
        if item_type == "message":
            for part in (_field(item, "content", []) or []):
                part_type = _field(part, "type")
                if part_type == "output_text":
                    text = _field(part, "text")
                    if text:
                        text_parts.append(str(text))
                elif part_type == "refusal":
                    refusal = _field(part, "refusal")
                    if refusal:
                        text_parts.append(str(refusal))
        elif item_type == "reasoning":
            for summary in (_field(item, "summary", []) or []):
                text = _field(summary, "text")
                if text:
                    reasoning.append(str(text).strip())
        elif item_type == "function_call":
            name = _field(item, "name")
            if not name:
                continue
            arguments, raw = _normalise_arguments(_field(item, "arguments", "{}"))
            call_id = _field(item, "call_id")
            item_id = _field(item, "id")
            tool_calls.append(ToolCall(
                id=call_id or item_id or f"call_{len(tool_calls)}",
                provider_call_id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=raw,
            ))

    status = _field(value, "status")
    incomplete = _field(value, "incomplete_details")
    incomplete_reason = _field(incomplete, "reason")
    return ProviderResponse(
        provider="openai",
        model=_field(value, "model") or model,
        content="\n".join(text_parts).strip(),
        reasoning=[trace for trace in reasoning if trace],
        tool_calls=tool_calls,
        finish_reason=incomplete_reason or ("stop" if status == "completed" else status),
        usage=_usage_from_openai_responses(_field(value, "usage")),
        provider_metadata={
            "id": _field(value, "id"),
            "status": status,
            "endpoint": "responses",
            **({"incomplete_details": _model_dump(incomplete)} if incomplete else {}),
        },
    )


def _message_extra(msg: Any) -> dict[str, Any]:
    extra = getattr(msg, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _extract_openai_reasoning(msg: Any) -> list[str]:
    extra = _message_extra(msg)
    candidates = [
        getattr(msg, "reasoning_content", None),
        getattr(msg, "reasoning", None),
        extra.get("reasoning_content"),
        extra.get("reasoning"),
    ]
    details = getattr(msg, "reasoning_details", None) or extra.get("reasoning_details")
    if isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                candidates.append(detail.get("text") or detail.get("content"))
            else:
                candidates.append(getattr(detail, "text", None) or getattr(detail, "content", None))
    return [str(item).strip() for item in candidates if isinstance(item, str) and item.strip()]


async def _execute_tool_call(call: ToolCall, tool_runtime: Dict[str, Any] | None) -> ToolResult:
    fn = (tool_runtime or {}).get(call.name)
    if fn is None:
        return ToolResult(
            tool_call_id=call.id,
            provider_call_id=call.provider_call_id,
            name=call.name,
            content=json_content({"error": "tool_not_found", "name": call.name}),
            is_error=True,
        )
    try:
        output = fn(call.arguments)
        if inspect.isawaitable(output):
            output = await output
        return ToolResult(
            tool_call_id=call.id,
            provider_call_id=call.provider_call_id,
            name=call.name,
            content=json_content(output),
        )
    except Exception as exc:
        return ToolResult(
            tool_call_id=call.id,
            provider_call_id=call.provider_call_id,
            name=call.name,
            content=json_content({"error": "tool_exception", "detail": str(exc)}),
            is_error=True,
        )

def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for part in value:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if text:
                texts.append(str(text))
        return "\n".join(texts)
    return "" if value is None else str(value)


def _openai_extra_body(provider: str, sampling: SamplingConfig,
                       options: ProviderOptions) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if provider in ("llama-server", "vllm", "unsloth"):
        if sampling.top_k is not None:
            body["top_k"] = sampling.top_k
        if sampling.min_p is not None:
            body["min_p"] = sampling.min_p
        if sampling.repetition_penalty is not None:
            body["repetition_penalty"] = sampling.repetition_penalty
    if provider == "llama-server" and options.reasoning.enabled is not None:
        body["chat_template_kwargs"] = {"enable_thinking": options.reasoning.enabled}
    if provider == "unsloth":
        if options.reasoning.enabled is not None:
            body["enable_thinking"] = options.reasoning.enabled
        if options.reasoning.effort:
            body["reasoning_effort"] = options.reasoning.effort
        if options.server_tools.enabled:
            body["enable_tools"] = True
            if options.server_tools.enabled_tools:
                body["enabled_tools"] = options.server_tools.enabled_tools
    if provider == "openrouter" and (
        options.reasoning.effort or options.reasoning.budget_tokens is not None
        or options.reasoning.enabled is not None
    ):
        reasoning: dict[str, Any] = {}
        if options.reasoning.budget_tokens is not None:
            reasoning["max_tokens"] = options.reasoning.budget_tokens
        elif options.reasoning.effort:
            reasoning["effort"] = options.reasoning.effort
        elif options.reasoning.enabled is not None:
            reasoning["enabled"] = options.reasoning.enabled
        reasoning["exclude"] = not options.reasoning.capture
        body["reasoning"] = reasoning
    body.update(options.extra_body)
    return body


async def _openai_compat_chat(*, provider: str, base_url: str | None, api_key: str,
                             model: str, msgs: list, sampling: SamplingConfig,
                             options: ProviderOptions, tools: list | None,
                             tool_choice: str | dict | None) -> tuple[ProviderResponse, dict]:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    max_tokens_key = (
        "max_completion_tokens"
        if provider in ("openai", "openrouter", "vllm", "unsloth")
        else "max_tokens"
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        max_tokens_key: sampling.max_output_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    if provider != "openai" or not _is_openai_reasoning_model(model):
        kwargs.update(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            frequency_penalty=sampling.frequency_penalty,
            presence_penalty=sampling.presence_penalty,
        )
    elif options.reasoning.effort:
        kwargs["reasoning_effort"] = options.reasoning.effort
    if provider in ("ollama", "llama-server") and (
        options.reasoning.effort or options.reasoning.enabled is not None
    ):
        kwargs["reasoning_effort"] = (
            options.reasoning.effort
            or ("medium" if options.reasoning.enabled else "none")
        )

    extra_body = _openai_extra_body(provider, sampling, options)
    if extra_body:
        kwargs["extra_body"] = extra_body
    r = await client.chat.completions.create(**kwargs)
    choice = r.choices[0]
    msg = choice.message
    tool_calls = _extract_openai_tool_calls(msg)
    message_extra = _message_extra(msg)
    raw_reasoning_details = getattr(msg, "reasoning_details", None) or message_extra.get("reasoning_details")
    response = ProviderResponse(
        provider=provider,
        model=getattr(r, "model", None) or model,
        content=_content_text(getattr(msg, "content", None)).strip(),
        reasoning=_extract_openai_reasoning(msg),
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=_usage_from_openai(getattr(r, "usage", None)),
        provider_metadata={
            "id": getattr(r, "id", None),
            "system_fingerprint": getattr(r, "system_fingerprint", None),
            **({"reasoning_details": _model_dump(raw_reasoning_details)} if raw_reasoning_details else {}),
        },
    )
    assistant_payload = {
        "role": "assistant",
        "content": getattr(msg, "content", None),
        "tool_calls": [_model_dump(tc) for tc in (getattr(msg, "tool_calls", None) or [])],
    }
    for field in ("reasoning_content", "reasoning", "reasoning_details"):
        value = getattr(msg, field, None)
        if value is None:
            value = message_extra.get(field)
        if value is not None:
            assistant_payload[field] = _model_dump(value)
    return response, assistant_payload

async def _openai_compat_call_with_auto_tools(*, provider: str, cfg: ProviderConfig,
                                             system_prompt: str, supplemental_system_context: str,
                                             user_content: object,
                                             sampling: SamplingConfig, options: ProviderOptions,
                                             tools_payload: dict, tool_runtime: Dict[str, Any] | None,
                                             max_rounds: int = 4) -> ProviderResponse:
    base_url = _openai_compat_base(provider, cfg.api_base)
    api_key = _openai_compat_key(provider, cfg.api_key)

    msgs = build_chat_messages(system_prompt, supplemental_system_context, user_content)
    tools = tools_payload.get("tools")
    tool_choice = tools_payload.get("tool_choice")

    collected_reasoning: list[str] = []
    last: ProviderResponse | None = None
    for _ in range(max_rounds):
        current, assistant_payload = await _openai_compat_chat(
            provider=provider, base_url=base_url, api_key=api_key,
            model=cfg.model_name, msgs=msgs, sampling=sampling,
            options=options, tools=tools, tool_choice=tool_choice,
        )
        last = current
        collected_reasoning.extend(current.reasoning)
        if not current.tool_calls or not tool_runtime:
            current.reasoning = collected_reasoning
            return current

        msgs.append(assistant_payload)
        results = await asyncio.gather(*(
            _execute_tool_call(call, tool_runtime) for call in current.tool_calls
        ))
        for result in results:
            msgs.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            })
    if last is None:
        return ProviderResponse(provider=provider, model=cfg.model_name, finish_reason="tool_round_limit")
    last.reasoning = collected_reasoning
    last.finish_reason = "tool_round_limit"
    return last


async def _openai_responses_call(*, cfg: ProviderConfig, system_prompt: str,
                                 supplemental_system_context: str, user_content: Any,
                                 sampling: SamplingConfig, options: ProviderOptions,
                                 tools: list[dict[str, Any]],
                                 tool_runtime: Dict[str, Any] | None,
                                 auto_execute_tools: bool,
                                 max_rounds: int = 4) -> ProviderResponse:
    """Call OpenAI's Responses API and preserve its item-based tool protocol."""
    client = openai.AsyncOpenAI(api_key=cfg.api_key)
    input_items: list[Any] = _openai_responses_input(user_content)
    instructions = "\n\n".join(
        value for value in (system_prompt, supplemental_system_context) if value
    )

    reasoning: dict[str, Any] = {}
    if options.reasoning.effort:
        reasoning["effort"] = options.reasoning.effort
    elif options.reasoning.enabled is False:
        reasoning["effort"] = "none"
    if options.reasoning.capture and options.reasoning.enabled is not False:
        reasoning["summary"] = "auto"

    collected_reasoning: list[str] = []
    total_usage: TokenUsage | None = None
    last: ProviderResponse | None = None
    for _ in range(max_rounds):
        kwargs: dict[str, Any] = {
            "model": cfg.model_name,
            "input": input_items,
            "max_output_tokens": sampling.max_output_tokens,
            "tools": tools,
            "tool_choice": "auto",
            # Chat Completions is stateless. Keep that harness expectation
            # while explicitly carrying Responses output items across rounds.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            kwargs["instructions"] = instructions
        if reasoning:
            kwargs["reasoning"] = reasoning
        # OpenAI reasoning models only accept sampling controls at ``none``
        # effort. Preserve the harness' dynamic values whenever that mode is
        # explicitly selected; higher/default efforts intentionally omit them.
        if reasoning.get("effort") == "none":
            kwargs["temperature"] = sampling.temperature
            kwargs["top_p"] = sampling.top_p
        if options.extra_body:
            kwargs["extra_body"] = options.extra_body

        raw = await client.responses.create(**kwargs)
        current = _parse_openai_responses(raw, cfg.model_name)
        last = current
        collected_reasoning.extend(current.reasoning)
        total_usage = _add_usage(total_usage, current.usage)

        if not current.tool_calls or not auto_execute_tools or not tool_runtime:
            current.reasoning = collected_reasoning
            current.usage = total_usage
            return current

        # Responses function-call and reasoning items are protocol state. They
        # must be returned intact before their matching function_call_output.
        input_items.extend(_field(raw, "output", []) or [])
        results = await asyncio.gather(*(
            _execute_tool_call(call, tool_runtime) for call in current.tool_calls
        ))
        for result in results:
            input_items.append({
                "type": "function_call_output",
                "call_id": result.provider_call_id or result.tool_call_id,
                "output": result.content,
            })

    if last is None:
        return ProviderResponse(
            provider="openai", model=cfg.model_name,
            finish_reason="tool_round_limit",
            provider_metadata={"endpoint": "responses"},
        )
    last.reasoning = collected_reasoning
    last.usage = total_usage
    last.finish_reason = "tool_round_limit"
    return last

# ───────────────────────────  main entry  ──────────────────────────────────
async def call_api_detailed(user_content: str, *, supplemental_system_context: str = "", system_prompt: str = "",
                            conversation_id=None, temperature: float | None = None,
                            top_p: float | None = None, frequency_penalty: float | None = None,
                            presence_penalty: float | None = None, top_k: int | None = None,
                            min_p: float | None = None, repetition_penalty: float | None = None,
                            max_tokens: int | None = None,
                            reasoning_enabled: bool | None = None,
                            reasoning_effort: str | None = None,
                            reasoning_budget_tokens: int | None = None,
                            capture_reasoning: bool | None = None,
                            image_paths: List[str] | None = None,
                            audio_paths: List[str] | None = None,
                            media_parts: Optional[List[Dict[str, Any] | MediaPart]] = None,
                            api_type_override: str | None = None, model_override: str | None = None,
                            tools: Optional[List[ToolSpec]] = None,
                            tool_runtime: Optional[Dict[str, Any]] = None,
                            auto_execute_tools: bool = False,
                            provider_options: ProviderOptions | Dict[str, Any] | None = None) -> ProviderResponse:
    """Send rendered user content plus any explicitly system-role instructions.

    Agent prompt schemas should interpolate ``assembled_context`` into
    ``user_content`` and leave ``supplemental_system_context`` empty.
    """
    temp = temperature if temperature is not None else api.temperature
    p_val = top_p if top_p is not None else api.top_p
    freq_pen = frequency_penalty if frequency_penalty is not None else api.frequency_penalty
    pres_pen = presence_penalty if presence_penalty is not None else api.presence_penalty
    provider = api_type_override or api.api_type
    model    = model_override or api.model_name
    if provider not in PROVIDER_TOOL_STYLE:
        raise ValueError(f"Unsupported API type: {provider}")

    prefix = provider.upper().replace("-", "_")
    state_applies = provider == api.api_type
    sampling = SamplingConfig(
        temperature=temp,
        top_p=p_val,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        top_k=top_k if top_k is not None else (api.top_k if state_applies else _env_int(f"{prefix}_TOP_K")),
        min_p=min_p if min_p is not None else (api.min_p if state_applies else _env_float(f"{prefix}_MIN_P")),
        repetition_penalty=(
            repetition_penalty if repetition_penalty is not None
            else (api.repetition_penalty if state_applies else _env_float(f"{prefix}_REPETITION_PENALTY"))
        ),
        max_output_tokens=(
            max_tokens if max_tokens is not None
            else (api.max_output_tokens if state_applies else (_env_int(f"{prefix}_MAX_OUTPUT_TOKENS", 12_000) or 12_000))
        ),
    )

    options = (
        provider_options
        if isinstance(provider_options, ProviderOptions)
        else ProviderOptions.model_validate(provider_options or {})
    )
    reasoning_fields = options.reasoning.model_fields_set
    state_reasoning_enabled = api.reasoning_enabled if state_applies else _env_bool(f"{prefix}_ENABLE_THINKING")
    state_reasoning_effort = api.reasoning_effort if state_applies else os.getenv(f"{prefix}_REASONING_EFFORT")
    state_reasoning_budget = api.reasoning_budget_tokens if state_applies else _env_int(f"{prefix}_REASONING_BUDGET_TOKENS")
    state_capture_reasoning = api.capture_reasoning if state_applies else True
    resolved_reasoning = ReasoningConfig(
        enabled=(
            reasoning_enabled if reasoning_enabled is not None
            else options.reasoning.enabled if "enabled" in reasoning_fields
            else state_reasoning_enabled
        ),
        effort=(
            reasoning_effort if reasoning_effort is not None
            else options.reasoning.effort if "effort" in reasoning_fields
            else state_reasoning_effort
        ),
        budget_tokens=(
            reasoning_budget_tokens if reasoning_budget_tokens is not None
            else options.reasoning.budget_tokens if "budget_tokens" in reasoning_fields
            else state_reasoning_budget
        ),
        capture=(
            capture_reasoning if capture_reasoning is not None
            else options.reasoning.capture if "capture" in reasoning_fields
            else state_capture_reasoning
        ),
    )
    options.reasoning = resolved_reasoning
    if provider_options is None and not options.server_tools.enabled:
        options.server_tools.enabled = bool(_env_bool(f"{prefix}_ENABLE_TOOLS", False))
    if provider_options is None and not options.server_tools.enabled_tools:
        options.server_tools.enabled_tools = _env_list(f"{prefix}_ENABLED_TOOLS")
    if provider == "unsloth":
        _apply_unsloth_media_extensions(
            options,
            audio_paths=audio_paths or [],
            media_parts=media_parts,
        )

    order_model = model
    if api_type_override and not model_override:
        try: order_model = get_api_config(provider, None).model_name
        except Exception: pass

    _console_preview("System prompt", system_prompt, Fore.LIGHTMAGENTA_EX)
    _console_preview("User content", user_content, Fore.LIGHTCYAN_EX)

    prepared_user_content, dims = prepare_multimodal_content(
        user_content,
        image_paths or [],
        audio_paths or [],
        provider,
        order_model,
        media_parts,
    )
    tools_payload = adapt_tools(tools, provider)
    responses_tools = adapt_openai_responses_tools(tools) if provider == "openai" else []

    async def dispatch() -> ProviderResponse:
        if api_type_override and not model_override:
            cfg = get_api_config(provider, None)
        elif model_override:
            cfg = get_api_config(provider, model_override)
        else:
            cfg = ProviderConfig(model_name=api.model_name, api_key=api.api_key, api_base=getattr(api,'api_base',None))
            if provider != api.api_type:
                cfg = get_api_config(provider, api.model_name)

        logging.info("Call → %s | model=%s T=%.2f P=%.2f FP=%.2f PP=%.2f",
                     provider, cfg.model_name, temp, p_val, freq_pen, pres_pen)

        if provider in ("openai", "ollama", "llama-server", "openrouter", "vllm", "unsloth"):
            if provider == "vllm" and not (cfg.api_base or "").rstrip("/").endswith(("/v1",)):
                pass
            if provider == "openai" and responses_tools and _is_openai_reasoning_model(cfg.model_name):
                return await _openai_responses_call(
                    cfg=cfg,
                    system_prompt=system_prompt,
                    supplemental_system_context=supplemental_system_context,
                    user_content=prepared_user_content,
                    sampling=sampling,
                    options=options,
                    tools=responses_tools,
                    tool_runtime=tool_runtime,
                    auto_execute_tools=auto_execute_tools,
                )
            if auto_execute_tools and tools_payload.get("tools"):
                return await _openai_compat_call_with_auto_tools(
                    provider=provider, cfg=cfg,
                    system_prompt=system_prompt,
                    supplemental_system_context=supplemental_system_context,
                    user_content=prepared_user_content,
                    sampling=sampling, options=options,
                    tools_payload=tools_payload, tool_runtime=tool_runtime
                )
            return await _call_openai_compat(
                provider,
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                             sampling=sampling, options=options,
                                             config=cfg, tools_payload=tools_payload)

        if provider == "anthropic":
            return await _call_anthropic(
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                         sampling=sampling, options=options,
                                         config=cfg, tools_payload=tools_payload,
                                         tool_runtime=tool_runtime,
                                         auto_execute_tools=auto_execute_tools)

        if provider == "gemini":
            return await _call_gemini(
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                      sampling=sampling, options=options,
                                      config=cfg, tools_payload=tools_payload,
                                      tool_runtime=tool_runtime,
                                      auto_execute_tools=auto_execute_tools)

        raise ValueError(provider)

    result = await retry_api_call(dispatch)
    result.provider_metadata["capture_reasoning"] = options.reasoning.capture
    response = result.as_agent_text(capture_reasoning=options.reasoning.capture)
    _console_preview("Response", response, Fore.MAGENTA)

    txt = (
        f"{system_prompt}\n{supplemental_system_context}\n{user_content}"
        if (system_prompt or supplemental_system_context)
        else user_content
    )
    input_tok  = count_tokens(txt) + sum(calculate_image_tokens(w, h) for w, h in dims)
    output_tok = count_tokens(response)
    logging.info("Tokens → Input: %d | Output: %d | Total: %d", input_tok, output_tok, input_tok + output_tok)

    logged_user_content = (
        f"[Media] {user_content}"
        if (image_paths or audio_paths or media_parts)
        else user_content
    )
    log_to_jsonl({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "api_type": provider,
        "model": model,
        "system_prompt": system_prompt,
        "supplemental_system_context": supplemental_system_context,
        "user_content": logged_user_content,
        "ai_output": response,
        "is_image": bool(image_paths),
        "is_audio": bool(audio_paths),
        "num_images": len(image_paths or []),
        "num_audio": len(audio_paths or []),
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "total_tokens": input_tok + output_tok,
        "provider_response": result.model_dump(mode="json"),
    })
    return result


async def call_api(*args, **kwargs) -> str:
    """Compatibility facade used by agent_core and every AgentRuntime."""
    result = await call_api_detailed(*args, **kwargs)
    capture = bool(result.provider_metadata.get("capture_reasoning", True))
    return result.as_agent_text(capture_reasoning=capture)

# ─────────────────────── openai-compatible (openai/ollama/llama-server/openrouter/vllm/unsloth) ─────────
async def _call_openai_compat(provider: str, user_content, *, system_prompt,
                              supplemental_system_context,
                              sampling: SamplingConfig, options: ProviderOptions,
                              config: ProviderConfig, tools_payload: dict) -> ProviderResponse:
    base_url = _openai_compat_base(provider, config.api_base)
    api_key = _openai_compat_key(provider, config.api_key)

    msgs = build_chat_messages(system_prompt, supplemental_system_context, user_content)
    response, _ = await _openai_compat_chat(
        provider=provider, base_url=base_url, api_key=api_key,
        model=config.model_name, msgs=msgs, sampling=sampling, options=options,
        tools=tools_payload.get("tools"), tool_choice=tools_payload.get("tool_choice"),
    )
    return response

# ─────────────────────── anthropic ─────────────────
async def _call_anthropic(user_content, *, system_prompt, supplemental_system_context,
                          sampling: SamplingConfig, options: ProviderOptions,
                          config: ProviderConfig, tools_payload: dict,
                          tool_runtime: Dict[str, Any] | None,
                          auto_execute_tools: bool, max_rounds: int = 4) -> ProviderResponse:
    client = anthropic.AsyncAnthropic(api_key=config.api_key)
    sys = "\n\n".join(
        [s for s in (system_prompt, supplemental_system_context) if s]
    ) or None
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    collected_reasoning: list[str] = []
    last: ProviderResponse | None = None

    for _ in range(max_rounds):
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": sampling.max_output_tokens,
        }
        if not _anthropic_rejects_sampling(config.model_name):
            kwargs["temperature"] = sampling.temperature
            kwargs["top_p"] = sampling.top_p
        if sys:
            kwargs["system"] = sys
        if (not _anthropic_rejects_sampling(config.model_name)
                and sampling.top_k is not None and sampling.top_k >= 0):
            kwargs["top_k"] = sampling.top_k
        if tools_payload.get("tools"):
            kwargs["tools"] = tools_payload["tools"]
            kwargs["tool_choice"] = {"type": "auto"}
        if options.reasoning.effort:
            kwargs["output_config"] = {"effort": options.reasoning.effort}
        if options.reasoning.enabled is False:
            kwargs["thinking"] = {"type": "disabled"}
        elif options.reasoning.enabled:
            configured_type = os.getenv("ANTHROPIC_THINKING_TYPE", "").strip().lower()
            thinking_type = configured_type or (
                "adaptive" if _anthropic_prefers_adaptive(config.model_name) else "enabled"
            )
            display = "summarized" if options.reasoning.capture else "omitted"
            if thinking_type == "adaptive":
                kwargs["thinking"] = {"type": "adaptive", "display": display}
            else:
                budget = options.reasoning.budget_tokens or 2048
                if budget < 1024 or budget >= sampling.max_output_tokens:
                    raise ValueError(
                        "Anthropic manual thinking budget must be at least 1024 "
                        "and less than max_tokens"
                    )
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                    "display": display,
                }
            if "temperature" in kwargs:
                kwargs["temperature"] = 1.0

        res = await client.messages.create(**kwargs)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_blocks: list[dict[str, Any]] = []
        for block in res.content or []:
            assistant_blocks.append(_model_dump(block))
            block_type = getattr(block, "type", None)
            if block_type == "text" and getattr(block, "text", None):
                text_parts.append(block.text)
            elif block_type == "thinking" and getattr(block, "thinking", None):
                reasoning_parts.append(block.thinking)
            elif block_type == "tool_use":
                args, raw = _normalise_arguments(getattr(block, "input", {}) or {})
                tool_calls.append(ToolCall(
                    id=getattr(block, "id", None) or f"toolu_{len(tool_calls)}",
                    provider_call_id=getattr(block, "id", None),
                    name=getattr(block, "name", None) or "unknown_tool",
                    arguments=args,
                    raw_arguments=raw,
                ))

        usage_obj = getattr(res, "usage", None)
        usage = None
        if usage_obj is not None:
            input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0),
            )
        current = ProviderResponse(
            provider="anthropic",
            model=getattr(res, "model", None) or config.model_name,
            content="\n".join(text_parts).strip(),
            reasoning=reasoning_parts,
            tool_calls=tool_calls,
            finish_reason=getattr(res, "stop_reason", None),
            usage=usage,
            provider_metadata={"id": getattr(res, "id", None)},
        )
        last = current
        collected_reasoning.extend(reasoning_parts)
        if not tool_calls or not auto_execute_tools or not tool_runtime:
            current.reasoning = collected_reasoning
            return current

        messages.append({"role": "assistant", "content": assistant_blocks})
        results = await asyncio.gather(*(
            _execute_tool_call(call, tool_runtime) for call in tool_calls
        ))
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": result.provider_call_id or result.tool_call_id,
                "content": result.content,
                **({"is_error": True} if result.is_error else {}),
            } for result in results],
        })

    if last is None:
        return ProviderResponse(provider="anthropic", model=config.model_name, finish_reason="tool_round_limit")
    last.reasoning = collected_reasoning
    last.finish_reason = "tool_round_limit"
    return last

# ─────────────────────── gemini ─────────────────
def _mime(p): return mimetypes.guess_type(p)[0] or "image/jpeg"

def _gemini_parts_from_paths(paths):
    ps=[]
    for p in paths:
        b=open(p,"rb").read()
        ps.append(types.Part.from_bytes(data=b,mime_type=_mime(p)))
    return ps

async def _call_gemini(user_content, *, system_prompt, supplemental_system_context,
                       sampling: SamplingConfig, options: ProviderOptions,
                       config: ProviderConfig, tools_payload: dict,
                       tool_runtime: Dict[str, Any] | None,
                       auto_execute_tools: bool, max_rounds: int = 4) -> ProviderResponse:
    client = genai.Client(api_key=config.api_key)
    sys = "\n\n".join(
        [s for s in (system_prompt, supplemental_system_context) if s]
    ) or None
    raw_parts = user_content if isinstance(user_content, list) else [str(user_content)]
    input_parts = [part if isinstance(part, types.Part) else types.Part(text=str(part)) for part in raw_parts]
    contents: list[types.Content] = [types.Content(role="user", parts=input_parts)]

    tool_defs = None
    if tools_payload.get("tools"):
        tool_defs = [
            item if isinstance(item, types.Tool) else types.Tool.model_validate(item)
            for item in tools_payload["tools"]
        ]

    thinking_config = None
    if options.reasoning.enabled is not None:
        thinking_kwargs: dict[str, Any] = {"include_thoughts": options.reasoning.capture}
        is_gemini_3 = "gemini-3" in config.model_name.lower()
        if is_gemini_3:
            thinking_kwargs["thinking_level"] = (
                options.reasoning.effort
                or ("medium" if options.reasoning.enabled else "minimal")
            )
        elif options.reasoning.budget_tokens is not None:
            thinking_kwargs["thinking_budget"] = options.reasoning.budget_tokens
        elif not options.reasoning.enabled:
            thinking_kwargs["thinking_budget"] = 0
        thinking_config = types.ThinkingConfig(**thinking_kwargs)

    config_kwargs: dict[str, Any] = {
        "system_instruction": sys,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_output_tokens": sampling.max_output_tokens,
        "frequency_penalty": sampling.frequency_penalty,
        "presence_penalty": sampling.presence_penalty,
        "response_mime_type": "text/plain",
    }
    if sampling.top_k is not None and sampling.top_k >= 0:
        config_kwargs["top_k"] = sampling.top_k
    if tool_defs:
        config_kwargs["tools"] = tool_defs
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    cfg = types.GenerateContentConfig(**config_kwargs)

    collected_reasoning: list[str] = []
    last: ProviderResponse | None = None
    try:
        for _ in range(max_rounds):
            response = await client.aio.models.generate_content(
                model=config.model_name,
                contents=contents,
                config=cfg,
            )
            candidates = getattr(response, "candidates", None) or []
            candidate = candidates[0] if candidates else None
            candidate_content = getattr(candidate, "content", None)
            response_parts = getattr(candidate_content, "parts", None) or []
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            signatures: list[str] = []
            for part in response_parts:
                text_value = getattr(part, "text", None)
                if text_value:
                    if getattr(part, "thought", False):
                        reasoning_parts.append(text_value)
                    else:
                        text_parts.append(text_value)
                signature = getattr(part, "thought_signature", None)
                if signature:
                    signatures.append(base64.b64encode(signature).decode() if isinstance(signature, bytes) else str(signature))
                function_call = getattr(part, "function_call", None)
                if function_call:
                    provider_call_id = getattr(function_call, "id", None)
                    args, raw = _normalise_arguments(getattr(function_call, "args", {}) or {})
                    tool_calls.append(ToolCall(
                        id=provider_call_id or f"gemini_{len(tool_calls)}",
                        provider_call_id=provider_call_id,
                        name=getattr(function_call, "name", None) or "unknown_tool",
                        arguments=args,
                        raw_arguments=raw,
                    ))

            usage_meta = getattr(response, "usage_metadata", None)
            usage = None
            if usage_meta is not None:
                input_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
                total_tokens = int(getattr(usage_meta, "total_token_count", input_tokens + output_tokens) or 0)
                usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    reasoning_tokens=int(getattr(usage_meta, "thoughts_token_count", 0) or 0),
                    cached_tokens=int(getattr(usage_meta, "cached_content_token_count", 0) or 0),
                )
            current = ProviderResponse(
                provider="gemini",
                model=getattr(response, "model_version", None) or config.model_name,
                content="\n".join(text_parts).strip(),
                reasoning=reasoning_parts,
                tool_calls=tool_calls,
                finish_reason=str(getattr(candidate, "finish_reason", "") or "") or None,
                usage=usage,
                provider_metadata={"thought_signatures": signatures},
            )
            last = current
            collected_reasoning.extend(reasoning_parts)
            if not tool_calls or not auto_execute_tools or not tool_runtime:
                current.reasoning = collected_reasoning
                return current

            if candidate_content is not None:
                contents.append(candidate_content)
            results = await asyncio.gather(*(
                _execute_tool_call(call, tool_runtime) for call in tool_calls
            ))
            response_parts_for_tools = []
            for result in results:
                try:
                    parsed_content = json.loads(result.content)
                except Exception:
                    parsed_content = {"result": result.content}
                if not isinstance(parsed_content, dict):
                    parsed_content = {"result": parsed_content}
                if result.is_error:
                    parsed_content.setdefault("error", True)
                response_parts_for_tools.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=result.provider_call_id,
                        name=result.name,
                        response=parsed_content,
                    )
                ))
            contents.append(types.Content(role="user", parts=response_parts_for_tools))
    finally:
        await client.aio.aclose()

    if last is None:
        return ProviderResponse(provider="gemini", model=config.model_name, finish_reason="tool_round_limit")
    last.reasoning = collected_reasoning
    last.finish_reason = "tool_round_limit"
    return last

# ─────────────────────── embeddings helper ────────────────────
async def get_embeddings(text: str | list[str],
                         provider: str | None = None,
                         model: str | None = None,
                         max_tokens: int = 256):
    """Create embeddings through each provider's documented native route.

    The return ABI is intentionally unchanged: one vector for a string input,
    or a list of vectors for a list input.
    """
    max_chars = max_tokens * 3
    if isinstance(text, str):
        if len(text) > max_chars: text = text[:max_chars // 2] + "..." + text[-max_chars // 2:]
    else:
        text = [t[:max_chars // 2] + "..." + t[-max_chars // 2:] if len(t) > max_chars else t for t in text]

    provider = provider or api.api_type
    if provider not in PROVIDER_TOOL_STYLE:
        raise ValueError(f"Unsupported API type: {provider}")

    state_applies = provider == api.api_type and bool(api.model_name)
    cfg = (
        ProviderConfig(model_name=api.model_name, api_key=api.api_key, api_base=api.api_base)
        if state_applies
        else get_api_config(provider)
    )
    prefix = provider.upper().replace("-", "_")

    if provider == "anthropic":
        raise ValueError("Anthropic does not expose an embeddings endpoint")

    if provider == "gemini":
        embed_model = model or os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        client = genai.Client(api_key=cfg.api_key)
        try:
            res = await client.aio.models.embed_content(model=embed_model, contents=text)
            vectors = [embedding.values for embedding in (res.embeddings or [])]
        finally:
            await client.aio.aclose()
        if not vectors:
            raise RuntimeError("Gemini returned no embedding vectors")
        return vectors[0] if isinstance(text, str) else vectors

    default_models = {
        "openai": "text-embedding-3-small",
        "ollama": "all-minilm:latest",
        "llama-server": cfg.model_name,
        "vllm": cfg.model_name,
        "openrouter": "openai/text-embedding-3-small",
        "unsloth": cfg.model_name,
    }
    embed_model = model or os.getenv(f"{prefix}_EMBED_MODEL") or default_models[provider]
    client = openai.AsyncOpenAI(
        api_key=_openai_compat_key(provider, cfg.api_key),
        **({"base_url": _openai_compat_base(provider, cfg.api_base)} if provider != "openai" else {}),
    )
    res = await client.embeddings.create(model=embed_model, input=text)
    ordered = sorted(res.data, key=lambda item: getattr(item, "index", 0))
    vectors = [item.embedding for item in ordered]
    if not vectors:
        raise RuntimeError(f"{provider} returned no embedding vectors")
    return vectors[0] if isinstance(text, str) else vectors

# ─────────────────────────── tool example ───────────────────────────
def tool_get_time(_: dict):
    return {"utc_time": datetime.now(timezone.utc).isoformat()}

TOOL_RUNTIME = {"get_time": tool_get_time}

TOOL_SPECS = [
    ToolSpec(
        name="get_time",
        description="Return current UTC time",
        parameters={"type":"object","properties":{},"required":[]}
    )
]

# ───────────────────────────  cli  ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, asyncio as _aio
    ap = argparse.ArgumentParser(description="Multi-API LLM client")
    ap.add_argument("--api", required=True,
                    choices=["ollama", "llama-server", "openai", "anthropic", "vllm", "openrouter", "gemini", "unsloth"])
    ap.add_argument("--model", help="model override")
    ap.add_argument("--tools", action="store_true", help="enable tool calling (example: get_time)")
    args = ap.parse_args()

    cfg = get_api_config(args.api, args.model)
    api.api_type   = args.api
    api.model_name = cfg.model_name
    api.api_base   = cfg.api_base
    api.api_key    = cfg.api_key

    while True:
        try:
            user_in = input(">>> ")
            if user_in.lower() in ("quit", "exit"): break
            _aio.run(call_api(
                user_in,
                tools=TOOL_SPECS if args.tools else None,
                tool_runtime=TOOL_RUNTIME if args.tools else None,
                auto_execute_tools=bool(args.tools) and args.api in ("openai","ollama","llama-server","openrouter","vllm","unsloth")
            ))
        except KeyboardInterrupt:
            break
