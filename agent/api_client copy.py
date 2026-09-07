import os, json, base64, asyncio, logging, mimetypes, atexit, queue, threading, time
from collections import defaultdict
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

import aiohttp
import openai
import anthropic
from google import genai
from google.genai import types

from PIL import Image
from dotenv import load_dotenv
from colorama import Fore, init as color_init
from pydantic import BaseModel, Field

from tokenizer import count_tokens, calculate_image_tokens

# ───────────────────────────  constants & init  ────────────────────────────
MAX_IMAGE_DIM = 640
CONSOLE_PREVIEW_CHARS = 4000
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

# ───────────────────────────  pydantic models  ─────────────────────────────
class ProviderConfig(BaseModel):
    model_name: str
    api_key: str | None = None
    api_base: str | None = None

class APIState(BaseModel):
    api_type:   str | None = None
    api_base:   str | None = None
    api_key:    str | None = None
    model_name: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.9, gt=0.0, le=1.0)
    frequency_penalty: float = Field(0.8, ge=-2.0, le=2.0)
    presence_penalty:  float = Field(0.5, ge=-2.0, le=2.0)

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict

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

# ───────────────────────────  helpers  ─────────────────────────────────────
def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val: raise EnvironmentError(f"Environment variable {var} is required")
    return val

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

def _with_v1_base(api_base: str | None) -> str:
    base = (api_base or "").rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"

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
                               media_parts: Optional[List[Dict[str, Any]]] = None) -> Tuple[object, list]:
    if not image_paths and not audio_paths and not media_parts:
        return user_content, []

    items = list(media_parts or [])
    items.extend({"type": "image", "path": p} for p in image_paths)
    items.extend({"type": "audio", "path": p} for p in audio_paths)
    media_first = api_type == "gemini" or (api_type in ("ollama", "llama-server", "unsloth") and _is_gemma4(model_name))
    if media_first:
        ordered = [x for x in items if x.get("type") in ("image", "video")]
        ordered.append({"type": "text", "text": user_content})
        ordered.extend(x for x in items if x.get("type") == "audio")
    else:
        ordered = [{"type": "text", "text": user_content}, *items]

    if api_type == "gemini":
        content, dims = [], []
        for item in ordered:
            typ = item.get("type")
            if typ == "text":
                content.append(item.get("text", ""))
            elif typ == "image":
                p = item["path"]
                img = Image.open(p); img.load()
                if max(img.size) > MAX_IMAGE_DIM:
                    ratio = MAX_IMAGE_DIM / max(img.size)
                    img = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
                content.append(img); dims.append(img.size)
            elif typ == "audio":
                content.append(types.Part.from_bytes(data=_read_bytes(item["path"]), mime_type=_mime(item["path"])))
        return content, dims

    dims = []
    if api_type == "anthropic":
        parts = []
        for item in ordered:
            typ = item.get("type")
            if typ == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif typ == "image":
                b, dim = encode_image(item["path"]); dims.append(dim)
                parts.append({"type": "image","source":{"type":"base64","media_type":"image/jpeg","data":b}})
            elif typ == "audio":
                raise ValueError("Audio input is not supported for anthropic")
        return parts, dims
    if api_type in ("openai", "ollama", "llama-server", "openrouter", "vllm", "unsloth"):
        parts = []
        for item in ordered:
            typ = item.get("type")
            if typ == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif typ == "image":
                b, dim = encode_image(item["path"]); dims.append(dim)
                parts.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}})
            elif typ == "audio":
                parts.append({"type":"input_audio","input_audio":{"data":_read_b64(item["path"]),"format":_audio_format(item["path"])}})
            elif typ == "video":
                raise ValueError("Video input should be expanded to frames before api_client")
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


def log_to_jsonl(data: dict, path: str = "api_calls.jsonl") -> None:
    """Queue an API log record; JSON serialization and file I/O stay off-thread."""
    _get_api_log_writer().submit(data, path)

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
        return ProviderConfig(api_base=os.getenv("UNSLOTH_API_BASE","http://localhost:8888"),
                              api_key=_require_env("UNSLOTH_API_KEY"),
                              model_name=model_override or os.getenv("UNSLOTH_MODEL_NAME","gemma-4-26B-A4B-it-GGUF"))
    if api_type == "gemini":
        return ProviderConfig(api_key=_require_env("GEMINI_API_KEY"),
                              model_name=model_override or os.getenv("GEMINI_MODEL_NAME","gemini-3-flash-preview"))
    if api_type == "openrouter":
        return ProviderConfig(api_base="https://openrouter.ai/api/v1",
                              api_key=_require_env("OPENROUTER_API_KEY"),
                              model_name=model_override or os.getenv("OPENROUTER_MODEL_NAME","moonshotai/kimi-k2:free"))
    raise ValueError(f"Unsupported API type: {api_type}")

def initialize_api_client(args):
    api.api_type  = args.api
    cfg = get_api_config(api.api_type, args.model)
    api.model_name = cfg.model_name
    api.api_base   = cfg.api_base
    api.api_key    = cfg.api_key
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

async def retry_api_call(func, *a, max_retries=3, retry_delay=1, **kw):
    for attempt in range(max_retries):
        try:
            return await func(*a, **kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600 and attempt < max_retries - 1:
                logging.warning("5xx from provider, retry %d/%d", attempt + 1, max_retries)
                await asyncio.sleep(retry_delay); continue
            raise

# ───────────────────────────  openai-compat tool loop  ─────────────────────
def _extract_openai_tool_calls(msg) -> List[dict]:
    tcs = getattr(msg, "tool_calls", None) or []
    out = []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", None),
            "name": getattr(fn, "name", None),
            "arguments": getattr(fn, "arguments", None) or "{}",
        })
    return [x for x in out if x["name"]]

async def _openai_compat_chat(*, base_url: str | None, api_key: str, model: str, msgs: list,
                             temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float,
                             tools: list | None, tool_choice: str | dict | None,
                             max_tokens_key: str, max_tokens_val: int,
                             gpt5_no_sampling: bool):
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    kwargs = {"model": model, "messages": msgs, max_tokens_key: max_tokens_val, "tools": tools, "tool_choice": tool_choice}
    if not gpt5_no_sampling:
        kwargs.update(temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
    r = await client.chat.completions.create(**kwargs)
    return r.choices[0].message

async def _openai_compat_call_with_auto_tools(*, provider: str, cfg: ProviderConfig,
                                             system_prompt: str, supplemental_system_context: str,
                                             user_content: object,
                                             temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float,
                                             tools_payload: dict, tool_runtime: Dict[str, Any] | None,
                                             max_rounds: int = 4):
    max_tokens_key = "max_completion_tokens" if provider == "openai" else "max_tokens"
    max_tokens_val = 12_000 if provider in ("openai", "ollama") else 12_000
    base_url = _openai_compat_base(provider, cfg.api_base)
    api_key = _openai_compat_key(provider, cfg.api_key)

    msgs = build_chat_messages(system_prompt, supplemental_system_context, user_content)
    tools = tools_payload.get("tools")
    tool_choice = tools_payload.get("tool_choice")

    for _ in range(max_rounds):
        m = await _openai_compat_chat(
            base_url=base_url, api_key=api_key, model=cfg.model_name, msgs=msgs,
            temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            tools=tools, tool_choice=tool_choice,
            max_tokens_key=max_tokens_key, max_tokens_val=max_tokens_val,
            gpt5_no_sampling=_is_gpt5(cfg.model_name) if provider == "openai" else False
        )
        tcs = _extract_openai_tool_calls(m)
        if not tcs: return m.content.strip()

        if not tool_runtime: return m.content.strip() if m.content else ""

        msgs.append({"role": "assistant", "content": m.content, "tool_calls": getattr(m, "tool_calls", None)})
        for tc in tcs:
            fn = tool_runtime.get(tc["name"])
            if not fn:
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"error": "tool_not_found", "name": tc["name"]})})
                continue
            try:
                args = json.loads(tc["arguments"] or "{}")
            except Exception:
                args = {}
            try:
                out = fn(args)
            except Exception as e:
                out = {"error": "tool_exception", "detail": str(e)}
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, ensure_ascii=False)})
    return ""

# ───────────────────────────  main entry  ──────────────────────────────────
async def call_api(user_content: str, *, supplemental_system_context: str = "", system_prompt: str = "",
                   conversation_id=None, temperature: float | None = None,
                   top_p: float | None = None, frequency_penalty: float | None = None,
                   presence_penalty: float | None = None, image_paths: List[str] | None = None,
                   audio_paths: List[str] | None = None,
                   media_parts: Optional[List[Dict[str, Any]]] = None,
                   api_type_override: str | None = None, model_override: str | None = None,
                   tools: Optional[List[ToolSpec]] = None,
                   tool_runtime: Optional[Dict[str, Any]] = None,
                   auto_execute_tools: bool = False):
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

    async def dispatch():
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
            if auto_execute_tools and tools_payload.get("tools"):
                return await _openai_compat_call_with_auto_tools(
                    provider=provider, cfg=cfg,
                    system_prompt=system_prompt,
                    supplemental_system_context=supplemental_system_context,
                    user_content=prepared_user_content,
                    temperature=temp, top_p=p_val, frequency_penalty=freq_pen, presence_penalty=pres_pen,
                    tools_payload=tools_payload, tool_runtime=tool_runtime
                )
            return await _call_openai_compat(
                provider,
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                             temperature=temp, top_p=p_val, frequency_penalty=freq_pen, presence_penalty=pres_pen,
                                             config=cfg, tools_payload=tools_payload)

        if provider == "anthropic":
            return await _call_anthropic(
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                         temperature=temp, top_p=p_val, frequency_penalty=freq_pen, presence_penalty=pres_pen,
                                         config=cfg, tools_payload=tools_payload)

        if provider == "gemini":
            return await _call_gemini(
                prepared_user_content,
                system_prompt=system_prompt,
                supplemental_system_context=supplemental_system_context,
                                      temperature=temp, top_p=p_val, frequency_penalty=freq_pen, presence_penalty=pres_pen,
                                      config=cfg, tools_payload=tools_payload)

        raise ValueError(provider)

    response = await retry_api_call(dispatch)
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
        "total_tokens": input_tok + output_tok
    })
    return response

# ─────────────────────── openai-compatible (openai/ollama/llama-server/openrouter/vllm/unsloth) ─────────
async def _call_openai_compat(provider: str, user_content, *, system_prompt,
                              supplemental_system_context,
                              temperature, top_p, frequency_penalty, presence_penalty,
                              config: ProviderConfig, tools_payload: dict):
    max_tokens_key = "max_completion_tokens" if provider == "openai" else "max_tokens"
    base_url = _openai_compat_base(provider, config.api_base)
    api_key = _openai_compat_key(provider, config.api_key)

    msgs = build_chat_messages(system_prompt, supplemental_system_context, user_content)
    m = await _openai_compat_chat(
        base_url=base_url, api_key=api_key, model=config.model_name, msgs=msgs,
        temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        tools=tools_payload.get("tools"), tool_choice=tools_payload.get("tool_choice"),
        max_tokens_key=max_tokens_key, max_tokens_val=12_000,
        gpt5_no_sampling=_is_gpt5(config.model_name) if provider == "openai" else False
    )
    return m.content.strip() if m.content else ""

# ─────────────────────── anthropic ─────────────────
async def _call_anthropic(user_content, *, system_prompt, supplemental_system_context,
                          temperature, top_p, frequency_penalty, presence_penalty,
                          config: ProviderConfig, tools_payload: dict):
    client = anthropic.AsyncAnthropic(api_key=config.api_key)
    sys = "\n\n".join(
        [s for s in (system_prompt, supplemental_system_context) if s]
    ) or None
    res = await client.messages.create(
        model=config.model_name,
        system=sys,
        messages=[{"role":"user","content":user_content}],
        max_tokens=4096,
        temperature=temperature,
        **({} if not tools_payload.get("tools") else {"tools": tools_payload["tools"]})
    )
    if res.content and hasattr(res.content[0], "text"): return res.content[0].text.strip()
    return str(res)

# ─────────────────────── gemini ─────────────────
def _mime(p): return mimetypes.guess_type(p)[0] or "image/jpeg"

def _gemini_parts_from_paths(paths):
    ps=[]
    for p in paths:
        b=open(p,"rb").read()
        ps.append(types.Part.from_bytes(data=b,mime_type=_mime(p)))
    return ps

async def _call_gemini(user_content, *, system_prompt, supplemental_system_context,
                       temperature, top_p, frequency_penalty, presence_penalty,
                       config: ProviderConfig, tools_payload: dict):
    client = genai.Client(api_key=config.api_key)
    sys = "\n\n".join(
        [s for s in (system_prompt, supplemental_system_context) if s]
    ) or None
    parts = user_content if isinstance(user_content, list) else [str(user_content)]
    cfg = types.GenerateContentConfig(
        system_instruction=sys,
        temperature=temperature,
        top_p=top_p,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        **({} if not tools_payload.get("tools") else {"tools": tools_payload["tools"]})
    )
    r = await asyncio.to_thread(client.models.generate_content, model=config.model_name, contents=parts, config=cfg)
    return (getattr(r, "text", None) or "").strip()

# ─────────────────────── embeddings helper ────────────────────
async def get_embeddings(text: str | list[str],
                         provider: str | None = None,
                         model: str | None = None,
                         max_tokens: int = 256):
    max_chars = max_tokens * 3
    if isinstance(text, str):
        if len(text) > max_chars: text = text[:max_chars // 2] + "..." + text[-max_chars // 2:]
    else:
        text = [t[:max_chars // 2] + "..." + t[-max_chars // 2:] if len(t) > max_chars else t for t in text]

    provider = provider or api.api_type
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
        model  = model or "text-embedding-3-small"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding if isinstance(text, str) else [d.embedding for d in res.data]
    if provider == "ollama":
        client = openai.AsyncOpenAI(base_url=f"{api.api_base}/v1", api_key="ollama")
        model = model or "all-minilm:latest"
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding if isinstance(text, str) else [d.embedding for d in res.data]
    if provider == "llama-server":
        base = api.api_base or os.getenv("LLAMA_SERVER_API_BASE", "http://127.0.0.1:8080")
        key = api.api_key or os.getenv("LLAMA_SERVER_API_KEY", "llama-server")
        client = openai.AsyncOpenAI(base_url=_with_v1_base(base), api_key=key)
        model = model or os.getenv("LLAMA_SERVER_EMBED_MODEL") or api.model_name
        res = await client.embeddings.create(model=model, input=text)
        return res.data[0].embedding if isinstance(text, str) else [d.embedding for d in res.data]
    if provider == "vllm":
        model = model or os.getenv("VLLM_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-en")
        async with aiohttp.ClientSession() as s:
            r = await s.post("http://localhost:8080/embed", json={"model": model, "inputs": text})
            if r.status != 200: raise RuntimeError(await r.text())
            data = await r.json()
            return data[0] if isinstance(data, list) else data["embeddings"][0]
    raise ValueError(f"Embeddings not supported for {provider}")

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
