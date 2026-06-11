"""Shared models, globals, and utility functions for the TUI."""

import os, sys, io, re, math, string, pickle, subprocess, contextlib, hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR / "agent"))

from api_client import PROVIDER_TOOL_STYLE, get_api_config
from bot_config import PromptSchema

SUPPORTED_APIS = list(PROVIDER_TOOL_STYLE.keys())

import yaml
from pydantic import BaseModel, Field, ConfigDict
from textual.app import ComposeResult
from textual.widgets import Label, ListItem


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PathConfig(BaseModel):
    root: Path = Field(default=SCRIPT_DIR)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def prompts_dir(self) -> Path: return self.root / "agent" / "prompts"
    @property
    def cache_dir(self) -> Path: return self.root / "cache"
    @property
    def discord_bot(self) -> Path: return self.root / "agent" / "discord_bot.py"

    def bot_prompts(self, name: str) -> Path: return self.prompts_dir / name
    def bot_memory(self, name: str) -> Path: return self.cache_dir / name / "memory_index" / "memory_cache.pkl"
    def bot_log(self, name: str) -> Path: return self.cache_dir / name / "logs" / f"bot_log_{name}.jsonl"
    def bot_system_prompts(self, name: str) -> Path: return self.bot_prompts(name) / "system_prompts.yaml"
    def bot_prompt_formats(self, name: str) -> Path: return self.bot_prompts(name) / "prompt_formats.yaml"
    def bot_pid(self, name: str) -> Path: return self.cache_dir / name / "bot.pid"
    def bot_tui_names(self, name: str) -> Path: return self.cache_dir / name / "tui_names.json"


@dataclass
class BotInstance:
    bot_name: str
    api: str
    model: str
    dmn_api: Optional[str] = None
    dmn_model: Optional[str] = None
    process: Optional[Any] = None
    running: bool = False
    worker: Optional[Any] = None
    external_pid: Optional[int] = None   # set for processes detected on TUI startup

    @property
    def instance_id(self) -> str:
        return self.bot_name.lower().replace(" ", "-")

    @property
    def pid(self) -> Optional[int]:
        """Active PID regardless of whether the process was launched or detected."""
        if self.process is not None:
            return self.process.pid
        return self.external_pid


class AppState:
    def __init__(self):
        self.selected_bot: Optional[str] = None
        self.selected_api: Optional[str] = None
        self.selected_model: Optional[str] = None
        self.dmn_api: Optional[str] = None
        self.dmn_model: Optional[str] = None
        self.instances: Dict[str, BotInstance] = {}

    @property
    def running_count(self) -> int:
        return sum(1 for inst in self.instances.values() if inst.running)

    def is_bot_running(self, bot_name: str) -> bool:
        return bot_name in self.instances and self.instances[bot_name].running


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════════════════════════════════════════

PATHS = PathConfig()
STATE = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_default_model(api: str) -> str:
    try:
        return get_api_config(api, None).model_name
    except Exception:
        return "unknown"


def get_api_env_key(api: str) -> Optional[str]:
    return {
        "ollama": None,
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "vllm": "VLLM_API_KEY",
        "unsloth": "UNSLOTH_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(api)


def check_api_available(api: str) -> bool:
    k = get_api_env_key(api)
    return True if k is None else bool(os.getenv(k))


def discover_bots() -> list[str]:
    if not PATHS.prompts_dir.exists():
        return ["default"]
    bots = [
        p.name
        for p in PATHS.prompts_dir.iterdir()
        if p.is_dir() and not p.name.startswith((".", "__", "archive"))
    ]
    return sorted(bots) if bots else ["default"]


def check_pid_running(pid: int) -> bool:
    """Return True if a process with the given PID is currently alive.

    Uses tasklist on Windows (signal 0 is unreliable there), os.kill(0) on Unix.
    """
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def get_bot_caches() -> list[str]:
    if not PATHS.cache_dir.exists():
        return []
    return sorted([d.name for d in PATHS.cache_dir.iterdir() if d.is_dir() and PATHS.bot_memory(d.name).exists()])


def tokenize(text: str) -> list[str]:
    text = re.sub(r"<\|[^|]+\|>", "", text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    stops = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
        "i","you","he","she","it","we","they","is","are","was","were","be","been",
        "have","has","had","this","that","these","those","into","their","from"
    }
    return [w for w in text.split() if w not in stops and len(w) >= 5]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LISTERS
# ═══════════════════════════════════════════════════════════════════════════════

def list_ollama_models() -> list[str]:
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return [l.split()[0] for l in r.stdout.strip().split("\n")[1:] if l.strip()]
    except Exception:
        pass
    return []


def list_openai_models() -> list[str]:
    k = os.getenv("OPENAI_API_KEY")
    if not k:
        return []
    try:
        import openai
        return sorted([m.id for m in openai.OpenAI(api_key=k).models.list().data if any(x in m.id for x in ["gpt", "o1", "o3", "o4"])])
    except Exception:
        return []


def list_anthropic_models() -> list[str]:
    k = os.getenv("ANTHROPIC_API_KEY")
    if not k:
        return []
    try:
        import anthropic
        return sorted([m.id for m in anthropic.Anthropic(api_key=k).models.list().data], reverse=True)
    except Exception:
        return ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-4-5"]


def list_vllm_models() -> list[str]:
    try:
        import urllib.request, json
        base = os.getenv("VLLM_API_BASE", "http://localhost:4000").rstrip("/")
        models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        with urllib.request.urlopen(models_url, timeout=5) as r:
            return [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        return []


def list_unsloth_models() -> list[str]:
    k = os.getenv("UNSLOTH_API_KEY")
    if not k:
        return []
    try:
        import urllib.request, json
        base = os.getenv("UNSLOTH_API_BASE", "http://localhost:8888").rstrip("/")
        models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        req = urllib.request.Request(models_url, headers={"Authorization": f"Bearer {k}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        return []


def list_openrouter_models() -> list[str]:
    k = os.getenv("OPENROUTER_API_KEY")
    if not k:
        return []
    try:
        import urllib.request, json
        req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {k}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            models = json.loads(r.read().decode()).get("data", [])
            free = [m["id"] for m in models if ":free" in m["id"]][:10]
            paid = [m["id"] for m in models if ":free" not in m["id"]][:10]
            return free + paid
    except Exception:
        return ["moonshotai/kimi-k2:free", "anthropic/claude-3.5-sonnet"]


def list_gemini_models() -> list[str]:
    k = os.getenv("GEMINI_API_KEY")
    if not k:
        return []
    try:
        from google import genai
        return sorted([
            m.name.replace("models/", "")
            for m in genai.Client(api_key=k).models.list()
            if "gemini" in m.name.lower() and "generateContent" in (m.supported_generation_methods or [])
        ])
    except Exception:
        return ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash"]


MODEL_LISTERS = {
    "ollama": list_ollama_models,
    "openai": list_openai_models,
    "anthropic": list_anthropic_models,
    "vllm": list_vllm_models,
    "unsloth": list_unsloth_models,
    "openrouter": list_openrouter_models,
    "gemini": list_gemini_models,
}


def get_models_for_api(api: str) -> list[str]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return MODEL_LISTERS.get(api, lambda: [])()


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

# Ground truth lives in PromptSchema (agent/bot_config.py) — imported above.
REQ_SYS = PromptSchema.required_system
REQ_FMT = PromptSchema.required_formats


def extract_tokens(s: str) -> set[str]:
    return set(re.findall(r"\{([a-zA-Z0-9_]+)\}", s or ""))


def load_yaml_file(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def save_yaml_file(p: Path, data: dict):
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_prompts(sys_prompts: dict, fmt_prompts: dict) -> dict:
    out = {"system": {}, "formats": {}, "valid": True}
    for k, v in sys_prompts.items():
        found = extract_tokens(v if isinstance(v, str) else str(v))
        need = REQ_SYS.get(k, set())
        missing = need - found
        out["system"][k] = {"found": found, "required": need, "missing": missing}
        if missing:
            out["valid"] = False
    for k, v in fmt_prompts.items():
        found = extract_tokens(v if isinstance(v, str) else str(v))
        need = REQ_FMT.get(k, set())
        missing = need - found
        out["formats"][k] = {"found": found, "required": need, "missing": missing}
        if missing:
            out["valid"] = False
    return out


def create_bot_stub(name: str) -> bool:
    p = PATHS.bot_prompts(name)
    if p.exists():
        return False
    p.mkdir(parents=True)
    sys_stub = {k: "stub. intensity {amygdala_response}%.\n" for k in REQ_SYS if k != "attention_triggers"}
    sys_stub["attention_triggers"] = []
    fmt_stub = {k: " ".join(f"{{{t}}}" for t in v) + "\n" for k, v in REQ_FMT.items()}
    save_yaml_file(p / "system_prompts.yaml", sys_stub)
    save_yaml_file(p / "prompt_formats.yaml", fmt_stub)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_memory_cache(bot_name: str) -> dict:
    path = PATHS.bot_memory(bot_name)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {
        "memories": data.get("memories", []),
        "user_memories": defaultdict(list, data.get("user_memories", {})),
        "inverted_index": defaultdict(list, data.get("inverted_index", {})),
        "path": str(path),
    }


def save_memory_cache(cache: dict) -> bool:
    if not cache.get("path"):
        return False
    try:
        tmp = cache["path"] + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({k: v for k, v in cache.items() if k != "path"}, f, protocol=5)
        os.replace(tmp, cache["path"])
        return True
    except Exception:
        return False


def load_tui_names(bot_name: str) -> dict:
    """Load cached user_id → display_name mapping for TUI chat."""
    path = PATHS.bot_tui_names(bot_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_tui_names(bot_name: str, names: dict) -> None:
    """Persist user_id → display_name mapping to disk."""
    path = PATHS.bot_tui_names(bot_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def search_memories(cache: dict, query: str, user_id: str = None, page: int = 1, per_page: int = 20) -> dict:
    if not cache.get("memories"):
        return {"memories": [], "pagination": {}}
    memories = cache["memories"]
    user_mems = cache["user_memories"]
    inv_idx = cache["inverted_index"]
    valid_ids = set(range(len(memories)))
    if user_id:
        valid_ids = set(user_mems.get(user_id, []))

    q_tokens = set(tokenize(query)) if query else set()

    if q_tokens:
        candidate_ids = set()
        for term in q_tokens:
            if term in inv_idx:
                candidate_ids.update(inv_idx[term])
        candidate_ids &= valid_ids

        total_docs = len([m for m in memories if m is not None])
        doc_freqs = {w: len(set(inv_idx.get(w, []))) for w in q_tokens if w in inv_idx}

        res = []
        for mid in candidate_ids:
            if mid >= len(memories) or memories[mid] is None:
                continue
            mem = memories[mid]
            toks = tokenize(mem)
            wc = defaultdict(int)
            for t in toks:
                wc[t] += 1
            score = 0.0
            for w in q_tokens:
                if w in wc and w in doc_freqs:
                    tf = wc[w]
                    df = doc_freqs[w]
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    score += idf * tf
            if score > 0:
                uid = next((u for u, mids in user_mems.items() if mid in mids), None)
                res.append({"id": mid, "text": mem, "user_id": uid, "score": score})
        res.sort(key=lambda x: x["score"], reverse=True)

    elif query and query.strip():
        q_lower = query.strip().lower()
        res = []
        for mid in sorted(valid_ids):
            if mid >= len(memories) or memories[mid] is None:
                continue
            mem = memories[mid]
            if q_lower in mem.lower():
                uid = next((u for u, mids in user_mems.items() if mid in mids), None)
                res.append({"id": mid, "text": mem, "user_id": uid, "score": 1.0})

    else:
        res = []
        for mid in sorted(valid_ids):
            if mid >= len(memories) or memories[mid] is None:
                continue
            uid = next((u for u, mids in user_mems.items() if mid in mids), None)
            res.append({"id": mid, "text": memories[mid], "user_id": uid, "score": 0.0})

    total = len(res)
    total_pages = max(1, math.ceil(total / per_page))
    page = min(max(1, page), total_pages)
    start = (page - 1) * per_page
    return {"memories": res[start : start + per_page], "pagination": {"page": page, "total_pages": total_pages, "total": total}}


def delete_memory(cache: dict, mid: int) -> bool:
    if mid >= len(cache["memories"]) or cache["memories"][mid] is None:
        return False
    cache["memories"][mid] = None
    for w in list(cache["inverted_index"].keys()):
        cache["inverted_index"][w] = [i for i in cache["inverted_index"][w] if i != mid]
        if not cache["inverted_index"][w]:
            del cache["inverted_index"][w]
    for uid in list(cache["user_memories"].keys()):
        if mid in cache["user_memories"][uid]:
            cache["user_memories"][uid].remove(mid)
        if not cache["user_memories"][uid]:
            del cache["user_memories"][uid]
    return True


def update_memory(cache: dict, mid: int, new_text: str) -> bool:
    """Update a single memory's text and rebuild its index entries."""
    if mid >= len(cache["memories"]) or cache["memories"][mid] is None:
        return False
    # Remove old index entries for this memory
    for w in list(cache["inverted_index"].keys()):
        cache["inverted_index"][w] = [i for i in cache["inverted_index"][w] if i != mid]
        if not cache["inverted_index"][w]:
            del cache["inverted_index"][w]
    # Update text
    cache["memories"][mid] = new_text
    # Add new index entries
    for tok in tokenize(new_text):
        cache["inverted_index"][tok].append(mid)
    return True


def _rebuild_index(cache: dict):
    """Rebuild inverted index from scratch."""
    cache["inverted_index"] = defaultdict(list)
    for mid, mem in enumerate(cache["memories"]):
        if mem is not None:
            for tok in tokenize(mem):
                cache["inverted_index"][tok].append(mid)


def find_replace_memories(cache: dict, find: str, replace: str, case_sensitive: bool = False, whole_words: bool = False, user_id: str = None) -> dict:
    """Find/replace across memories, rebuild index."""
    if not cache.get("memories") or not find:
        return {"changes": 0, "processed": 0}
    pattern = re.escape(find)
    if whole_words:
        pattern = r"\b" + pattern + r"\b"
    flags = 0 if case_sensitive else re.IGNORECASE
    valid_ids = set(range(len(cache["memories"])))
    if user_id:
        valid_ids = set(cache["user_memories"].get(user_id, []))
    changes, processed = 0, 0
    for mid in valid_ids:
        if mid >= len(cache["memories"]) or cache["memories"][mid] is None:
            continue
        processed += 1
        old = cache["memories"][mid]
        new = re.sub(pattern, replace, old, flags=flags)
        if new != old:
            cache["memories"][mid] = new
            changes += 1
    if changes > 0:
        _rebuild_index(cache)
    return {"changes": changes, "processed": processed, "user_scope": user_id or "ALL"}


def delete_user_cascade(cache: dict, user_id: str) -> dict:
    """Delete user and all their memories, reindex."""
    if user_id not in cache.get("user_memories", {}):
        return {"removed": 0, "error": "user not found"}
    mids = list(cache["user_memories"][user_id])
    for mid in mids:
        if mid < len(cache["memories"]):
            cache["memories"][mid] = None
    del cache["user_memories"][user_id]
    _rebuild_index(cache)
    for uid in list(cache["user_memories"].keys()):
        cache["user_memories"][uid] = [m for m in cache["user_memories"][uid] if m < len(cache["memories"]) and cache["memories"][m] is not None]
        if not cache["user_memories"][uid]:
            del cache["user_memories"][uid]
    return {"removed": len(mids), "user": user_id}


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VizNode:
    """A node in the memory visualization."""
    mid: int
    x: float
    y: float
    text: str
    user_id: Optional[str]
    score: float = 0.0
    grid_x: int = 0
    grid_y: int = 0


_SVD_COMPONENTS = 50  # latent dims before final PCA — captures global structure


def build_tfidf_vectors(cache: dict, memory_ids: List[int]) -> Tuple[Any, List[str], np.ndarray]:
    """Build a sparse L2-normalised float32 TF-IDF matrix for the given memory IDs."""
    from scipy.sparse import csr_matrix, diags  # lazy — only loaded when viz is used

    memories = cache["memories"]
    inv_idx  = cache["inverted_index"]

    all_terms   = sorted(inv_idx.keys())
    term_to_idx = {t: i for i, t in enumerate(all_terms)}
    total_docs  = sum(1 for m in memories if m is not None)

    idf_map = {
        t: math.log((total_docs + 1) / (len(set(postings)) + 1)) + 1
        for t, postings in inv_idx.items()
    }

    rows, cols, vals = [], [], []
    for row, mid in enumerate(memory_ids):
        if mid >= len(memories) or memories[mid] is None:
            continue
        tc: dict[str, int] = defaultdict(int)
        for t in tokenize(memories[mid]):
            tc[t] += 1
        for t, count in tc.items():
            if t in term_to_idx:
                rows.append(row)
                cols.append(term_to_idx[t])
                vals.append(float(count) * idf_map[t])

    sparse = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(memory_ids), len(all_terms)),
        dtype=np.float32,
    )

    raw_magnitudes = np.sqrt(np.asarray(sparse.power(2).sum(axis=1)).ravel())
    norms = raw_magnitudes.copy()
    norms[norms == 0] = 1.0
    sparse = diags(1.0 / norms.astype(np.float32)) @ sparse

    return sparse, all_terms, raw_magnitudes


def reduce_dimensions_sparse(sparse: Any, method: str = "pca") -> np.ndarray:
    """Reduce a sparse L2-normalised TF-IDF matrix to 2-D coordinates.

    Pipeline:
      TruncatedSVD(50)  — sparse-native, finds global structure across full vocab
      PCA(2) / UMAP(2)  — properly centered 2-D projection of those 50 dense dims
    """
    from sklearn.decomposition import TruncatedSVD, PCA  # lazy
    from sklearn.preprocessing import StandardScaler      # lazy

    n_samples = sparse.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, 2))

    n_svd = min(_SVD_COMPONENTS, n_samples - 1, sparse.shape[1] - 1)
    reduced = TruncatedSVD(n_components=n_svd, random_state=42).fit_transform(sparse)

    # Equalise SVD component contributions before final projection.
    # Without this, the first singular vector dominates and PCA/UMAP collapses
    # everything onto a single axis.
    reduced = StandardScaler().fit_transform(reduced)

    if method == "umap":
        import umap as _umap  # lazy — numba JIT only triggered on first viz with UMAP
        return _umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            random_state=42,
        ).fit_transform(reduced)

    return PCA(n_components=2, random_state=42).fit_transform(reduced)


def reduce_dimensions(vectors: Any, method: str = "pca") -> np.ndarray:
    return reduce_dimensions_sparse(vectors, method)


# ─── Viz coord cache ──────────────────────────────────────────────────────────

def _viz_cache_dir(bot_name: str) -> Path:
    return PATHS.cache_dir / bot_name / "viz_cache"


def _viz_fingerprint(bot_name: str, method: str, memory_ids: List[int]) -> str:
    """Fast, stable fingerprint for a (bot, method, memory_id_set) triple."""
    ids_bytes = np.array(sorted(memory_ids), dtype=np.int32).tobytes()
    ids_hash  = hashlib.md5(ids_bytes).hexdigest()[:10]
    key = f"{bot_name}:{method}:{len(memory_ids)}:{ids_hash}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_viz_cache(
    bot_name: str,
    method: str,
    memory_ids: List[int],
    coords: np.ndarray,
    mags: np.ndarray,
) -> bool:
    """Persist 2-D coords + magnitudes to a compressed .npz file."""
    try:
        cache_dir = _viz_cache_dir(bot_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = _viz_fingerprint(bot_name, method, memory_ids)
        np.savez_compressed(
            str(cache_dir / f"{fp}.npz"),
            coords=coords.astype(np.float32),
            mags=mags.astype(np.float32),
            memory_ids=np.array(memory_ids, dtype=np.int32),
        )
        (cache_dir / f"{fp}.meta.json").write_text(
            json.dumps({"bot": bot_name, "method": method, "n": len(memory_ids), "fp": fp}),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def load_viz_cache(
    bot_name: str,
    method: str,
    memory_ids: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load cached (coords, mags) if a valid file exists, else return None."""
    try:
        fp      = _viz_fingerprint(bot_name, method, memory_ids)
        npz_path = _viz_cache_dir(bot_name) / f"{fp}.npz"
        if not npz_path.exists():
            return None
        data = np.load(str(npz_path))
        if data["memory_ids"].tolist() != memory_ids:
            return None
        return data["coords"], data["mags"]
    except Exception:
        return None


def find_connections(
    cache: dict,
    mid: int,
    top_k: int = 5,
    valid_mids: Optional[set] = None,
) -> List[Tuple[int, float, List[str]]]:
    """Find memories connected to the given memory by shared keywords.

    If valid_mids is provided, only connections to memories in that set are
    returned — used to isolate results to the currently visible user scope.
    """
    memories = cache.get("memories", [])
    inv_idx = cache.get("inverted_index", {})

    if mid >= len(memories) or memories[mid] is None:
        return []

    source_toks = set(tokenize(memories[mid]))
    connections = defaultdict(lambda: {"score": 0.0, "terms": []})

    for term in source_toks:
        if term in inv_idx:
            for other_mid in inv_idx[term]:
                if other_mid == mid:
                    continue
                if other_mid >= len(memories) or memories[other_mid] is None:
                    continue
                if valid_mids is not None and other_mid not in valid_mids:
                    continue
                connections[other_mid]["score"] += 1
                connections[other_mid]["terms"].append(term)

    for other_mid in connections:
        other_toks = set(tokenize(memories[other_mid]))
        union_size = len(source_toks | other_toks)
        if union_size > 0:
            connections[other_mid]["score"] = len(connections[other_mid]["terms"]) / union_size

    result = [(mid, data["score"], data["terms"][:5]) for mid, data in connections.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result[:top_k]


def render_ascii_viz(nodes: List[VizNode], width: int = 60, height: int = 20,
                     selected_idx: int = -1, connections: List[int] = None) -> str:
    """Render nodes as ASCII visualization."""
    if not nodes:
        return "╔" + "═" * width + "╗\n" + \
               ("║" + " " * width + "║\n") * height + \
               "╚" + "═" * width + "╝\n[dim]no data[/dim]"

    connections = connections or []

    grid = [[" " for _ in range(width)] for _ in range(height)]
    node_positions = {}

    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max(max_x - min_x, 0.001)
    range_y = max(max_y - min_y, 0.001)

    for i, node in enumerate(nodes):
        gx = int((node.x - min_x) / range_x * (width - 2)) + 1
        gy = int((node.y - min_y) / range_y * (height - 2)) + 1
        gx = max(1, min(width - 2, gx))
        gy = max(1, min(height - 2, gy))
        node.grid_x = gx
        node.grid_y = gy

        for offset in range(10):
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                test_x = gx + dx * offset
                test_y = gy + dy * offset
                if 1 <= test_x < width - 1 and 1 <= test_y < height - 1:
                    if (test_x, test_y) not in node_positions:
                        node.grid_x = test_x
                        node.grid_y = test_y
                        node_positions[(test_x, test_y)] = i
                        break
            else:
                continue
            break

    if selected_idx >= 0 and selected_idx < len(nodes) and connections:
        sel_node = nodes[selected_idx]
        for conn_mid in connections:
            for i, node in enumerate(nodes):
                if node.mid == conn_mid:
                    _draw_line(grid, sel_node.grid_x, sel_node.grid_y,
                              node.grid_x, node.grid_y, width, height)
                    break

    for i, node in enumerate(nodes):
        gx, gy = node.grid_x, node.grid_y
        if i == selected_idx:
            char = "◉"
            color = "bold"
        elif node.mid in connections:
            char = "◎"
            color = "bold"
        elif node.score > 0.7:
            char = "●"
            color = "bold"
        elif node.score > 0.3:
            char = "○"
            color = ""
        else:
            char = "·"
            color = "dim"
        if color:
            grid[gy][gx] = f"[{color}]{char}[/{color}]"
        else:
            grid[gy][gx] = char

    lines = ["╔" + "═" * width + "╗"]
    for row in grid:
        lines.append("║" + "".join(row) + "║")
    lines.append("╚" + "═" * width + "╝")

    legend = "[bold]●[/bold] high  ○ mid  [dim]·[/dim] low  "
    legend += "[bold]◉[/bold] selected  [bold]◎[/bold] connected  "
    legend += f"[dim][{len(nodes)} nodes][/dim]"
    lines.append(legend)

    return "\n".join(lines)


def _draw_line(grid: List[List[str]], x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    """Draw line between two points using box-drawing characters."""
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return

    x_inc = dx / steps
    y_inc = dy / steps

    if abs(dx) > abs(dy) * 2:
        char = "─"
    elif abs(dy) > abs(dx) * 2:
        char = "│"
    elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
        char = "╲"
    else:
        char = "╱"

    _density = {"░", "▒", "▓"}
    _lines = {"─", "│", "╲", "╱"}

    x, y = float(x1), float(y1)
    for _ in range(steps):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            cur = grid[iy][ix]
            if cur == " " or cur in _density:
                grid[iy][ix] = char
            elif cur in _lines and cur != char:
                grid[iy][ix] = "┼"
        x += x_inc
        y += y_inc


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class SelectableItem(ListItem):
    def __init__(self, label: str, value: str, subtitle: str = "", available: bool = True):
        super().__init__()
        self.label, self.value, self.subtitle, self.available = label, value, subtitle, available

    def compose(self) -> ComposeResult:
        from rich.markup import escape as _esc
        status = "✓" if self.available else "✗"
        marker = f"[bold]{status}[/bold]" if self.available else f"[dim]{status}[/dim]"
        t = f"{marker} [bold]{_esc(self.label)}[/bold]"
        if self.subtitle:
            t += f"  [dim]{_esc(self.subtitle)}[/dim]"
        yield Label(t)
