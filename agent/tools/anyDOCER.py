"""
A single-entry document decoder built on `anydoc` (firecrawl/anydoc).

Converts binary document formats (Word, PowerPoint, Excel, OpenDocument, RTF,
EPUB, CSV, PDF) into GitHub-Flavored Markdown so they can enter the agent's
context the same way a plain `.txt` attachment does.

It guarantees the same five keys in the returned dict:

    {filename, content, content_type, doc_format, error_info}

Where:
* `content_type` is **never** "error". It is one of:
    - "document"   -> decoded successfully, `content` is Markdown
    - "none"       -> nothing could be decoded (unsupported/encrypted/corrupt)
* `error_info` is always an **empty dict** so downstream clients never see
  decoder internals.
* `doc_format` is the format anydoc detected ("pdf", "docx", ...) or "" when
  nothing was decoded.

The decoder is *not* a budgeter. `to_markdown_bytes` returns the whole
document; use `decode_and_compress` (or the caller's own
`text_ingestion_mode` block) to fit it into the context window. Compression
reuses the chunked chronomic path so large PDFs are not fed to
`chronomic_filter` in a single shot.

Format is detected from the **bytes**, not the extension, so a mislabeled
upload still converts. Signature-less formats (CSV) carry no magic bytes and
fall back to the extension hint.

The module still logs everything (`logging.warning`/`logging.exception`) to aid
server-side debugging.
"""

import os
import asyncio
import logging
import argparse
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor

from .chronpression import chronomic_filter

# Optional anydoc import - document ingestion degrades to "none" without it
try:
    import anydoc
    ANYDOC_AVAILABLE = True
except ImportError:
    ANYDOC_AVAILABLE = False
    logging.info(
        "anydoc not installed - document ingestion unavailable. "
        "Install with: pip install firecrawl-anydoc"
    )

# Formats anydoc decodes. Mirrors config.files.allowed_document_extensions;
# kept here so the tool is usable standalone (CLI, tests) without bot config.
DOCUMENT_EXTENSIONS = {
    '.doc', '.docx', '.docm',
    '.ppt', '.pptx', '.pptm',
    '.xls', '.xlsx', '.xlsm', '.xlsb',
    '.odt', '.ods', '.odp',
    '.rtf', '.epub', '.csv', '.pdf',
}

DOCUMENT_MIME_TYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.oasis.opendocument.text',
    'application/vnd.oasis.opendocument.spreadsheet',
    'application/vnd.oasis.opendocument.presentation',
    'application/rtf',
    'text/rtf',
    'application/epub+zip',
    'text/csv',
}

MAX_DOCUMENT_BYTES = 25_000_000
CHUNK_TARGET_CHARS = 8000
MAX_PARALLEL_CHUNKS = 8
MIN_COMPRESSION = 0.3
MAX_COMPRESSION = 0.92
MIN_CONTENT_LENGTH = 1  # Only a wholly empty decode counts as a failure

_decode_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS)


class DocumentResult(TypedDict):
    filename: str
    content: str
    content_type: str
    doc_format: str
    error_info: dict


def _empty_result(filename: str) -> DocumentResult:
    return DocumentResult(
        filename=filename,
        content="",
        content_type="none",
        doc_format="",
        error_info={},
    )


def is_document(filename: str, content_type: str | None = None) -> bool:
    """True when `filename`/`content_type` names a format anydoc can decode.

    Used as the routing gate before reading attachment bytes. Extension is the
    cheap pre-filter; anydoc re-detects from bytes at decode time, so a
    mislabeled file still converts once it reaches `decode_bytes`.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext in DOCUMENT_EXTENSIONS:
        return True
    if content_type:
        ct = content_type.split(';')[0].strip().lower()
        return ct in DOCUMENT_MIME_TYPES
    return False


def _sync_decode(data: bytes, filename: str) -> DocumentResult:
    """Blocking decode. Runs on `_decode_executor`, never on the event loop."""
    if not ANYDOC_AVAILABLE:
        logging.warning("Document decode skipped for %s - anydoc not installed", filename)
        return _empty_result(filename)

    if not data:
        logging.warning("Document decode skipped for %s - empty payload", filename)
        return _empty_result(filename)

    if len(data) > MAX_DOCUMENT_BYTES:
        logging.warning(
            "Document decode skipped for %s - %d bytes exceeds limit %d",
            filename, len(data), MAX_DOCUMENT_BYTES
        )
        return _empty_result(filename)

    # Bytes win; extension is the fallback for signature-less formats (CSV).
    detected = anydoc.format_from_bytes(data)
    ext_format = anydoc.format_from_extension(os.path.splitext(filename.lower())[1])
    doc_format = detected or ext_format

    try:
        if detected:
            markdown = anydoc.to_markdown_bytes(data)
        elif ext_format:
            markdown = anydoc.to_markdown_bytes(data, ext_format)
        else:
            logging.warning(
                "Document decode failed for %s - unrecognized content and unknown extension",
                filename
            )
            return _empty_result(filename)
    except anydoc.EncryptedError:
        logging.warning("Document decode failed for %s - file is encrypted", filename)
        return _empty_result(filename)
    except anydoc.ConvertError as e:
        # UnsupportedError / MalformedError / MissingPartError / ResourceLimitError
        logging.warning("Document decode failed for %s - %s: %s", filename, type(e).__name__, e)
        return _empty_result(filename)
    except Exception:
        logging.exception("Document decode failed unexpectedly for %s", filename)
        return _empty_result(filename)

    markdown = (markdown or "").strip()
    if len(markdown) < MIN_CONTENT_LENGTH:
        logging.warning(
            "Document decode produced no usable content for %s (format=%s)",
            filename, doc_format
        )
        return _empty_result(filename)

    logging.info(
        "Decoded %s (format=%s): %d bytes -> %d chars markdown",
        filename, doc_format, len(data), len(markdown)
    )
    return DocumentResult(
        filename=filename,
        content=markdown,
        content_type="document",
        doc_format=doc_format or "",
        error_info={},
    )


async def decode_bytes(data: bytes, filename: str) -> DocumentResult:
    """Decode an in-memory document to Markdown. Never raises."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_decode_executor, _sync_decode, data, filename)
    except Exception:
        logging.exception("Document decode dispatch failed for %s", filename)
        return _empty_result(filename)


async def decode_path(path: str) -> DocumentResult:
    """Decode a document from disk. Never raises."""
    filename = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        logging.warning("Could not read document %s: %s", path, e)
        return _empty_result(filename)
    return await decode_bytes(data, filename)


def get_compression_for_target(raw_chars: int, target_chars: int) -> float:
    """Compression ratio that brings `raw_chars` down to roughly `target_chars`."""
    if raw_chars <= target_chars:
        return MIN_COMPRESSION
    needed_reduction = 1.0 - (target_chars / raw_chars)
    compression = needed_reduction + 0.05
    return min(MAX_COMPRESSION, max(MIN_COMPRESSION, compression))


def chunk_text_by_chars(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> list[str]:
    """Split on blank lines so Markdown blocks (tables, lists) stay intact."""
    blocks = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for block in blocks:
        if not block.strip():
            continue
        block_chars = len(block)
        if current_chars + block_chars > target_chars and current:
            chunks.append("\n\n".join(current))
            current = [block]
            current_chars = block_chars
        else:
            current.append(block)
            current_chars += block_chars
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _sync_compress_all(text: str, compression: float, fuzzy_strength: float) -> str:
    """Chunked chronomic compression - large documents are never single-shot."""
    if len(text) < CHUNK_TARGET_CHARS * 1.5:
        try:
            return chronomic_filter(
                text, compression=compression, fuzzy_strength=fuzzy_strength, horizon=6
            )
        except Exception:
            logging.warning("Chronomic filtering failed")
            return text

    chunks = chunk_text_by_chars(text, CHUNK_TARGET_CHARS)
    compressed = []
    for chunk in chunks:
        try:
            compressed.append(
                chronomic_filter(
                    chunk, compression=compression, fuzzy_strength=fuzzy_strength, horizon=6
                )
            )
        except Exception:
            logging.warning("Chronomic filtering failed on chunk")
            compressed.append(chunk)
    return "\n\n".join(c for c in compressed if c)


async def compress_markdown(
    text: str, compression: float = 0.5, fuzzy_strength: float = 1.0
) -> tuple[str, bool]:
    """Compress decoded Markdown. Returns (text, succeeded)."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _decode_executor, _sync_compress_all, text, compression, fuzzy_strength
        )
        return result, True
    except Exception:
        logging.exception("Chronomic compression dispatch failed")
        return text, False


async def decode_and_compress(
    data: bytes,
    filename: str,
    threshold_chars: int,
    target_chars: int,
    fuzzy_strength: float = 1.0,
) -> DocumentResult:
    """Decode a document and chronpress it if it exceeds `threshold_chars`.

    The one call an ingestion path needs: bytes in, context-sized Markdown out.
    Documents under the threshold pass through untouched.
    """
    result = await decode_bytes(data, filename)
    if result["content_type"] == "none":
        return result

    content = result["content"]
    if len(content) > threshold_chars:
        compression = get_compression_for_target(len(content), target_chars)
        compressed, ok = await compress_markdown(
            content, compression=compression, fuzzy_strength=fuzzy_strength
        )
        if ok:
            logging.info(
                "Chronpressed %s: %d -> %d chars (compression=%.2f)",
                filename, len(content), len(compressed), compression
            )
            result["content"] = compressed

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="anydoc -> Markdown decoder")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output")
    p.add_argument("-c", "--compress", action="store_true", help="Apply chronomic compression")
    p.add_argument("--threshold", type=int, default=16000)
    p.add_argument("--target", type=int, default=8000)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)

    async def _main():
        with open(a.input, "rb") as f:
            data = f.read()
        if a.compress:
            return await decode_and_compress(
                data, os.path.basename(a.input), a.threshold, a.target
            )
        return await decode_bytes(data, os.path.basename(a.input))

    res = asyncio.run(_main())
    if res["content_type"] == "none":
        print(f"Could not decode {a.input}")
        raise SystemExit(1)

    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(res["content"])
        if a.verbose:
            print(f"wrote {a.output} (format={res['doc_format']}, {len(res['content'])} chars)")
    else:
        print(res["content"])
