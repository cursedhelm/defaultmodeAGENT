from __future__ import annotations

import hashlib
import mimetypes
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from tokenizer import count_tokens, decode_tokens, encode_text
from tools.anyDOCER import ANYDOC_AVAILABLE, anydoc

from .models import BookAsset, BookChunk


EPUB_CONVERSION_VERSION = 3


@dataclass
class ConvertedBook:
    title: str
    author: str | None
    markdown: str
    metadata: dict = field(default_factory=dict)
    assets: list[BookAsset] = field(default_factory=list)


def _slug(value: str, fallback: str = "book") -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()
    return clean[:80] or fallback


def _asset_extension(media_type: str) -> str:
    known = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
    }
    return known.get(media_type, mimetypes.guess_extension(media_type) or ".bin")


def _epub_metadata(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                element for element in container.iter()
                if element.tag.rsplit("}", 1)[-1] == "rootfile"
            )
            opf_path = rootfile.attrib["full-path"]
            opf = ElementTree.fromstring(archive.read(opf_path))
            for element in opf.iter():
                name = element.tag.rsplit("}", 1)[-1]
                if name in {"title", "creator", "language", "publisher", "identifier"}:
                    value = " ".join((element.text or "").split())
                    if value and name not in result:
                        result[name] = value
    except Exception:
        return result
    return result


def _style_text(text: str, style) -> str:
    if not style:
        return text
    if getattr(style, "code", False):
        text = f"`{text}`"
    if getattr(style, "bold", False):
        text = f"**{text}**"
    if getattr(style, "italic", False):
        text = f"*{text}*"
    if getattr(style, "strike", False):
        text = f"~~{text}~~"
    return text


def _render_inlines(inlines, asset_paths: dict[int, str]) -> str:
    output: list[str] = []
    for inline in inlines or []:
        kind = getattr(inline, "kind", "")
        if kind == "text":
            output.append(_style_text(getattr(inline, "text", "") or "", getattr(inline, "style", None)))
        elif kind == "line_break":
            output.append("  \n")
        elif kind == "anchor":
            # EPUB anchors describe package navigation and synthetic page
            # boundaries. They are not prose and leak XHTML noise into both
            # Markdown and embedding chunks, so do not render them.
            continue
        elif kind == "note_ref":
            output.append(f"[^{getattr(inline, 'note_id', '')}]")
        elif kind == "link":
            label = _render_inlines(getattr(inline, "content", None), asset_paths)
            target = getattr(inline, "target", None)
            value = getattr(target, "value", "") if target else ""
            target_kind = getattr(target, "kind", "") if target else ""
            # Relative/anchor EPUB links depend on discarded XHTML anchors.
            # Keep their readable label; preserve genuine external links.
            output.append(f"[{label}]({value})" if value and target_kind == "external" else label)
        elif kind == "image":
            alt = getattr(inline, "alt", None) or "image"
            source = getattr(inline, "source", None)
            source_kind = getattr(source, "kind", "") if source else ""
            if source_kind == "asset":
                value = asset_paths.get(int(getattr(source, "asset_id", -1)))
            elif source_kind == "external":
                value = getattr(source, "url", None)
            else:
                value = None
            output.append(f"![{alt}]({value})" if value else f"*[{alt}]*")
    return "".join(output)


def _epub_locator(anchor: str) -> str:
    """Turn an anydoc package anchor into a quiet, human-readable locator."""

    path, _, fragment = (anchor or "").partition("#")
    page = re.fullmatch(r"page([0-9]+|[ivxlcdm]+)", fragment, re.IGNORECASE)
    if page:
        return f"page {page.group(1).casefold()}"
    stem = Path(path).stem
    numbered = re.fullmatch(r"(chapter|part)0*(\d+)", stem, re.IGNORECASE)
    if numbered:
        return f"{numbered.group(1).casefold()} {int(numbered.group(2))}"
    labels = {
        "toc": "contents",
        "half": "half title",
        "title": "title page",
        "titlepage": "title page",
        "dedication": "dedication",
        "epigraph": "epigraph",
        "copyright": "copyright",
        "note_author": "note on the author",
    }
    return labels.get(stem.casefold(), stem.replace("_", " ").replace("-", " ") or "book")


def _block_epub_markers(block) -> list[str]:
    markers: list[str] = []
    for inline in getattr(block, "content", None) or []:
        if getattr(inline, "kind", "") != "anchor":
            continue
        anchor = getattr(inline, "anchor", "") or ""
        if anchor:
            markers.append(
                f"<!-- bookshelf:epub-locator={_epub_locator(anchor)} -->"
            )
    return markers


def clean_epub_markdown(markdown: str) -> str:
    """Remove EPUB/XHTML navigation debris while preserving Markdown prose."""

    value = re.sub(r"<a\b[^>]*>\s*</a>", "", markdown, flags=re.IGNORECASE)
    value = re.sub(r"</?a\b[^>]*>", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\[([^\]]+)\]\((?:[^)\s]*\.xhtml(?:#[^)]*)?|#[^)]+)\)",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _render_blocks(blocks, asset_paths: dict[int, str], indent: int = 0) -> str:
    rendered: list[str] = []
    for block in blocks or []:
        # anydoc also exposes anchors nested inside TOC list items. Those are
        # link destinations, not document boundaries; only top-level anchors
        # describe the spine/page position of the prose.
        if indent == 0:
            rendered.extend(_block_epub_markers(block))
        kind = getattr(block, "kind", "")
        if kind == "heading":
            level = max(1, min(6, int(getattr(block, "level", 1) or 1)))
            rendered.append(f"{'#' * level} {_render_inlines(getattr(block, 'content', None), asset_paths)}")
        elif kind == "paragraph":
            rendered.append(_render_inlines(getattr(block, "content", None), asset_paths))
        elif kind == "code_block":
            lang = getattr(block, "lang", None) or ""
            rendered.append(f"```{lang}\n{getattr(block, 'text', '') or ''}\n```")
        elif kind == "rule":
            rendered.append("---")
        elif kind == "block_quote":
            value = _render_blocks(getattr(block, "blocks", None), asset_paths, indent)
            rendered.append("\n".join(f"> {line}" for line in value.splitlines()))
        elif kind == "list":
            value = getattr(block, "list", None)
            lines: list[str] = []
            for index, item in enumerate(getattr(value, "items", []) or []):
                marker = "-"
                if getattr(value, "marker", "bullet") != "bullet":
                    marker = f"{int(getattr(value, 'start', 1) or 1) + index}."
                body = _render_blocks(getattr(item, "blocks", None), asset_paths, indent + 2).strip()
                body_lines = body.splitlines() or [""]
                lines.append(" " * indent + f"{marker} {body_lines[0]}")
                lines.extend(" " * (indent + 2) + line for line in body_lines[1:])
            rendered.append("\n".join(lines))
        elif kind == "table":
            table = getattr(block, "table", None)
            grid = getattr(table, "grid", []) or []
            rows: list[list[str]] = []
            for row in grid:
                cells: list[str] = []
                for slot in row:
                    if getattr(slot, "kind", "") != "origin":
                        cells.append("")
                        continue
                    cell = getattr(slot, "cell", None)
                    text = _render_blocks(getattr(cell, "blocks", None), asset_paths).replace("\n", " ").strip()
                    cells.append(text.replace("|", "\\|"))
                rows.append(cells)
            if rows:
                width = max(len(row) for row in rows)
                rows = [row + [""] * (width - len(row)) for row in rows]
                table_lines = [
                    "| " + " | ".join(rows[0]) + " |",
                    "| " + " | ".join("---" for _ in range(width)) + " |",
                ]
                table_lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
                rendered.append("\n".join(table_lines))
    return "\n\n".join(value for value in rendered if value.strip())


def _convert_epub(source: Path, output_dir: Path, book_id: str) -> ConvertedBook:
    if not ANYDOC_AVAILABLE or anydoc is None:
        raise RuntimeError("firecrawl-anydoc is required for EPUB ingestion")
    data = source.read_bytes()
    metadata = _epub_metadata(data)
    document = anydoc.to_document(data, "epub")
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    assets: list[BookAsset] = []
    asset_paths: dict[int, str] = {}
    for ordinal, asset in enumerate(getattr(document, "assets", []) or []):
        asset_id = int(getattr(asset, "id", ordinal))
        media_type = getattr(asset, "media_type", None) or "application/octet-stream"
        name = f"asset-{ordinal + 1:04d}{_asset_extension(media_type)}"
        relative = f"media/{name}"
        (output_dir / relative).write_bytes(bytes(getattr(asset, "data", b"")))
        asset_paths[asset_id] = relative
        assets.append(BookAsset(
            id=f"{book_id}:asset:{ordinal}", book_id=book_id, ordinal=ordinal,
            relative_path=relative, media_type=media_type,
            source_name=getattr(asset, "origin_part", None),
        ))
    markdown = _render_blocks(getattr(document, "blocks", []), asset_paths)
    notes = []
    for note in getattr(document, "notes", []) or []:
        notes.append(f"[^{getattr(note, 'id', '')}]: " + _render_blocks(
            getattr(note, "blocks", []), asset_paths
        ).replace("\n", " "))
    if notes:
        markdown += "\n\n" + "\n".join(notes)
    markdown = clean_epub_markdown(markdown)
    return ConvertedBook(
        title=metadata.get("title") or source.stem,
        author=metadata.get("creator"), markdown=markdown.strip(),
        metadata={
            **metadata,
            "format": "epub",
            "conversion_version": EPUB_CONVERSION_VERSION,
        },
        assets=assets,
    )


def _convert_pdf(source: Path, output_dir: Path, book_id: str) -> ConvertedBook:
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is None:
        if not ANYDOC_AVAILABLE or anydoc is None:
            raise RuntimeError("PyMuPDF or firecrawl-anydoc is required for PDF ingestion")
        markdown = anydoc.to_markdown(str(source))
        return ConvertedBook(
            title=source.stem, author=None,
            markdown="<!-- bookshelf:document -->\n\n" + markdown,
            metadata={"format": "pdf", "media_extraction": "unavailable"},
        )

    document = fitz.open(str(source))
    try:
        metadata = {key: value for key, value in (document.metadata or {}).items() if value}
        title = metadata.get("title") or source.stem
        author = metadata.get("author")
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        markdown_parts: list[str] = []
        assets: list[BookAsset] = []
        seen_xrefs: dict[int, str] = {}
        for page_number, page in enumerate(document, start=1):
            markdown_parts.append(f'<!-- bookshelf:page={page_number} -->')
            text = (page.get_text("text") or "").strip()
            if text:
                paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
                markdown_parts.append("\n\n".join(paragraphs))
            for image_number, image in enumerate(page.get_images(full=True), start=1):
                xref = int(image[0])
                if xref in seen_xrefs:
                    relative = seen_xrefs[xref]
                else:
                    extracted = document.extract_image(xref)
                    extension = extracted.get("ext", "bin")
                    name = f"page-{page_number:04d}-image-{image_number:02d}.{extension}"
                    relative = f"media/{name}"
                    (output_dir / relative).write_bytes(extracted["image"])
                    seen_xrefs[xref] = relative
                    media_type = mimetypes.guess_type(name)[0] or f"image/{extension}"
                    ordinal = len(assets)
                    assets.append(BookAsset(
                        id=f"{book_id}:asset:{ordinal}", book_id=book_id, ordinal=ordinal,
                        relative_path=relative, media_type=media_type,
                        locator=f"page {page_number}", source_name=f"xref:{xref}",
                    ))
                markdown_parts.append(f"![Page {page_number} image {image_number}]({relative})")
        metadata.update({"format": "pdf", "page_count": document.page_count})
        return ConvertedBook(
            title=title, author=author, markdown="\n\n".join(markdown_parts).strip(),
            metadata=metadata, assets=assets,
        )
    finally:
        document.close()


def convert_book(source_path: str | Path, output_dir: str | Path, book_id: str) -> ConvertedBook:
    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extension = source.suffix.casefold()
    if extension == ".epub":
        return _convert_epub(source, output, book_id)
    if extension == ".pdf":
        return _convert_pdf(source, output, book_id)
    raise ValueError(f"unsupported bookshelf format: {extension}")


def chunk_markdown(markdown: str, book_id: str, target_tokens: int) -> list[BookChunk]:
    """Create stable, sequential reading chunks while retaining source locators."""

    page_pattern = re.compile(r"<!--\s*bookshelf:page=(\d+)\s*-->")
    epub_locator_pattern = re.compile(
        r"<!--\s*bookshelf:epub-locator=([^>]+?)\s*-->"
    )
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    expanded: list[str] = []
    for block in blocks:
        tokens = encode_text(block)
        if len(tokens) <= target_tokens:
            expanded.append(block)
        else:
            for offset in range(0, len(tokens), target_tokens):
                expanded.append(decode_tokens(tokens[offset:offset + target_tokens]).strip())

    chunks: list[BookChunk] = []
    current: list[str] = []
    current_page: int | None = None
    current_locator: str | None = None
    pending_locator: str | None = None
    current_heading: str | None = None

    def flush() -> None:
        if not current:
            return
        text = "\n\n".join(current).strip()
        if not text:
            current.clear()
            return
        ordinal = len(chunks)
        locator = current_locator or (f"page {current_page}" if current_page is not None else (
            f"chapter {current_heading}" if current_heading else f"section {ordinal + 1}"
        ))
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
        media = list(dict.fromkeys(image_pattern.findall(text)))
        chunks.append(BookChunk(
            id=f"{book_id}:chunk:{ordinal}:{digest}", book_id=book_id,
            ordinal=ordinal, text=text, locator=locator,
            heading=current_heading, media_paths=media,
        ))
        current.clear()

    for block in expanded:
        page_match = page_pattern.fullmatch(block)
        epub_locator_match = epub_locator_pattern.fullmatch(block)
        if page_match:
            current_page = int(page_match.group(1))
            pending_locator = f"page {current_page}"
            if current_locator is None:
                current_locator = pending_locator
            continue
        if epub_locator_match:
            pending_locator = epub_locator_match.group(1).strip()
            if current_locator is None:
                current_locator = pending_locator
            continue
        heading_match = re.match(r"^#{1,6}\s+(.+)$", block.splitlines()[0])
        if heading_match:
            current_heading = heading_match.group(1).strip()
        prospective = "\n\n".join((*current, block))
        if current and count_tokens(prospective) > target_tokens:
            flush()
            current_locator = pending_locator
        current.append(block)
    flush()
    return chunks


def refresh_epub_chunks(markdown: str, chunks: list[BookChunk]) -> list[BookChunk]:
    """Clean stable chunks and recover their locator from EPUB marker offsets."""

    marker_pattern = re.compile(
        r"<!--\s*bookshelf:epub-locator=([^>]+?)\s*-->"
    )
    markers = [
        (match.start(), match.group(1).strip())
        for match in marker_pattern.finditer(markdown)
    ]
    search_from = 0
    marker_index = 0
    active_locator: str | None = None
    refreshed: list[BookChunk] = []
    for chunk in chunks:
        text = clean_epub_markdown(chunk.text) or "---"
        heading = clean_epub_markdown(chunk.heading) if chunk.heading else None
        first_block = next(
            (part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()),
            text,
        )
        needle = first_block[:200]
        position = markdown.find(needle, search_from) if needle else -1
        if position < 0 and needle:
            position = markdown.find(needle)
        if position >= 0:
            search_from = position + len(needle)
            while marker_index < len(markers) and markers[marker_index][0] <= position:
                active_locator = markers[marker_index][1]
                marker_index += 1
        locator = active_locator or clean_epub_markdown(chunk.locator) or f"section {chunk.ordinal + 1}"
        refreshed.append(chunk.model_copy(update={
            "text": text,
            "heading": heading,
            "locator": locator,
        }))
    return refreshed
