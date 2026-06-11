"""
Platform-agnostic agent core.

process_message, process_files, and generate_and_save_thought live here.
They receive a NormalizedMessage + PlatformAdapter + AgentRuntime and have
no imports from discord or discord_bot.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from bot_config import config
from chunker import clean_response, truncate_middle
from context import (
    build_conversation_context,
    build_memory_context,
    process_history_dual,
    rerank_if_enabled,
)
from adapters.base import NormalizedMessage, PlatformAdapter
from runtime import AgentRuntime
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
from tools.webSCRAPE import scrape_webpage
from attention import format_themes_for_prompt

# Module-level temporal parser (stateless utility)
_temporal_parser = TemporalParser()

# Per-task themes cache (same pattern as discord_bot.py)
_themes_ctx: contextvars.ContextVar[Dict] = contextvars.ContextVar(
    "themes_ctx", default={}
)

# Config constants — mirrors what discord_bot.py reads at module level
MAX_CONVERSATION_HISTORY = config.conversation.max_history
MINIMAL_CONVERSATION_HISTORY = config.conversation.minimal_history
TRUNCATION_LENGTH = config.conversation.truncation_length
WEB_CONTENT_TRUNCATION_LENGTH = config.conversation.web_content_truncation_length
HARSH_TRUNCATION_LENGTH = config.conversation.harsh_truncation_length
MEMORY_CAPACITY = config.persona.memory_capacity
ALLOWED_EXTENSIONS = config.files.allowed_extensions
ALLOWED_IMAGE_EXTENSIONS = config.files.allowed_image_extensions
ALLOWED_AUDIO_EXTENSIONS = config.files.allowed_audio_extensions
ALLOWED_VIDEO_EXTENSIONS = config.files.allowed_video_extensions


def _currentmoment() -> str:
    return datetime.now().strftime("%H:%M [%d/%m/%y]")


def _fallback_prompt(kind: str, *, context: str, filename: str, user_message: str, user_name: str,
                     text_files: str = "", image_files: str = "", audio_files: str = "", video_files: str = "") -> str:
    if kind == "audio":
        return f"{context}\nAudio files: {filename}\n@{user_name}: {user_message or 'Please analyze this audio.'}"
    if kind == "video":
        return f"{context}\nVideo files: {filename}\n@{user_name}: {user_message or 'Please analyze this video.'}"
    return (
        f"{context}\nImages:\n{image_files}\nAudio:\n{audio_files}\nVideo:\n{video_files}\n"
        f"Text files:\n{text_files}\n@{user_name}: {user_message or 'Please analyze these files.'}"
    )


def _themes_memoized(memory_index, user_id: str, mode: str = "sections") -> str:
    d = _themes_ctx.get()
    k = (user_id, mode)
    if k in d:
        return d[k]
    s = format_themes_for_prompt(memory_index, user_id, mode=mode)
    d[k] = s
    _themes_ctx.set(d)
    return s


def _log(runtime: AgentRuntime, data: dict) -> None:
    logger = runtime.logger
    if hasattr(logger, "log"):
        logger.log(data)
    else:
        logger.info(str(data))


def _build_url_context(url_results: list, truncation_len: int):
    """Format scraped URL content; return (ctx_str, errors, image_paths)."""
    contents, errors, image_paths = [], [], []
    for data in url_results:
        ctype = data.get("content_type", "none")
        if data.get("image_paths"):
            image_paths.extend(data["image_paths"])
        if ctype not in ("error", "none", "html_preview"):
            contents.append(
                f"URL Content: {data['url']}\nTitle: {data['title']}\n"
                f"Description: {data['description']}\n\nContent:\n{data['content']}"
            )
        elif ctype == "html_preview" and data.get("content"):
            contents.append(
                f"URL Content (partial): {data['url']}\nTitle: {data['title']}\n\n"
                f"Content:\n{data['content']}"
            )
        elif ctype == "none":
            errors.append((data["url"], data.get("description") or "Could not fetch content"))
    if not contents:
        return "", errors, image_paths
    ctx = "\nWeb Page Content:\n<web_content>\n"
    for c in contents:
        ctx += f"{truncate_middle(c, max_tokens=truncation_len)}\n"
    ctx += "</web_content>\n\n"
    return ctx, errors, image_paths


# ------------------------------------------------------------------ #
# Core processing functions                                           #
# ------------------------------------------------------------------ #

async def process_message(
    msg: NormalizedMessage,
    adapter: PlatformAdapter,
    runtime: AgentRuntime,
    memory_index,
    prompt_formats: dict,
    system_prompts: dict,
    github_repo=None,
) -> None:
    """Main message processing — platform-agnostic replacement for discord_bot.process_message."""
    runtime.logger.debug(f"Processing message from {msg.author_name}")

    if not runtime.processing_enabled:
        return

    user_id = msg.author_id
    user_name = msg.author_name

    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        msg.content,
    )
    is_first_interaction = not bool(memory_index.user_memories.get(user_id, []))

    # File routing — check msg.attachments (includes reply attachments via normalize())
    all_attachments = list(msg.attachments) + (msg.reply_to.attachments if msg.reply_to else [])
    has_supported_files = _has_supported_files(all_attachments)
    if has_supported_files:
        await process_files(
            msg=msg,
            adapter=adapter,
            runtime=runtime,
            memory_index=memory_index,
            prompt_formats=prompt_formats,
            system_prompts=system_prompts,
            user_message=msg.content,
            attachments=all_attachments,
        )
        return

    sanitized_content = adapter.sanitize_content(msg.content, msg)

    context_parts = [sanitized_content, f"@{msg.author_name}"]
    if not msg.is_dm:
        context_parts.append(f"#{msg.channel_name}")
    search_query = " ".join(context_parts)

    try:
        history_task = asyncio.create_task(
            adapter.fetch_history(msg.channel_id, MAX_CONVERSATION_HISTORY, skip_id=msg.id)
        )
        memory_task = asyncio.create_task(
            memory_index.search_async(
                search_query,
                k=MEMORY_CAPACITY,
                user_id=(user_id if msg.is_dm else None),
            )
        )
        url_tasks = [
            asyncio.create_task(
                scrape_webpage(url, cache=getattr(runtime, "cache", None), user_id=user_id)
            )
            for url in urls
        ]

        history_result, candidate_memories = await asyncio.gather(history_task, memory_task)
        history_msgs, reactions_map = history_result
        url_results = await asyncio.gather(*url_tasks) if url_tasks else []

        simple_ctx, formatted_msgs = process_history_dual(
            history_msgs, reactions_map, _temporal_parser, TRUNCATION_LENGTH
        )

        relevant_memories = await rerank_if_enabled(
            runtime, candidate_memories, search_query, logger=runtime.logger
        )

        context = adapter.format_context_header(msg)
        context += build_memory_context(relevant_memories, _temporal_parser, TRUNCATION_LENGTH)

        url_ctx, url_errors, url_image_paths = _build_url_context(
            url_results, WEB_CONTENT_TRUNCATION_LENGTH
        )
        for url, err in url_errors:
            await adapter.send(msg.channel_id, f"Error scraping URL {url}: {err}")
        context += url_ctx
        context += build_conversation_context(formatted_msgs)

        prompt_key = "introduction" if is_first_interaction else "chat_with_memory"
        sanitized_context = adapter.sanitize_content(context, msg)
        prompt = prompt_formats[prompt_key].format(
            context=sanitized_context,
            user_name=user_name,
            user_message=sanitized_content,
        )

        themes = _themes_memoized(memory_index, user_id, mode="sections")
        system_prompt = (
            system_prompts["default_chat"]
            .replace("{amygdala_response}", str(runtime.amygdala_response))
            .replace("{themes}", themes)
        )

        response_content = None
        async with adapter.thinking(msg.channel_id):
            response_content = await runtime.call_api(
                prompt=prompt,
                context=context,
                system_prompt=system_prompt,
                temperature=runtime.amygdala_response / 100,
                image_paths=url_image_paths if url_image_paths else None,
            )
            response_content, thinking_traces = separate_thinking_traces(response_content)
            await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
            response_content = clean_response(response_content)

        if response_content:
            formatted_content = adapter.format_response(response_content, msg)
            await adapter.send(msg.channel_id, formatted_content)

            if runtime.spike_processor:
                runtime.spike_processor.log_engagement(int(msg.channel_id))

            # Embedded command dispatch (Discord-specific; adapter provides no-op default)
            await adapter.invoke_embedded_commands(response_content, msg)

            timestamp = _currentmoment()
            if not msg.is_dm:
                memory_text = (
                    f"@{user_name} in {msg.guild_name} #{msg.channel_name} ({timestamp}): "
                    f"{sanitized_content}\n@{runtime.agent_name}: {response_content}"
                )
            else:
                memory_text = (
                    f"@{user_name} in DM ({timestamp}): "
                    f"{sanitized_content}\n@{runtime.agent_name}: {response_content}"
                )

            await memory_index.add_memory_async(user_id, memory_text)

            asyncio.create_task(
                generate_and_save_thought(
                    memory_index=memory_index,
                    user_id=user_id,
                    user_name=user_name,
                    memory_text=memory_text,
                    prompt_formats=prompt_formats,
                    system_prompts=system_prompts,
                    runtime=runtime,
                    conversation_context=simple_ctx,
                )
            )

            reply_context = msg.reply_to.content if msg.reply_to else None
            _log(runtime, {
                "event": "chat_interaction",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "user_name": user_name,
                "channel": msg.channel_name,
                "user_message": sanitized_content,
                "reply_to": reply_context,
                "ai_response": response_content,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "temperature": runtime.amygdala_response / 100,
            })

    except Exception as e:
        await adapter.send(msg.channel_id, f"An error occurred: {str(e)}")
        runtime.logger.error(
            f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}"
        )
        _log(runtime, {
            "event": "chat_error",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_name": user_name,
            "channel": msg.channel_name,
            "error": str(e),
        })


async def process_files(
    msg: NormalizedMessage,
    adapter: PlatformAdapter,
    runtime: AgentRuntime,
    memory_index,
    prompt_formats: dict,
    system_prompts: dict,
    user_message: str = "",
    attachments: Optional[List] = None,
) -> None:
    """File/image processing — platform-agnostic replacement for discord_bot.process_files."""
    if not runtime.processing_enabled:
        await adapter.send(msg.channel_id, "Processing currently disabled.")
        return

    user_id = msg.author_id
    user_name = msg.author_name

    if attachments is None:
        attachments = msg.attachments

    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        msg.content,
    )
    if not attachments and not urls:
        await adapter.send(msg.channel_id, "No attachments or URLs found.")
        return

    if not user_message:
        user_message = adapter.sanitize_content(msg.content, msg)

    runtime.logger.info(
        f"Processing {len(attachments)} files from {user_name} (ID: {user_id}) "
        f"with message: {user_message}"
    )

    history_task = asyncio.create_task(
        adapter.fetch_history(msg.channel_id, MINIMAL_CONVERSATION_HISTORY, skip_id=msg.id)
    )
    url_tasks = [
        asyncio.create_task(
            scrape_webpage(url, cache=getattr(runtime, "cache", None), user_id=user_id)
        )
        for url in urls
    ]

    image_files: List[str] = []
    audio_files: List[str] = []
    video_files: List[str] = []
    text_contents: List[dict] = []
    temp_paths: List[str] = []
    audio_paths: List[str] = []
    video_frame_paths: List[str] = []
    media_source_paths: List[str] = []
    has_images = False
    has_audio = False
    has_video = False
    has_text = False

    try:
        cache = getattr(runtime, "cache", None)

        for att in attachments:
            ext = os.path.splitext(att.filename.lower())[1]
            is_potentially_image = (
                att.content_type
                and att.content_type.startswith("image/")
                and ext in ALLOWED_IMAGE_EXTENSIONS
            )
            is_potentially_audio = (
                ext in ALLOWED_AUDIO_EXTENSIONS
                or (att.content_type and att.content_type.startswith("audio/"))
            )
            is_potentially_video = (
                ext in ALLOWED_VIDEO_EXTENSIONS
                or (att.content_type and att.content_type.startswith("video/"))
            )
            is_potentially_text = ext in ALLOWED_EXTENSIONS
            data_to_save = None
            processed_as_image = False
            processed_as_audio = False
            processed_as_video = False
            processed_as_text = False

            if att.size > 1_000_000:
                if is_potentially_audio or is_potentially_video:
                    pass
                elif is_potentially_image:
                    try:
                        from PIL import Image
                        import io
                        image_data = await att.read()
                        img = Image.open(io.BytesIO(image_data))
                        img.load()
                        img.thumbnail((512, 512))
                        output_buffer = io.BytesIO()
                        save_format = "PNG" if img.mode == "RGBA" else "JPEG"
                        if img.mode == "P":
                            img = img.convert("RGB")
                            save_format = "JPEG"
                        elif img.mode == "LA":
                            img = img.convert("RGBA")
                            save_format = "PNG"
                        img.save(output_buffer, format=save_format)
                        resized_data = output_buffer.getvalue()
                        if len(resized_data) > 1_000_000:
                            runtime.logger.warning(
                                f"Image {att.filename} still too large after resizing."
                            )
                            await adapter.send(
                                msg.channel_id,
                                f"Sorry, could not resize {att.filename} sufficiently. Skipping.",
                            )
                            continue
                        data_to_save = resized_data
                        processed_as_image = True
                    except Exception as e:
                        runtime.logger.error(f"Error resizing image {att.filename}: {str(e)}")
                        await adapter.send(
                            msg.channel_id,
                            f"Error processing large image {att.filename}. Skipping.",
                        )
                        continue
                else:
                    runtime.logger.warning(f"Skipping oversized non-image file: {att.filename}")
                    await adapter.send(
                        msg.channel_id,
                        f"Skipping {att.filename} - file is over 1MB and not a resizable image.",
                    )
                    continue
            if not data_to_save and not (processed_as_image or processed_as_text):
                if is_potentially_image:
                    try:
                        from PIL import Image
                        import io
                        image_data = await att.read()
                        data_to_save = image_data
                        processed_as_image = True
                        try:
                            img = Image.open(io.BytesIO(data_to_save))
                            img.verify()
                        except Exception as e:
                            runtime.logger.warning(
                                f"Small image {att.filename} failed verification: {e}. "
                                "Still attempting to use."
                            )
                    except Exception as e:
                        runtime.logger.error(f"Error processing small image {att.filename}: {str(e)}")
                        continue
                elif is_potentially_audio:
                    try:
                        data_to_save = await att.read()
                        processed_as_audio = True
                    except Exception as e:
                        runtime.logger.error(f"Error reading audio {att.filename}: {str(e)}")
                        continue
                elif is_potentially_video:
                    try:
                        data_to_save = await att.read()
                        processed_as_video = True
                    except Exception as e:
                        runtime.logger.error(f"Error reading video {att.filename}: {str(e)}")
                        continue
                elif is_potentially_text:
                    try:
                        content = (await att.read()).decode("utf-8")
                        mode = config.files.text_ingestion_mode
                        if mode == "truncate":
                            if len(content) > config.files.truncate_length:
                                content = content[: config.files.truncate_length]
                        elif mode == "chronpress":
                            if len(content) > config.files.chronpress_threshold:
                                content = await _smart_compress(content)
                        elif mode == "hybrid":
                            if len(content) > config.files.chronpress_threshold:
                                content = await _smart_compress(content)
                            if len(content) > config.files.truncate_length:
                                content = content[: config.files.truncate_length]
                        text_contents.append({"filename": att.filename, "content": content})
                        processed_as_text = True
                    except UnicodeDecodeError:
                        continue
                else:
                    await adapter.send(
                        msg.channel_id,
                        f"Skipping {att.filename} - unsupported type. "
                        f"Supported: {', '.join(ALLOWED_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS)}",
                    )
                    continue

            if processed_as_image and data_to_save and cache:
                try:
                    temp_path, _ = cache.create_temp_file(
                        user_id=user_id,
                        prefix="img_",
                        suffix=os.path.splitext(att.filename)[1],
                        content=data_to_save,
                    )
                    if not os.path.exists(temp_path):
                        runtime.logger.error(f"Failed to save image to temp file: {temp_path}")
                        continue
                    image_files.append(att.filename)
                    temp_paths.append(temp_path)
                    has_images = True
                except Exception as e:
                    runtime.logger.error(f"Error saving temp image file {att.filename}: {str(e)}")
                    continue
            elif processed_as_text:
                has_text = True
            elif processed_as_audio and data_to_save and cache:
                src_path, _ = cache.create_temp_file(user_id=user_id, prefix="audio_src_", suffix=ext, content=data_to_save)
                media_source_paths.append(src_path)
                out_path = await _compress_audio(src_path, cache, user_id, att.filename)
                audio_files.append(att.filename); audio_paths.append(out_path); has_audio = True
            elif processed_as_video and data_to_save and cache:
                src_path, _ = cache.create_temp_file(user_id=user_id, prefix="video_src_", suffix=ext, content=data_to_save)
                media_source_paths.append(src_path)
                frames = await _compress_video(src_path, cache, user_id, att.filename)
                video_files.append(att.filename); video_frame_paths.extend(frames); has_video = True

        history_result = await history_task
        history_msgs, reactions_map = history_result
        url_results = await asyncio.gather(*url_tasks) if url_tasks else []

        for data in url_results:
            ctype = data.get("content_type", "none")
            if data.get("image_paths"):
                for img_path in data["image_paths"]:
                    if img_path and img_path not in temp_paths:
                        temp_paths.append(img_path)
                        image_files.append(os.path.basename(img_path))
                        has_images = True
            if ctype not in ("error", "none"):
                text_contents.append({
                    "filename": f"webpage_{data['title']}",
                    "content": (
                        f"URL: {data['url']}\nTitle: {data['title']}\n"
                        f"Description: {data['description']}\n\nContent:\n{data['content']}"
                    ),
                })
                has_text = True
            elif ctype == "none":
                await adapter.send(
                    msg.channel_id,
                    f"Error scraping URL {data['url']}: {data.get('description', 'Unknown error')}",
                )

        if not (has_images or has_audio or has_video or has_text):
            await adapter.send(msg.channel_id, "No valid files found to analyze after processing.")
            return

        _, formatted_msgs = process_history_dual(
            history_msgs, reactions_map, _temporal_parser, HARSH_TRUNCATION_LENGTH
        )

        context = f"Current channel: #{msg.channel_name}\n\n"
        context += "<conversation>\n"
        for m in formatted_msgs:
            context += f"{m}\n"
        context += "</conversation>\n"

        amygdala = str(runtime.amygdala_response)
        themes = ", ".join(_themes_memoized(memory_index, user_id, mode="just_user").split())

        media_count = sum(bool(x) for x in (has_images, has_audio, has_video, has_text))
        if media_count > 1:
            if "analyze_combined" not in prompt_formats or "combined_analysis" not in system_prompts:
                raise ValueError("Missing required combined analysis prompts")
            prompt_tpl = prompt_formats.get("analyze_combined")
            prompt = prompt_tpl.format(
                context=context,
                image_files="\n".join(image_files),
                audio_files="\n".join(audio_files),
                video_files="\n".join(video_files),
                text_files="\n".join(
                    f"{t['filename']}: {truncate_middle(t['content'], 1000)}"
                    for t in text_contents
                ),
                user_message=user_message or "Please analyze these files.",
                user_name=user_name,
            ) if prompt_tpl else _fallback_prompt(
                "combined", context=context, image_files="\n".join(image_files),
                audio_files="\n".join(audio_files), video_files="\n".join(video_files),
                text_files="\n".join(f"{t['filename']}: {truncate_middle(t['content'], 1000)}" for t in text_contents),
                filename="", user_message=user_message, user_name=user_name
            )
            system_prompt = (
                system_prompts["combined_analysis"]
                .replace("{amygdala_response}", amygdala)
                .replace("{themes}", themes)
            )
        elif has_images:
            if "analyze_image" not in prompt_formats or "image_analysis" not in system_prompts:
                raise ValueError("Missing required image analysis prompts")
            prompt = prompt_formats["analyze_image"].format(
                context=context,
                filename=", ".join(image_files),
                user_message=user_message or "Please analyze these images.",
                user_name=user_name,
            )
            system_prompt = (
                system_prompts["image_analysis"]
                .replace("{amygdala_response}", amygdala)
                .replace("{themes}", themes)
            )
        elif has_audio:
            prompt_tpl = prompt_formats.get("analyze_audio")
            prompt = prompt_tpl.format(
                context=context,
                filename=", ".join(audio_files),
                user_message=user_message or "Please analyze this audio.",
                user_name=user_name,
            ) if prompt_tpl else _fallback_prompt("audio", context=context, filename=", ".join(audio_files), user_message=user_message, user_name=user_name)
            system_prompt = (
                system_prompts.get("audio_analysis", system_prompts["combined_analysis"])
                .replace("{amygdala_response}", amygdala)
                .replace("{themes}", themes)
            )
        elif has_video:
            prompt_tpl = prompt_formats.get("analyze_video")
            prompt = prompt_tpl.format(
                context=context,
                filename=", ".join(video_files),
                user_message=user_message or "Please analyze this video.",
                user_name=user_name,
            ) if prompt_tpl else _fallback_prompt("video", context=context, filename=", ".join(video_files), user_message=user_message, user_name=user_name)
            system_prompt = (
                system_prompts.get("video_analysis", system_prompts["combined_analysis"])
                .replace("{amygdala_response}", amygdala)
                .replace("{themes}", themes)
            )
        else:
            if "analyze_file" not in prompt_formats or "file_analysis" not in system_prompts:
                raise ValueError("Missing required file analysis prompts")
            combined_text = "\n\n".join(
                f"=== {t['filename']} ===\n{t['content']}" for t in text_contents
            )
            prompt = prompt_formats["analyze_file"].format(
                context=context,
                filename=", ".join(t["filename"] for t in text_contents),
                file_content=combined_text,
                user_message=user_message,
                user_name=user_name,
            )
            system_prompt = (
                system_prompts["file_analysis"]
                .replace("{amygdala_response}", amygdala)
                .replace("{themes}", themes)
            )

        response_content = None
        async with adapter.thinking(msg.channel_id):
            response_content = await runtime.call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                image_paths=(temp_paths + video_frame_paths) if (temp_paths or video_frame_paths) else None,
                audio_paths=audio_paths if audio_paths else None,
                temperature=runtime.amygdala_response / 100,
            )
            response_content, thinking_traces = separate_thinking_traces(response_content)
            await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
            response_content = clean_response(response_content)

        if response_content:
            formatted_content = adapter.format_response(response_content, msg)
            await adapter.send(msg.channel_id, formatted_content)

            if runtime.spike_processor:
                runtime.spike_processor.log_engagement(int(msg.channel_id))

            await adapter.invoke_embedded_commands(response_content, msg)

            files_desc = []
            if image_files:
                files_desc.append(f"{len(image_files)} images: {', '.join(image_files)}")
            if audio_files:
                files_desc.append(f"{len(audio_files)} audio files: {', '.join(audio_files)}")
            if video_files:
                files_desc.append(f"{len(video_files)} videos: {', '.join(video_files)}")
            if text_contents:
                files_desc.append(
                    f"{len(text_contents)} text files: "
                    f"{', '.join(t['filename'] for t in text_contents)}"
                )

            timestamp = _currentmoment()
            sanitized_msg = adapter.sanitize_content(user_message, msg)
            memory_text = (
                f"({timestamp}) Grokking {' and '.join(files_desc)} for User @{user_name} "
                f"in #{msg.channel_name}. User's message: {sanitized_msg}\n"
                f"@{runtime.agent_name}: {response_content}"
            )

            await memory_index.add_memory_async(user_id, memory_text)

            file_context = ""
            if text_contents:
                file_context += "File Contents:\n"
                for fd in text_contents:
                    file_context += (
                        f"--- {fd['filename']} ---\n"
                        f"{truncate_middle(fd['content'], max_tokens=TRUNCATION_LENGTH)}\n\n"
                    )
            if image_files:
                file_context += f"Images analyzed: {', '.join(image_files)}\n"
            if audio_files:
                file_context += f"Audio analyzed: {', '.join(audio_files)}\n"
            if video_files:
                file_context += f"Video analyzed as frames: {', '.join(video_files)}\n"

            paths_to_cleanup = list(temp_paths) + list(audio_paths) + list(video_frame_paths) + list(media_source_paths)

            def _cleanup():
                for p in paths_to_cleanup:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                        meta = f"{p}.meta"
                        if os.path.exists(meta):
                            os.remove(meta)
                    except Exception as e:
                        runtime.logger.debug(f"temp.cleanup path={p} err={e}")

            asyncio.create_task(
                generate_and_save_thought(
                    memory_index=memory_index,
                    user_id=user_id,
                    user_name=user_name,
                    memory_text=memory_text,
                    prompt_formats=prompt_formats,
                    system_prompts=system_prompts,
                    runtime=runtime,
                    file_context=file_context,
                    image_paths=(temp_paths + video_frame_paths) if (temp_paths or video_frame_paths) else None,
                    audio_paths=audio_paths if audio_paths else None,
                    cleanup_callback=_cleanup,
                )
            )

            _log(runtime, {
                "event": "file_analysis",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "user_name": user_name,
                "files_processed": {
                    "images": image_files,
                    "audio": audio_files,
                    "video": video_files,
                    "text_files": [t["filename"] for t in text_contents],
                },
                "user_message": user_message,
                "ai_response": response_content,
            })

    except Exception as e:
        await adapter.send(msg.channel_id, f"An error occurred while analyzing files: {str(e)}")
        runtime.logger.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        runtime.logger.error(traceback.format_exc())


async def generate_and_save_thought(
    memory_index,
    user_id: str,
    user_name: str,
    memory_text: str,
    prompt_formats: dict,
    system_prompts: dict,
    runtime: AgentRuntime,
    file_context: str = "",
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    cleanup_callback=None,
    conversation_context: str = "",
) -> None:
    """Generate a reflective thought about a memory and store it."""
    current_time = datetime.now()
    storage_timestamp = current_time.strftime("%H:%M [%d/%m/%y]")
    temporal_expr = _temporal_parser.get_temporal_expression(current_time)
    temporal_timestamp = temporal_expr.base_expression
    if temporal_expr.time_context:
        temporal_timestamp = f"{temporal_timestamp} in the {temporal_expr.time_context}"

    timestamp_pattern = r"\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)"
    temporal_memory_text = re.sub(
        timestamp_pattern,
        lambda m: (
            f"({_temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})"
        ),
        memory_text,
    )

    thought_prompt = prompt_formats["generate_thought"].format(
        user_name=user_name,
        memory_text=temporal_memory_text,
        timestamp=temporal_timestamp,
        conversation_context=conversation_context,
    )
    if file_context:
        thought_prompt += f"\n\nAdditional File Context:\n{file_context}"

    themes = _themes_memoized(memory_index, user_id, mode="sections")
    thought_system_prompt = (
        system_prompts["thought_generation"]
        .replace("{amygdala_response}", str(runtime.amygdala_response))
        .replace("{themes}", themes)
    )

    thought_response = await runtime.call_api(
        prompt=thought_prompt,
        context="",
        system_prompt=thought_system_prompt,
        image_paths=image_paths,
        audio_paths=audio_paths,
        temperature=runtime.amygdala_response / 100,
    )
    thought_response, thinking_traces = separate_thinking_traces(thought_response)
    await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
    thought_response = clean_response(thought_response)

    memory_string = (
        f"Reflections on interactions with @{user_name} ({storage_timestamp}):\n {thought_response}"
    )
    runtime.logger.debug(f"Pre-memory addition string: {memory_string}")
    await memory_index.add_memory_async(user_id, memory_string)
    runtime.logger.debug(f"Post-memory addition: {memory_index.user_memories[user_id][-1]}")

    _log(runtime, {
        "event": "thought_generation",
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "user_name": user_name,
        "memory_text": memory_text,
        "thought_response": thought_response,
    })

    if cleanup_callback:
        cleanup_callback()


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

async def _run_media_command(args: List[str], logger=None) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-500:]
        if logger:
            logger.error(f"media.compress.err cmd={args[0]} msg={detail}")
        raise RuntimeError(detail or f"{args[0]} exited with {proc.returncode}")


async def _compress_audio(src_path: str, cache, user_id: str, filename: str) -> str:
    out_dir = cache.get_user_temp_dir(user_id)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(filename)[0])[:80] or "audio"
    out_path = os.path.join(out_dir, f"audio_{stem}.wav")
    await _run_media_command([
        "ffmpeg", "-y",
        "-i", src_path,
        "-t", str(config.files.audio_max_seconds),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        out_path,
    ], getattr(cache, "logger", None))
    return out_path


async def _compress_video(src_path: str, cache, user_id: str, filename: str) -> List[str]:
    out_dir = cache.get_user_temp_dir(user_id)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(filename)[0])[:80] or "video"
    pattern = os.path.join(out_dir, f"video_{stem}_%03d.jpg")
    await _run_media_command([
        "ffmpeg", "-y",
        "-i", src_path,
        "-t", str(config.files.video_max_seconds),
        "-vf", f"fps={config.files.video_frame_rate},scale=640:-2:force_original_aspect_ratio=decrease",
        "-q:v", "4",
        pattern,
    ], getattr(cache, "logger", None))
    return sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(f"video_{stem}_") and f.endswith(".jpg")
    )

def _has_supported_files(attachments: List) -> bool:
    for att in attachments:
        ext = os.path.splitext(att.filename.lower())[1]
        if ext in ALLOWED_EXTENSIONS or (
            att.content_type
            and att.content_type.startswith("image/")
            and ext in ALLOWED_IMAGE_EXTENSIONS
        ) or ext in ALLOWED_AUDIO_EXTENSIONS or ext in ALLOWED_VIDEO_EXTENSIONS or (
            att.content_type and (att.content_type.startswith("audio/") or att.content_type.startswith("video/"))
        ):
            return True
    return False


async def _smart_compress(text: str) -> str:
    target_chars = config.files.chronpress_target_chars
    if len(text) <= target_chars:
        return text
    ratio = 1.0 - (target_chars / len(text)) + 0.05
    compression = max(0.3, min(ratio, 0.90))
    try:
        from tools.chronpression import chronomic_filter
        return await asyncio.to_thread(chronomic_filter, text, compression=compression, fuzzy_strength=1.0)
    except Exception:
        return text
