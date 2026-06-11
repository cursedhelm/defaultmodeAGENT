#discord
import discord
from discord import app_commands
from discord.ext import commands
from discord.ext.commands.view import StringView
# standard libraries
import asyncio
import os
import contextvars
import mimetypes
import yaml
from datetime import datetime
import argparse
import threading
import re
import importlib.util
import sys
from typing import Optional
# api import and hyperparameter handlers
from hippocampus import Hippocampus, HippocampusConfig
from context import (
    fetch_history_with_reactions,
    process_history_dual,
    build_memory_context,
    build_conversation_context,
    get_or_create_hippocampus,
    rerank_if_enabled
)
# image handling
from PIL import Image
import io
import traceback
# import tools
from tools.discordSUMMARISER import ChannelSummarizer
from tools.discordGITHUB import GitHubRepo, RepoIndex, process_repo_contents, repo_processing_event
from tools.webSCRAPE import scrape_webpage
from tools.chronpression import chronomic_filter
# import memory module
from memory import UserMemoryIndex, CacheManager
from defaultmode import DMNProcessor
from chunker import truncate_middle, clean_response, balance_wraps
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
# Discord Format Handling
from discord_utils import sanitize_mentions, format_discord_mentions
from attention import check_attention_triggers_fuzzy, get_current_themes, format_themes_for_prompt
# Action generation
from spike import SpikeProcessor
# Configuration imports
from bot_config import (
    config,
    init_logging,
    apply_overrides,
)
# libraries logging import for jsonl, sqlite and info logging
from logger import BotLogger

init_logging()

script_dir = os.path.dirname(os.path.abspath(__file__))

_themes_ctx = contextvars.ContextVar('themes_ctx', default={})

def format_themes_for_prompt_memoized(mi, uid, mode="just_user"):
    d=_themes_ctx.get()
    k=(uid,mode)
    if k in d:
        return d[k]
    s=format_themes_for_prompt(mi,uid,mode=mode)
    d[k]=s
    _themes_ctx.set(d)
    return s

# Access config values
HIPPOCAMPUS_BANDWIDTH = config.persona.hippocampus_bandwidth
MAX_CONVERSATION_HISTORY = config.conversation.max_history
MINIMAL_CONVERSATION_HISTORY = config.conversation.minimal_history
TRUNCATION_LENGTH = config.conversation.truncation_length
WEB_CONTENT_TRUNCATION_LENGTH = config.conversation.web_content_truncation_length
HARSH_TRUNCATION_LENGTH = config.conversation.harsh_truncation_length
TEMPERATURE = config.persona.temperature
DEFAULT_AMYGDALA_RESPONSE = config.persona.default_amygdala_response
ALLOWED_EXTENSIONS = config.files.allowed_extensions
ALLOWED_IMAGE_EXTENSIONS = config.files.allowed_image_extensions
DISCORD_BOT_MANAGER_ROLE = config.discord.bot_manager_role
TICK_RATE = config.system.tick_rate
MEMORY_CAPACITY = config.persona.memory_capacity
MOOD_COEFF = config.persona.mood_coefficient

# Attention system control
attention_enabled = True

def log_to_jsonl(data, bot_id=None):
    """Log data to JSONL file and SQLite database with consistent timestamp format.
    
    Args:
        data (dict): Data to log
        bot_id (str, optional): Bot identifier for log filename. If None, uses current bot's name.
    """
    # Get or create logger instance using bot's name if not specified
    if not hasattr(log_to_jsonl, '_logger'):
        log_to_jsonl._logger = None    
    # Update logger if bot_id changes or not initialized
    current_bot_id = bot_id or (bot.user.name if bot and bot.user else "default")
    if not log_to_jsonl._logger or log_to_jsonl._logger.bot_id != current_bot_id:
        log_to_jsonl._logger = BotLogger(current_bot_id)
    # Use logger to handle both JSONL and SQLite logging
    log_to_jsonl._logger.log(data)

def update_temperature(intensity: int) -> None:
    """Update the bot's temperature based on intensity value. Manages Amagdala response and API client temperature/top p."""
    TEMPERATURE = intensity / 100.0
    bot.update_api_temperature(TEMPERATURE)
    if hasattr(bot, 'dmn_processor') and bot.dmn_processor:
        bot.dmn_processor.amygdala_response = intensity
        bot.dmn_processor.temperature = TEMPERATURE
    bot.logger.info(f"Updated bot temperature to {TEMPERATURE} across all components")

async def interaction_send(interaction: discord.Interaction, content: str, *, ephemeral: bool = True) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(content, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(content, ephemeral=ephemeral)

async def require_interaction_permission(command_name: str, interaction: discord.Interaction) -> bool:
    if config.discord.has_interaction_permission(command_name, interaction):
        return True
    await interaction_send(interaction, "You don't have permission to use this command.", ephemeral=True)
    return False

def currentmoment():
    return datetime.now().strftime("%H:%M [%d/%m/%y]")

def start_background_processing_thread(repo, memory_index, max_depth=None, branch='main', channel=None):
    thread = threading.Thread(target=run_background_processing, args=(repo, memory_index, max_depth, branch))
    thread.start()
    bot.logger.info(f"Started background processing of repository contents in a separate thread (Branch: {branch}, Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")

def run_background_processing(repo, memory_index, max_depth=None, branch='main', channel=None):
    global repo_processing_event
    repo_processing_event.clear()
    try:
        asyncio.run(process_repo_contents(repo, '', memory_index, max_depth, branch))
        memory_index.save_cache()  # Save the cache after indexing
        if channel:
            asyncio.get_event_loop().create_task(channel.send(f"Repository indexing completed for branch '{branch}'"))
    except Exception as e:
        bot.logger.error(f"Error in background processing for branch '{branch}': {str(e)}")
    finally:
        repo_processing_event.set()

async def maintain_typing_state(channel):
    """Maintains typing state in channel by refreshing before timeout."""
    try:
        async with channel.typing():
            # Keep typing state active for up to 5 minutes
            await asyncio.sleep(300)
    except Exception as e:
        bot.logger.debug(f"Typing state maintenance ended: {str(e)}")


async def extract_content_and_reply(message, is_command, bot=None):
    """extract user content, handle reply context, return (content, reply_context, reply_attachments)"""
    reply_context = None
    reply_attachments = []
    
    if is_command:
        parts = message.content.split(maxsplit=1)
        return (parts[1] if len(parts) > 1 else "", None, [])
    
    if message.guild and message.guild.me:
        content = message.content.replace(f'<@!{message.guild.me.id}>', '').replace(f'<@{message.guild.me.id}>', '').strip()
    else:
        content = message.content.strip()
    
    if message.reference:
        try:
            original = await message.channel.fetch_message(message.reference.message_id)
            original_content = original.content.strip()
            
            if original.attachments:
                for att in original.attachments:
                    ext = os.path.splitext(att.filename.lower())[1]
                    is_image = att.content_type and att.content_type.startswith('image/') and ext in ALLOWED_IMAGE_EXTENSIONS
                    is_text = ext in ALLOWED_EXTENSIONS
                    if is_image or is_text:
                        reply_attachments.append(att)
            
            if original_content:
                for m in original.mentions:
                    original_content = original_content.replace(f'<@{m.id}>', f'@{m.name}').replace(f'<@!{m.id}>', f'@{m.name}')
                for ch in original.channel_mentions:
                    original_content = original_content.replace(f'<#{ch.id}>', f'#{ch.name}')
                reply_context = original_content
                content = f"[@{message.author.name} replying to @{original.author.name}'s message: {original_content}]\n\n@{message.author.name}: {content}"
        except (discord.NotFound, discord.Forbidden):
            pass
    
    return content, reply_context, reply_attachments


def build_url_context(url_results, truncation_len):
    """format scraped url content and collect image paths"""
    contents = []
    errors = []
    image_paths = []
    for data in url_results:
        ctype = data.get('content_type', 'none')
        # Collect image paths from scraped results
        if data.get('image_paths'):
            image_paths.extend(data['image_paths'])
        if ctype not in ('error', 'none', 'html_preview'):
            contents.append(f"URL Content: {data['url']}\nTitle: {data['title']}\nDescription: {data['description']}\n\nContent:\n{data['content']}")
        elif ctype == 'html_preview' and data.get('content'):
            contents.append(f"URL Content (partial): {data['url']}\nTitle: {data['title']}\n\nContent:\n{data['content']}")
        elif ctype == 'none':
            errors.append((data['url'], data.get('description') or 'Could not fetch content'))
    if not contents:
        return "", errors, image_paths
    ctx = "\nWeb Page Content:\n<web_content>\n"
    for c in contents:
        ctx += f"{truncate_middle(c, max_tokens=truncation_len)}\n"
    ctx += "</web_content>\n\n"
    return ctx, errors, image_paths


async def smart_compress_text(text: str) -> str:
    target_chars = config.files.chronpress_target_chars
    if len(text) <= target_chars:
        return text
    ratio = 1.0 - (target_chars / len(text)) + 0.05
    compression = max(0.3, min(ratio, 0.90))
    try:
        return await asyncio.to_thread(
            chronomic_filter,
            text,
            compression=compression,
            fuzzy_strength=1.0
        )
    except Exception:
        return text


async def process_message(message, memory_index, prompt_formats, system_prompts, github_repo, is_command=False):
    """main message processing with parallel i/o and single history fetch"""
    bot.logger.debug(f"Processing message from {message.author.name}")
    
    if not getattr(bot, 'processing_enabled', True):
        return
    
    user_id = str(message.author.id)
    user_name = message.author.name
    is_dm = isinstance(message.channel, discord.DMChannel)
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)
    is_first_interaction = not bool(memory_index.user_memories.get(user_id, []))
    content, reply_context, reply_attachments = await extract_content_and_reply(message, is_command, bot)
    
    all_attachments = list(message.attachments) + reply_attachments
    has_supported_files = False
    for att in all_attachments:
        ext = os.path.splitext(att.filename.lower())[1]
        if (ext in ALLOWED_EXTENSIONS) or (att.content_type and att.content_type.startswith('image/') and ext in ALLOWED_IMAGE_EXTENSIONS):
            has_supported_files = True
            break
    
    if has_supported_files:
        await process_files(
            message=message,
            memory_index=memory_index,
            prompt_formats=prompt_formats,
            system_prompts=system_prompts,
            user_message=content,
            bot=bot,
            attachments=all_attachments
        )
        return
    
    combined_mentions = list(message.mentions) + list(message.channel_mentions) + list(message.role_mentions)
    sanitized_content = sanitize_mentions(content, combined_mentions)
    
    context_parts = [sanitized_content, f"@{sanitize_mentions(user_name, combined_mentions)}"]
    if not is_dm:
        channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
        context_parts.append(f"#{sanitize_mentions(channel_name, combined_mentions)}")
    search_query = " ".join(context_parts)
    
    try:
        response_content = None
        
        history_task = asyncio.create_task(fetch_history_with_reactions(message.channel, MAX_CONVERSATION_HISTORY, skip_id=message.id))
        memory_task = asyncio.create_task(memory_index.search_async(search_query, k=MEMORY_CAPACITY, user_id=(user_id if is_dm else None)))
        url_tasks = [asyncio.create_task(scrape_webpage(url, cache=bot.cache, user_id=user_id)) for url in urls]
        
        history_result, candidate_memories = await asyncio.gather(history_task, memory_task)
        history_msgs, reactions_map = history_result
        url_results = await asyncio.gather(*url_tasks) if url_tasks else []
        
        simple_ctx, formatted_msgs = process_history_dual(history_msgs, reactions_map, bot.temporal_parser, TRUNCATION_LENGTH)
        
        relevant_memories = await rerank_if_enabled(bot, candidate_memories, search_query, logger=bot.logger)
        
        if hasattr(message.channel, 'name'):
            context = f"Current Discord server: {message.guild.name}, channel: #{message.channel.name}\n"
        else:
            context = "Current channel: Direct Message\n"
        
        context += build_memory_context(relevant_memories, bot.temporal_parser, TRUNCATION_LENGTH)
        
        url_ctx, url_errors, url_image_paths = build_url_context(url_results, WEB_CONTENT_TRUNCATION_LENGTH)
        for url, err in url_errors:
            await message.channel.send(f"Error scraping URL {url}: {err}")
        context += url_ctx

        context += build_conversation_context(formatted_msgs)

        prompt_key = 'introduction' if is_first_interaction else 'chat_with_memory'
        prompt = prompt_formats[prompt_key].format(
            context=sanitize_mentions(context, combined_mentions),
            user_name=user_name,
            user_message=sanitize_mentions(sanitized_content, combined_mentions)
        )

        themes = format_themes_for_prompt_memoized(bot.memory_index, user_id, mode="sections")
        system_prompt = system_prompts['default_chat'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)

        typing_task = asyncio.create_task(maintain_typing_state(message.channel))
        try:
            # Pass image paths from scraped URLs to the API for vision processing
            response_content = await bot.call_api(
                prompt,
                context=context,
                system_prompt=system_prompt,
                temperature=bot.amygdala_response/100,
                image_paths=url_image_paths if url_image_paths else None
            )
            response_content, thinking_traces = separate_thinking_traces(response_content)
            await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
            response_content = clean_response(response_content)
        finally:
            typing_task.cancel()
        
        if response_content:
            formatted_content = format_discord_mentions(response_content, message.guild, bot.mentions_enabled, bot)
            await send_long_message(message.channel, formatted_content, bot=bot)
            if hasattr(bot, 'spike_processor') and bot.spike_processor:
                bot.spike_processor.log_engagement(message.channel.id)
            await invoke_embedded_commands(response_content, message, bot)

            timestamp = currentmoment()
            channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'

            if hasattr(message.channel, 'name'):
                memory_text = f"@{user_name} in {message.guild.name} #{channel_name} ({timestamp}): {sanitize_mentions(sanitized_content, combined_mentions)}\n@{bot.user.name}: {response_content}"
            else:
                memory_text = f"@{user_name} in DM ({timestamp}): {sanitize_mentions(sanitized_content, combined_mentions)}\n@{bot.user.name}: {response_content}"
            
            await memory_index.add_memory_async(user_id, memory_text)
            
            asyncio.create_task(generate_and_save_thought(
                memory_index=memory_index,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot,
                conversation_context=simple_ctx
            ))
            
            log_to_jsonl({
                'event': 'chat_interaction',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'channel': channel_name,
                'user_message': sanitized_content,
                'reply_to': reply_context,
                'ai_response': response_content,
                'system_prompt': system_prompt,
                'prompt': prompt,
                'temperature': bot.amygdala_response/100
            }, bot_id=bot.user.name)
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await send_long_message(message.channel, error_message, bot=bot)
        bot.logger.error(f"Error in message processing for {user_name} (ID: {user_id}): {str(e)}")
        log_to_jsonl({
            'event': 'chat_error',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'channel': message.channel.name if hasattr(message.channel, 'name') else 'DM',
            'error': str(e)
        }, bot_id=bot.user.name)


async def process_files(message, memory_index, prompt_formats, system_prompts, user_message="", bot=None, temperature=TEMPERATURE, attachments=None):
    """file processing with parallel url scraping and single history fetch"""
    if not getattr(bot, 'processing_enabled', True):
        await message.channel.send("Processing currently disabled.")
        return
    
    user_id = str(message.author.id)
    user_name = message.author.name
    
    attachments = attachments if attachments is not None else list(message.attachments)
    
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content)
    if not attachments and not urls:
        await message.channel.send("No attachments or URLs found.")
        return
    
    if not user_message:
        if message.guild and message.guild.me:
            user_message = message.content.replace(f'<@!{message.guild.me.id}>', '').replace(f'<@{message.guild.me.id}>', '').strip()
        else:
            user_message = message.content.strip()
    combined_mentions = list(message.mentions) + list(message.channel_mentions) + list(message.role_mentions)
    user_message = sanitize_mentions(user_message, combined_mentions)
    
    bot.logger.info(f"Processing {len(attachments)} files from {user_name} (ID: {user_id}) with message: {user_message}")
    
    history_task = asyncio.create_task(fetch_history_with_reactions(message.channel, MINIMAL_CONVERSATION_HISTORY, skip_id=message.id))
    url_tasks = [asyncio.create_task(scrape_webpage(url, cache=bot.cache, user_id=user_id)) for url in urls]
    
    image_files = []
    text_contents = []
    temp_paths = []
    has_images = False
    has_text = False
    
    try:
        for attachment in attachments:
            ext = os.path.splitext(attachment.filename.lower())[1]
            is_potentially_image = (attachment.content_type and attachment.content_type.startswith('image/') and ext in ALLOWED_IMAGE_EXTENSIONS)
            is_potentially_text = ext in ALLOWED_EXTENSIONS
            data_to_save = None
            processed_as_image = False
            processed_as_text = False
            
            if attachment.size > 1000000:
                if is_potentially_image:
                    try:
                        image_data = await attachment.read()
                        img = Image.open(io.BytesIO(image_data))
                        img.load()
                        img.thumbnail((512, 512))
                        output_buffer = io.BytesIO()
                        save_format = 'PNG' if img.mode == 'RGBA' else 'JPEG'
                        if img.mode == 'P':
                            img = img.convert('RGB')
                            save_format = 'JPEG'
                        elif img.mode == 'LA':
                            img = img.convert('RGBA')
                            save_format = 'PNG'
                        img.save(output_buffer, format=save_format)
                        resized_data = output_buffer.getvalue()
                        if len(resized_data) > 1000000:
                            bot.logger.warning(f"Image {attachment.filename} still too large after resizing.")
                            await message.channel.send(f"Sorry, could not resize {attachment.filename} sufficiently. Skipping.")
                            continue
                        data_to_save = resized_data
                        processed_as_image = True
                    except Exception as e:
                        bot.logger.error(f"Error resizing image {attachment.filename}: {str(e)}")
                        await message.channel.send(f"Error processing large image {attachment.filename}. Skipping.")
                        continue
                else:
                    bot.logger.warning(f"Skipping oversized non-image file: {attachment.filename}")
                    await message.channel.send(f"Skipping {attachment.filename} - file is over 1MB and not a resizable image.")
                    continue
            else:
                if is_potentially_image:
                    try:
                        image_data = await attachment.read()
                        data_to_save = image_data
                        processed_as_image = True
                        try:
                            img = Image.open(io.BytesIO(data_to_save))
                            img.verify()
                        except Exception as e:
                            bot.logger.warning(f"Small image {attachment.filename} failed verification: {e}. Still attempting to use.")
                    except Exception as e:
                        bot.logger.error(f"Error processing small image {attachment.filename}: {str(e)}")
                        continue
                elif is_potentially_text:
                    try:
                        content = (await attachment.read()).decode("utf-8")
                        mode = config.files.text_ingestion_mode

                        if mode == "truncate":
                            if len(content) > config.files.truncate_length:
                                content = content[:config.files.truncate_length]

                        elif mode == "chronpress":
                            if len(content) > config.files.chronpress_threshold:
                                content = await smart_compress_text(content)

                        elif mode == "hybrid":
                            if len(content) > config.files.chronpress_threshold:
                                content = await smart_compress_text(content)
                            if len(content) > config.files.truncate_length:
                                content = content[:config.files.truncate_length]

                        text_contents.append({"filename": attachment.filename, "content": content})
                        processed_as_text = True

                    except UnicodeDecodeError:
                        continue

                else:
                    await message.channel.send(f"Skipping {attachment.filename} - unsupported type. Supported types: {', '.join(ALLOWED_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS)}")
                    continue
            
            if processed_as_image and data_to_save:
                try:
                    temp_path, file_id = bot.cache.create_temp_file(
                        user_id=user_id,
                        prefix="img_",
                        suffix=os.path.splitext(attachment.filename)[1],
                        content=data_to_save
                    )
                    if not os.path.exists(temp_path):
                        bot.logger.error(f"Failed to save image to temp file: {temp_path}")
                        continue
                    image_files.append(attachment.filename)
                    temp_paths.append(temp_path)
                    has_images = True
                except Exception as e:
                    bot.logger.error(f"Error saving temp image file {attachment.filename}: {str(e)}")
                    continue
            elif processed_as_text:
                has_text = True
        
        history_result = await history_task
        history_msgs, reactions_map = history_result
        url_results = await asyncio.gather(*url_tasks) if url_tasks else []
        
        for data in url_results:
            ctype = data.get('content_type', 'none')
            # Collect image paths from scraped URLs
            if data.get('image_paths'):
                for img_path in data['image_paths']:
                    if img_path and img_path not in temp_paths:
                        temp_paths.append(img_path)
                        image_files.append(os.path.basename(img_path))
                        has_images = True
            if ctype not in ('error', 'none'):
                text_contents.append({
                    'filename': f"webpage_{data['title']}",
                    'content': f"URL: {data['url']}\nTitle: {data['title']}\nDescription: {data['description']}\n\nContent:\n{data['content']}"
                })
                has_text = True
            elif ctype == 'none':
                await message.channel.send(f"Error scraping URL {data['url']}: {data.get('description', 'Unknown error')}")
        
        if not (has_images or has_text):
            if not message.channel.last_message or message.channel.last_message.author != bot.user:
                await message.channel.send("No valid files found to analyze after processing.")
            return
        
        _, formatted_msgs = process_history_dual(history_msgs, reactions_map, bot.temporal_parser, HARSH_TRUNCATION_LENGTH)
        
        context = f"Current channel: #{message.channel.name if hasattr(message.channel, 'name') else 'Direct Message'}\n\n"
        context += "<conversation>\n"
        for msg in formatted_msgs:
            context += f"{msg}\n"
        context += "</conversation>\n"
        
        amygdala_response = str(bot.amygdala_response if bot else DEFAULT_AMYGDALA_RESPONSE)
        themes = ", ".join(get_current_themes(bot.memory_index))
        
        if has_images and has_text:
            if 'analyze_combined' not in prompt_formats or 'combined_analysis' not in system_prompts:
                raise ValueError("Missing required combined analysis prompts")
            prompt = prompt_formats['analyze_combined'].format(
                context=context,
                image_files="\n".join(image_files),
                text_files="\n".join(f"{t['filename']}: {truncate_middle(t['content'], 1000)}" for t in text_contents),
                user_message=user_message if user_message else "Please analyze these files.",
                user_name=user_name
            )
            system_prompt = system_prompts['combined_analysis'].replace('{amygdala_response}', amygdala_response).replace('{themes}', themes)
        elif has_images:
            if 'analyze_image' not in prompt_formats or 'image_analysis' not in system_prompts:
                raise ValueError("Missing required image analysis prompts")
            prompt = prompt_formats['analyze_image'].format(
                context=context,
                filename=", ".join(image_files),
                user_message=user_message if user_message else "Please analyze these images.",
                user_name=user_name
            )
            system_prompt = system_prompts['image_analysis'].replace('{amygdala_response}', amygdala_response).replace('{themes}', themes)
        else:
            if 'analyze_file' not in prompt_formats or 'file_analysis' not in system_prompts:
                raise ValueError("Missing required file analysis prompts")
            combined_text = "\n\n".join(f"=== {t['filename']} ===\n{t['content']}" for t in text_contents)
            prompt = prompt_formats['analyze_file'].format(
                context=context,
                filename=", ".join(t['filename'] for t in text_contents),
                file_content=combined_text,
                user_message=user_message,
                user_name=user_name
            )
            system_prompt = system_prompts['file_analysis'].replace('{amygdala_response}', amygdala_response).replace('{themes}', themes)
        
        typing_task = asyncio.create_task(maintain_typing_state(message.channel))
        try:
            response_content = await bot.call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                image_paths=temp_paths if temp_paths else None,
                temperature=bot.amygdala_response/100
            )
            response_content, thinking_traces = separate_thinking_traces(response_content)
            await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
            response_content = clean_response(response_content)
        finally:
            typing_task.cancel()
        
        if response_content:
            formatted_content = format_discord_mentions(response_content, message.guild, bot.mentions_enabled, bot)
            await send_long_message(message.channel, formatted_content, bot=bot)
            if hasattr(bot, 'spike_processor') and bot.spike_processor:
                bot.spike_processor.log_engagement(message.channel.id)
            await invoke_embedded_commands(response_content, message, bot)

            files_description = []
            if image_files:
                files_description.append(f"{len(image_files)} images: {', '.join(image_files)}")
            if text_contents:
                files_description.append(f"{len(text_contents)} text files: {', '.join(t['filename'] for t in text_contents)}")
            
            timestamp = currentmoment()
            channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
            memory_text = f"({timestamp}) Grokking {' and '.join(files_description)} for User @{user_name} in #{channel_name}. User's message: {sanitize_mentions(user_message, combined_mentions)}\n@{bot.user.name}: {response_content}"
            
            await memory_index.add_memory_async(user_id, memory_text)
            
            file_context = ""
            if text_contents:
                file_context += "File Contents:\n"
                for file_data in text_contents:
                    truncated_content = truncate_middle(file_data['content'], max_tokens=TRUNCATION_LENGTH)
                    file_context += f"--- {file_data['filename']} ---\n{truncated_content}\n\n"
            if image_files:
                file_context += f"Images analyzed: {', '.join(image_files)}\n"

            # Capture the specific paths to clean up - don't use force=True which nukes ALL temp files
            paths_to_cleanup = list(temp_paths) if temp_paths else []
            def cleanup_temp_files():
                for p in paths_to_cleanup:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                        meta_path = f"{p}.meta"
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                    except Exception as e:
                        bot.logger.debug(f"temp.cleanup.specific path={p} err={e}")

            asyncio.create_task(generate_and_save_thought(
                memory_index=memory_index,
                user_id=user_id,
                user_name=user_name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot,
                file_context=file_context,
                image_paths=temp_paths if temp_paths else None,
                cleanup_callback=cleanup_temp_files
            ))
            
            log_to_jsonl({
                'event': 'file_analysis',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'files_processed': {'images': image_files, 'text_files': [t['filename'] for t in text_contents]},
                'user_message': user_message,
                'ai_response': response_content
            }, bot_id=bot.user.name)
    
    except Exception as e:
        error_message = f"An error occurred while analyzing files: {str(e)}"
        await send_long_message(message.channel, error_message, bot=bot)
        bot.logger.error(f"Error in file analysis for {user_name} (ID: {user_id}): {str(e)}")
        bot.logger.error(traceback.format_exc())
        
async def send_long_message(channel: discord.TextChannel, text: str, max_length=1800, bot=None):
    '''Send a long message to a Discord channel, splitting it into chunks if necessary while preserving formatting.'''
    if not text:
        return
    formatted_text = text
    segments = []
    lines = formatted_text.split('\n')
    current_segment = []
    in_code_block = False
    tag_stack = []
    for line in lines:
        if '```' in line:
            if not in_code_block:
                if current_segment:
                    segments.append(('\n'.join(current_segment), False))
                    current_segment = []
                in_code_block = True
            else:
                in_code_block = False
                current_segment.append(line)
                segments.append(('\n'.join(current_segment), True))
                current_segment = []
                continue
        if not in_code_block:
            opens = line.count('<')
            closes = line.count('>')
            if opens > closes:
                tag_stack.extend(['<'] * (opens - closes))
            elif closes > opens and tag_stack:
                tag_stack = tag_stack[:(opens - closes)]
                
            if tag_stack and not current_segment:
                if current_segment:
                    segments.append(('\n'.join(current_segment), False))
                    current_segment = []
            elif not tag_stack and current_segment and any('<' in s or '>' in s for s in current_segment):
                current_segment.append(line)
                segments.append(('\n'.join(current_segment), True))
                current_segment = []
                continue
        current_segment.append(line)
    if current_segment:
        segments.append(('\n'.join(current_segment), in_code_block or bool(tag_stack)))
    chunks = []
    current_chunk = []
    current_length = 0
    for content, is_wrapped in segments:
        if is_wrapped:
            if len(content) > max_length:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                if '\n' not in content:
                    remaining = content
                    while remaining:
                        chunk_size = max_length - 6
                        if remaining.startswith('```'):
                            chunk = remaining[:chunk_size] + '\n```'
                            remaining = '```\n' + remaining[chunk_size:] if remaining[chunk_size:] else ''
                        else:
                            chunk = remaining[:chunk_size]
                            remaining = remaining[chunk_size:]
                        chunks.append(chunk)
                else:
                    balanced = balance_wraps(content)
                    while balanced:
                        original_length = len(balanced)
                        split_point = balanced.rfind('\n', 0, max_length)
                        if split_point == -1:
                            split_point = max_length - 6
                        chunk = balanced[:split_point]
                        if '```' in chunk and chunk.count('```') % 2 != 0:
                            chunk += '\n```'
                        chunks.append(chunk)
                        balanced = balanced[split_point:].lstrip()
                        # Safety check - ensure we're making progress
                        if len(balanced) >= original_length:
                            # Force split if we're stuck
                            chunks.append(balanced[:max_length-6])
                            balanced = balanced[max_length-6:].lstrip()
                        # Emergency break if balanced is not getting smaller
                        if len(balanced) < 6:  # Minimum viable remainder
                            if balanced:
                                chunks.append(balanced)
                            break
                        if '```' in chunk and chunk.endswith('```') and balanced:
                            balanced = '```\n' + balanced
            else:
                if current_length + len(content) + 1 > max_length:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [content]
                    current_length = len(content)
                else:
                    current_chunk.append(content)
                    current_length += len(content) + 1
        else:
            lines = content.split('\n')
            for line in lines:
                while len(line) > max_length:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                    current_length = 0
                if current_length + len(line) + 1 > max_length:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line) + 1
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    # Send chunks with rate limit handling
    for chunk in chunks:
        if not chunk.strip():  # Skip empty chunks
            continue
        max_retries = 3
        retry_count = 0
        base_delay = 0.5
        while retry_count < max_retries:
            try:
                await channel.send(chunk.strip())
                await asyncio.sleep(0.1)  
                break
            except discord.HTTPException as e:
                if e.status == 429:  # Rate limit hit
                    retry_count += 1
                    if retry_count == max_retries:
                        if bot and bot.logger:
                            bot.logger.error("Max retries reached for message chunk. Skipping.")
                        break
                        
                    retry_after = getattr(e, 'retry_after', base_delay * (2 ** retry_count))
                    if bot and bot.logger:
                        bot.logger.warning(f"Rate limited. Waiting {retry_after:.2f}s before retry {retry_count}/{max_retries}")
                    await asyncio.sleep(retry_after)
                else:
                    if bot and bot.logger:
                        bot.logger.error(f"Error sending message chunk: {str(e)}")
                    break

async def generate_and_save_thought(memory_index, user_id, user_name, memory_text, prompt_formats, system_prompts, bot, file_context=None, image_paths=None, cleanup_callback=None, conversation_context=None):
    """
    Generates a thought about a memory and saves both to the memory index.
    """
    current_time = datetime.now()
    storage_timestamp = current_time.strftime("%H:%M [%d/%m/%y]")
    temporal_expr = bot.temporal_parser.get_temporal_expression(current_time)
    temporal_timestamp = temporal_expr.base_expression
    if temporal_expr.time_context:
        temporal_timestamp = f"{temporal_timestamp} in the {temporal_expr.time_context}"
    timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
    temporal_memory_text = re.sub(
        timestamp_pattern,
        lambda m: f"({bot.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
        memory_text
    )
    thought_prompt = prompt_formats['generate_thought'].format(
        user_name=user_name,
        memory_text=temporal_memory_text,
        timestamp=temporal_timestamp,
        conversation_context=conversation_context if conversation_context else ""
    )
    if file_context:
        thought_prompt += f"\n\nAdditional File Context:\n{file_context}"
    context = ""
    themes=format_themes_for_prompt_memoized(bot.memory_index,user_id,mode="sections")
    thought_system_prompt = system_prompts['thought_generation'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
    thought_response = await bot.call_api(
        thought_prompt,
        context=context,
        system_prompt=thought_system_prompt,
        image_paths=image_paths,
        temperature=bot.amygdala_response/100
    )
    thought_response, thinking_traces = separate_thinking_traces(thought_response)
    await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
    thought_response = clean_response(thought_response)
    memory_string = f"Reflections on interactions with @{user_name} ({storage_timestamp}):\n {thought_response}"
    bot.logger.debug(f"Pre-memory addition string: {memory_string}")
    await memory_index.add_memory_async(user_id, memory_string)
    bot.logger.debug(f"Post-memory addition: {memory_index.user_memories[user_id][-1]}")
    log_to_jsonl({
        'event': 'thought_generation',
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'user_name': user_name,
        'memory_text': memory_text,
        'thought_response': thought_response
    }, bot_id=bot.user.name)

    if cleanup_callback:
        cleanup_callback()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = os.path.basename(sanitized)
    return sanitized

class FakeMessage:
    """Creates a fake Discord message for bot self-invocation."""
    def __init__(self, original, bot, content):
        # copy only what's needed
        self._state = original._state
        self.id = original.id
        self.channel = original.channel
        self.guild = getattr(original, 'guild', None)
        # override for self-invoke
        self.author = bot.user
        self.content = content
        self.mentions = []
        self.channel_mentions = []
        self.role_mentions = []
        self.attachments = []
        self.reference = None
        self.interaction_metadata = None

async def invoke_embedded_commands(response_content: str, message, bot) -> None:
    """Scan response for whitelisted commands and invoke them."""
    whitelist = config.discord.bot_action_commands
    for line in response_content.split('\n'):
        line = line.strip()
        if not line.startswith('!'):
            continue
        parts = line[1:].split(None, 1)
        if not parts:
            continue
        cmd_name = parts[0]
        cmd_args = parts[1] if len(parts) > 1 else ''
        if cmd_name not in whitelist:
            continue
        cmd = bot.get_command(cmd_name)
        if not cmd:
            bot.logger.warning(f"self-invoke: command '{cmd_name}' not found")
            continue
        bot.logger.info(f"self-invoke: !{cmd_name} args='{cmd_args}'")
        try:
            fake_msg = FakeMessage(message, bot, f"!{cmd_name} {cmd_args}")
            ctx = await bot.get_context(fake_msg)
            ctx.command = cmd
            ctx.invoked_with = cmd_name
            ctx.prefix = '!'
            ctx.view = StringView(cmd_args)
            await cmd.invoke(ctx)
        except Exception as e:
            bot.logger.error(f"self-invoke failed: {e}")

class CustomHelpCommand(commands.HelpCommand):
    async def send_bot_help(self, mapping):
        bot = self.context.bot
        is_self = self.context.author.id == bot.user.id
        
        is_manager = False
        if not is_self:
            try:
                if isinstance(self.context.channel, discord.DMChannel):
                    for guild in bot.guilds:
                        member = guild.get_member(self.context.author.id)
                        if member and (member.guild_permissions.administrator or member.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in member.roles)):
                            is_manager = True
                            break
                elif self.context.guild:
                    member = self.context.guild.get_member(self.context.author.id)
                    if member:
                        is_manager = (member.guild_permissions.administrator or member.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in member.roles))
            except (AttributeError, TypeError):
                pass
        
        lines = [f"**{bot.user.name}**"]
        
        if is_self or is_manager:
            status = f"api={bot.api.api_type} model={bot.api.model_name} persona={bot.amygdala_response}%"
            toggles = []
            if getattr(bot, 'github_enabled', False): toggles.append("github")
            if bot.dmn_processor.enabled: toggles.append("dmn")
            if getattr(bot, 'spike_processor', None) and bot.spike_processor.enabled: toggles.append("spike")
            if getattr(bot, 'processing_enabled', True): toggles.append("processing")
            if getattr(bot, 'mentions_enabled', False): toggles.append("mentions")
            if getattr(bot, 'attention_enabled', True): toggles.append("attention")
            status += f" [{'+'.join(toggles) if toggles else 'none'}]"
            lines.append(status)
        
        lines.append("")
        
        for cog, cog_commands in mapping.items():
            try:
                filtered = await self.filter_commands(cog_commands, sort=True)
            except Exception:
                filtered = list(cog_commands) if cog_commands else []
            for cmd in filtered:
                if is_self and cmd.name not in config.discord.bot_action_commands:
                    continue
                if not cmd.hidden or is_manager or is_self:
                    sig = f"!{cmd.name}"
                    for param in cmd.clean_params.values():
                        if param.default == param.empty:
                            sig += f" <{param.name}>"
                        else:
                            sig += f" [{param.name}]"
                    desc = cmd.help.split('\n')[0] if cmd.help else ""
                    if len(desc) > 128:
                        desc = desc[:125] + "..."
                    lines.append(f"`{sig}` {desc}")
        
        await self.get_destination().send('\n'.join(lines))
    
    async def send_command_help(self, command):
        sig = f"!{command.name}"
        for param in command.clean_params.values():
            if param.default == param.empty:
                sig += f" <{param.name}>"
            else:
                sig += f" [{param.name}]"
        lines = [f"**{command.name}**", f"`{sig}`", command.help or "No description."]
        if command.aliases:
            lines.append(f"aliases: {', '.join(command.aliases)}")
        await self.get_destination().send('\n'.join(lines))
'''

class CustomHelpCommand(commands.HelpCommand):
    async def send_bot_help(self, mapping):
        embed = discord.Embed(title=f"🤖 {self.context.bot.user.name} Commands", description="Here are all available commands:", color=discord.Color.blue())
        # Check permissions
        is_manager = False
        if isinstance(self.context.channel, discord.DMChannel):
            for guild in self.context.bot.guilds:
                member = guild.get_member(self.context.author.id)
                if member and (member.guild_permissions.administrator or member.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in member.roles)):
                    is_manager = True
                    break
        else:
            is_manager = (self.context.author.guild_permissions.administrator or self.context.author.guild_permissions.manage_guild or any(role.name == DISCORD_BOT_MANAGER_ROLE for role in self.context.author.roles))
        if is_manager:
            api_settings = [
                f"**API Type**: {self.context.bot.api.api_type}",
                f"**Model**: {self.context.bot.api.model_name}",
                f"**Amygdala Response**: {self.context.bot.amygdala_response}%"]
            embed.add_field(name="🔧 Current Settings", value="\n".join(api_settings), inline=False)
            embed.add_field(name="📚 GitHub Integration " + ("✅" if getattr(self.context.bot, 'github_enabled', False) else "❌"), value="", inline=False)
            embed.add_field(name="🧠 DMN Processor " + ("✅" if self.context.bot.dmn_processor.enabled else "❌"), value="", inline=False)
            embed.add_field(name="⚡ Processing " + ("✅" if getattr(self.context.bot, 'processing_enabled', True) else "❌"), value="", inline=False)
            embed.add_field(name="🔗 Mentions " + ("✅" if getattr(self.context.bot, 'mentions_enabled', True) else "❌"), value="", inline=False)
            embed.add_field(name="👁️ Attention " + ("✅" if getattr(self.context.bot, 'attention_enabled', True) else "❌"), value="", inline=False)
        for cog, cog_commands in mapping.items():
            filtered = await self.filter_commands(cog_commands, sort=True)
            if filtered:
                category = "General" if cog is None else cog.qualified_name
                command_list = []
                for cmd in filtered:
                    if not cmd.hidden or is_manager:
                        brief = cmd.help.split('\n')[0] if cmd.help else "No description"
                        if len(brief) > 60:
                            brief = brief[:57] + "..."
                        command_list.append(f"`!{cmd.name}` - {brief}")
                if command_list:
                    embed.add_field(name=f"📑 {category}", value="\n".join(command_list), inline=False)
        await self.get_destination().send(embed=embed)
    async def send_command_help(self, command):
        """Handles help for a specific command."""
        embed = discord.Embed(title=f"Command: {command.name}", description=command.help or "No description available.", color=discord.Color.green())
        signature = self.get_command_signature(command)
        embed.add_field(name="Usage", value=f"```{signature}```", inline=False)
        if command.aliases:
            embed.add_field(name="Aliases", value=", ".join(f"`{alias}`" for alias in command.aliases),  inline=False)
        if command.checks:
            checks = []
            for check in command.checks:
                check_name = check.__qualname__.split('.')[0]
                if 'has_guild_permissions' in check_name:
                    checks.append("Requires server management permissions")
                elif 'is_owner' in check_name:
                    checks.append("Bot owner only")
                else:
                    checks.append(check_name)           
            if checks:
                embed.add_field(name="Requirements", value="\n".join(f"• {check}" for check in checks), inline=False)
        await self.get_destination().send(embed=embed)
'''
def load_private_api_client(bot_id: str, args):
    """
    Return an independent copy of api_client (api object + helpers)
    without touching the original module.
    """
    module_name = f"api_client_{bot_id}"
    if module_name in sys.modules:               # already loaded?
        return sys.modules[module_name]          # reuse

    spec = importlib.util.find_spec("api_client")
    if spec is None:
        raise ImportError("Could not locate api_client.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod

    mod.initialize_api_client(args)

    return mod
    
async def initialize_themes_cache(memory_index, logger):
    """Initialize themes cache in background to avoid blocking startup."""
    try:
        themes = await asyncio.to_thread(get_current_themes, memory_index)
        logger.info(f"Startup themes cache loaded with {len(themes)} existing themes")
    except Exception as e:
        logger.error(f"Failed to initialize themes cache: {e}")

async def dynamic_prefix(bot, message):
    if isinstance(message.channel, discord.DMChannel):
        return ['!']
    if not message.guild or not message.guild.me:
        return ['!']
    tokens = [f'<@{bot.user.id}>', f'<@!{bot.user.id}>'] + [f'<@&{r.id}>' for r in message.guild.me.roles]
    prefixes = []
    for t in tokens:
        prefixes.append(f'{t}!')
        prefixes.append(f'{t} !')
    return prefixes

def setup_bot(prompt_path=None, bot_id=None):
    """Initialize the Discord bot with specified configuration."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    help_command = CustomHelpCommand()
    
    bot = commands.Bot(
        command_prefix=dynamic_prefix,
        intents=intents,
        status=discord.Status.online,
        help_command=help_command
    )
    # Initialize central logger for this bot instance
    bot.logger = BotLogger(bot_id if bot_id else "default")
    # Create base cache directory for this bot
    bot_cache_dir = bot_id if bot_id else "default"
    # Initialize with specific cache types under the bot's directory
    memory_index = UserMemoryIndex(f'{bot_cache_dir}/memory_index', logger=bot.logger)
    repo_index = RepoIndex(bot_id)
    bot.cache = CacheManager(bot_id or "default")

    
    # Initialize GitHub repository with validation
    try:
        if bot_id:
            github_token = os.getenv(f'GITHUB_TOKEN_{bot_id.upper()}')
            github_repo_name = os.getenv(f'GITHUB_REPO_{bot_id.upper()}')
            bot.logger.info(f"Attempting GitHub init with token env: GITHUB_TOKEN_{bot_id.upper()}")
        else:
            github_token = os.getenv('GITHUB_TOKEN')
            github_repo_name = os.getenv('GITHUB_REPO')
            bot.logger.info("Attempting GitHub init with default token env")

        if not github_token or not github_repo_name:
            bot.github_enabled = False
            github_repo = None
            bot.logger.warning(f"GitHub credentials not found for bot {bot_id or 'default'}. Required env vars: " + 
                          f"GITHUB_TOKEN_{bot_id.upper() if bot_id else ''}, " +
                          f"GITHUB_REPO_{bot_id.upper() if bot_id else ''}")
        else:
            github_repo = GitHubRepo(github_token, github_repo_name)
            github_repo.repo.get_contents('/')
            bot.github_enabled = True
            bot.github_repo = github_repo
            bot.logger.info(f"GitHub integration enabled for repository: {github_repo_name}")
    except Exception as e:
        bot.github_enabled = False
        github_repo = None
        bot.logger.warning(f"GitHub initialization failed for bot {bot_id or 'default'}: {str(e)}. GitHub features will be disabled.")

    try:
        with open(os.path.join(prompt_path, 'prompt_formats.yaml'), 'r', encoding='utf-8') as file:
            prompt_formats = yaml.safe_load(file)
        
        with open(os.path.join(prompt_path, 'system_prompts.yaml'), 'r', encoding='utf-8') as file:
            system_prompts = yaml.safe_load(file)
    except Exception as e:
        bot.logger.error(f"Error loading prompt files from {prompt_path}: {str(e)}")
        raise

    bot.amygdala_response = DEFAULT_AMYGDALA_RESPONSE

    bot.memory_index = memory_index
    bot.prompt_formats = prompt_formats
    bot.system_prompts = system_prompts
    bot.temporal_parser = TemporalParser()
    bot.processing_enabled = True
    bot.mentions_enabled = False
    bot.attention_enabled = True
    bot._slash_commands_synced = False

    # AgentRuntime helpers — satisfy runtime.AgentRuntime protocol
    async def _resolve_user(user_id: str) -> str:
        try:
            user = await bot.fetch_user(int(user_id))
            return user.name if user else f"User({user_id})"
        except Exception:
            return f"User({user_id})"
    bot.resolve_user = _resolve_user

    # Build the DiscordAdapter now; agent_id/agent_name set after on_ready
    from adapters.discord_adapter import DiscordAdapter
    bot._adapter = DiscordAdapter(bot)

    @bot.event
    async def on_ready():
        bot.logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

        # Wire AgentRuntime identity properties now that user is known
        bot.agent_id = str(bot.user.id)
        bot.agent_name = bot.user.name

        bot.dmn_processor.logger = BotLogger(bot.user.name)
        bot.loop.create_task(bot.dmn_processor.start())
        bot.logger.info('DMN processor started')

        if config.discord.sync_slash_commands and not bot._slash_commands_synced:
            try:
                if config.discord.slash_guild_id:
                    guild = discord.Object(id=int(config.discord.slash_guild_id))
                    bot.tree.copy_global_to(guild=guild)
                    synced = await bot.tree.sync(guild=guild)
                    bot.logger.info(f"Synced {len(synced)} slash commands to guild {config.discord.slash_guild_id}")
                elif config.discord.global_slash_commands:
                    synced = await bot.tree.sync()
                    bot.logger.info(f"Synced {len(synced)} global slash commands")
                else:
                    total_synced = 0
                    for guild in bot.guilds:
                        bot.tree.copy_global_to(guild=guild)
                        synced = await bot.tree.sync(guild=guild)
                        total_synced += len(synced)
                        bot.logger.info(f"Synced {len(synced)} slash commands to guild {guild.id} ({guild.name})")
                    bot.logger.info(f"Synced {total_synced} slash command registrations across {len(bot.guilds)} guilds")
                bot._slash_commands_synced = True
            except Exception as e:
                bot.logger.error(f"Slash command sync failed: {str(e)}")
                bot.logger.error(traceback.format_exc())

        log_to_jsonl({
            'event': 'bot_ready',
            'timestamp': datetime.now().isoformat(),
            'bot_name': bot.user.name,
            'bot_id': bot.user.id
        }, bot_id=bot.user.name)

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        ctx = await bot.get_context(message)
        if ctx.command is not None:
            await bot.invoke(ctx)
            return

        uid = str(message.author.id)
        attn = False
        if bot.attention_enabled:
            attn = await asyncio.to_thread(
                check_attention_triggers_fuzzy,
                message.content,
                system_prompts.get('attention_triggers', []),
                memory_index=memory_index,
                user_id=uid
            )

        if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions or any(r in message.guild.me.roles for r in message.role_mentions) or attn:

            from agent_core import process_message as _core_process_message
            norm_msg = await bot._adapter.normalize(message, is_command=False)

            try:
                await _core_process_message(
                    msg=norm_msg,
                    adapter=bot._adapter,
                    runtime=bot,
                    memory_index=memory_index,
                    prompt_formats=prompt_formats,
                    system_prompts=system_prompts,
                    github_repo=github_repo,
                )
            except Exception as e:
                await message.channel.send(f"Error processing message: {str(e)}")
                bot.logger.error(f"Error during on_message dispatch: {str(e)}")
                bot.logger.error(traceback.format_exc())

    async def handle_persona(intensity: Optional[int], actor=None) -> str:
        if intensity is None:
            return f"Current amygdala arousal is {bot.amygdala_response}%."
        if 0 <= intensity <= 100:
            bot.amygdala_response = intensity
            update_temperature(intensity)
            success_msg = f"Amygdala arousal set to {intensity}%"
            if hasattr(bot, 'dmn_processor') and bot.dmn_processor:
                success_msg += ". DMN processor synchronized."
            return success_msg
        if actor:
            bot.logger.warning(f"Invalid amygdala arousal attempted by {actor}: {intensity}")
        else:
            bot.logger.warning(f"Invalid amygdala arousal attempted: {intensity}")
        return "Please provide a valid intensity between 0 and 100."

    async def handle_attention(state: Optional[str]) -> str:
        if state is None or state.lower() == 'status':
            status = "enabled" if bot.attention_enabled else "disabled"
            bot.logger.info(f"Attention status queried: {status}")
            return f"Attention triggers are currently **{status}**"
        if state.lower() in ['on', 'enable', 'true', '1']:
            bot.attention_enabled = True
            return "Attention triggers **enabled** - I'll respond to topic-based triggers"
        if state.lower() in ['off', 'disable', 'false', '0']:
            bot.attention_enabled = False
            return "Attention triggers **disabled** - I'll only respond to mentions and DMs"
        bot.logger.warning(f"Invalid attention command attempted: {state}")
        return "Usage: `!attention on` or `!attention off`"

    async def handle_spike(action: Optional[str]) -> str:
        sp = getattr(bot, 'spike_processor', None)
        if not sp:
            return "Spike processor not initialized."
        if action is None or action.lower() == 'status':
            status = "enabled" if sp.enabled else "disabled"
            surfaces = sp.get_recent_surfaces()
            cooldown_remaining = max(0, sp.config.cooldown_seconds - (datetime.now() - sp.last_spike).total_seconds())
            lines = [
                f"**spike status:** {status}",
                f"**surfaces:** {len(surfaces)} active",
                f"**cooldown:** {cooldown_remaining:.0f}s remaining" if cooldown_remaining > 0 else "**cooldown:** ready",
                f"**threshold:** {sp.config.match_threshold:.2f}"
            ]
            if surfaces:
                lines.append("\n**recent surfaces:**")
                for s in surfaces[:5]:
                    name = f"#{s.channel.name}" if hasattr(s.channel, 'name') else "DM"
                    ago = (datetime.now() - s.last_engaged).total_seconds() / 60
                    lines.append(f"  {name} ({ago:.0f}m ago)")
            return '\n'.join(lines)
        action = action.lower()
        if action in ('on', 'enable', 'start'):
            sp.enabled = True
            return "spike processor **enabled**"
        if action in ('off', 'disable', 'stop'):
            sp.enabled = False
            return "spike processor **disabled**"
        return "usage: `!spike [on|off|status]`"

    async def handle_mentions(state: Optional[str]) -> str:
        if state is None or state.lower() == 'status':
            return f"Mention conversion is currently {'enabled' if bot.mentions_enabled else 'disabled'}."
        state = state.lower()
        if state in ('on', 'true', 'enable'):
            bot.mentions_enabled = True
            return "Mention conversion enabled - usernames will be converted to mentions."
        if state in ('off', 'false', 'disable'):
            bot.mentions_enabled = False
            return "Mention conversion disabled - usernames will remain as plain text."
        return "Invalid state. Use: on/off/status"

    async def handle_reranking(setting: Optional[str]) -> str:
        if setting is None or setting.lower() == 'status':
            status = "on" if config.persona.use_hippocampus_reranking else "off"
            return f"**Memory Reranking:** {status}"
        if setting.lower() in ('on', 'true', 'enable'):
            config.persona.use_hippocampus_reranking = True
            return "Memory reranking enabled"
        if setting.lower() in ('off', 'false', 'disable'):
            config.persona.use_hippocampus_reranking = False
            return "Memory reranking disabled"
        return "Invalid value. Use: on/off"

    @bot.command(name='persona')
    @commands.check(lambda ctx: config.discord.has_command_permission('persona', ctx))
    async def set_amygdala_response(ctx, intensity: int = None):
        """Set emotional intensity 0-100. Lower=calm/focused, higher=creative/volatile."""
        await ctx.send(await handle_persona(intensity, ctx.author))

    @bot.tree.command(name='persona', description='Set or view emotional intensity.')
    @app_commands.describe(intensity='Optional intensity from 0 to 100')
    @app_commands.rename(intensity='intensity')
    async def persona_slash(interaction: discord.Interaction, intensity: app_commands.Range[int, 0, 100] = None):
        if not await require_interaction_permission('persona', interaction):
            return
        await interaction_send(interaction, await handle_persona(intensity, interaction.user), ephemeral=True)

    @bot.command(name='attention')
    @commands.check(lambda ctx: config.discord.has_command_permission('attention', ctx))
    async def toggle_attention(ctx, state: str = None):
        """Toggle topic-based triggers. When on, responds to relevant topics without @mention."""
        await ctx.send(await handle_attention(state))

    @bot.tree.command(name='attention', description='Control topic-based attention triggers.')
    @app_commands.describe(state='Show status or turn attention on/off')
    @app_commands.choices(state=[
        app_commands.Choice(name='status', value='status'),
        app_commands.Choice(name='on', value='on'),
        app_commands.Choice(name='off', value='off'),
    ])
    async def attention_slash(interaction: discord.Interaction, state: app_commands.Choice[str] = None):
        if not await require_interaction_permission('attention', interaction):
            return
        await interaction_send(interaction, await handle_attention(state.value if state else None), ephemeral=True)

    @bot.command(name='spike')
    @commands.check(lambda ctx: config.discord.has_command_permission('spike', ctx))
    async def spike_control(ctx, action: str = None):
        """Control spike processor (orphaned memory outreach)."""
        await ctx.send(await handle_spike(action))

    @bot.tree.command(name='spike', description='Control spike orphaned-memory outreach.')
    @app_commands.describe(action='Show status or turn spike on/off')
    @app_commands.choices(action=[
        app_commands.Choice(name='status', value='status'),
        app_commands.Choice(name='on', value='on'),
        app_commands.Choice(name='off', value='off'),
    ])
    async def spike_slash(interaction: discord.Interaction, action: app_commands.Choice[str] = None):
        if not await require_interaction_permission('spike', interaction):
            return
        await interaction_send(interaction, await handle_spike(action.value if action else None), ephemeral=True)

    @bot.command(name='add_memory')
    @commands.check(lambda ctx: config.discord.has_command_permission('add_memory', ctx))
    async def add_memory(ctx, *, memory_text):
        """Store a custom memory for the invoking user."""
        memory_index.add_memory(str(ctx.author.id), memory_text)
        await ctx.send("Memory added successfully.")
        log_to_jsonl({
            'event': 'add_memory',
            'timestamp': datetime.now().isoformat(),
            'user_id': str(ctx.author.id),
            'user_name': ctx.author.name,
            'memory_text': memory_text
        }, bot_id=bot.user.name)

    @bot.command(name='clear_memories')
    @commands.check(lambda ctx: config.discord.has_command_permission('clear_memories', ctx))
    async def clear_memories(ctx):
        """Clear all memories of the invoking user."""
        user_id = str(ctx.author.id)
        memory_index.clear_user_memories(user_id)
        await ctx.send("Your memories have been cleared.")
        log_to_jsonl({
            'event': 'clear_user_memories',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': ctx.author.name
        }, bot_id=bot.user.name)

    @bot.command(name='summarize')
    @commands.check(lambda ctx: config.discord.has_command_permission('summarize', ctx))
    async def summarize(ctx, *, args=None):
        """Summarize the last [n] messages in a specified channel and send the summary to DM."""
        try:
            n = MAX_CONVERSATION_HISTORY
            channel = None
            if args:
                parts = args.split()
                if len(parts) >= 1:
                    if parts[0].startswith('<#') and parts[0].endswith('>'):
                        channel_id = int(parts[0][2:-1])
                    elif parts[0].isdigit():
                        channel_id = int(parts[0])
                    else:
                        await ctx.send("Please provide a valid channel ID or mention.")
                        return
                    
                    channel = bot.get_channel(channel_id)
                    if channel is None:
                        await ctx.send(f"Invalid channel. Channel ID: {channel_id}")
                        return
                    parts = parts[1:]  

                    if parts:
                        try:
                            n = int(parts[0])
                        except ValueError:
                            await ctx.send("Invalid input. Please provide a number for the amount of messages to summarize.")
                            return
            else:
                await ctx.send("Please specify a channel ID or mention to summarize.")
                return
            member = channel.guild.get_member(ctx.author.id)
            if member is None or not channel.permissions_for(member).read_messages:
                await ctx.send("You don't have permission to read messages in the specified channel.")
                return
            if not channel.permissions_for(channel.guild.me).read_message_history:
                await ctx.send("I don't have permission to read message history in the specified channel.")
                return
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                summarizer = ChannelSummarizer(bot, prompt_formats, system_prompts, max_entries=n)
                summary = await summarizer.summarize_channel(channel.id)
            finally:
                typing_task.cancel()
            try:
                await send_long_message(ctx.author, f"**Channel Summary for #{channel.name} (Last {n} messages)**\n\n{summary}", bot=bot)
                if isinstance(ctx.channel, discord.DMChannel):
                    await ctx.send(f"I've sent you the summary of #{channel.name}.")
                else:
                    await ctx.send(f"{ctx.author.mention}, I've sent you a DM with the summary of #{channel.name}.")
            except discord.Forbidden:
                await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
            memory_text = f"Summarized {n} messages from #{channel.name}. Summary: {summary}"
            await generate_and_save_thought(
                memory_index=memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )
        except discord.Forbidden as e:
            await ctx.send(f"I don't have permission to perform this action. Error: {str(e)}")
        except Exception as e:
            error_message = f"An error occurred while summarizing the channel: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(f"Error in channel summarization: {str(e)}")

    @bot.command(name='index_repo')
    @commands.check(lambda ctx: config.discord.has_command_permission('index_repo', ctx))
    async def index_repo(ctx, option: str = None, branch: str = 'main'):
        """Index the GitHub repository contents, list indexed files, or check indexing status with optional list, status and branch."""
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        global repo_processing_event
        if option == 'list':
            if repo_processing_event.is_set():
                indexed_files = set()
                for file_paths in repo_index.repo_index.values():
                    indexed_files.update(file_paths)
                if indexed_files:
                    file_list = f"# Indexed Repository Files (Branch: {branch})\n\n"
                    for file in sorted(indexed_files):
                        file_list += f"- `{file}`\n"
                    temp_path, _ = bot.cache_managers['file'].create_temp_file(
                        user_id=str(ctx.author.id),
                        prefix="repo_index_",
                        suffix=".md", 
                        content=file_list
                    )
                    await ctx.send(f"Here's the list of indexed files from the '{branch}' branch:", file=discord.File(temp_path))
                else:
                    await ctx.send(f"No files have been indexed yet on the '{branch}' branch.")
            else:
                await ctx.send(f"Repository indexing has not been completed for the '{branch}' branch. Please run `!index_repo` first.")
        elif option == 'status':
            if repo_processing_event.is_set():
                await ctx.send("Repository indexing is complete.")
            else:
                await ctx.send("Repository indexing is still in progress.")
        else:
            try:
                repo_processing_event.clear()
                await ctx.send(f"Starting to index the repository on the '{branch}' branch... This may take a while.")
                repo_index.clear_cache()
                start_background_processing_thread(github_repo.repo, repo_index, max_depth=None, branch=branch)
                await ctx.send(f"Repository indexing has started in the background for the '{branch}' branch.")
            except Exception as e:
                error_message = f"An error occurred while starting the repository indexing on the '{branch}' branch: {str(e)}"
                await ctx.send(error_message)
                bot.logger.error(error_message)
                
    @bot.command(name='repo_file_chat')
    @commands.check(lambda ctx: config.discord.has_command_permission('repo_file_chat', ctx))
    async def repo_file_chat(ctx, *, input_text: str = None):
        """Discuss a repo file."""
        if not input_text:
            await ctx.send("Usage: !repo_file_chat <file_path> <task_description>")
            return
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        parts = input_text.split(maxsplit=1)
        if len(parts) < 2:
            await ctx.send("Error: Please provide both a file path and a task description.")
            return
        file_path = parts[0]
        user_task_description = parts[1]
        if not repo_processing_event.is_set():
            await ctx.send("Repository indexing is not complete. Please run !index_repo first.")
            return
        try:
            file_path = file_path.strip().replace('\\', '/')
            if file_path.startswith('/'):
                file_path = file_path[1:]  # Remove leading slash if present
            indexed_files = set()
            for file_set in repo_index.repo_index.values():
                indexed_files.update(file_set)
            if file_path not in indexed_files:
                await ctx.send(f"Error: The file '{file_path}' is not in the indexed repository.")
                return
            response_content = None
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                repo_code = github_repo.get_file_content(file_path)
                if repo_code.startswith("Error fetching file:"):
                    await ctx.send(f"Error: {repo_code}")
                    return
                elif repo_code == "File is too large to fetch content directly.":
                    await ctx.send(repo_code)
                    return
                _, file_extension = os.path.splitext(file_path)
                code_type = mimetypes.types_map.get            
                if 'repo_file_chat' not in prompt_formats or 'repo_file_chat' not in system_prompts:
                    await ctx.send("Error: Required prompt templates are missing.")
                    return
                # Build context
                context = f"Current discord channel: #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
                context += "Ongoing Chatroom Conversation:\n\n"
                context += "<conversation>\n"
                messages = []
                async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                    if msg.id != ctx.message.id:  # Skip the command message
                        #combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                        combined_mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)

                        
                        msg_content = sanitize_mentions(msg.content, combined_mentions)
                        truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                        clean_name = msg.author.name
                        formatted_msg = f" @{clean_name}: {truncated_content}"
                        # Add reactions if present
                        if msg.reactions:
                            reaction_parts = []
                            for reaction in msg.reactions:
                                reaction_emoji = str(reaction.emoji)
                                async for user in reaction.users():
                                    reaction_user_name = user.name
                                    reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                            if reaction_parts:
                                formatted_msg += f" (Reactions: {' '.join(reaction_parts)})"
                        messages.append(formatted_msg)

                for msg in reversed(messages):
                    context += f"{msg}\n"
                context += "</conversation>\n"

                prompt = prompt_formats['repo_file_chat'].format(
                    file_path=file_path,
                    code_type=code_type,
                    repo_code=repo_code,
                    user_task_description=user_task_description,
                    context=context
                )

                themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
                system_prompt = system_prompts['repo_file_chat'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
                response_content = await bot.call_api(prompt, system_prompt=system_prompt)
                response_content, thinking_traces = separate_thinking_traces(response_content)
                await store_thinking_traces(memory_index, str(ctx.author.id), ctx.author.name, thinking_traces)
                response_content = clean_response(response_content)
                response_content = balance_wraps(response_content)
            finally:
                typing_task.cancel()
            if response_content:
                formatted_response = f"# Analysis for {file_path}\n\n"
                formatted_response += f"**Task**: {user_task_description}\n\n"
                formatted_response += response_content
                await send_long_message(ctx.channel, formatted_response, bot=bot)
                memory_text = f"Recollection of'{file_path}' discussing '{user_task_description}'.\n {response_content}"
                asyncio.create_task(generate_and_save_thought(
                    memory_index=memory_index,
                    user_id=str(ctx.author.id),
                    user_name=ctx.author.name,
                    memory_text=memory_text,
                    prompt_formats=prompt_formats,
                    system_prompts=system_prompts,
                    bot=bot
                ))
        except Exception as e:
            error_message = f"Error generating file summary and querying AI: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(error_message)

    @bot.command(name='ask_repo')
    @commands.check(lambda ctx: isinstance(ctx.channel, discord.DMChannel) or ctx.author.guild_permissions.manage_messages)
    async def ask_repo(ctx, *, question: str = None):
        """Search GitHub repo and discuss results."""
        if not question:
            await ctx.send("Usage: !ask_repo <question>")
            return
        if not bot.github_enabled:
            await ctx.send("GitHub integration is currently disabled. Please check bot logs for details.")
            return
        response = None  
        try:
            if not repo_processing_event.is_set():
                await ctx.send("Repository indexing is not complete. Please wait or run !index_repo first.")
                return
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                relevant_files = await repo_index.search_repo_async(question)
            finally:
                typing_task.cancel()
            if not relevant_files:
                await ctx.send("No relevant files found in the repository for this question.")
                return
            context = "Relevant files in the repository:\n"
            file_links = []  
            for file_path, score in relevant_files:
                context += f"- {file_path} (Relevance: {score:.2f})\n"
                file_content = github_repo.get_file_content(file_path)
                context += f"Content preview: {truncate_middle(file_content, 1000)}\n\n"
                file_links.append(f"{file_path}")
            context += f"\nCurrent channel: #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'}\n\n"
            context += "**Ongoing Chatroom Conversation:**\n\n"
            context += "<conversation>\n"
            messages = []
            async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                if msg.id != ctx.message.id:  # Skip the question message
                    combined_mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)
                    msg_content = sanitize_mentions(msg.content, combined_mentions)
                    truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                    clean_name = msg.author.name
                    formatted_msg = f" @{clean_name}: {truncated_content}"
                    if msg.reactions:
                        reaction_parts = []
                        for reaction in msg.reactions:
                            reaction_emoji = str(reaction.emoji)
                            async for user in reaction.users():
                                reaction_user_name = user.name
                                reaction_parts.append(f"@{reaction_user_name}: {reaction_emoji}")
                        
                        if reaction_parts:
                            formatted_msg += f"\n(Message Reactions: {' '.join(reaction_parts)})"
                    
                    messages.append(formatted_msg)
            for msg in reversed(messages):
                context += f"{msg}\n"
            context += "</conversation>\n"
            prompt = prompt_formats['ask_repo'].format(
                context=context,
                question=question
            )
            #themes = ", ".join(get_current_themes(bot.memory_index))
            themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
            system_prompt = system_prompts['ask_repo'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                response = await bot.call_api(prompt, context=context, system_prompt=system_prompt)
                response, thinking_traces = separate_thinking_traces(response)
                await store_thinking_traces(memory_index, str(ctx.author.id), ctx.author.name, thinking_traces)
                response = clean_response(response)
            finally:
                typing_task.cancel()
            if response:
                response += "\n\nReferenced Files:\n```md\n" + "\n".join(file_links) + "\n```"
                await send_long_message(ctx, response, bot=bot)
        except Exception as e:
            error_message = f"An error occurred while processing the repo chat: {str(e)}"
            await ctx.send(error_message)
            bot.logger.error(f"Error in repo chat: {str(e)}")
            return
        if response:
            timestamp = currentmoment()
            memory_text = f"({timestamp}) Asked repo question '{question}'. Response: {response}"
            await generate_and_save_thought(
                memory_index=memory_index,
                user_id=str(ctx.author.id),
                user_name=ctx.author.name,
                memory_text=memory_text,
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
                bot=bot
            )

    @bot.command(name='search_memories')
    @commands.check(lambda ctx: config.discord.has_command_permission('search_memories', ctx))
    async def search_memories(ctx, *, query):
        """Search stored memories."""
        is_dm = isinstance(ctx.channel, discord.DMChannel)
        user_id = str(ctx.author.id) if is_dm else None
        typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
        try:
            results = await memory_index.search_async(query, user_id=user_id)
        finally:
            typing_task.cancel()
        if not results:
            await ctx.send("No results found.")
            return
        current_chunk = f"Search results for '{query}':\n"
        for memory, score in results:
            truncated_memory = truncate_middle(memory, 800)
            result_line = f"[Relevance: {score:.2f}] {truncated_memory}\n"
            if len(result_line) > 1800:
                result_line = result_line[:1896] + "...\n"
            if len(current_chunk) + len(result_line) > 1800:
                await ctx.send(current_chunk)
                current_chunk = result_line
            else:
                current_chunk += result_line
        if current_chunk:
            await ctx.send(current_chunk)

    @bot.command(name='dmn')
    @commands.check(lambda ctx: config.discord.has_command_permission('dmn', ctx))
    async def dmn_control(ctx, action: str = None):
        """Control background thoughts generation and consolidation."""
        if not action:
            await ctx.send("Please specify an action: start, stop, or status")
            return
        action = action.lower()
        if action == "start":
            if not bot.dmn_processor.enabled:
                await bot.dmn_processor.start()
                await ctx.send("DMN processor started. Bot will now generate periodic reflective thoughts.")
            else:
                await ctx.send("DMN processor is already running.")
        elif action == "stop":
            if bot.dmn_processor.enabled:
                await bot.dmn_processor.stop()
                await ctx.send("DMN processor stopped. Bot will no longer generate background thoughts.")
            else:
                await ctx.send("DMN processor is not currently running.")
        elif action == "status":
            status = "running" if bot.dmn_processor.enabled else "stopped"
            await ctx.send(f"DMN processor is currently {status}.")
        else:
            await ctx.send("Invalid action. Please use: start, stop, or status")

    @bot.command(name='kill')
    @commands.check(lambda ctx: config.discord.has_command_permission('kill', ctx))
    async def kill_tasks(ctx):
        """Gracefully terminate API processing while maintaining Discord connection"""
        try:
            bot.processing_enabled = False
            if bot.dmn_processor.enabled:
                await bot.dmn_processor.stop()
            if hasattr(bot, 'spike_processor') and bot.spike_processor:
                bot.spike_processor.enabled = False
            await ctx.send("Processing disabled. Ongoing API calls will complete but no new calls will be initiated.")
            bot.logger.info(f"Kill command initiated by {ctx.author.name} (ID: {ctx.author.id})")
        except Exception as e:
            await ctx.send(f"Error in kill command: {str(e)}")
            bot.logger.error(f"Kill command error: {str(e)}")

    @bot.command(name='resume')
    @commands.check(lambda ctx: config.discord.has_command_permission('resume', ctx))
    async def resume_tasks(ctx):
        """Resume API processing"""
        bot.processing_enabled = True
        if hasattr(bot, 'spike_processor') and bot.spike_processor:
            bot.spike_processor.enabled = True
        await ctx.send("Processing resumed.")
        bot.logger.info(f"Processing resumed by {ctx.author.name} (ID: {ctx.author.id})")

    @bot.command(name='mentions')
    @commands.check(lambda ctx: config.discord.has_command_permission('mentions', ctx))
    async def toggle_mentions(ctx, state: str = None):
        """Toggle or check mention conversion state."""
        await ctx.send(await handle_mentions(state))

    @bot.tree.command(name='mentions', description='Control username-to-mention conversion.')
    @app_commands.describe(state='Show status or turn mention conversion on/off')
    @app_commands.choices(state=[
        app_commands.Choice(name='status', value='status'),
        app_commands.Choice(name='on', value='on'),
        app_commands.Choice(name='off', value='off'),
    ])
    async def mentions_slash(interaction: discord.Interaction, state: app_commands.Choice[str] = None):
        if not await require_interaction_permission('mentions', interaction):
            return
        await interaction_send(interaction, await handle_mentions(state.value if state else None), ephemeral=True)

    @bot.command(name='get_logs')
    @commands.check(lambda ctx: config.discord.has_command_permission('get_logs', ctx))
    async def get_logs(ctx):
        """Request bot logs via DM (most recent entries up to size limit)."""
        try:
            log_dir = os.path.join(config.logging.base_log_dir, bot.user.name, 'logs')
            log_path = os.path.join(
                log_dir,
                config.logging.jsonl_pattern.format(bot_id=bot.user.name)
            )
            temp_path = os.path.join(
                log_dir,
                f'temp_{config.logging.jsonl_pattern.format(bot_id=bot.user.name)}'
            )
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                MAX_SIZE = 1 * 1024 * 1024  # 1MB size limit
                with open(log_path, 'r', encoding='utf-8') as source:
                    lines = source.readlines()
                    lines.reverse()
                    size = 0
                    recent_lines = []
                    for line in lines:
                        line_size = len(line.encode('utf-8'))
                        if size + line_size > MAX_SIZE:
                            break
                        recent_lines.append(line)
                        size += line_size
                if recent_lines:
                    with open(temp_path, 'w', encoding='utf-8') as temp:
                        temp.writelines(recent_lines)
                    try:
                        await ctx.author.send(
                            f"Most recent logs ({len(recent_lines)} entries)",
                            file=discord.File(temp_path, filename=f"{bot.user.name}_recent_logs.jsonl")
                        )
                        if not isinstance(ctx.channel, discord.DMChannel):
                            await ctx.send(f"{ctx.author.mention}, I've sent you the logs via DM.")
                    except discord.Forbidden:
                        await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
                    finally:
                        # Cleanup temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    await ctx.send("No logs available within size limit.")
            else:
                await ctx.send("No logs available.")
        except Exception as e:
            bot.logger.error(f"Error retrieving logs: {str(e)}")
            await ctx.send(f"An error occurred while retrieving the logs: {str(e)}")

    @bot.command(name='reranking')
    @commands.check(lambda ctx: config.discord.has_command_permission('reranking', ctx))
    async def toggle_reranking(ctx, setting: str = None):
        """
        Control hippocampus memory reranking.
        """
        await ctx.send(await handle_reranking(setting))

    @bot.tree.command(name='reranking', description='Control hippocampus memory reranking.')
    @app_commands.describe(setting='Show status or turn reranking on/off')
    @app_commands.choices(setting=[
        app_commands.Choice(name='status', value='status'),
        app_commands.Choice(name='on', value='on'),
        app_commands.Choice(name='off', value='off'),
    ])
    async def reranking_slash(interaction: discord.Interaction, setting: app_commands.Choice[str] = None):
        if not await require_interaction_permission('reranking', interaction):
            return
        await interaction_send(interaction, await handle_reranking(setting.value if setting else None), ephemeral=True)
    return bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['ollama', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter', 'unsloth'], 
                        default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, 
                        help='Specify the model to use. If not provided, defaults will be used based on the API.')
    parser.add_argument('--prompt-path', type=str, 
                        default='agent/prompts',
                        help='Path to prompt files directory (default: agent/prompts)')
    parser.add_argument('--bot-name', type=str,
                        help='Name of the bot to run (used for token and cache management)')
    parser.add_argument('--dmn-api', choices=['ollama', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter', 'unsloth'], 
                        help='Choose the API to use for DMN processor (default: use main API)')
    parser.add_argument('--dmn-model', type=str,
                        help='Specify the model to use for DMN processor (default: use main model)')
    parser.add_argument('--use-chronpression', action='store_true',
                        help='Use chronomic compression instead of LLM for DMN thought distillation')

    args = parser.parse_args()
    # Initialize global logger
    apply_overrides(args.bot_name or "default")
    logger = BotLogger(args.bot_name if args.bot_name else "default")
    # Get base prompt path and combine with bot name if provided
    base_prompt_path = os.path.abspath(args.prompt_path)
    prompt_path = os.path.join(base_prompt_path, args.bot_name.lower() if args.bot_name else 'default')
    if not os.path.exists(prompt_path):
        logger.critical(f"Prompt path does not exist: {prompt_path}")
        exit(1)
    #logger.info(f"Using prompt path: {prompt_path}")
    # Select appropriate token based on bot name
    if args.bot_name:
        token_env_var = f'DISCORD_TOKEN_{args.bot_name.upper()}'
        TOKEN = os.getenv(token_env_var)
        if not TOKEN:
            logger.critical(f"No token found for bot '{args.bot_name}' (Environment variable: {token_env_var})")
            exit(1)
        logger.info(f"Running as {args.bot_name}")
    else:
        TOKEN = os.getenv('DISCORD_TOKEN')  # Fall back to default token
        if not TOKEN:
            logger.critical("No Discord token found in environment variables")
            exit(1)
    # Create private API client for this bot
    private_api = load_private_api_client(args.bot_name or "default", args)
    # Override DMN config with command line arguments if provided
    if args.use_chronpression:
        config.dmn.use_chronpression = True
    if args.dmn_api or args.dmn_model:
        config.dmn.dmn_api_type = args.dmn_api or config.dmn.dmn_api_type
        config.dmn.dmn_model = args.dmn_model or config.dmn.dmn_model
        #logger.info(f"DMN API overridden: {config.dmn.dmn_api_type}, Model: {config.dmn.dmn_model}")
    bot = setup_bot(prompt_path=prompt_path, bot_id=args.bot_name)
    # Attach per-bot API handles
    bot.api = private_api.api
    bot.call_api = private_api.call_api
    bot.update_api_temperature = private_api.update_api_temperature
    bot.update_api_top_p = private_api.update_api_top_p
    # Initialize DMN processor after API client is attached
    bot.dmn_processor = DMNProcessor(
        memory_index=bot.memory_index,
        prompt_formats=bot.prompt_formats,
        system_prompts=bot.system_prompts,
        runtime=bot,
        dmn_api_type=config.dmn.dmn_api_type,
        dmn_model=config.dmn.dmn_model
    )
    # Sync initial amygdala arousal
    bot.dmn_processor.set_amygdala_response(bot.amygdala_response)
    # Initialize spike processor (enabled by default, toggle via !spike on/off)
    bot.spike_processor = SpikeProcessor(
        bot,
        bot.memory_index,
        cache_path=os.path.join('cache', args.bot_name or 'default', 'spike')
    )
    # Run the configured bot; discord.py handles reconnect internally
    try:
        bot.run(TOKEN, reconnect=True)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down gracefully...")
    except discord.errors.LoginFailure as e:
        logger.critical(f"Login failed (invalid token): {str(e)}")
    except Exception as e:
        logger.critical(f"Critical error occurred: {str(e)}")
        raise
