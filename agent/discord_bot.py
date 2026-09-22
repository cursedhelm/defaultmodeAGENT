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
from io import BytesIO
from typing import Optional
import traceback
# import tools
from tools.discordSUMMARISER import ChannelSummarizer
from tools.discordGITHUB import GitHubRepo, RepoIndex, process_repo_contents, repo_processing_event
from tools.todos.factory import create_todo_service
from tools.todos.migration import migrate_todont_directory
from tools.todos.models import Principal, TodoRequestContext
from tools.todos.toolset import build_todo_tool_bundle
from tools.todos.discord_commands import register_todo_commands
from tools.bookshelf.factory import create_bookshelf_service
from tools.bookshelf.toolset import build_bookshelf_tool_bundle
from tools.bundle import merge_tool_bundles
# import memory module
from memory import UserMemoryIndex, CacheManager
from defaultmode import DMNProcessor
from reading import ReadingProcessor
from chunker import truncate_middle, clean_response, balance_wraps
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
# Discord Format Handling
from discord_utils import sanitize_mentions
from attention import check_attention_triggers_fuzzy, format_themes_for_prompt, warm_theme_cache, snapshot_theme_cache
from viz_live import VizLiveServer, make_live_payload
# Action generation
from spike import SpikeProcessor
from discord_api_worker import DiscordAPIWorker
# Configuration imports
from bot_config import (
    config,
    init_logging,
    apply_overrides,
)
# libraries logging import for jsonl, sqlite and info logging
from logger import BotLogger
from log_export import read_recent_jsonl
from pydantic import BaseModel, Field

init_logging()


class BotPrompts(BaseModel):
    """Hardcoded prompt scaffolding for the command paths that still live in
    this module (repo commands, channel summarize, thought generation).

    The chat/file paths route through agent_core, whose CorePrompts owns the
    equivalents; reflection_memory here deliberately duplicates
    agent_core.PROMPTS.reflection_memory — keep them in sync.
    """
    # Thought generation (used by summarize / repo commands)
    reflection_memory: str = Field(default="Reflections on interactions with @{user_name} ({timestamp}):\n {thought}")
    file_context_suffix: str = Field(default="\n\nAdditional File Context:\n{file_context}")
    # Memory-string templates
    summarize_memory: str = Field(default="Summarized {count} messages from #{channel_name}. Summary: {summary}")
    repo_chat_memory: str = Field(default="Recollection of'{file_path}' discussing '{task}'.\n {response}")
    ask_repo_memory: str = Field(default="({timestamp}) Asked repo question '{question}'. Response: {response}")
    # Repo command context frames
    repo_chat_context_header: str = Field(default="Current discord channel: #{channel_name}\n\nOngoing Chatroom Conversation:\n\n<conversation>\n")
    ask_repo_files_header: str = Field(default="Relevant files in the repository:\n")
    ask_repo_file_line: str = Field(default="- {file_path} (Relevance: {score:.2f})\n")
    ask_repo_file_preview: str = Field(default="Content preview: {content}\n\n")
    ask_repo_context_header: str = Field(default="\nCurrent channel: #{channel_name}\n\n**Ongoing Chatroom Conversation:**\n\n<conversation>\n")
    conversation_close: str = Field(default="</conversation>\n")
    history_message_line: str = Field(default=" @{name}: {content}")
    reaction_entry: str = Field(default="@{user}: {emoji}")
    repo_chat_reactions_suffix: str = Field(default=" (Reactions: {reactions})")
    ask_repo_reactions_suffix: str = Field(default="\n(Message Reactions: {reactions})")


PROMPTS = BotPrompts()

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
ALLOWED_DOCUMENT_EXTENSIONS = config.files.allowed_document_extensions
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
    rendered_user_content = prompt_formats['generate_thought'].format(
        user_name=user_name,
        memory_text=temporal_memory_text,
        timestamp=temporal_timestamp,
        conversation_context=conversation_context if conversation_context else ""
    )
    if file_context:
        rendered_user_content += PROMPTS.file_context_suffix.format(file_context=file_context)
    themes=format_themes_for_prompt_memoized(bot.memory_index,user_id,mode="sections")
    thought_system_prompt = system_prompts['thought_generation'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
    thought_response = await bot.call_api(
        user_content=rendered_user_content,
        system_prompt=thought_system_prompt,
        image_paths=image_paths,
        temperature=bot.amygdala_response/100
    )
    thought_response, thinking_traces = separate_thinking_traces(thought_response)
    await store_thinking_traces(memory_index, user_id, user_name, thinking_traces)
    thought_response = clean_response(thought_response)
    memory_string = PROMPTS.reflection_memory.format(
        user_name=user_name, timestamp=storage_timestamp, thought=thought_response,
    )
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
    bot.todo_service = None
    bot.bookshelf_service = None
    bot.reading_processor = None

    def _is_todo_manager(author) -> bool:
        permissions = getattr(author, 'guild_permissions', None)
        if permissions and (permissions.administrator or permissions.manage_guild):
            return True
        return any(
            role.name == config.discord.bot_manager_role
            for role in getattr(author, 'roles', [])
        )

    def _principal(user) -> Principal:
        return Principal(
            key=f"discord:{user.id}",
            display_name=getattr(user, 'display_name', None) or user.name,
            is_bot=bool(getattr(user, 'bot', False)),
        )

    async def _build_tools_for_message(msg):
        if msg.raw is None:
            return None
        raw = msg.raw
        actor = _principal(raw.author)
        agent = _principal(bot.user)
        targets = {}
        users = list(getattr(raw, 'mentions', []) or [])
        if msg.reply_to is not None and msg.reply_to.raw is not None:
            users.append(msg.reply_to.raw.author)
        for user in users:
            principal = _principal(user)
            targets[str(user.id)] = principal
            targets[principal.key] = principal
            targets[principal.display_name.casefold()] = principal
            targets[getattr(user, 'name', principal.display_name).casefold()] = principal
        todo_context = TodoRequestContext(
            actor=actor,
            agent=agent,
            guild_id=str(raw.guild.id) if getattr(raw, 'guild', None) else None,
            channel_id=str(raw.channel.id),
            is_manager=_is_todo_manager(raw.author),
            source="agent_tool",
            known_targets=targets,
        )
        todo_bundle = (
            build_todo_tool_bundle(bot.todo_service, todo_context)
            if config.todo.enabled and bot.todo_service is not None else None
        )
        bookshelf_bundle = None
        if config.bookshelf.enabled and bot.bookshelf_service is not None:
            supported = {}
            all_attachments = list(msg.attachments) + (
                list(msg.reply_to.attachments) if msg.reply_to else []
            )
            for attachment in all_attachments:
                if os.path.splitext(attachment.filename)[1].casefold() not in {'.pdf', '.epub'}:
                    continue
                if attachment.size > config.bookshelf.max_file_bytes:
                    continue
                supported[attachment.filename] = await attachment.read()
            bookshelf_bundle = build_bookshelf_tool_bundle(
                bot.bookshelf_service, _reader_id(bot), supported
            )
        return merge_tool_bundles(todo_bundle, bookshelf_bundle)

    def _reader_id(runtime_bot):
        return str(getattr(runtime_bot, 'reader_id', None) or getattr(runtime_bot, 'agent_name', None) or bot_id or 'default')

    bot.build_tools_for_message = _build_tools_for_message
    if config.todo.enabled:
        register_todo_commands(bot, config)

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
        async def _start_spike_and_dmn():
            if bot.spike_processor is not None:
                await bot.spike_processor.initialize()
            await bot.dmn_processor.start()
            bot.logger.info('DMN processor started')
        bot.loop.create_task(_start_spike_and_dmn())
        if bot.reading_processor is not None:
            bot.reading_processor.logger = BotLogger(bot.user.name)
            bot.loop.create_task(bot.reading_processor.start())
            bot.logger.info('READER processor starting')

        # Warm the global theme cache off the request path. No-op when the
        # pickle already exists; on a cold corpus this moves full trigram
        # extraction off the first user's message.
        async def _warm_themes():
            count = await asyncio.to_thread(warm_theme_cache, memory_index)
            bot.logger.info(f"Theme cache warmed: {count} themes")
        bot.loop.create_task(_warm_themes())

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
            latest = await asyncio.to_thread(sp.repository.latest)
            if latest:
                lines.append(
                    f"**last action:** {latest.action or 'pending'} / {latest.status} "
                    f"({'grounded' if latest.grounded else 'ungrounded'})"
                )
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
            memory_text = PROMPTS.summarize_memory.format(count=n, channel_name=channel.name, summary=summary)
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
                # Build supplemental system context
                assembled_context = PROMPTS.repo_chat_context_header.format(
                    channel_name=ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'
                )
                messages = []
                async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                    if msg.id != ctx.message.id:  # Skip the command message
                        #combined_mentions = list(msg.mentions) + list(msg.channel_mentions)
                        combined_mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)

                        
                        msg_content = sanitize_mentions(msg.content, combined_mentions)
                        truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                        clean_name = msg.author.name
                        formatted_msg = PROMPTS.history_message_line.format(name=clean_name, content=truncated_content)
                        # Add reactions if present
                        if msg.reactions:
                            reaction_parts = []
                            for reaction in msg.reactions:
                                reaction_emoji = str(reaction.emoji)
                                async for user in reaction.users():
                                    reaction_user_name = user.name
                                    reaction_parts.append(PROMPTS.reaction_entry.format(user=reaction_user_name, emoji=reaction_emoji))
                            if reaction_parts:
                                formatted_msg += PROMPTS.repo_chat_reactions_suffix.format(reactions=' '.join(reaction_parts))
                        messages.append(formatted_msg)

                for msg in reversed(messages):
                    assembled_context += f"{msg}\n"
                assembled_context += PROMPTS.conversation_close

                rendered_user_content = prompt_formats['repo_file_chat'].format(
                    file_path=file_path,
                    code_type=code_type,
                    repo_code=repo_code,
                    user_task_description=user_task_description,
                    assembled_context=assembled_context,
                ).lstrip()

                themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
                system_prompt = system_prompts['repo_file_chat'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
                response_content = await bot.call_api(
                    user_content=rendered_user_content,
                    system_prompt=system_prompt,
                )
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
                memory_text = PROMPTS.repo_chat_memory.format(
                    file_path=file_path, task=user_task_description, response=response_content,
                )
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
            assembled_context = PROMPTS.ask_repo_files_header
            file_links = []
            for file_path, score in relevant_files:
                assembled_context += PROMPTS.ask_repo_file_line.format(file_path=file_path, score=score)
                file_content = github_repo.get_file_content(file_path)
                assembled_context += PROMPTS.ask_repo_file_preview.format(content=truncate_middle(file_content, 1000))
                file_links.append(f"{file_path}")
            assembled_context += PROMPTS.ask_repo_context_header.format(
                channel_name=ctx.channel.name if hasattr(ctx.channel, 'name') else 'Direct Message'
            )
            messages = []
            async for msg in ctx.channel.history(limit=MAX_CONVERSATION_HISTORY):
                if msg.id != ctx.message.id:  # Skip the question message
                    combined_mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)
                    msg_content = sanitize_mentions(msg.content, combined_mentions)
                    truncated_content = truncate_middle(msg_content, max_tokens=TRUNCATION_LENGTH)
                    clean_name = msg.author.name
                    formatted_msg = PROMPTS.history_message_line.format(name=clean_name, content=truncated_content)
                    if msg.reactions:
                        reaction_parts = []
                        for reaction in msg.reactions:
                            reaction_emoji = str(reaction.emoji)
                            async for user in reaction.users():
                                reaction_user_name = user.name
                                reaction_parts.append(PROMPTS.reaction_entry.format(user=reaction_user_name, emoji=reaction_emoji))
                        
                        if reaction_parts:
                            formatted_msg += PROMPTS.ask_repo_reactions_suffix.format(reactions=' '.join(reaction_parts))
                    
                    messages.append(formatted_msg)
            for msg in reversed(messages):
                assembled_context += f"{msg}\n"
            assembled_context += PROMPTS.conversation_close
            rendered_user_content = prompt_formats['ask_repo'].format(
                assembled_context=assembled_context,
                question=question
            ).lstrip()
            #themes = ", ".join(get_current_themes(bot.memory_index))
            themes=format_themes_for_prompt_memoized(bot.memory_index,str(ctx.author.id),mode="sections")
            system_prompt = system_prompts['ask_repo'].replace('{amygdala_response}', str(bot.amygdala_response)).replace('{themes}', themes)
            typing_task = asyncio.create_task(maintain_typing_state(ctx.channel))
            try:
                response = await bot.call_api(
                    user_content=rendered_user_content,
                    system_prompt=system_prompt,
                )
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
            memory_text = PROMPTS.ask_repo_memory.format(timestamp=timestamp, question=question, response=response)
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

    @bot.command(name='reader')
    @commands.check(lambda ctx: config.discord.has_command_permission('reader', ctx))
    async def reader_control(ctx, action: str = 'status'):
        """Control or inspect this agent's background book reader."""
        if bot.reading_processor is None:
            await ctx.send("READER is disabled for this agent.")
            return
        action = action.casefold()
        if action == 'start':
            await bot.reading_processor.start()
            await ctx.send("READER processor started.")
        elif action == 'stop':
            await bot.reading_processor.stop()
            await ctx.send("READER processor stopped; its book position was preserved.")
        elif action == 'status':
            status = await bot.bookshelf_service.status(bot.reader_id)
            if status.current_book and status.progress:
                await ctx.send(
                    f"READER is {'running' if bot.reading_processor.enabled else 'stopped'}: "
                    f"{status.current_book.title} at {status.current_locator or 'the end'} "
                    f"({status.percent_complete:.1f}%)."
                )
            else:
                await ctx.send(
                    f"READER is {'running' if bot.reading_processor.enabled else 'stopped'}; "
                    f"{status.available_books} ready, {status.pending_books} pending, "
                    f"{status.failed_books} failed books."
                )
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
            if bot.reading_processor and bot.reading_processor.enabled:
                await bot.reading_processor.stop()
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
        if bot.reading_processor and not bot.reading_processor.enabled:
            await bot.reading_processor.start()
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
    async def get_logs(ctx, source: str = 'all'):
        """DM recent bot/API JSONL without loading the complete logs into memory."""
        aliases = {'calls': 'api', 'both': 'all'}
        source = aliases.get(source.casefold().strip(), source.casefold().strip())
        if source not in {'all', 'bot', 'api'}:
            await ctx.send("Usage: `!get_logs [all|bot|api]`")
            return

        try:
            max_bytes = 1 * 1024 * 1024
            requested = []
            if source in {'all', 'bot'}:
                requested.append(('bot', getattr(bot.logger, 'jsonl_path', None)))
            if source in {'all', 'api'}:
                requested.append(('api', getattr(bot, 'api_log_path', None)))

            exports = []
            unavailable = []
            for label, path in requested:
                if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
                    unavailable.append(label)
                    continue
                export = await asyncio.to_thread(read_recent_jsonl, path, max_bytes)
                if not export.payload:
                    unavailable.append(label)
                    continue
                exports.append((label, export))

            if not exports:
                await ctx.send(
                    "No requested logs are available yet. "
                    "API logs begin in the per-bot log directory after restart."
                )
                return

            safe_bot_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', bot.logger.bot_id)
            streams = []
            files = []
            summary = []
            for label, export in exports:
                stream = BytesIO(export.payload)
                streams.append(stream)
                files.append(discord.File(
                    stream,
                    filename=f"{safe_bot_name}_recent_{label}_logs.jsonl",
                ))
                scope = "tail" if export.truncated else "complete file"
                summary.append(
                    f"{label}: {export.entry_count} entries, {scope}, "
                    f"source {export.source_bytes:,} bytes"
                )
            if unavailable:
                summary.append(f"unavailable: {', '.join(unavailable)}")

            try:
                await ctx.author.send("Recent logs\n" + "\n".join(summary), files=files)
            finally:
                for stream in streams:
                    stream.close()

            if not isinstance(ctx.channel, discord.DMChannel):
                await ctx.send(f"{ctx.author.mention}, I've sent you the logs via DM.")
        except discord.Forbidden:
            await ctx.send("I couldn't send you a DM. Please check your privacy settings and try again.")
        except discord.HTTPException as e:
            bot.logger.error(f"Discord rejected log export: {e}")
            await ctx.send("Discord rejected the log attachment. Try `!get_logs bot` or `!get_logs api` separately.")
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
    parser.add_argument('--api', choices=['ollama', 'llama-server', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter', 'unsloth'], 
                        default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, 
                        help='Specify the model to use. If not provided, defaults will be used based on the API.')
    parser.add_argument('--prompt-path', type=str, 
                        default='agent/prompts',
                        help='Path to prompt files directory (default: agent/prompts)')
    parser.add_argument('--bot-name', type=str,
                        help='Name of the bot to run (used for token and cache management)')
    parser.add_argument('--dmn-api', choices=['ollama', 'llama-server', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter', 'unsloth'], 
                        help='Choose the API to use for DMN processor (default: use main API)')
    parser.add_argument('--dmn-model', type=str,
                        help='Specify the model to use for DMN processor (default: use main model)')
    parser.add_argument('--reader-api', choices=['ollama', 'llama-server', 'openai', 'anthropic', 'vllm', 'gemini', 'openrouter', 'unsloth'],
                        help='Choose the API for the background READER (default: use main API)')
    parser.add_argument('--reader-model', type=str,
                        help='Specify the model for the background READER (default: use main model)')
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
    args.api_log_path = os.path.join(
        logger.log_dir,
        f"api_calls_{logger.bot_id}.jsonl",
    )
    private_api = load_private_api_client(args.bot_name or "default", args)
    # Override DMN config with command line arguments if provided
    if args.use_chronpression:
        config.dmn.use_chronpression = True
    if args.dmn_api or args.dmn_model:
        config.dmn.dmn_api_type = args.dmn_api or config.dmn.dmn_api_type
        config.dmn.dmn_model = args.dmn_model or config.dmn.dmn_model
        #logger.info(f"DMN API overridden: {config.dmn.dmn_api_type}, Model: {config.dmn.dmn_model}")
    bot = setup_bot(prompt_path=prompt_path, bot_id=args.bot_name)
    bot.reader_id = args.bot_name or 'default'
    # Keep every Discord-triggered inference path off the gateway event loop.
    # All channel, reflection, DMN, spike, summary, and repo calls share this
    # async facade and are serialized on its dedicated worker thread.
    api_worker = DiscordAPIWorker(
        private_api.call_api,
        private_api.api,
        name=args.bot_name or "default",
    )
    # Attach per-bot API handles
    bot.api = private_api.api
    bot.api_log_path = private_api.api.api_log_path
    bot.api_worker = api_worker
    bot.call_api = api_worker.call_api
    bot.update_api_temperature = private_api.update_api_temperature
    bot.update_api_top_p = private_api.update_api_top_p
    if config.todo.enabled:
        bot.todo_service = create_todo_service(
            config.todo, private_api.get_embeddings, bot.logger
        )
        if config.todo.import_directory:
            migration_result = asyncio.run(
                migrate_todont_directory(bot.todo_service, config.todo.import_directory)
            )
            bot.logger.info(f"Todo migration: {migration_result}")
    if args.reader_api or args.reader_model:
        config.reading.reader_api_type = args.reader_api or config.reading.reader_api_type
        config.reading.reader_model = args.reader_model or config.reading.reader_model
    if config.bookshelf.enabled:
        bot.bookshelf_service = create_bookshelf_service(
            bot.cache.get_cache_dir('bookshelf'), config.bookshelf,
            private_api.get_embeddings, bot.logger,
        )
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
    if config.bookshelf.enabled and bot.bookshelf_service is not None:
        bot.reading_processor = ReadingProcessor(
            bookshelf=bot.bookshelf_service,
            memory_index=bot.memory_index,
            prompt_formats=bot.prompt_formats,
            system_prompts=bot.system_prompts,
            runtime=bot,
            reader_id=bot.reader_id,
            reading_config=config.reading,
        )
    # Initialize spike processor (enabled by default, toggle via !spike on/off)
    bot.spike_processor = SpikeProcessor(
        bot,
        bot.memory_index,
        cache_path=bot.cache.get_cache_dir('spike')
    )
    # Publish the same versioned, read-only state hook used by in-process TUI
    # Chat. The endpoint is localhost-only and never writes memory_cache.pkl.
    def _viz_live_snapshot():
        runtime_state = {
            'agent_id': getattr(bot, 'agent_id', None),
            'agent_name': getattr(bot, 'agent_name', args.bot_name or 'default'),
            'amygdala_response': getattr(bot, 'amygdala_response', None),
            'processing_enabled': getattr(bot, 'processing_enabled', None),
            'transport': 'discord',
        }
        return make_live_payload(
            bot.memory_index._snapshot(),
            runtime=runtime_state,
            themes=snapshot_theme_cache(),
        )
    bot._viz_live_server = None
    try:
        bot._viz_live_server = VizLiveServer(
            args.bot_name or 'default',
            'cache',
            _viz_live_snapshot,
            logger=logger,
        )
        bot._viz_live_server.start()
    except Exception as e:
        bot._viz_live_server = None
        logger.warning(f"Viz live hook unavailable: {e}")
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
    finally:
        if bot._viz_live_server is not None:
            bot._viz_live_server.stop()
        api_worker.shutdown()
        private_api.shutdown_api_logger()
