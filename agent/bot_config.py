import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import ClassVar, Set, Dict
from logger import BotLogger
from datetime import datetime, timedelta
import discord

# Force reload of .env file
if os.path.exists('.env'):
    load_dotenv(override=True)

class LogConfig(BaseModel):
    """Logging configuration and paths"""
    base_log_dir: str = Field(default="cache")
    jsonl_pattern: str = Field(default="bot_log_{bot_id}.jsonl")
    db_pattern: str = Field(default="bot_log_{bot_id}.db")
    log_level: str = Field(default=os.getenv('LOGLEVEL', 'INFO'))
    log_format: str = Field(default='%(asctime)s - %(levelname)s - %(message)s')

    enable_console: bool = Field(default=True, description="Enable console logging")
    enable_jsonl: bool = Field(default=True, description="Enable JSONL file logging")
    enable_sql: bool = Field(default=False, description="Enable SQLite database logging")


class APIConfig(BaseModel):
    """API authentication and endpoint configurations"""
    discord_token: str = Field(default=os.getenv('DISCORD_TOKEN'))
    github_token: str = Field(default=os.getenv('GITHUB_TOKEN'))
    github_repo: str = Field(default=os.getenv('GITHUB_REPO'))
    notion_api_key: str = Field(default=os.getenv('NOTION_API_KEY'))
    ollama_api_base: str = Field(default=os.getenv('OLLAMA_API_BASE', 'http://localhost:11434'))
    ollama_model: str = Field(default=os.getenv('OLLAMA_MODEL'))


class FileConfig(BaseModel):
    allowed_extensions: Set[str] = Field(default={'.py','.js','.html','.css','.json','.md','.txt'})
    allowed_image_extensions: Set[str] = Field(default={'.jpg','.jpeg','.png','.gif','.bmp'})
    allowed_audio_extensions: Set[str] = Field(default={'.mp3','.wav','.m4a','.ogg','.flac'})
    allowed_video_extensions: Set[str] = Field(default={'.mp4','.mov','.webm','.mkv'})
    allowed_document_extensions: Set[str] = Field(default={
        '.doc','.docx','.docm',
        '.ppt','.pptx','.pptm',
        '.xls','.xlsx','.xlsm','.xlsb',
        '.odt','.ods','.odp',
        '.rtf','.epub','.csv','.pdf',
    })

    # single source of truth
    text_ingestion_mode: str = Field(default="hybrid")
    truncate_length: int = Field(default=8000)
    chronpress_threshold: int = Field(default=16000)
    chronpress_target_chars: int = Field(default=8000)
    audio_max_seconds: int = Field(default=30)
    video_max_seconds: int = Field(default=60)
    video_frame_rate: int = Field(default=1)


class SearchConfig(BaseModel):
    """Search and indexing configuration"""
    max_tokens: int = Field(default=8000)
    context_chunks: int = Field(default=4)
    chunk_percentage: int = Field(default=10)

class ConversationConfig(BaseModel):
    """Conversation handling configuration"""
    max_history: int = Field(default=32)
    minimal_history: int = Field(default=12)
    truncation_length: int = Field(default=768)
    harsh_truncation_length: int = Field(default=256)
    web_content_truncation_length: int = Field(default=8000)

class PersonaConfig(BaseModel):
    """Persona and response configuration"""
    default_amygdala_response: int = Field(default=70)
    temperature: float = Field(default_factory=lambda: 70/100.0)
    hippocampus_bandwidth: float = Field(default=0.70) 
    memory_capacity: int = Field(default=32)
    use_hippocampus_reranking: bool = Field(default=True)
    reranking_blend_factor: float = Field(default=0.5, description="Weight for blending initial search scores with reranking similarity (0-1)") 
    minimum_reranking_threshold: float = Field(default=0.64, description="Minimum threshold for reranked memories") 
    mood_coefficient: float = Field(default=0.15, description="Coefficient (0-1) that controls how strongly amygdala state lowers or raises the memory-selection threshold")

class SystemConfig(BaseModel):
    """System-wide configuration"""
    poll_interval: int = Field(default=int(os.getenv('POLL_INTERVAL', 120)))
    tick_rate: int = Field(default=800)

class AttentionConfig(BaseModel):
    """Attention mechanism configuration"""
    threshold: int = Field(default=60, description="Fuzzy match threshold for attention triggers (0-100)")
    default_top_n: int = Field(default=32, description="Default number of top trigrams to extract from memory")
    default_min_occ: int = Field(default=8, description="Minimum occurrence count for trigrams to be considered")
    refresh_interval_hours: int = Field(default=24, description="Hours between trigram cache refreshes")
    cooldown_minutes: float = Field(default=0.30, description="Minutes between attention trigger activations")

    stop_words: Set[str] = Field(default_factory=lambda: {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
        'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours',
        'theirs', 'a', 'an', 'some', 'any', 'all', 'each', 'every', 'no',
        'none', 'one', 'two', 'three', 'first', 'second', 'last', 'next',
        'other', 'another', 'more', 'most', 'much', 'many', 'few', 'little',
        'less', 'least', 'only', 'just', 'even', 'also', 'too', 'very',
        'quite', 'rather', 'so', 'such', 'how', 'what', 'when', 'where',
        'why', 'who', 'which', 'whose', 'whom', 'if', 'unless', 'until',
        'while', 'since', 'because', 'as', 'than', 'then', 'now', 'here',
        'there', 'yes', 'no', 'not', 'dont', 'doesnt', 'didnt', 'wont',
        'wouldnt', 'couldnt', 'shouldnt', 'cant', 'isnt', 'arent', 'wasnt',
        'werent', 'hasnt', 'havent', 'hadnt'
    })

    @property
    def refresh_interval(self) -> timedelta:
        """Get refresh interval as timedelta"""
        return timedelta(hours=self.refresh_interval_hours)

    @property
    def cooldown(self) -> timedelta:
        """Get cooldown as timedelta"""
        return timedelta(minutes=self.cooldown_minutes)

class DMNConfig(BaseModel):
    """DMN configuration"""
    tick_rate: int = Field(default=240, description="Time between thought generations in seconds")
    temperature: float = Field(default=0.7, description="Base creative temperature")
    temperature_max: float = Field(default=1.8)
    combination_threshold: float = Field(default=0.2, description="Minimum relevance score for memory combinations")
    decay_rate: float = Field(default=0.1, description="Rate at which used memory weights decrease")
    top_k: int = Field(default=24, description="Top k memories to consider for combination")
    density_multiplier: float = Field(default=2.1, description="Multiplier for density-based temperature scaling")
    fuzzy_overlap_threshold: int = Field(default=80, description="Minimum fuzzy overlap threshold for memory combination")
    fuzzy_search_threshold: int = Field(default=90, description="Minimum fuzzy search threshold for term matching")
    max_memory_length: int = Field(default=64, description="Maximum length of a memory based on truncate_middle function")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity score for memory relevance")

    # DMN-specific API settings
    dmn_api_type: str = Field(default=None, description="API type for DMN processor (ollama, openai, anthropic, etc.)")
    dmn_model: str = Field(default=None, description="Model name for DMN processor")

    # Chronomic distillation mode (replaces LLM call with chronomic_filter)
    use_chronpression: bool = Field(default=False, description="Use chronomic compression instead of LLM for DMN thought distillation")
    chron_compression_max: float = Field(default=0.99, description="Maximum compression ratio for chronomic distillation at full amygdala arousal (0.0-1.0)")

    # Neighbour thinning probability bounds (similarity-graded)
    thin_p_min: float = Field(default=0.05, description="Min per-term removal probability for distant neighbours")
    thin_p_max: float = Field(default=0.6, description="Max per-term removal probability for close neighbours (s≈1)")

    # Memory presets
    modes: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "forgetful": {
            "combination_threshold": 0.02,
            "similarity_threshold": 0.2,
            "decay_rate": 0.5,
            "top_k": 24,
            "fuzzy_overlap_threshold": 70,
            "fuzzy_search_threshold": 80,
            "thin_p_min": 0.1,
            "thin_p_max": 0.8,
        },
        "homeostatic": {
            "combination_threshold": 0.3,
            "similarity_threshold": 0.3,
            "decay_rate": 0.25,
            "top_k": 16,
            "fuzzy_overlap_threshold": 80,
            "fuzzy_search_threshold": 90,
            "thin_p_min": 0.05,
            "thin_p_max": 0.6,
        },
        "conservative": {
            "combination_threshold": 0.8,
            "similarity_threshold": 0.4,
            "decay_rate": 0.15,
            "top_k": 8,
            "fuzzy_overlap_threshold": 90,
            "fuzzy_search_threshold": 95,
            "thin_p_min": 0.02,
            "thin_p_max": 0.4,
        }
    })

class SpikeConfig(BaseModel):
    """Spike processor configuration - handles orphaned memory outreach"""
    context_n: int = Field(default=50, description="Initial message count to compress per surface")
    max_expansion: int = Field(default=150, description="Maximum message count for tie-breaking expansion")
    expansion_step: int = Field(default=25, description="Step size when expanding context for ties")
    match_threshold: float = Field(default=0.35, description="Minimum score for surface to be viable")
    compression_ratio: float = Field(default=0.6, description="Chronpression ratio for surface context")
    cooldown_seconds: int = Field(default=120, description="Minimum seconds between spike fires")
    max_surfaces: int = Field(default=8, description="Maximum recent surfaces to consider")
    recency_window_hours: int = Field(default=512, description="Hours to look back for engaged surfaces")
    memory_k: int = Field(default=12, description="Number of memories to retrieve for context")
    memory_truncation: int = Field(default=512, description="Max tokens per memory in context")
    theme_weight: float = Field(default=0.3, description="Weight for theme resonance in scoring (0-1)")

class EmbeddingConfig(BaseModel):
    """Pydantic model for embedding configuration."""
    provider: str = Field( default='ollama', description="Provider for embedding service" )
    model: str = Field( default='all-minilm', description="Specific model for embeddings" )
    api_base: str = Field( default='http://localhost:11434', description="Base URL for Ollama API" )
    max_embed_tokens: int = Field(default=160, description="Maximum tokens per embedding request")
    dimensions: int = Field( default=384, description="Expected embedding dimensions" )

class HippocampusConfig(BaseModel):
    """Pydantic model for Hippocampus configuration - provides vector embeddings for downstream search."""
    embedding_provider: str = Field(default='ollama', description="Provider for embedding service")
    embedding_model: str = Field(default='all-minilm:latest', description="Model to use for embeddings")
    blend_factor: float = Field(default=0.7, description="Weight for blending initial search scores with embedding similarity (0-1)")

class DiscordConfig(BaseModel):
    """Discord-specific configuration"""
    channel_id: str = Field(default=os.getenv('DISCORD_CHANNEL_ID'))
    bot_manager_role: str = Field(default='Ally')
    sync_slash_commands: bool = Field(default=os.getenv('DISCORD_SYNC_SLASH_COMMANDS', 'true').lower() in ('1', 'true', 'yes', 'on'))
    slash_guild_id: str = Field(default=os.getenv('DISCORD_SLASH_GUILD_ID'))
    global_slash_commands: bool = Field(default=os.getenv('DISCORD_GLOBAL_SLASH_COMMANDS', 'false').lower() in ('1', 'true', 'yes', 'on'))
    
    system_commands: Set[str] = Field(default={ 'kill', 'resume', 'get_logs', 'dmn', 'mentions', 'persona', 'search_memories', 'spike' })
    management_commands: Set[str] = Field(default={ 'add_memory', 'index_repo', 'reranking', 'clear_memories', 'attention', 'spike' })
    general_commands: Set[str] = Field(default={ 'summarize', 'ask_repo', 'repo_file_chat', 'analyze_file' })
    bot_action_commands: Set[str] = Field(default={ 'help', 'dmn', 'persona', 'add_memory', 'ask_repo', 'search_memories', 'kill', 'attention' })

    def has_command_permission(self, command_name: str, ctx) -> bool:
        if command_name not in (
            self.system_commands | 
            self.management_commands | 
            self.general_commands
        ):
            return False
        # bot self-invocation - restricted to bot_action_commands
        if ctx.author.bot:
            return command_name in self.bot_action_commands
        if command_name in self.general_commands:
            return True
        if isinstance(ctx.channel, discord.DMChannel):
            has_admin = False
            has_ally = False
            for guild in ctx.bot.guilds:
                member = guild.get_member(ctx.author.id)
                if not member:
                    continue
                if (member.guild_permissions.administrator or 
                    member.guild_permissions.manage_guild):
                    has_admin = True
                    break
                if any(role.name == self.bot_manager_role for role in member.roles):
                    has_ally = True
            if command_name in self.system_commands:
                return has_admin
            if command_name in self.management_commands:
                return has_admin or has_ally
            return False
        if (ctx.author.guild_permissions.administrator or 
            ctx.author.guild_permissions.manage_guild):
            return True
        if (command_name in self.management_commands and
            any(role.name == self.bot_manager_role for role in ctx.author.roles)):
            return True
        return False

    def has_interaction_permission(self, command_name: str, interaction) -> bool:
        if command_name not in (
            self.system_commands |
            self.management_commands |
            self.general_commands
        ):
            return False
        user = interaction.user
        if getattr(user, 'bot', False):
            return command_name in self.bot_action_commands
        if command_name in self.general_commands:
            return True
        if isinstance(interaction.channel, discord.DMChannel) or interaction.guild is None:
            has_admin = False
            has_ally = False
            for guild in interaction.client.guilds:
                member = guild.get_member(user.id)
                if not member:
                    continue
                if (member.guild_permissions.administrator or
                    member.guild_permissions.manage_guild):
                    has_admin = True
                    break
                if any(role.name == self.bot_manager_role for role in member.roles):
                    has_ally = True
            if command_name in self.system_commands:
                return has_admin
            if command_name in self.management_commands:
                return has_admin or has_ally
            return False
        permissions = getattr(user, 'guild_permissions', None)
        if permissions and (permissions.administrator or permissions.manage_guild):
            return True
        if (command_name in self.management_commands and
            any(role.name == self.bot_manager_role for role in getattr(user, 'roles', []))):
            return True
        return False
    
class PromptSchema(BaseModel):
    """Single source of truth for required prompt template variables.

    Use PromptSchema.required_system and PromptSchema.required_formats
    in both the TUI validator and discord_bot.py format-string checks.
    """

    required_system: ClassVar[Dict[str, Set[str]]] = {
        "default_chat": {"amygdala_response"},
        "default_web_chat": {"amygdala_response"},
        "repo_file_chat": {"amygdala_response"},
        "channel_summarization": {"amygdala_response"},
        "ask_repo": {"amygdala_response"},
        "thought_generation": {"amygdala_response"},
        "file_analysis": {"amygdala_response"},
        "image_analysis": {"amygdala_response"},
        "combined_analysis": {"amygdala_response"},
        "spike_engagement": {"amygdala_response", "themes"},
        "attention_triggers": set(),
    }
    required_formats: ClassVar[Dict[str, Set[str]]] = {
        "chat_with_memory": {"context", "user_name", "user_message"},
        "introduction": {"context", "user_name", "user_message"},
        "introduction_web": {"context", "user_name", "user_message"},
        "analyze_code": {"context", "code_content", "user_name", "user_message"},
        "summarize_channel": {"context", "content"},
        "ask_repo": {"context", "question"},
        "repo_file_chat": {"file_path", "code_type", "repo_code", "user_task_description", "context"},
        "generate_thought": {"user_name", "memory_text"},
        "analyze_image": {"context", "filename", "user_message", "user_name"},
        "analyze_file": {"context", "filename", "file_content", "user_message", "user_name"},
        "analyze_combined": {"context", "image_files", "text_files", "user_message", "user_name"},
        "spike_engagement": {"tension_desc", "memory", "memory_context", "conversation_context", "location", "timestamp"},
    }


class BotConfig(BaseModel):
    """Main configuration container"""
    api: APIConfig = Field(default_factory=APIConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    files: FileConfig = Field(default_factory=FileConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    dmn: DMNConfig = Field(default_factory=DMNConfig)
    spike: SpikeConfig = Field(default_factory=SpikeConfig)

# Create global config instance
config = BotConfig()

def apply_overrides(bot_name: str) -> None:
    """Merge cache/{bot_name}/config_overrides.json into the global config object."""
    import json
    from pathlib import Path
    p = Path(config.logging.base_log_dir) / bot_name / "config_overrides.json"
    if not p.exists():
        return
    try:
        overrides = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for section, values in overrides.items():
        sub = getattr(config, section, None)
        if sub is None or not isinstance(values, dict):
            continue
        for k, v in values.items():
            try:
                setattr(sub, k, v)
            except Exception:
                pass

def init_logging():
    """Initialize global logging after config is fully loaded."""
    BotLogger.setup_global_logging()
