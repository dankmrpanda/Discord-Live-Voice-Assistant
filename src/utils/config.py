"""Configuration management for the Discord bot."""

import os
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Any
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from config.yaml and environment variables.
    
    Configuration sources:
    - .env file: Secrets only (DISCORD_BOT_TOKEN, DISCORD_APPLICATION_ID, GEMINI_API_KEY)
    - config.yaml: All other settings (voice, wake word, behavior, logging, etc.)
    
    Changes to config.yaml are auto-reloaded without restart.
    Changes to .env require a container/bot restart.
    """
    
    # Discord (from .env only - secrets)
    discord_bot_token: str
    discord_application_id: Optional[str]
    
    # Gemini (API key from .env, other settings from config.yaml)
    gemini_api_key: str
    gemini_voice: str = "Puck"
    gemini_model: str = "gemini-2.0-flash-live-001"
    
    # Wake Word
    wake_phrase: str = "hey_jarvis"
    wake_word_threshold: float = 0.5
    
    # Bot Behavior
    capture_duration: float = 5.0
    silence_threshold: float = 0.5
    
    # System Prompt
    system_prompt: str = ""
    
    # Logging
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_audio: bool = False
    
    # Audio settings
    discord_sample_rate: int = 48000  # Discord uses 48kHz
    gemini_input_sample_rate: int = 16000  # Gemini expects 16kHz
    gemini_output_sample_rate: int = 24000  # Gemini outputs 24kHz
    audio_channels: int = 1  # Mono
    
    # Internal: config file path for reloading
    _config_path: Optional[str] = field(default=None, repr=False)
    
    # Change listeners
    _change_listeners: List[Callable[["Config", List[str]], Any]] = field(default_factory=list, repr=False)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, env_path: Optional[str] = None) -> "Config":
        """Load configuration from config.yaml and environment variables.
        
        Args:
            config_path: Optional path to config.yaml file.
            env_path: Optional path to .env file.
            
        Returns:
            Config instance with loaded values.
            
        Raises:
            ValueError: If required settings are missing.
        """
        # Load .env file for secrets
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        # Load config.yaml for user settings and resolve the path
        yaml_config, resolved_config_path = cls._load_yaml_with_path(config_path)
        
        # Required secrets from environment
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is required")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Get wake word settings (from config.yaml only)
        wake_word_config = yaml_config.get("wake_word", {})
        wake_phrase = wake_word_config.get("phrase", "hey_jarvis")
        wake_threshold = wake_word_config.get("threshold", 0.5)
        
        # Get voice settings (from config.yaml only)
        voice_config = yaml_config.get("voice", {})
        gemini_voice = voice_config.get("name", "Puck")
        
        # Get Gemini model settings (from config.yaml only)
        gemini_config = yaml_config.get("gemini", {})
        gemini_model = gemini_config.get("model", "gemini-2.0-flash-live-001")
        
        # Get behavior settings
        behavior_config = yaml_config.get("behavior", {})
        
        # Get audio settings
        audio_config = yaml_config.get("audio", {})
        
        # Get logging settings (from config.yaml only)
        logging_config = yaml_config.get("logging", {})
        log_level = logging_config.get("level", "INFO")
        log_audio = logging_config.get("log_audio", False)
        
        # Get system prompt
        system_prompt = yaml_config.get("system_prompt", cls._default_system_prompt())
        
        return cls(
            discord_bot_token=discord_token,
            discord_application_id=os.getenv("DISCORD_APPLICATION_ID"),
            gemini_api_key=gemini_key,
            gemini_voice=gemini_voice,
            gemini_model=gemini_model,
            wake_phrase=wake_phrase,
            wake_word_threshold=float(wake_threshold),
            capture_duration=behavior_config.get("capture_duration", 5.0),
            silence_threshold=behavior_config.get("silence_threshold", 0.5),
            system_prompt=system_prompt,
            log_level=log_level,
            log_directory=logging_config.get("directory", "logs"),
            log_audio=log_audio,
            discord_sample_rate=audio_config.get("discord_sample_rate", 48000),
            gemini_input_sample_rate=audio_config.get("gemini_input_sample_rate", 16000),
            gemini_output_sample_rate=audio_config.get("gemini_output_sample_rate", 24000),
            _config_path=resolved_config_path,
        )
    
    def add_change_listener(self, listener: Callable[["Config", List[str]], Any]) -> None:
        """Add a listener that will be called when config changes.
        
        Args:
            listener: Callback function that receives (config, changed_fields).
        """
        if listener not in self._change_listeners:
            self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[["Config", List[str]], Any]) -> None:
        """Remove a change listener.
        
        Args:
            listener: The listener to remove.
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def reload(self) -> List[str]:
        """Reload configuration from the config file.
        
        Returns:
            List of field names that changed.
        """
        if not self._config_path:
            return []
        
        yaml_config = self._load_yaml(self._config_path)
        changed_fields = []
        
        # Track old values and update
        # Wake word settings (from config.yaml only)
        wake_word_config = yaml_config.get("wake_word", {})
        new_wake_phrase = wake_word_config.get("phrase", "hey_jarvis")
        new_wake_threshold = wake_word_config.get("threshold", 0.5)
        
        if self.wake_phrase != new_wake_phrase:
            changed_fields.append("wake_phrase")
            self.wake_phrase = new_wake_phrase
        if self.wake_word_threshold != new_wake_threshold:
            changed_fields.append("wake_word_threshold")
            self.wake_word_threshold = float(new_wake_threshold)
        
        # Voice settings (from config.yaml only)
        voice_config = yaml_config.get("voice", {})
        new_voice = voice_config.get("name", "Puck")
        if self.gemini_voice != new_voice:
            changed_fields.append("gemini_voice")
            self.gemini_voice = new_voice
        
        # Gemini model settings (from config.yaml only)
        gemini_config = yaml_config.get("gemini", {})
        new_model = gemini_config.get("model", "gemini-2.0-flash-live-001")
        if self.gemini_model != new_model:
            changed_fields.append("gemini_model")
            self.gemini_model = new_model
        
        # Behavior settings
        behavior_config = yaml_config.get("behavior", {})
        new_capture_duration = behavior_config.get("capture_duration", 5.0)
        new_silence_threshold = behavior_config.get("silence_threshold", 0.5)
        
        if self.capture_duration != new_capture_duration:
            changed_fields.append("capture_duration")
            self.capture_duration = new_capture_duration
        if self.silence_threshold != new_silence_threshold:
            changed_fields.append("silence_threshold")
            self.silence_threshold = new_silence_threshold
        
        # System prompt
        new_system_prompt = yaml_config.get("system_prompt", self._default_system_prompt())
        if self.system_prompt != new_system_prompt:
            changed_fields.append("system_prompt")
            self.system_prompt = new_system_prompt
        
        # Logging settings (from config.yaml only)
        logging_config = yaml_config.get("logging", {})
        new_log_level = logging_config.get("level", "INFO")
        new_log_audio = logging_config.get("log_audio", False)
        
        if self.log_level != new_log_level:
            changed_fields.append("log_level")
            self.log_level = new_log_level
        if self.log_audio != new_log_audio:
            changed_fields.append("log_audio")
            self.log_audio = new_log_audio
        
        # Notify listeners if there were changes
        if changed_fields:
            for listener in self._change_listeners:
                try:
                    listener(self, changed_fields)
                except Exception:
                    pass  # Don't let listener errors break reload
        
        return changed_fields
    
    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """Load configuration from environment variables only (legacy method).
        
        Args:
            env_path: Optional path to .env file.
            
        Returns:
            Config instance with loaded values.
        """
        return cls.load(env_path=env_path)
    
    @staticmethod
    def _load_yaml(config_path: Optional[str] = None) -> dict:
        """Load YAML configuration file.
        
        Args:
            config_path: Path to config.yaml file.
            
        Returns:
            Dictionary of configuration values.
        """
        config, _ = Config._load_yaml_with_path(config_path)
        return config
    
    @staticmethod
    def _load_yaml_with_path(config_path: Optional[str] = None) -> tuple:
        """Load YAML configuration file and return the resolved path.
        
        Args:
            config_path: Path to config.yaml file.
            
        Returns:
            Tuple of (config dict, resolved path string or None).
        """
        try:
            import yaml
        except ImportError:
            # If PyYAML not installed, return empty dict
            return {}, None
        
        # Default paths to check
        if config_path:
            paths = [Path(config_path)]
        else:
            paths = [
                Path("config.yaml"),
                Path("config.yml"),
                Path("/app/config.yaml"),  # Docker path
            ]
        
        for path in paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        return (config if config else {}, str(path.absolute()))
                except Exception:
                    pass
        
        return {}, None
    
    @staticmethod
    def _default_system_prompt() -> str:
        """Get default system prompt."""
        return """You are a helpful voice assistant in a Discord voice channel.
Keep your responses concise and conversational since they will be spoken aloud.
Be friendly, helpful, and natural in your responses.
If you don't understand something, ask for clarification.
Avoid using markdown formatting, bullet points, or numbered lists since this is voice output."""
    
    @property
    def wake_phrase_display(self) -> str:
        """Get human-readable wake phrase for display."""
        return self.wake_phrase.replace("_", " ").title()


class ConfigWatcher:
    """Watches config file for changes and triggers reload.
    
    Uses file modification time to detect changes efficiently.
    """
    
    def __init__(self, config: Config, check_interval: float = 2.0):
        """Initialize the config watcher.
        
        Args:
            config: The Config instance to watch and reload.
            check_interval: How often to check for changes (seconds).
        """
        self.config = config
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_mtime: Optional[float] = None
        
        # Get initial modification time
        if config._config_path:
            try:
                self._last_mtime = Path(config._config_path).stat().st_mtime
            except (OSError, IOError):
                pass
    
    async def start(self) -> None:
        """Start watching for config changes."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
    
    async def stop(self) -> None:
        """Stop watching for config changes."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _watch_loop(self) -> None:
        """Main loop that checks for config file changes."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                if not self.config._config_path:
                    continue
                
                try:
                    current_mtime = Path(self.config._config_path).stat().st_mtime
                except (OSError, IOError):
                    continue
                
                if self._last_mtime is not None and current_mtime > self._last_mtime:
                    # File was modified, reload config
                    changed = self.config.reload()
                    if changed:
                        # Logging will be handled by the change listeners
                        pass
                
                self._last_mtime = current_mtime
                
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't let watcher errors crash the bot
                pass
