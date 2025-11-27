"""Configuration management for the Discord bot."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from config.yaml and environment variables.
    
    Priority order (highest to lowest):
    1. Environment variables (for secrets like API keys)
    2. config.yaml file (for user settings)
    3. Default values
    """
    
    # Discord (from .env only - secrets)
    discord_bot_token: str
    discord_application_id: Optional[str]
    
    # Gemini (API key from .env, other settings from config.yaml)
    gemini_api_key: str
    gemini_voice: str
    
    # Wake Word
    wake_phrase: str
    wake_word_threshold: float
    
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
        
        # Load config.yaml for user settings
        yaml_config = cls._load_yaml(config_path)
        
        # Required secrets from environment
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is required")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Get wake word settings (env overrides yaml)
        wake_word_config = yaml_config.get("wake_word", {})
        wake_phrase = os.getenv("WAKE_PHRASE") or wake_word_config.get("phrase", "hey_jarvis")
        wake_threshold = os.getenv("WAKE_WORD_THRESHOLD")
        if wake_threshold:
            wake_threshold = float(wake_threshold)
        else:
            wake_threshold = wake_word_config.get("threshold", 0.5)
        
        # Get voice settings (env overrides yaml)
        voice_config = yaml_config.get("voice", {})
        gemini_voice = os.getenv("GEMINI_VOICE") or voice_config.get("name", "Puck")
        
        # Get behavior settings
        behavior_config = yaml_config.get("behavior", {})
        
        # Get audio settings
        audio_config = yaml_config.get("audio", {})
        
        # Get logging settings (env overrides yaml)
        logging_config = yaml_config.get("logging", {})
        log_level = os.getenv("LOG_LEVEL") or logging_config.get("level", "INFO")
        log_audio = logging_config.get("log_audio", False)
        
        # Get system prompt
        system_prompt = yaml_config.get("system_prompt", cls._default_system_prompt())
        
        return cls(
            discord_bot_token=discord_token,
            discord_application_id=os.getenv("DISCORD_APPLICATION_ID"),
            gemini_api_key=gemini_key,
            gemini_voice=gemini_voice,
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
        )
    
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
        try:
            import yaml
        except ImportError:
            # If PyYAML not installed, return empty dict
            return {}
        
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
                        return config if config else {}
                except Exception:
                    pass
        
        return {}
    
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
