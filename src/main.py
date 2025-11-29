"""Main entry point for the Discord Live VC Bot."""

import asyncio
import sys
import platform
from pathlib import Path

from .utils.config import Config, ConfigWatcher
from .utils.logger import setup_logger, get_logger
from .bot.client import DiscordBot, set_bot


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Load configuration from config.yaml and .env
    try:
        config = Config.load()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure all required environment variables are set.")
        print("See .env.example for reference.")
        return 1
    
    # Set up logging
    logger = setup_logger(
        name="discord_bot",
        level=config.log_level,
        enable_debug_file=True,
    )
    
    # Add config change listener for logging
    def log_config_change(cfg: Config, changed: list) -> None:
        logger.info(f"📝 Configuration reloaded: {', '.join(changed)}")
    
    config.add_change_listener(log_config_change)
    
    logger.info("=" * 60)
    logger.info("Discord Live VC Bot Starting")
    logger.info("=" * 60)
    
    # Log system info
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {platform.platform()}")
    logger.debug(f"Machine: {platform.machine()}")
    
    # Log configuration sources
    config_file = Path("config.yaml")
    if config_file.exists():
        logger.info(f"Config file: {config_file.absolute()}")
    else:
        logger.info("Config file: using defaults (config.yaml not found)")
    
    # Log configuration
    logger.info(f"Wake phrase: '{config.wake_phrase_display}'")
    logger.info(f"Wake word threshold: {config.wake_word_threshold}")
    logger.info(f"Gemini model: {config.gemini_model}")
    logger.info(f"Gemini voice: {config.gemini_voice}")
    logger.info(f"Capture duration: {config.capture_duration}s")
    logger.info(f"Log level: {config.log_level}")
    logger.debug(f"Discord sample rate: {config.discord_sample_rate} Hz")
    logger.debug(f"Gemini input sample rate: {config.gemini_input_sample_rate} Hz")
    logger.debug(f"Gemini output sample rate: {config.gemini_output_sample_rate} Hz")
    
    # Create bot
    logger.debug("Creating DiscordBot instance")
    bot = DiscordBot(config)
    
    # Set global reference for slash commands
    set_bot(bot)
    logger.debug("Bot reference set for slash commands")
    
    # Run the bot
    try:
        logger.info("Starting Discord bot connection...")
        # py-cord doesn't use log_handler argument
        bot.run(config.discord_bot_token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(f"Fatal error details: {type(e).__name__}: {e}", exc_info=True)
        return 1
    
    logger.info("Bot shutdown complete")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
