"""Logging utilities for the Discord bot."""

import logging
import sys
from datetime import datetime
from pathlib import Path


# Global logger cache
_loggers: dict[str, logging.Logger] = {}

# Log directory
LOG_DIR = Path(__file__).parent.parent.parent / "logs"


def setup_logger(
    name: str = "discord_bot",
    level: str = "INFO",
    enable_debug_file: bool = True,
) -> logging.Logger:
    """Set up and configure a logger with console and session file handlers.
    
    Creates a single session log file per bot startup that captures all logs.
    
    Args:
        name: Logger name.
        level: Log level for console (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        enable_debug_file: Whether to enable debug file logging (default True).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if name in _loggers:
        return _loggers[name]
    
    # Always set logger to DEBUG to capture all messages
    # Handlers will filter based on their own levels
    logger.setLevel(logging.DEBUG)
    
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    # Detailed formatter for file logging
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(lineno)4d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Simple formatter for console
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler - respects the specified level
    log_level = getattr(logging, level.upper(), logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Session log file - new file for each bot session, captures ALL logs
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_path = LOG_DIR / f"session_{session_timestamp}.log"
    session_file_handler = logging.FileHandler(
        session_log_path,
        encoding="utf-8",
    )
    session_file_handler.setLevel(logging.DEBUG)  # Capture everything
    session_file_handler.setFormatter(detailed_formatter)
    logger.addHandler(session_file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _loggers[name] = logger
    
    # Log startup info
    logger.info(f"Logger initialized: {name}")
    logger.debug(f"Log directory: {LOG_DIR}")
    logger.debug(f"Session log file: {session_log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a child logger.
    
    Args:
        name: Logger name (will be prefixed with 'discord_bot.' if not already).
        
    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create as child of main logger
    full_name = f"discord_bot.{name}" if not name.startswith("discord_bot") else name
    logger = logging.getLogger(full_name)
    _loggers[name] = logger
    return logger
