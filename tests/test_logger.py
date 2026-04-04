"""Tests for logger configuration behavior."""

import logging
import tempfile
import unittest
from pathlib import Path

from src.utils.logger import setup_logger


class LoggerSetupTests(unittest.TestCase):
    def _cleanup_logger(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_setup_logger_can_disable_file_logging(self) -> None:
        logger = setup_logger(name="discord_bot.test.no_file", enable_debug_file=False)
        try:
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(file_handlers, [])
        finally:
            self._cleanup_logger(logger)

    def test_setup_logger_uses_configured_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "my_logs"
            logger = setup_logger(
                name="discord_bot.test.custom_dir",
                enable_debug_file=True,
                log_directory=str(log_dir),
            )
            try:
                file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
                self.assertEqual(len(file_handlers), 1)
                file_path = Path(file_handlers[0].baseFilename).resolve()
                self.assertEqual(file_path.parent, log_dir.resolve())
            finally:
                self._cleanup_logger(logger)


if __name__ == "__main__":
    unittest.main()
