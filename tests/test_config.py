"""Tests for configuration reload behavior."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from src.utils.config import Config


class ConfigReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_discord_token = os.environ.get("DISCORD_BOT_TOKEN")
        self._old_gemini_key = os.environ.get("GEMINI_API_KEY")
        os.environ["DISCORD_BOT_TOKEN"] = "test_discord_token"
        os.environ["GEMINI_API_KEY"] = "test_gemini_key"

    def tearDown(self) -> None:
        if self._old_discord_token is None:
            os.environ.pop("DISCORD_BOT_TOKEN", None)
        else:
            os.environ["DISCORD_BOT_TOKEN"] = self._old_discord_token

        if self._old_gemini_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self._old_gemini_key

    @unittest.skipUnless(importlib.util.find_spec("yaml") is not None, "PyYAML is required for config YAML reload test")
    def test_reload_notifies_and_survives_listener_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "wake_word:",
                        "  phrase: \"hey_jarvis\"",
                        "  threshold: 0.3",
                        "voice:",
                        "  name: \"Puck\"",
                        "behavior:",
                        "  capture_duration: 5.0",
                        "  silence_threshold: 0.5",
                        "logging:",
                        "  level: \"INFO\"",
                        "  directory: \"logs\"",
                        "  log_audio: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = Config.load(config_path=str(config_path))
            notified = []

            def good_listener(cfg: Config, changed_fields: list[str]) -> None:
                notified.append(tuple(changed_fields))

            def bad_listener(cfg: Config, changed_fields: list[str]) -> None:
                raise RuntimeError("listener failed")

            config.add_change_listener(good_listener)
            config.add_change_listener(bad_listener)

            config_path.write_text(
                "\n".join(
                    [
                        "wake_word:",
                        "  phrase: \"alexa\"",
                        "  threshold: 0.6",
                        "voice:",
                        "  name: \"Charon\"",
                        "behavior:",
                        "  capture_duration: 7.0",
                        "  silence_threshold: 1.0",
                        "logging:",
                        "  level: \"DEBUG\"",
                        "  directory: \"custom_logs\"",
                        "  log_audio: true",
                    ]
                ),
                encoding="utf-8",
            )

            changed = config.reload()

            self.assertIn("wake_phrase", changed)
            self.assertIn("wake_word_threshold", changed)
            self.assertIn("gemini_voice", changed)
            self.assertIn("capture_duration", changed)
            self.assertIn("silence_threshold", changed)
            self.assertIn("log_level", changed)
            self.assertIn("log_audio", changed)
            self.assertGreaterEqual(len(notified), 1)


if __name__ == "__main__":
    unittest.main()
