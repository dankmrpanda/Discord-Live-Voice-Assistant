"""Tests for voice runtime session helpers."""

import asyncio
import importlib.util
import unittest
from pathlib import Path


def _load_session_class():
    module_path = Path(__file__).resolve().parents[1] / "src" / "bot" / "session.py"
    spec = importlib.util.spec_from_file_location("session_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.VoiceRuntimeSession


HAS_DISCORD = importlib.util.find_spec("discord") is not None
VoiceRuntimeSession = _load_session_class() if HAS_DISCORD else None


@unittest.skipUnless(HAS_DISCORD, "discord package is required for session tests")
class VoiceRuntimeSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_task_clears_attribute(self) -> None:
        session = VoiceRuntimeSession()

        session.audio_loop_task = asyncio.create_task(asyncio.sleep(5))
        await session.cancel_task("audio_loop_task")

        self.assertIsNone(session.audio_loop_task)

    async def test_cancel_task_handles_missing_task(self) -> None:
        session = VoiceRuntimeSession()
        session.capture_task = None

        await session.cancel_task("capture_task")
        self.assertIsNone(session.capture_task)

    async def test_cancel_task_already_finished(self) -> None:
        """Cancelling a task that has already completed should clear the attr."""
        session = VoiceRuntimeSession()
        session.audio_loop_task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.sleep(0.05)  # let it finish
        await session.cancel_task("audio_loop_task")
        self.assertIsNone(session.audio_loop_task)

    def test_default_field_values(self) -> None:
        session = VoiceRuntimeSession()
        self.assertIsNone(session.voice_client)
        self.assertIsNone(session.target_channel)
        self.assertFalse(session.is_capturing_for_gemini)
        self.assertEqual(session.audio_chunks_sent, 0)
        self.assertIsInstance(session.speech_buffer, list)
        self.assertEqual(len(session.speech_buffer), 0)

    def test_events_not_set_initially(self) -> None:
        session = VoiceRuntimeSession()
        self.assertFalse(session.connection_ready.is_set())
        self.assertFalse(session.connection_failed.is_set())
        self.assertFalse(session.streaming_complete.is_set())


if __name__ == "__main__":
    unittest.main()
