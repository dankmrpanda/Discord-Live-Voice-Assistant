"""Unit tests for VoiceHandler streaming pipeline helpers."""

import asyncio
import importlib.util
from types import SimpleNamespace
import unittest


HAS_DISCORD = importlib.util.find_spec("discord") is not None
HAS_VOICE_RECV = False
if HAS_DISCORD:
    try:
        HAS_VOICE_RECV = importlib.util.find_spec("discord.ext.voice_recv") is not None
    except ModuleNotFoundError:
        HAS_VOICE_RECV = False

if HAS_DISCORD and HAS_VOICE_RECV:
    from src.bot.voice_handler import VoiceHandler


class _NoAudioCapture:
    def is_silence_detected(self) -> bool:
        return False

    async def get_streaming_chunk(self, timeout: float = 0.1):
        await asyncio.sleep(0)
        return None


class _CaptureWithChunks:
    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)

    def is_silence_detected(self) -> bool:
        return False

    async def get_streaming_chunk(self, timeout: float = 0.1):
        await asyncio.sleep(0)
        if self._chunks:
            return self._chunks.pop(0)
        return None


class _GeminiRecorder:
    def __init__(self) -> None:
        self.sent_chunks = 0
        self.end_turn_called = 0

    async def send_audio(self, audio_data: bytes) -> bool:
        self.sent_chunks += 1
        return True

    async def end_turn(self) -> bool:
        self.end_turn_called += 1
        return True


class _GeminiReceiveTimeout:
    async def receive_responses(self):
        await asyncio.sleep(1.0)
        if False:
            yield b""


class _GeminiReceiveIdleTimeout:
    async def receive_responses(self):
        yield b"first"
        await asyncio.sleep(1.0)
        if False:
            yield b""


class _GeminiReceiveError:
    async def receive_responses(self):
        raise RuntimeError("boom")
        if False:
            yield b""


class _PlaybackStub:
    def __init__(self) -> None:
        self.finished = False
        self.chunks = 0

    def add_streaming_chunk(self, gemini_audio: bytes) -> bool:
        self.chunks += 1
        return True

    def finish_streaming(self) -> None:
        self.finished = True


@unittest.skipUnless(HAS_DISCORD and HAS_VOICE_RECV, "discord.py + discord-ext-voice-recv are required")
class VoicePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_audio_loop_no_audio_exits_without_end_turn(self) -> None:
        gemini = _GeminiRecorder()
        dummy = SimpleNamespace(
            _is_capturing_for_gemini=True,
            capture_duration=0.01,
            _capture=_NoAudioCapture(),
            _gemini=gemini,
            _audio_chunks_sent=0,
            _streaming_complete=asyncio.Event(),
        )

        result = await VoiceHandler._send_audio_loop(dummy)

        self.assertEqual(result.chunks_sent, 0)
        self.assertEqual(result.total_bytes, 0)
        self.assertFalse(result.ended_turn)
        self.assertEqual(result.reason, "no_audio_captured")
        self.assertEqual(gemini.end_turn_called, 0)

    async def test_send_audio_loop_ends_turn_when_audio_present(self) -> None:
        gemini = _GeminiRecorder()
        dummy = SimpleNamespace(
            _is_capturing_for_gemini=True,
            capture_duration=0.1,
            _capture=_CaptureWithChunks([b"a" * 640]),
            _gemini=gemini,
            _audio_chunks_sent=0,
            _streaming_complete=asyncio.Event(),
        )

        result = await VoiceHandler._send_audio_loop(dummy)

        self.assertGreaterEqual(result.chunks_sent, 1)
        self.assertTrue(result.ended_turn)
        self.assertEqual(gemini.end_turn_called, 1)

    async def test_receive_response_loop_first_chunk_timeout(self) -> None:
        playback = _PlaybackStub()
        dummy = SimpleNamespace(
            gemini_first_chunk_timeout=0.05,
            gemini_chunk_idle_timeout=0.05,
            gemini_max_turn_duration=1.0,
            _gemini=_GeminiReceiveTimeout(),
            _playback=playback,
            _streaming_complete=asyncio.Event(),
        )

        result = await VoiceHandler._receive_response_loop(dummy)

        self.assertTrue(result.timed_out)
        self.assertEqual(result.reason, "first_chunk_timeout")
        self.assertEqual(result.chunks_received, 0)
        self.assertTrue(playback.finished)

    async def test_receive_response_loop_idle_timeout_after_first_chunk(self) -> None:
        playback = _PlaybackStub()
        dummy = SimpleNamespace(
            gemini_first_chunk_timeout=0.2,
            gemini_chunk_idle_timeout=0.05,
            gemini_max_turn_duration=1.0,
            _gemini=_GeminiReceiveIdleTimeout(),
            _playback=playback,
            _streaming_complete=asyncio.Event(),
        )

        result = await VoiceHandler._receive_response_loop(dummy)

        self.assertTrue(result.timed_out)
        self.assertEqual(result.reason, "chunk_idle_timeout")
        self.assertEqual(result.chunks_received, 1)
        self.assertEqual(playback.chunks, 1)

    async def test_receive_response_loop_marks_playback_complete_on_error(self) -> None:
        playback = _PlaybackStub()
        dummy = SimpleNamespace(
            gemini_first_chunk_timeout=0.2,
            gemini_chunk_idle_timeout=0.2,
            gemini_max_turn_duration=1.0,
            _gemini=_GeminiReceiveError(),
            _playback=playback,
            _streaming_complete=asyncio.Event(),
        )

        result = await VoiceHandler._receive_response_loop(dummy)

        self.assertTrue(result.had_error)
        self.assertTrue(result.reason.startswith("receive_error:"))
        self.assertTrue(playback.finished)


if __name__ == "__main__":
    unittest.main()
