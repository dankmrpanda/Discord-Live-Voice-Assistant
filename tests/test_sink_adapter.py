"""Tests for voice_recv sink adapter behavior."""

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
    from src.audio.sink import WakeWordSink


class _CaptureRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, bool]] = []

    async def process_discord_audio_per_user(self, pcm_data: bytes, user_id: int, is_stereo: bool = True) -> None:
        self.calls.append((pcm_data, user_id, is_stereo))


@unittest.skipUnless(HAS_DISCORD and HAS_VOICE_RECV, "discord.py + discord-ext-voice-recv are required")
class SinkAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_sink_routes_pcm_to_capture_per_user(self) -> None:
        capture = _CaptureRecorder()
        sink = WakeWordSink(capture=capture)

        user = SimpleNamespace(id=1234)
        voice_data = SimpleNamespace(pcm=b"\x00" * 3840)
        sink.write(user, voice_data)

        await asyncio.sleep(0.05)

        self.assertEqual(len(capture.calls), 1)
        payload, user_id, is_stereo = capture.calls[0]
        self.assertEqual(user_id, 1234)
        self.assertEqual(payload, b"\x00" * 3840)
        self.assertTrue(is_stereo)

    async def test_sink_ignores_missing_pcm_payload(self) -> None:
        capture = _CaptureRecorder()
        sink = WakeWordSink(capture=capture)

        user = SimpleNamespace(id=999)
        voice_data = SimpleNamespace(pcm=None)
        sink.write(user, voice_data)

        await asyncio.sleep(0.05)
        self.assertEqual(capture.calls, [])

    async def test_sink_routes_opus_after_manual_decode(self) -> None:
        capture = _CaptureRecorder()
        sink = WakeWordSink(capture=capture)
        sink._decrypt_opus_for_user = lambda user_id, payload: b"decrypted-opus"  # type: ignore[method-assign]
        sink._decode_opus_for_user = lambda user_id, payload: b"\x01" * 3840  # type: ignore[method-assign]

        user = SimpleNamespace(id=555)
        voice_data = SimpleNamespace(pcm=None, opus=b"encrypted")
        sink.write(user, voice_data)

        await asyncio.sleep(0.05)
        self.assertEqual(len(capture.calls), 1)
        payload, user_id, is_stereo = capture.calls[0]
        self.assertEqual(user_id, 555)
        self.assertEqual(payload, b"\x01" * 3840)
        self.assertTrue(is_stereo)

    async def test_sink_drops_opus_frame_when_decode_fails(self) -> None:
        capture = _CaptureRecorder()
        sink = WakeWordSink(capture=capture)
        sink._decrypt_opus_for_user = lambda user_id, payload: payload  # type: ignore[method-assign]
        sink._decode_opus_for_user = lambda user_id, payload: None  # type: ignore[method-assign]

        user = SimpleNamespace(id=123)
        voice_data = SimpleNamespace(pcm=None, opus=b"bad-opus")
        sink.write(user, voice_data)

        await asyncio.sleep(0.05)
        self.assertEqual(capture.calls, [])


if __name__ == "__main__":
    unittest.main()
