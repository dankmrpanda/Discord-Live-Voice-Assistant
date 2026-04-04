"""Tests for Gemini live response parsing logic."""

from types import SimpleNamespace
import unittest

from src.ai.gemini_client import GeminiLiveClient, GeminiSessionState


class _FakeSession:
    def __init__(self, messages):
        self._messages = messages

    async def receive(self):
        for message in self._messages:
            yield message


class GeminiParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_receive_responses_parses_output_transcription_field(self) -> None:
        audio_bytes = b"\x01\x02\x03\x04"
        part = SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))
        model_turn = SimpleNamespace(parts=[part])
        transcription = SimpleNamespace(text="hello world", transcriptions=None)
        server_content = SimpleNamespace(
            model_turn=model_turn,
            output_transcription=transcription,
            turn_complete=True,
            generation_complete=False,
            interrupted=False,
        )
        message = SimpleNamespace(server_content=server_content, text=None)

        text_events: list[str] = []
        completion_calls = {"count": 0}

        async def _on_text(text: str) -> None:
            text_events.append(text)

        async def _on_completion() -> None:
            completion_calls["count"] += 1

        updates = {"count": 0}

        def _update_activity() -> None:
            updates["count"] += 1

        dummy = SimpleNamespace(
            _session=_FakeSession([message]),
            _audio_buffer=[],
            _audio_callback=None,
            _text_callback=_on_text,
            _completion_callback=_on_completion,
            _update_activity=_update_activity,
            _record_error=lambda: None,
            _state=GeminiSessionState.STREAMING,
        )

        chunks = []
        async for chunk in GeminiLiveClient.receive_responses(dummy):
            chunks.append(chunk)

        self.assertEqual(chunks, [audio_bytes])
        self.assertIn("hello world", text_events)
        self.assertEqual(completion_calls["count"], 1)
        self.assertGreaterEqual(updates["count"], 1)
        self.assertEqual(dummy._state, GeminiSessionState.CONNECTED)

    async def test_receive_responses_fallback_output_audio_transcription_field(self) -> None:
        audio_bytes = b"\xAA\xBB"
        part = SimpleNamespace(inline_data=SimpleNamespace(data=audio_bytes))
        model_turn = SimpleNamespace(parts=[part])
        transcription_items = [SimpleNamespace(text="hello"), SimpleNamespace(text="there")]
        old_transcription = SimpleNamespace(transcriptions=transcription_items, text=None)
        server_content = SimpleNamespace(
            model_turn=model_turn,
            output_audio_transcription=old_transcription,
            turn_complete=True,
            generation_complete=False,
            interrupted=False,
        )
        message = SimpleNamespace(server_content=server_content, text=None)

        text_events: list[str] = []

        async def _on_text(text: str) -> None:
            text_events.append(text)

        dummy = SimpleNamespace(
            _session=_FakeSession([message]),
            _audio_buffer=[],
            _audio_callback=None,
            _text_callback=_on_text,
            _completion_callback=None,
            _update_activity=lambda: None,
            _record_error=lambda: None,
            _state=GeminiSessionState.STREAMING,
        )

        chunks = []
        async for chunk in GeminiLiveClient.receive_responses(dummy):
            chunks.append(chunk)

        self.assertEqual(chunks, [audio_bytes])
        self.assertIn("hello there", text_events)


if __name__ == "__main__":
    unittest.main()
