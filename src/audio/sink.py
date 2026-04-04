"""Discord voice receive sink for wake word detection."""

import asyncio
from typing import Optional, Callable, Awaitable, TYPE_CHECKING, Dict

import discord
from discord.ext import voice_recv

from ..utils.logger import get_logger

try:
    import davey
except ImportError:  # pragma: no cover - discord.py voice requires davey at runtime
    davey = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .capture import AudioCapture

logger = get_logger("audio.sink")


class WakeWordSink(voice_recv.AudioSink):
    """Receive per-user Discord PCM audio and forward it to AudioCapture."""

    def __init__(
        self,
        *,
        capture: Optional["AudioCapture"] = None,
        audio_callback: Optional[Callable[[bytes, int], Awaitable[None]]] = None,
    ):
        super().__init__()
        self._capture = capture
        self._audio_callback = audio_callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunk_count = 0
        self._per_user_chunk_count: Dict[int, int] = {}
        self._opus_decoders: Dict[int, discord.opus.Decoder] = {}
        self._decode_error_count: Dict[int, int] = {}

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # Sink writes occur in a background thread, loop may not exist yet.
            pass

        logger.info("WakeWordSink initialized (voice_recv)")

    def wants_opus(self) -> bool:
        """Request Opus frames and decode manually with DAVE-aware logic."""
        return True

    def set_capture(self, capture: "AudioCapture") -> None:
        """Set the audio capture instance."""
        self._capture = capture
        logger.debug("AudioCapture set on sink")

    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes, int], Awaitable[None]]],
    ) -> None:
        """Set callback for raw audio data."""
        self._audio_callback = callback

    def _ensure_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        if self._loop is not None:
            return self._loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return self._loop

    def _decrypt_opus_for_user(self, user_id: int, opus_payload: bytes) -> bytes:
        """Decrypt DAVE-wrapped Opus payload when available."""
        if not opus_payload:
            return b""

        if davey is None:
            return opus_payload

        voice_client = getattr(self, "voice_client", None)
        connection = getattr(voice_client, "_connection", None)
        dave_session = getattr(connection, "dave_session", None)
        if dave_session is None or not getattr(dave_session, "ready", False):
            return opus_payload

        try:
            decrypted = dave_session.decrypt(user_id, davey.MediaType.audio, opus_payload)
        except Exception as exc:
            # Decrypt errors can happen around transitions. Fall back to original payload.
            count = self._decode_error_count.get(user_id, 0) + 1
            self._decode_error_count[user_id] = count
            if count == 1 or count % 50 == 0:
                logger.warning(
                    f"DAVE decrypt failed for user {user_id} ({count} errors): {exc}"
                )
            return opus_payload

        if isinstance(decrypted, (bytes, bytearray, memoryview)):
            return bytes(decrypted)
        return opus_payload

    def _decode_opus_for_user(self, user_id: int, opus_payload: bytes) -> Optional[bytes]:
        """Decode Opus to PCM and drop corrupt frames instead of killing the listener."""
        if not opus_payload:
            return None

        decoder = self._opus_decoders.get(user_id)
        if decoder is None:
            decoder = discord.opus.Decoder()
            self._opus_decoders[user_id] = decoder

        try:
            return decoder.decode(opus_payload, fec=False)
        except Exception as exc:
            count = self._decode_error_count.get(user_id, 0) + 1
            self._decode_error_count[user_id] = count
            if count == 1 or count % 50 == 0:
                logger.warning(
                    f"Dropping corrupted Opus frame for user {user_id} ({count} errors): {exc}"
                )
            return None

    def _get_pcm_payload(self, user_id: int, data: voice_recv.VoiceData) -> Optional[bytes]:
        """Extract PCM payload from VoiceData, supporting both PCM and Opus modes."""
        # Compatibility path for tests or environments where PCM is already present.
        pcm_data = getattr(data, "pcm", None)
        if isinstance(pcm_data, (bytes, bytearray, memoryview)) and pcm_data:
            return bytes(pcm_data)

        opus_data = getattr(data, "opus", None)
        if not isinstance(opus_data, (bytes, bytearray, memoryview)) or not opus_data:
            return None

        decrypted_opus = self._decrypt_opus_for_user(user_id, bytes(opus_data))
        return self._decode_opus_for_user(user_id, decrypted_opus)

    def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData) -> None:
        """Receive Discord voice frames and enqueue async processing."""
        if user is None:
            return

        user_id = user.id
        payload = self._get_pcm_payload(user_id, data)
        if not payload:
            return

        self._chunk_count += 1
        self._per_user_chunk_count[user_id] = self._per_user_chunk_count.get(user_id, 0) + 1

        if self._per_user_chunk_count[user_id] == 1:
            logger.info(
                f"First audio chunk from user {user_id}: {len(payload)} bytes (Discord audio flowing)"
            )
        elif self._per_user_chunk_count[user_id] % 500 == 1:
            logger.debug(
                f"Audio chunk #{self._per_user_chunk_count[user_id]} from user {user_id}: {len(payload)} bytes"
            )

        loop = self._ensure_loop()
        if loop is None:
            logger.warning("No event loop available, cannot process audio")
            return

        if self._capture is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._capture.process_discord_audio_per_user(payload, user_id, is_stereo=True),
                    loop,
                )
            except Exception as exc:
                logger.error(f"Error scheduling audio processing for user {user_id}: {exc}")

        if self._audio_callback is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._audio_callback(payload, user_id),
                    loop,
                )
            except Exception as exc:
                logger.error(f"Error scheduling audio callback for user {user_id}: {exc}")

    def cleanup(self) -> None:
        """Clean up sink resources."""
        logger.info(f"WakeWordSink cleanup - processed {self._chunk_count} total audio chunks")
        for user_id, count in self._per_user_chunk_count.items():
            logger.debug(f"  User {user_id}: {count} chunks")
        self._per_user_chunk_count.clear()
        self._opus_decoders.clear()
        self._decode_error_count.clear()

    def get_active_users(self) -> list[int]:
        """Get users who have sent audio in this sink session."""
        return list(self._per_user_chunk_count.keys())
