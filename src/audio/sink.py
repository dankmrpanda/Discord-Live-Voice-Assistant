"""Audio stats and diagnostics helper for per-user audio processing."""

import time as _time
import wave
from pathlib import Path
from typing import Dict

from ..utils.logger import get_logger

logger = get_logger("audio.sink")


class WakeWordSink:
    """Helper class that tracks audio stats and handles raw audio dumping.

    This is NOT a discord-ext-voice-recv AudioSink subclass. Instead, the
    VoiceHandler uses voice_recv.BasicSink(callback) for audio reception
    and forwards audio here for stats tracking and diagnostics.

    Each user's audio is tracked separately to enable proper wake word
    detection with 3+ users in the voice channel.
    """

    def __init__(
        self,
        *,
        dump_raw_audio: bool = False,
        dump_audio_dir: str = "out/raw_audio",
    ):
        """Initialize the audio stats helper.

        Args:
            dump_raw_audio: If True, save raw Discord PCM to WAV files per user.
            dump_audio_dir: Directory to write raw audio dumps to.
        """
        self._chunk_count = 0
        self._per_user_chunk_count: Dict[int, int] = {}
        self._total_bytes = 0
        self._last_stats_time: float = 0.0
        self._last_stats_chunks: int = 0
        self._last_stats_bytes: int = 0
        self._first_write_logged = False

        # Raw audio dump
        self._dump_raw_audio = dump_raw_audio
        self._dump_audio_dir = dump_audio_dir
        self._dump_buffers: Dict[int, list[bytes]] = {}
        self._dump_session_ts = _time.strftime("%Y%m%d_%H%M%S")

        if self._dump_raw_audio:
            logger.info(f"WakeWordSink: raw audio dump ENABLED -> {self._dump_audio_dir}")

        logger.info("WakeWordSink initialized (per-user audio processing enabled)")

    def handle_audio(self, pcm_data: bytes, user_id: int) -> None:
        """Process an audio chunk for stats and diagnostics.

        Called from the VoiceHandler's BasicSink callback on the
        PacketRouter thread. This method must be thread-safe and fast.

        Args:
            pcm_data: Raw PCM audio bytes (48kHz, 16-bit, stereo from Discord).
            user_id: Discord user ID this audio came from (0 if unknown).
        """
        # Log the very first chunk for debugging "no audio" issues
        if not self._first_write_logged:
            self._first_write_logged = True
            logger.info(
                f"[sink] FIRST audio chunk received: user={user_id}, "
                f"size={len(pcm_data)} bytes"
            )

        self._chunk_count += 1
        self._total_bytes += len(pcm_data)

        # Track per-user chunk counts
        if user_id not in self._per_user_chunk_count:
            self._per_user_chunk_count[user_id] = 0
            logger.info(f"New user detected in voice: {user_id}")
        self._per_user_chunk_count[user_id] += 1

        # Raw audio dump: collect chunks per user
        if self._dump_raw_audio:
            if user_id not in self._dump_buffers:
                self._dump_buffers[user_id] = []
            self._dump_buffers[user_id].append(pcm_data)

        # Per-second throughput stats (log every ~1 s worth of chunks)
        now = _time.monotonic()
        if self._last_stats_time == 0.0:
            self._last_stats_time = now
        elif now - self._last_stats_time >= 1.0:
            d_chunks = self._chunk_count - self._last_stats_chunks
            d_bytes = self._total_bytes - self._last_stats_bytes
            logger.debug(
                f"[sink] {d_chunks} chunks/s, {d_bytes:,} B/s, "
                f"users={len(self._per_user_chunk_count)}"
            )
            self._last_stats_time = now
            self._last_stats_chunks = self._chunk_count
            self._last_stats_bytes = self._total_bytes

    def cleanup(self) -> None:
        """Clean up the sink resources and flush dumps."""
        logger.info(f"WakeWordSink cleanup - processed {self._chunk_count} total audio chunks")

        if self._chunk_count == 0:
            logger.warning(
                "[sink] WARNING: Zero audio chunks received from Discord during this session. "
                "Possible causes: voice_recv not delivering audio, bot lacks permissions, "
                "or no users were speaking."
            )

        for uid, count in self._per_user_chunk_count.items():
            logger.debug(f"  User {uid}: {count} chunks")

        # Flush raw audio dumps
        if self._dump_raw_audio:
            self._flush_audio_dumps()

        self._per_user_chunk_count.clear()
        self._dump_buffers.clear()

    def _flush_audio_dumps(self) -> None:
        """Write accumulated raw audio to WAV files (one per user)."""
        if not self._dump_buffers:
            logger.info("[sink] No raw audio to dump (no chunks received)")
            return

        dump_dir = Path(self._dump_audio_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

        for user_id, chunks in self._dump_buffers.items():
            pcm = b"".join(chunks)
            if not pcm:
                continue

            filename = f"user_{user_id}_{self._dump_session_ts}.wav"
            filepath = dump_dir / filename

            # Discord sends 48kHz, 16-bit, stereo PCM
            sample_rate = 48000
            channels = 2
            sample_width = 2
            duration = len(pcm) / (sample_rate * channels * sample_width)

            try:
                with wave.open(str(filepath), "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm)
                logger.info(
                    f"[sink] Dumped raw audio: {filepath} "
                    f"({len(chunks)} chunks, {len(pcm):,} bytes, {duration:.1f}s)"
                )
            except Exception as e:
                logger.error(f"[sink] Failed to write audio dump {filepath}: {e}")

    def get_active_users(self) -> list:
        """Get list of users who have sent audio.

        Returns:
            List of user IDs that have sent audio.
        """
        return list(self._per_user_chunk_count.keys())
