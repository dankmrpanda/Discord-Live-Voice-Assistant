"""Custom Discord audio sink for wake word detection with per-user audio processing."""

import asyncio
from typing import Optional, Callable, Awaitable, TYPE_CHECKING, Dict
import discord

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .capture import AudioCapture

logger = get_logger("audio.sink")


class WakeWordSink(discord.sinks.Sink):
    """Custom sink that receives Discord audio and processes per-user wake word detection.
    
    This sink receives raw PCM audio from Discord voice connections and
    forwards it to the AudioCapture system for per-user processing.
    Each user's audio is processed separately to enable proper wake word
    detection with 3+ users in the voice channel.
    """
    
    def __init__(
        self,
        *,
        capture: Optional["AudioCapture"] = None,
        audio_callback: Optional[Callable[[bytes, int], Awaitable[None]]] = None,
        filters=None,
    ):
        """Initialize the wake word sink.
        
        Args:
            capture: AudioCapture instance to receive audio.
            audio_callback: Optional async callback for raw audio data (data, user_id).
            filters: Optional filters for the sink.
        """
        super().__init__(filters=filters)
        self._capture = capture
        self._audio_callback = audio_callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunk_count = 0
        self._per_user_chunk_count: Dict[int, int] = {}
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet, will be set later
            pass
        
        logger.info("WakeWordSink initialized (per-user audio processing enabled)")
    
    def set_capture(self, capture: "AudioCapture") -> None:
        """Set the audio capture instance.
        
        Args:
            capture: AudioCapture instance to receive audio.
        """
        self._capture = capture
        logger.debug("AudioCapture set on sink")
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes, int], Awaitable[None]]],
    ) -> None:
        """Set callback for raw audio data.
        
        Args:
            callback: Async function to call with raw audio bytes and user ID.
        """
        self._audio_callback = callback

    @discord.sinks.Filters.container
    def write(self, data: bytes, user: int) -> None:
        """Receive audio data from Discord for a specific user.
        
        This method is called by Discord's voice system when audio is received.
        The @Filters.container decorator handles user filtering.
        Audio is processed PER-USER to enable proper wake word detection
        even with multiple users speaking.
        
        Args:
            data: Raw PCM audio bytes (48kHz, 16-bit, stereo).
            user: User ID that the audio came from.
        """
        self._chunk_count += 1
        
        # Track per-user chunk counts
        if user not in self._per_user_chunk_count:
            self._per_user_chunk_count[user] = 0
            logger.info(f"New user detected in voice: {user}")
        self._per_user_chunk_count[user] += 1
        
        # Log first chunk from each user and then periodically
        if self._per_user_chunk_count[user] == 1:
            logger.info(f"First audio chunk from user {user}: {len(data)} bytes (Discord audio flowing)")
            # === DEEP DIAGNOSTIC: Verify audio format (stereo vs mono) ===
            try:
                import numpy as _np
                raw_samples = _np.frombuffer(data, dtype=_np.int16)
                n_samples = len(raw_samples)
                # At 48kHz stereo, 20ms = 960 frames × 2 channels = 1920 samples = 3840 bytes
                # At 48kHz stereo, 80ms = 3840 frames × 2 channels = 7680 samples = 15360 bytes
                # At 48kHz mono,  80ms = 3840 frames = 3840 samples = 7680 bytes
                expected_stereo_frames = n_samples // 2  # frame count if stereo
                expected_stereo_duration_ms = (expected_stereo_frames / 48000) * 1000
                expected_mono_duration_ms = (n_samples / 48000) * 1000
                logger.info(f"🔍 AUDIO FORMAT DIAGNOSTIC for user {user}:")
                logger.info(f"   Raw bytes: {len(data)}, int16 samples: {n_samples}")
                logger.info(f"   If STEREO: {expected_stereo_frames} frames, {expected_stereo_duration_ms:.1f}ms @ 48kHz")
                logger.info(f"   If MONO:   {n_samples} frames, {expected_mono_duration_ms:.1f}ms @ 48kHz")
                # Check for stereo: in interleaved stereo, adjacent samples should be
                # correlated (L and R channels of the same instant). In mono, they'd be
                # sequential time samples. Log first 20 raw samples for manual inspection.
                first_20 = raw_samples[:20].tolist()
                logger.info(f"   First 20 int16 values: {first_20}")
                # Stereo test: if stereo, even-indexed and odd-indexed samples are different channels
                # They should have similar magnitude but may differ. If mono, consecutive samples
                # change smoothly (small differences between adjacent samples for speech).
                if n_samples >= 20:
                    even_samples = raw_samples[0:20:2]  # L channel if stereo
                    odd_samples = raw_samples[1:20:2]   # R channel if stereo
                    diff_adjacent = _np.abs(_np.diff(raw_samples[:20].astype(_np.float32)))
                    diff_channels = _np.abs((even_samples - odd_samples).astype(_np.float32))
                    logger.info(f"   Avg diff between adjacent samples: {diff_adjacent.mean():.1f}")
                    logger.info(f"   Avg diff between even/odd (L/R if stereo): {diff_channels.mean():.1f}")
                    logger.info(f"   >> If L/R diff ≈ 0, audio is likely MONO (duplicate channels) or MONO data")
                    logger.info(f"   >> If L/R diff >> adjacent diff, likely TRUE STEREO")
            except Exception as diag_e:
                logger.warning(f"Audio format diagnostic failed: {diag_e}")
        elif self._per_user_chunk_count[user] % 500 == 1:
            logger.debug(f"Audio chunk #{self._per_user_chunk_count[user]} from user {user}: {len(data)} bytes")
        
        # Schedule async processing in the event loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No event loop available, cannot process audio")
                return
        
        # Process audio asynchronously - PER USER
        if self._capture is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._capture.process_discord_audio_per_user(data, user, is_stereo=True),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio processing for user {user}: {e}")
        
        # Call raw audio callback if set (now includes user ID)
        if self._audio_callback is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._audio_callback(data, user),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio callback for user {user}: {e}")
    
    def cleanup(self) -> None:
        """Clean up the sink resources."""
        logger.info(f"WakeWordSink cleanup - processed {self._chunk_count} total audio chunks")
        for user_id, count in self._per_user_chunk_count.items():
            logger.debug(f"  User {user_id}: {count} chunks")
        self._per_user_chunk_count.clear()
        self.finished = True
    
    def get_all_audio(self):
        """Get all recorded audio (required by Sink interface).
        
        We don't store audio, so this returns empty list.
        """
        return []
    
    def format_audio(self, audio):
        """Format audio (required by Sink interface).
        
        We process audio in real-time, so this is a no-op.
        """
        pass
    
    def get_active_users(self) -> list:
        """Get list of users who have sent audio.
        
        Returns:
            List of user IDs that have sent audio.
        """
        return list(self._per_user_chunk_count.keys())
