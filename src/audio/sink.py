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
        
        # Log occasionally (per user)
        if self._per_user_chunk_count[user] % 500 == 1:
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
