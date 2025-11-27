"""Custom Discord audio sink for wake word detection."""

import asyncio
from typing import Optional, Callable, Awaitable, TYPE_CHECKING
import discord

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .capture import AudioCapture

logger = get_logger("audio.sink")


class WakeWordSink(discord.sinks.Sink):
    """Custom sink that receives Discord audio and feeds it to wake word detection.
    
    This sink receives raw PCM audio from Discord voice connections and
    forwards it to the AudioCapture system for processing.
    """
    
    def __init__(
        self,
        *,
        capture: Optional["AudioCapture"] = None,
        audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None,
        filters=None,
    ):
        """Initialize the wake word sink.
        
        Args:
            capture: AudioCapture instance to receive audio.
            audio_callback: Optional async callback for raw audio data.
            filters: Optional filters for the sink.
        """
        super().__init__(filters=filters)
        self._capture = capture
        self._audio_callback = audio_callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunk_count = 0
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet, will be set later
            pass
        
        logger.info("WakeWordSink initialized")
    
    def set_capture(self, capture: "AudioCapture") -> None:
        """Set the audio capture instance.
        
        Args:
            capture: AudioCapture instance to receive audio.
        """
        self._capture = capture
        logger.debug("AudioCapture set on sink")
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes], Awaitable[None]]],
    ) -> None:
        """Set callback for raw audio data.
        
        Args:
            callback: Async function to call with raw audio bytes.
        """
        self._audio_callback = callback

    @discord.sinks.Filters.container
    def write(self, data: bytes, user: int) -> None:
        """Receive audio data from Discord.
        
        This method is called by Discord's voice system when audio is received.
        The @Filters.container decorator handles user filtering.
        
        Args:
            data: Raw PCM audio bytes (48kHz, 16-bit, stereo).
            user: User ID that the audio came from.
        """
        self._chunk_count += 1
        
        # Log occasionally
        if self._chunk_count % 500 == 1:  # Log every ~10 seconds (500 * 20ms frames)
            logger.debug(f"Received audio chunk #{self._chunk_count}: {len(data)} bytes from user {user}")
        
        # Schedule async processing in the event loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No event loop available, cannot process audio")
                return
        
        # Process audio asynchronously
        if self._capture is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._capture.process_discord_audio(data, is_stereo=True),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio processing: {e}")
        
        # Call raw audio callback if set
        if self._audio_callback is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._audio_callback(data),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio callback: {e}")
    
    def cleanup(self) -> None:
        """Clean up the sink resources."""
        logger.info(f"WakeWordSink cleanup - processed {self._chunk_count} audio chunks")
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
