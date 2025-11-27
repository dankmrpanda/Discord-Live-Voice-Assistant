"""Audio playback to Discord voice channels."""

import asyncio
import io
from typing import Optional, Callable

import discord

from ..utils.logger import get_logger
from .processor import AudioProcessor

logger = get_logger("audio.playback")


class PCMVolumeTransformer(discord.AudioSource):
    """An audio source that reads from PCM data with volume control."""
    
    def __init__(
        self,
        pcm_data: bytes,
        volume: float = 1.0,
    ):
        """Initialize the PCM audio source.
        
        Args:
            pcm_data: Raw PCM audio (48kHz, 16-bit, stereo).
            volume: Volume multiplier (0.0 to 2.0).
        """
        self._pcm_data = pcm_data
        self._position = 0
        self._volume = volume
        # Discord expects 20ms frames of stereo 48kHz 16-bit audio
        # 48000 * 2 bytes * 2 channels * 0.02 seconds = 3840 bytes
        self._frame_size = 3840
    
    def read(self) -> bytes:
        """Read the next frame of audio.
        
        Returns:
            20ms of audio data, or empty bytes if finished.
        """
        if self._position >= len(self._pcm_data):
            return b""
        
        end = self._position + self._frame_size
        frame = self._pcm_data[self._position:end]
        self._position = end
        
        # Pad if needed
        if len(frame) < self._frame_size:
            frame += b"\x00" * (self._frame_size - len(frame))
        
        return frame
    
    def is_opus(self) -> bool:
        """Check if the audio source is Opus encoded."""
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._pcm_data = b""


class AudioPlayback:
    """Manages audio playback to Discord voice channels."""
    
    def __init__(
        self,
        processor: AudioProcessor,
        volume: float = 1.0,
    ):
        """Initialize the audio playback.
        
        Args:
            processor: AudioProcessor instance for format conversion.
            volume: Default volume (0.0 to 2.0).
        """
        self.processor = processor
        self.volume = volume
        
        self._voice_client: Optional[discord.VoiceClient] = None
        self._is_playing = False
        self._playback_complete_event: Optional[asyncio.Event] = None
        self._after_callback: Optional[Callable[[], None]] = None
    
    def set_voice_client(self, voice_client: Optional[discord.VoiceClient]) -> None:
        """Set the Discord voice client for playback.
        
        Args:
            voice_client: Discord VoiceClient instance.
        """
        self._voice_client = voice_client
    
    def set_after_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback to run after playback completes.
        
        Args:
            callback: Function to call when playback finishes.
        """
        self._after_callback = callback
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
    
    async def play_gemini_audio(self, pcm_data: bytes) -> None:
        """Play audio received from Gemini.
        
        Args:
            pcm_data: Raw PCM audio from Gemini (24kHz, mono).
        """
        if not self._voice_client or not self._voice_client.is_connected():
            logger.warning("Cannot play audio: not connected to voice channel")
            return
        
        if self._voice_client.is_playing():
            logger.debug("Stopping current playback for new audio")
            self._voice_client.stop()
        
        # Convert Gemini format to Discord format
        discord_pcm = self.processor.gemini_to_discord(pcm_data)
        
        # Create audio source
        source = PCMVolumeTransformer(discord_pcm, volume=self.volume)
        
        # Set up completion tracking
        self._is_playing = True
        self._playback_complete_event = asyncio.Event()
        
        def after_playback(error):
            if error:
                logger.error(f"Playback error: {error}")
            self._is_playing = False
            if self._playback_complete_event:
                self._playback_complete_event.set()
            if self._after_callback:
                self._after_callback()
        
        # Start playback
        self._voice_client.play(source, after=after_playback)
        logger.debug(f"Started playback of {len(discord_pcm)} bytes")
    
    async def wait_for_playback(self, timeout: Optional[float] = None) -> bool:
        """Wait for current playback to complete.
        
        Args:
            timeout: Maximum time to wait (seconds).
            
        Returns:
            True if playback completed, False if timed out.
        """
        if not self._playback_complete_event:
            return True
        
        try:
            await asyncio.wait_for(
                self._playback_complete_event.wait(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def stop(self) -> None:
        """Stop current playback."""
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()
        self._is_playing = False
