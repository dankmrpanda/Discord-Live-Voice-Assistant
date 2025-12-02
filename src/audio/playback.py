"""Audio playback to Discord voice channels with streaming support."""

import asyncio
import io
from collections import deque
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


class StreamingPCMSource(discord.AudioSource):
    """Audio source that streams from a queue for low-latency playback.
    
    This source reads from an asyncio Queue, allowing audio to be played
    as soon as chunks arrive from Gemini, rather than waiting for the
    full response.
    """
    
    # Discord expects 20ms frames of stereo 48kHz 16-bit audio
    FRAME_SIZE = 3840  # 48000 * 2 bytes * 2 channels * 0.02 seconds
    
    def __init__(self, processor: AudioProcessor, volume: float = 1.0):
        """Initialize the streaming audio source.
        
        Args:
            processor: AudioProcessor for format conversion.
            volume: Volume multiplier (0.0 to 2.0).
        """
        self._processor = processor
        self._volume = volume
        self._buffer = bytearray()
        self._chunk_queue: deque[bytes] = deque()
        self._is_finished = False
        self._lock = asyncio.Lock()
    
    def add_chunk(self, gemini_audio: bytes) -> None:
        """Add a chunk of Gemini audio to the playback queue.
        
        Args:
            gemini_audio: PCM audio from Gemini (24kHz mono).
        """
        # Convert to Discord format (48kHz stereo)
        discord_pcm = self._processor.gemini_to_discord(gemini_audio)
        self._chunk_queue.append(discord_pcm)
    
    def mark_finished(self) -> None:
        """Mark that no more chunks will be added."""
        self._is_finished = True
    
    def read(self) -> bytes:
        """Read the next frame of audio.
        
        Returns:
            20ms of audio data, or empty bytes if finished.
        """
        # Fill buffer from queue if needed
        while len(self._buffer) < self.FRAME_SIZE and self._chunk_queue:
            chunk = self._chunk_queue.popleft()
            self._buffer.extend(chunk)
        
        # If buffer has enough data, return a frame
        if len(self._buffer) >= self.FRAME_SIZE:
            frame = bytes(self._buffer[:self.FRAME_SIZE])
            del self._buffer[:self.FRAME_SIZE]
            return frame
        
        # If finished and buffer has remaining data, pad and return
        if self._is_finished:
            if len(self._buffer) > 0:
                frame = bytes(self._buffer)
                self._buffer.clear()
                # Pad to frame size
                if len(frame) < self.FRAME_SIZE:
                    frame += b"\x00" * (self.FRAME_SIZE - len(frame))
                return frame
            return b""
        
        # Not finished but buffer empty - return silence to keep stream alive
        return b"\x00" * self.FRAME_SIZE
    
    def is_opus(self) -> bool:
        """Check if the audio source is Opus encoded."""
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._buffer.clear()
        self._chunk_queue.clear()


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
        
        # Streaming playback support
        self._streaming_source: Optional[StreamingPCMSource] = None
        self._is_streaming = False
    
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
        self._is_streaming = False
        self._streaming_source = None
    
    # =========================================================================
    # Streaming playback methods for low-latency response
    # =========================================================================
    
    def start_streaming_playback(self) -> bool:
        """Start streaming playback mode.
        
        This starts playing immediately and audio chunks can be added
        as they arrive from Gemini.
        
        Returns:
            True if streaming started successfully.
        """
        if not self._voice_client or not self._voice_client.is_connected():
            logger.warning("Cannot start streaming: not connected to voice channel")
            return False
        
        if self._voice_client.is_playing():
            logger.debug("Stopping current playback for streaming")
            self._voice_client.stop()
        
        # Create streaming source
        self._streaming_source = StreamingPCMSource(self.processor, volume=self.volume)
        
        # Set up completion tracking
        self._is_playing = True
        self._is_streaming = True
        self._playback_complete_event = asyncio.Event()
        
        def after_playback(error):
            if error:
                logger.error(f"Streaming playback error: {error}")
            self._is_playing = False
            self._is_streaming = False
            self._streaming_source = None
            if self._playback_complete_event:
                self._playback_complete_event.set()
            if self._after_callback:
                self._after_callback()
        
        # Start playback with streaming source
        self._voice_client.play(self._streaming_source, after=after_playback)
        logger.debug("Started streaming playback")
        return True
    
    def add_streaming_chunk(self, gemini_audio: bytes) -> bool:
        """Add an audio chunk to the streaming playback.
        
        Args:
            gemini_audio: PCM audio from Gemini (24kHz mono).
            
        Returns:
            True if chunk was added successfully.
        """
        if not self._streaming_source or not self._is_streaming:
            logger.warning("Cannot add chunk: not in streaming mode")
            return False
        
        self._streaming_source.add_chunk(gemini_audio)
        return True
    
    def finish_streaming(self) -> None:
        """Mark streaming as complete (no more chunks will be added)."""
        if self._streaming_source:
            self._streaming_source.mark_finished()
            logger.debug("Streaming marked as finished")
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently in streaming playback mode."""
        return self._is_streaming
