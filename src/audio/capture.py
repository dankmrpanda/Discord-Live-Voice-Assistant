"""Audio capture from Discord voice channels."""

import asyncio
from collections import deque
from typing import Optional, Callable, Awaitable
import numpy as np

from ..utils.logger import get_logger
from .processor import AudioProcessor

logger = get_logger("audio.capture")


class AudioCapture:
    """Captures and buffers audio from Discord voice channels.
    
    This class manages incoming audio from Discord, converts it to the
    format needed by wake word detection and Gemini, and provides
    buffering for continuous audio processing.
    """
    
    def __init__(
        self,
        processor: AudioProcessor,
        buffer_duration: float = 30.0,
        chunk_duration: float = 0.1,
    ):
        """Initialize the audio capture.
        
        Args:
            processor: AudioProcessor instance for format conversion.
            buffer_duration: Maximum duration of audio to buffer (seconds).
            chunk_duration: Duration of each audio chunk for processing (seconds).
        """
        self.processor = processor
        self.buffer_duration = buffer_duration
        self.chunk_duration = chunk_duration
        
        # Calculate buffer sizes
        self._samples_per_chunk = int(
            processor.gemini_input_sample_rate * chunk_duration
        )
        self._max_buffer_samples = int(
            processor.gemini_input_sample_rate * buffer_duration
        )
        
        # Audio buffer (stores 16kHz mono PCM for wake word / Gemini)
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        
        # Callback for processed audio chunks
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        
        # State
        self._is_capturing = False
        self._lock = asyncio.Lock()
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes], Awaitable[None]]],
    ) -> None:
        """Set callback for processed audio chunks.
        
        The callback receives 16kHz mono PCM audio chunks.
        
        Args:
            callback: Async function to call with each audio chunk.
        """
        self._audio_callback = callback
    
    async def start(self) -> None:
        """Start capturing audio."""
        self._is_capturing = True
        logger.info("Audio capture started")
    
    async def stop(self) -> None:
        """Stop capturing audio and clear buffer."""
        self._is_capturing = False
        async with self._lock:
            self._buffer.clear()
            self._buffer_samples = 0
        logger.info("Audio capture stopped")
    
    async def process_discord_audio(
        self,
        pcm_data: bytes,
        is_stereo: bool = True,
    ) -> None:
        """Process incoming audio from Discord.
        
        Converts Discord format (48kHz stereo) to Gemini format (16kHz mono)
        and stores in buffer.
        
        Args:
            pcm_data: Raw PCM audio from Discord.
            is_stereo: Whether the audio is stereo.
        """
        if not self._is_capturing:
            return
        
        # Convert to Gemini format
        gemini_pcm = self.processor.discord_to_gemini(pcm_data, is_stereo)
        audio = self.processor.pcm_to_numpy(gemini_pcm)
        
        async with self._lock:
            # Add to buffer
            self._buffer.append(audio)
            self._buffer_samples += len(audio)
            
            # Trim buffer if too large
            while self._buffer_samples > self._max_buffer_samples:
                removed = self._buffer.popleft()
                self._buffer_samples -= len(removed)
        
        # Call callback with converted audio
        if self._audio_callback:
            await self._audio_callback(gemini_pcm)
    
    async def get_recent_audio(self, duration: float) -> bytes:
        """Get the most recent audio from the buffer.
        
        Args:
            duration: Duration of audio to retrieve (seconds).
            
        Returns:
            PCM audio bytes (16kHz mono).
        """
        samples_needed = int(self.processor.gemini_input_sample_rate * duration)
        
        async with self._lock:
            if self._buffer_samples == 0:
                return b""
            
            # Collect samples from buffer
            all_samples = np.concatenate(list(self._buffer))
            
            # Get the most recent samples
            if len(all_samples) > samples_needed:
                all_samples = all_samples[-samples_needed:]
            
            return self.processor.numpy_to_pcm(all_samples)
    
    async def get_chunk_for_wake_word(self) -> Optional[np.ndarray]:
        """Get audio chunk sized for wake word detection.
        
        Returns:
            Numpy array of audio samples, or None if buffer is empty.
        """
        async with self._lock:
            if self._buffer_samples < self._samples_per_chunk:
                return None
            
            # Collect enough samples
            all_samples = np.concatenate(list(self._buffer))
            
            # Return the most recent chunk
            return all_samples[-self._samples_per_chunk:]
    
    async def get_and_consume_chunk(self) -> Optional[np.ndarray]:
        """Get all accumulated audio and clear buffer (for streaming to Gemini).
        
        This returns all audio accumulated since the last call and clears the buffer,
        preventing re-sending the same audio data during real-time streaming.
        
        Returns:
            Numpy array of all accumulated audio samples, or None if buffer is empty.
        """
        async with self._lock:
            if self._buffer_samples == 0:
                return None
            
            # Collect all samples in buffer
            all_samples = np.concatenate(list(self._buffer))
            
            # Clear the buffer after consuming
            self._buffer.clear()
            self._buffer_samples = 0
            
            return all_samples
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer synchronously."""
        self._buffer.clear()
        self._buffer_samples = 0
