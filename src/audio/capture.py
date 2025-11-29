"""Audio capture from Discord voice channels with per-user support."""

import asyncio
from collections import deque
from typing import Optional, Callable, Awaitable, Dict
import numpy as np

from ..utils.logger import get_logger
from .processor import AudioProcessor

logger = get_logger("audio.capture")


class AudioCapture:
    """Captures and buffers audio from Discord voice channels.
    
    This class manages incoming audio from Discord, converts it to the
    format needed by wake word detection and Gemini, and provides
    buffering for continuous audio processing.
    
    Supports per-user audio buffers for accurate wake word detection
    when multiple users are in the voice channel.
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
        
        # Shared audio buffer (legacy, for mixed audio)
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        
        # Per-user audio buffers for wake word detection
        self._user_buffers: Dict[int, deque] = {}
        self._user_buffer_samples: Dict[int, int] = {}
        self._user_locks: Dict[int, asyncio.Lock] = {}
        
        # Track which user triggered wake word (for prompt capture)
        self._active_user_id: Optional[int] = None
        
        # Callback for processed audio chunks (takes user_id now)
        self._audio_callback: Optional[Callable[[bytes, int], Awaitable[None]]] = None
        
        # State
        self._is_capturing = False
        self._lock = asyncio.Lock()
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes, int], Awaitable[None]]],
    ) -> None:
        """Set callback for processed audio chunks.
        
        The callback receives 16kHz mono PCM audio chunks and the user ID.
        
        Args:
            callback: Async function to call with each audio chunk and user_id.
        """
        self._audio_callback = callback
    
    def set_active_user(self, user_id: Optional[int]) -> None:
        """Set the active user whose audio should be captured for Gemini.
        
        Args:
            user_id: Discord user ID, or None to clear.
        """
        self._active_user_id = user_id
        if user_id:
            logger.info(f"Active user set to {user_id} - only capturing their audio for prompt")
        else:
            logger.debug("Active user cleared")
    
    def get_active_user(self) -> Optional[int]:
        """Get the currently active user ID.
        
        Returns:
            The active user's Discord ID, or None if no user is active.
        """
        return self._active_user_id
    
    def _get_user_lock(self, user_id: int) -> asyncio.Lock:
        """Get or create a lock for a specific user's buffer.
        
        Args:
            user_id: Discord user ID.
            
        Returns:
            asyncio.Lock for this user's buffer.
        """
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]
    
    def _ensure_user_buffer(self, user_id: int) -> None:
        """Ensure a buffer exists for the given user.
        
        Args:
            user_id: Discord user ID.
        """
        if user_id not in self._user_buffers:
            self._user_buffers[user_id] = deque()
            self._user_buffer_samples[user_id] = 0
            logger.debug(f"Created audio buffer for user {user_id}")
    
    async def start(self) -> None:
        """Start capturing audio."""
        self._is_capturing = True
        logger.info("Audio capture started")
    
    async def stop(self) -> None:
        """Stop capturing audio and clear all buffers."""
        self._is_capturing = False
        async with self._lock:
            # Clear shared buffer
            self._buffer.clear()
            self._buffer_samples = 0
            
            # Clear all user buffers
            for user_id in list(self._user_buffers.keys()):
                self._user_buffers[user_id].clear()
                self._user_buffer_samples[user_id] = 0
            
            self._active_user_id = None
        logger.info("Audio capture stopped")
    
    async def process_discord_audio_per_user(
        self,
        pcm_data: bytes,
        user_id: int,
        is_stereo: bool = True,
    ) -> None:
        """Process incoming audio from a specific Discord user.
        
        This method maintains separate buffers per user to enable accurate
        wake word detection when multiple users are in the voice channel.
        
        Args:
            pcm_data: Raw PCM audio from Discord.
            user_id: Discord user ID this audio came from.
            is_stereo: Whether the audio is stereo.
        """
        if not self._is_capturing:
            return
        
        # Convert to Gemini format (16kHz mono)
        gemini_pcm = self.processor.discord_to_gemini(pcm_data, is_stereo)
        audio = self.processor.pcm_to_numpy(gemini_pcm)
        
        # Get or create user's buffer
        self._ensure_user_buffer(user_id)
        user_lock = self._get_user_lock(user_id)
        
        async with user_lock:
            # Add to user's buffer
            self._user_buffers[user_id].append(audio)
            self._user_buffer_samples[user_id] += len(audio)
            
            # Trim buffer if too large
            while self._user_buffer_samples[user_id] > self._max_buffer_samples:
                removed = self._user_buffers[user_id].popleft()
                self._user_buffer_samples[user_id] -= len(removed)
        
        # Also add to shared buffer if this is the active user (for Gemini streaming)
        if self._active_user_id is not None and user_id == self._active_user_id:
            async with self._lock:
                self._buffer.append(audio)
                self._buffer_samples += len(audio)
                
                # Trim shared buffer if too large
                while self._buffer_samples > self._max_buffer_samples:
                    removed = self._buffer.popleft()
                    self._buffer_samples -= len(removed)
        
        # Call callback with converted audio and user ID
        if self._audio_callback:
            await self._audio_callback(gemini_pcm, user_id)
    
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
        
        # Call callback with converted audio (legacy - no user_id)
        if self._audio_callback:
            # Try to call with user_id=0 for backwards compatibility
            try:
                await self._audio_callback(gemini_pcm, 0)
            except TypeError:
                # Old callback signature without user_id
                await self._audio_callback(gemini_pcm)
    
    async def get_recent_audio(self, duration: float, user_id: Optional[int] = None) -> bytes:
        """Get the most recent audio from the buffer.
        
        Args:
            duration: Duration of audio to retrieve (seconds).
            user_id: If provided, get audio from this user's buffer only.
            
        Returns:
            PCM audio bytes (16kHz mono).
        """
        samples_needed = int(self.processor.gemini_input_sample_rate * duration)
        
        if user_id is not None and user_id in self._user_buffers:
            user_lock = self._get_user_lock(user_id)
            async with user_lock:
                if self._user_buffer_samples.get(user_id, 0) == 0:
                    return b""
                
                all_samples = np.concatenate(list(self._user_buffers[user_id]))
                if len(all_samples) > samples_needed:
                    all_samples = all_samples[-samples_needed:]
                
                return self.processor.numpy_to_pcm(all_samples)
        
        # Fall back to shared buffer
        async with self._lock:
            if self._buffer_samples == 0:
                return b""
            
            # Collect samples from buffer
            all_samples = np.concatenate(list(self._buffer))
            
            # Get the most recent samples
            if len(all_samples) > samples_needed:
                all_samples = all_samples[-samples_needed:]
            
            return self.processor.numpy_to_pcm(all_samples)
    
    async def get_chunk_for_wake_word_per_user(self, user_id: int) -> Optional[bytes]:
        """Get audio chunk from a specific user for wake word detection.
        
        Args:
            user_id: Discord user ID.
            
        Returns:
            PCM audio bytes (16kHz mono), or None if buffer is empty.
        """
        if user_id not in self._user_buffers:
            return None
        
        user_lock = self._get_user_lock(user_id)
        async with user_lock:
            buffer_samples = self._user_buffer_samples.get(user_id, 0)
            if buffer_samples < self._samples_per_chunk:
                return None
            
            # Collect enough samples
            all_samples = np.concatenate(list(self._user_buffers[user_id]))
            
            # Return the most recent chunk as bytes
            chunk = all_samples[-self._samples_per_chunk:]
            return self.processor.numpy_to_pcm(chunk)
    
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
    
    def clear_buffer(self, user_id: Optional[int] = None) -> None:
        """Clear the audio buffer synchronously.
        
        Args:
            user_id: If provided, clear only this user's buffer. Otherwise clear shared buffer.
        """
        if user_id is not None and user_id in self._user_buffers:
            self._user_buffers[user_id].clear()
            self._user_buffer_samples[user_id] = 0
            logger.debug(f"Cleared audio buffer for user {user_id}")
        else:
            self._buffer.clear()
            self._buffer_samples = 0
    
    def clear_all_user_buffers(self) -> None:
        """Clear all per-user buffers."""
        for user_id in list(self._user_buffers.keys()):
            self._user_buffers[user_id].clear()
            self._user_buffer_samples[user_id] = 0
        logger.debug("Cleared all user audio buffers")
    
    def cleanup_user(self, user_id: int) -> None:
        """Clean up resources for a user who left the channel.
        
        Args:
            user_id: Discord user ID to clean up.
        """
        if user_id in self._user_buffers:
            del self._user_buffers[user_id]
        if user_id in self._user_buffer_samples:
            del self._user_buffer_samples[user_id]
        if user_id in self._user_locks:
            del self._user_locks[user_id]
        
        if self._active_user_id == user_id:
            self._active_user_id = None
            logger.info(f"Active user {user_id} left, cleared active user")
        
        logger.debug(f"Cleaned up audio resources for user {user_id}")
    
    def get_active_users(self) -> list[int]:
        """Get list of users with active audio buffers.
        
        Returns:
            List of user IDs with audio buffers.
        """
        return list(self._user_buffers.keys())
    
    def get_user_buffer_duration(self, user_id: int) -> float:
        """Get the duration of audio buffered for a user.
        
        Args:
            user_id: Discord user ID.
            
        Returns:
            Duration in seconds, or 0 if user has no buffer.
        """
        samples = self._user_buffer_samples.get(user_id, 0)
        return samples / self.processor.gemini_input_sample_rate
