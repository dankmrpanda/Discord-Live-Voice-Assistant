"""Audio capture from Discord voice channels with per-user support."""

import asyncio
import threading
import time as _time
from collections import deque
from typing import Optional, Callable, Awaitable, Dict
import numpy as np

from ..utils.logger import get_logger
from .processor import AudioProcessor
from .contract import GEMINI_INPUT_FORMAT, validate_pcm_chunk, compute_audio_stats

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.audio_logger import AudioDebugLogger

logger = get_logger("audio.capture")

# VAD/Energy detection constants
VAD_ENERGY_THRESHOLD = 0.01  # RMS energy threshold for speech detection
DEFAULT_SILENCE_DURATION = 0.5  # Default seconds of silence before ending (configurable)
FRAME_DURATION_MS = 20  # Approximate duration of each Discord audio frame in ms
MIN_SPEECH_DURATION = 1.0  # Minimum seconds of speech before allowing silence detection
WAKE_WORD_GRACE_PERIOD = 0.5  # Seconds to ignore audio after wake word (skip the wake word tail)


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
        silence_threshold: float = DEFAULT_SILENCE_DURATION,
        audio_debug_logger: "AudioDebugLogger | None" = None,
    ):
        """Initialize the audio capture.
        
        Args:
            processor: AudioProcessor instance for format conversion.
            buffer_duration: Maximum duration of audio to buffer (seconds).
            chunk_duration: Duration of each audio chunk for processing (seconds).
            silence_threshold: Seconds of silence before ending capture (configurable).
        """
        self.processor = processor
        self.buffer_duration = buffer_duration
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        
        # Calculate VAD silence frames from threshold
        # Each frame is ~20ms, so frames = threshold_seconds / 0.02
        self._vad_silence_frames = int(silence_threshold / (FRAME_DURATION_MS / 1000))
        logger.debug(f"VAD configured: silence_threshold={silence_threshold}s -> {self._vad_silence_frames} frames")
        
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
        self._user_locks_lock = threading.Lock()  # Thread-safe access to user locks dict
        
        # Track which user triggered wake word (for prompt capture)
        self._active_user_id: Optional[int] = None
        
        # Callback for processed audio chunks (takes user_id now)
        self._audio_callback: Optional[Callable[[bytes, int], Awaitable[None]]] = None
        
        # Real-time streaming ring buffer for Gemini (low-latency path)
        # Uses deque with maxlen as a ring buffer - oldest frames are discarded when full
        # This prevents frame drops of NEW audio while still bounding memory usage
        self._streaming_buffer: deque[bytes] = deque(maxlen=100)
        self._streaming_buffer_lock = asyncio.Lock()
        self._streaming_data_available = asyncio.Event()
        self._is_streaming_to_gemini = False

        # --- Streaming pipeline metrics ---
        self._streaming_drops = 0
        self._streaming_total_pushed = 0
        self._streaming_total_consumed = 0
        self._streaming_total_bytes_pushed = 0
        self._streaming_seq = 0  # monotonic sequence number per streaming session
        
        # VAD state for silence detection
        self._consecutive_silent_frames = 0
        self._speech_detected = False
        self._silence_detected_event: Optional[asyncio.Event] = None
        
        # Timing state for minimum speech duration and grace period
        self._streaming_start_time: float = 0.0
        self._speech_start_time: float = 0.0  # When speech first started (after streaming)
        
        # State
        self._is_capturing = False
        self._lock = asyncio.Lock()
        
        # Audio debug logger (optional)
        self._audio_debug_logger = audio_debug_logger

        # Contract validation toggle (set from config.diagnostics.audio_contract_validation)
        self.contract_validation: bool = False
    
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
        with self._user_locks_lock:
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
        
        When streaming mode is active and this is the active user, frames are
        pushed directly to the streaming queue for low-latency Gemini streaming.
        
        Args:
            pcm_data: Raw PCM audio from Discord.
            user_id: Discord user ID this audio came from.
            is_stereo: Whether the audio is stereo.
        """
        if not self._is_capturing:
            return
        
        # Convert to Gemini format (16kHz mono)
        gemini_pcm = self.processor.discord_to_gemini(pcm_data, is_stereo)

        # Optional contract validation (toggled by diagnostics config)
        if self.contract_validation:
            validate_pcm_chunk(
                gemini_pcm, GEMINI_INPUT_FORMAT,
                label=f"capture_user_{user_id}", warn_only=True,
            )

        audio = self.processor.pcm_to_numpy(gemini_pcm)
        
        # Audio debug logging: stage 1 (raw) and stage 2 (converted)
        if self._audio_debug_logger and self._audio_debug_logger.enabled:
            self._audio_debug_logger.log_raw_discord(pcm_data, user_id)
            self._audio_debug_logger.log_converted(gemini_pcm, user_id)
        
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
        
        # Real-time streaming path for active user
        if self._active_user_id is not None and user_id == self._active_user_id:
            # If streaming mode is active, push directly to queue (no concatenation!)
            if self._is_streaming_to_gemini:
                import time
                current_time = time.time()
                
                # Grace period: skip VAD processing during the wake word tail
                time_since_start = current_time - self._streaming_start_time
                in_grace_period = time_since_start < WAKE_WORD_GRACE_PERIOD
                
                if not in_grace_period:
                    # Perform VAD on this frame (after grace period)
                    rms_energy = self._compute_rms_energy(audio)
                    is_speech = rms_energy > VAD_ENERGY_THRESHOLD
                    
                    if is_speech:
                        if not self._speech_detected:
                            # First speech detected after grace period
                            self._speech_start_time = current_time
                            logger.debug(f"VAD: Speech started (after {time_since_start:.2f}s)")
                        self._speech_detected = True
                        self._consecutive_silent_frames = 0
                    else:
                        if self._speech_detected:
                            self._consecutive_silent_frames += 1
                            # Check if we've detected end of speech
                            if self._consecutive_silent_frames >= self._vad_silence_frames:
                                # Only trigger if we've had minimum speech duration
                                speech_duration = current_time - self._speech_start_time if self._speech_start_time > 0 else 0
                                if speech_duration >= MIN_SPEECH_DURATION:
                                    logger.debug(f"VAD: Silence detected after {self._consecutive_silent_frames} frames ({speech_duration:.2f}s of speech)")
                                    if self._silence_detected_event:
                                        self._silence_detected_event.set()
                                else:
                                    # Not enough speech yet, reset and keep listening
                                    logger.debug(f"VAD: Silence detected but only {speech_duration:.2f}s of speech (need {MIN_SPEECH_DURATION}s), resetting")
                                    self._consecutive_silent_frames = 0
                                    self._speech_detected = False
                                    self._speech_start_time = 0.0
                
                # Push frame directly to streaming ring buffer
                # deque with maxlen automatically discards oldest if full
                async with self._streaming_buffer_lock:
                    if len(self._streaming_buffer) == self._streaming_buffer.maxlen:
                        self._streaming_drops += 1
                        if self._streaming_drops % 50 == 1:
                            logger.warning(
                                f"⚠️ Streaming buffer full — dropping oldest frame "
                                f"(total drops: {self._streaming_drops}, "
                                f"depth: {len(self._streaming_buffer)}/{self._streaming_buffer.maxlen})"
                            )
                    self._streaming_seq += 1
                    self._streaming_total_pushed += 1
                    self._streaming_total_bytes_pushed += len(gemini_pcm)
                    self._streaming_buffer.append(gemini_pcm)
                    self._streaming_data_available.set()
            else:
                # Legacy path: buffer for later concatenation
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
    
    def cleanup_user(self, user_id: int) -> None:
        """Clean up resources for a user who left the channel.
        
        Args:
            user_id: Discord user ID to clean up.
        """
        if user_id in self._user_buffers:
            del self._user_buffers[user_id]
        if user_id in self._user_buffer_samples:
            del self._user_buffer_samples[user_id]
        with self._user_locks_lock:
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
    
    # =========================================================================
    # Real-time streaming and VAD methods
    # =========================================================================
    
    def start_streaming_to_gemini(self) -> None:
        """Start streaming mode for real-time audio to Gemini.
        
        In streaming mode, audio frames are pushed to a ring buffer immediately
        instead of being buffered for later concatenation.
        
        Includes a grace period to skip the wake word tail audio and 
        minimum speech duration before silence detection kicks in.
        """
        import time
        self._is_streaming_to_gemini = True
        self._consecutive_silent_frames = 0
        self._speech_detected = False
        self._streaming_start_time = time.time()
        self._speech_start_time = 0.0  # Will be set when speech is detected
        self._silence_detected_event = asyncio.Event()
        # Clear the ring buffer and reset per-session metrics
        self._streaming_buffer.clear()
        self._streaming_data_available.clear()
        self._streaming_drops = 0
        self._streaming_total_pushed = 0
        self._streaming_total_consumed = 0
        self._streaming_total_bytes_pushed = 0
        self._streaming_seq = 0
        logger.debug(f"Started streaming mode for Gemini (grace period: {WAKE_WORD_GRACE_PERIOD}s, min speech: {MIN_SPEECH_DURATION}s)")
    
    def stop_streaming_to_gemini(self) -> None:
        """Stop streaming mode."""
        self._is_streaming_to_gemini = False
        self._speech_detected = False
        self._consecutive_silent_frames = 0
        self._streaming_start_time = 0.0
        self._speech_start_time = 0.0
        if self._silence_detected_event:
            self._silence_detected_event.set()  # Unblock any waiters
        logger.debug("Stopped streaming mode for Gemini")
    
    async def get_streaming_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get the next audio chunk from the streaming ring buffer.

        This is the low-latency path for real-time streaming to Gemini.
        Frames are pushed directly without concatenation.

        Args:
            timeout: Maximum time to wait for a chunk.

        Returns:
            PCM audio bytes, or None if timeout.
        """
        # Try to pop immediately under lock
        async with self._streaming_buffer_lock:
            if self._streaming_buffer:
                chunk = self._streaming_buffer.popleft()
                self._streaming_total_consumed += 1
                return chunk
            # Buffer empty — clear event BEFORE waiting (under lock) to avoid race
            self._streaming_data_available.clear()

        # Wait for producer to signal new data
        try:
            await asyncio.wait_for(
                self._streaming_data_available.wait(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

        # Try again after wakeup
        async with self._streaming_buffer_lock:
            if self._streaming_buffer:
                chunk = self._streaming_buffer.popleft()
                self._streaming_total_consumed += 1
                return chunk
        return None
    
    def _compute_rms_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio samples.
        
        Args:
            audio: Audio samples as numpy array.
            
        Returns:
            RMS energy value.
        """
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def is_silence_detected(self) -> bool:
        """Check if silence has been detected after speech.
        
        Returns:
            True if user stopped speaking (speech detected, then silence,
            and minimum speech duration requirement met).
        """
        import time
        if not self._speech_detected:
            return False
        if self._consecutive_silent_frames < self._vad_silence_frames:
            return False
        # Check minimum speech duration
        if self._speech_start_time > 0:
            speech_duration = time.time() - self._speech_start_time
            if speech_duration < MIN_SPEECH_DURATION:
                return False
        return True
    
    def reset_vad_state(self) -> None:
        """Reset VAD state for new detection."""
        self._consecutive_silent_frames = 0
        self._speech_detected = False
        self._speech_start_time = 0.0
        if self._silence_detected_event:
            self._silence_detected_event.clear()
    
    def set_silence_threshold(self, threshold_seconds: float) -> None:
        """Update the silence detection threshold.
        
        Args:
            threshold_seconds: New silence threshold in seconds.
        """
        self.silence_threshold = threshold_seconds
        self._vad_silence_frames = int(threshold_seconds / (FRAME_DURATION_MS / 1000))
        logger.info(f"VAD silence threshold updated: {threshold_seconds}s -> {self._vad_silence_frames} frames")

    # =========================================================================
    # Pipeline health reporting
    # =========================================================================

    def get_pipeline_health(self) -> dict:
        """Return a snapshot of streaming pipeline metrics.

        Designed to be called once per second by the health reporter.
        """
        return {
            "streaming_active": self._is_streaming_to_gemini,
            "buffer_depth": len(self._streaming_buffer),
            "buffer_max": self._streaming_buffer.maxlen,
            "total_pushed": self._streaming_total_pushed,
            "total_consumed": self._streaming_total_consumed,
            "total_drops": self._streaming_drops,
            "total_bytes_pushed": self._streaming_total_bytes_pushed,
            "active_users": len(self._user_buffers),
            "active_user_id": self._active_user_id,
            "speech_detected": self._speech_detected,
            "silent_frames": self._consecutive_silent_frames,
            "seq": self._streaming_seq,
        }
