"""Wake word detection using OpenWakeWord with per-user support."""

import asyncio
import concurrent.futures
from typing import Optional, Callable, Awaitable, Dict, Tuple
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("wake_word.detector")

# OpenWakeWord models available by default
AVAILABLE_MODELS = {
    "hey_jarvis": "hey_jarvis_v0.1",
    "alexa": "alexa_v0.1", 
    "hey_mycroft": "hey_mycroft_v0.1",
    "timer": "timer_v0.1",
    "weather": "weather_v0.1",
}

# Thread pool for CPU-bound wake word inference
# This prevents blocking the event loop during model.predict()
_inference_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_inference_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the thread pool executor for wake word inference."""
    global _inference_executor
    if _inference_executor is None:
        # Use 2 threads - one for inference, one spare for user model creation
        _inference_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="wake_word_inference"
        )
        logger.debug("Created wake word inference thread pool")
    return _inference_executor


class WakeWordDetector:
    """Detects wake words in audio using OpenWakeWord with per-user support.
    
    This class wraps OpenWakeWord to provide async-friendly wake word
    detection for the Discord voice bot. Each user gets their own detector
    state to prevent audio mixing issues with 3+ users.
    
    Inference is offloaded to a thread pool to avoid blocking the event loop.
    """
    
    def __init__(
        self,
        wake_phrase: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        verbose: bool = False,
    ):
        """Initialize the wake word detector.
        
        Args:
            wake_phrase: Wake word model to use (e.g., "hey_jarvis", "alexa").
            threshold: Detection threshold (0.0 to 1.0, higher = stricter).
            sample_rate: Expected audio sample rate (should be 16kHz).
            verbose: If True, log every audio chunk's wake word scores.
        """
        logger.debug(f"Initializing WakeWordDetector: phrase={wake_phrase}, threshold={threshold}, sample_rate={sample_rate}, verbose={verbose}")
        
        self.wake_phrase = wake_phrase
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        # Main shared model (for single-user compatibility)
        self._model = None
        
        # Per-user model instances for multi-user support
        self._user_models: Dict[int, object] = {}
        self._user_process_counts: Dict[int, int] = {}
        self._user_last_scores: Dict[int, dict] = {}
        
        # Pre-warmed models ready for new users
        self._prewarmed_models: list = []
        self._prewarm_count = 2  # Number of models to keep pre-warmed
        
        self._is_enabled = True
        self._detection_callback: Optional[Callable[[int], Awaitable[None]]] = None  # Now takes user_id
        self._process_count = 0
        self._last_scores: dict = {}
        
        # Model name for creating instances
        self._model_name = None
        
        # Load the shared model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the OpenWakeWord model and pre-warm user models."""
        try:
            logger.debug("Importing openwakeword library")
            import openwakeword
            from openwakeword.model import Model
            
            # Get model name
            self._model_name = AVAILABLE_MODELS.get(
                self.wake_phrase,
                self.wake_phrase,  # Allow custom model paths
            )
            
            logger.info(f"Loading wake word model: {self._model_name}")
            logger.debug(f"Available models: {list(AVAILABLE_MODELS.keys())}")
            
            # Load the shared model
            self._model = Model(
                wakeword_models=[self._model_name],
                inference_framework="onnx",
            )
            
            logger.info(f"Wake word detector initialized for '{self.wake_phrase}'")
            logger.info(f"Detection threshold: {self.threshold}")
            logger.info("Per-user detection enabled for multi-user voice channels")
            
            # Pre-warm models for new users (to avoid load-time latency)
            self._prewarm_models()
            
        except ImportError as e:
            logger.error(f"Failed to import openwakeword: {e}")
            logger.error("Install with: pip install openwakeword")
            raise
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            logger.debug(f"Model load error details: {type(e).__name__}: {e}")
            raise
    
    def _prewarm_models(self) -> None:
        """Pre-warm wake word models for new users.
        
        This creates a pool of ready-to-use models so that when a new user
        joins, we don't have to wait for model initialization.
        """
        try:
            from openwakeword.model import Model
            
            models_to_create = self._prewarm_count - len(self._prewarmed_models)
            if models_to_create > 0:
                logger.info(f"Pre-warming {models_to_create} wake word models...")
                for _ in range(models_to_create):
                    model = Model(
                        wakeword_models=[self._model_name],
                        inference_framework="onnx",
                    )
                    self._prewarmed_models.append(model)
                logger.info(f"Pre-warmed {models_to_create} models (total pool: {len(self._prewarmed_models)})")
        except Exception as e:
            logger.warning(f"Failed to pre-warm models: {e}")
    
    def _get_or_create_user_model(self, user_id: int) -> object:
        """Get or create a model instance for a specific user.
        
        Uses pre-warmed models when available to avoid load-time latency.
        
        Args:
            user_id: Discord user ID.
            
        Returns:
            OpenWakeWord Model instance for this user.
        """
        if user_id not in self._user_models:
            try:
                # Try to use a pre-warmed model first (instant)
                if self._prewarmed_models:
                    model = self._prewarmed_models.pop()
                    self._user_models[user_id] = model
                    logger.info(f"Assigned pre-warmed wake word model to user {user_id} (pool remaining: {len(self._prewarmed_models)})")
                    
                    # Asynchronously replenish the pool
                    asyncio.get_event_loop().call_soon(self._prewarm_models)
                else:
                    # Fall back to creating a new model (has load latency)
                    from openwakeword.model import Model
                    logger.debug(f"Creating new wake word model for user {user_id} (no pre-warmed models available)")
                    self._user_models[user_id] = Model(
                        wakeword_models=[self._model_name],
                        inference_framework="onnx",
                    )
                    logger.info(f"Created wake word model for user {user_id}")
                
                self._user_process_counts[user_id] = 0
                self._user_last_scores[user_id] = {}
                
            except Exception as e:
                logger.error(f"Failed to create model for user {user_id}: {e}")
                # Fall back to shared model
                return self._model
        
        return self._user_models[user_id]
    
    def set_detection_callback(
        self,
        callback: Optional[Callable[[int], Awaitable[None]]],
    ) -> None:
        """Set callback to run when wake word is detected.
        
        Args:
            callback: Async function to call on detection. Takes user_id as parameter.
        """
        self._detection_callback = callback
    
    @property
    def is_enabled(self) -> bool:
        """Check if wake word detection is enabled."""
        return self._is_enabled
    
    def enable(self) -> None:
        """Enable wake word detection."""
        self._is_enabled = True
        logger.debug("Wake word detection enabled")
    
    def disable(self) -> None:
        """Disable wake word detection."""
        self._is_enabled = False
        logger.debug("Wake word detection disabled")
    
    def reset(self, user_id: Optional[int] = None) -> None:
        """Reset the detector state.
        
        Args:
            user_id: If provided, reset only this user's model. Otherwise reset all.
        """
        if user_id is not None:
            if user_id in self._user_models:
                self._user_models[user_id].reset()
                self._user_process_counts[user_id] = 0
                logger.debug(f"Reset wake word model for user {user_id}")
        else:
            # Reset all user models
            for uid, model in self._user_models.items():
                model.reset()
                self._user_process_counts[uid] = 0
            if self._model:
                self._model.reset()
            logger.debug("Reset all wake word models")
    
    def cleanup_user(self, user_id: int) -> None:
        """Clean up model for a user who left the channel.
        
        Args:
            user_id: Discord user ID to clean up.
        """
        if user_id in self._user_models:
            del self._user_models[user_id]
            del self._user_process_counts[user_id]
            if user_id in self._user_last_scores:
                del self._user_last_scores[user_id]
            logger.info(f"Cleaned up wake word model for user {user_id}")
    
    async def process_audio_for_user(self, audio_data: bytes, user_id: int) -> bool:
        """Process audio chunk for a specific user and check for wake word.
        
        This method maintains separate model state per user, preventing
        audio mixing issues when multiple users are in the voice channel.
        
        Inference is run in a thread executor to avoid blocking the event loop.
        
        Args:
            audio_data: PCM audio bytes (16kHz, 16-bit, mono).
            user_id: Discord user ID this audio came from.
            
        Returns:
            True if wake word was detected for this user, False otherwise.
        """
        if not self._is_enabled:
            return False
        
        # Get or create user-specific model
        model = self._get_or_create_user_model(user_id)
        if model is None:
            return False
        
        self._user_process_counts[user_id] = self._user_process_counts.get(user_id, 0) + 1
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Run prediction in thread executor to avoid blocking event loop
        # This is CPU-bound work that can take several milliseconds
        loop = asyncio.get_event_loop()
        executor = _get_inference_executor()
        try:
            prediction = await loop.run_in_executor(executor, model.predict, audio_np)
        except Exception as e:
            logger.error(f"Error running wake word prediction for user {user_id}: {e}")
            return False
        
        self._user_last_scores[user_id] = prediction
        
        # Check if any model detected wake word above threshold
        detected = False
        for model_name, score in prediction.items():
            if score >= self.threshold:
                logger.info(f"🎤 WAKE WORD DETECTED from USER {user_id}! Model: {model_name}, Score: {score:.3f}, Threshold: {self.threshold}")
                logger.debug(f"Detection after {self._user_process_counts[user_id]} chunks for this user")
                detected = True
                break
        
        if detected and self._detection_callback:
            # Reset this user's model to avoid multiple detections
            self.reset(user_id)
            await self._detection_callback(user_id)
        
        return detected
    
    async def process_audio(self, audio_data: bytes) -> bool:
        """Process audio chunk and check for wake word (legacy, non-per-user).
        
        Args:
            audio_data: PCM audio bytes (16kHz, 16-bit, mono).
            
        Returns:
            True if wake word was detected, False otherwise.
        """
        if not self._is_enabled or not self._model:
            return False
        
        self._process_count += 1
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Run prediction (this is CPU-bound, but fast)
        prediction = self._model.predict(audio_np)
        self._last_scores = prediction
        
        # Check if any model detected wake word above threshold
        detected = False
        for model_name, score in prediction.items():
            if score >= self.threshold:
                logger.info(f"🎤 WAKE WORD DETECTED! Model: {model_name}, Score: {score:.3f}, Threshold: {self.threshold}")
                logger.debug(f"Detection after {self._process_count} audio chunks processed")
                detected = True
                break
        
        if detected and self._detection_callback:
            # Reset to avoid multiple detections
            self.reset()
            self._process_count = 0
            # Call with user_id=0 for legacy compatibility
            await self._detection_callback(0)
        
        return detected
    
    def process_audio_sync(self, audio_data: bytes) -> bool:
        """Process audio chunk synchronously.
        
        Args:
            audio_data: PCM audio bytes (16kHz, 16-bit, mono).
            
        Returns:
            True if wake word was detected, False otherwise.
        """
        if not self._is_enabled or not self._model:
            return False
        
        self._process_count += 1
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Run prediction
        prediction = self._model.predict(audio_np)
        self._last_scores = prediction
        
        # Check threshold
        for model_name, score in prediction.items():
            if score >= self.threshold:
                logger.info(f"🎤 WAKE WORD DETECTED (sync)! Model: {model_name}, Score: {score:.3f}")
                self.reset()
                self._process_count = 0
                return True
        
        return False
    
    def get_last_scores(self, user_id: Optional[int] = None) -> dict:
        """Get the last prediction scores for debugging.
        
        Args:
            user_id: If provided, get scores for this user. Otherwise get shared scores.
        
        Returns:
            Dictionary of model names to scores.
        """
        if user_id is not None and user_id in self._user_last_scores:
            return self._user_last_scores[user_id].copy()
        return self._last_scores.copy()
    
    def get_available_models(self) -> list[str]:
        """Get list of available wake word models.
        
        Returns:
            List of model names.
        """
        return list(AVAILABLE_MODELS.keys())
    
    def get_active_users(self) -> list[int]:
        """Get list of users with active wake word models.
        
        Returns:
            List of user IDs with models.
        """
        return list(self._user_models.keys())
