"""Wake word detection using OpenWakeWord."""

import asyncio
from typing import Optional, Callable, Awaitable
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


class WakeWordDetector:
    """Detects wake words in audio using OpenWakeWord.
    
    This class wraps OpenWakeWord to provide async-friendly wake word
    detection for the Discord voice bot.
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
        
        self._model = None
        self._is_enabled = True
        self._detection_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._process_count = 0  # Track how many chunks processed
        self._last_scores: dict = {}  # Store last prediction scores for debugging
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the OpenWakeWord model."""
        try:
            logger.debug("Importing openwakeword library")
            import openwakeword
            from openwakeword.model import Model
            
            # Get model name
            model_name = AVAILABLE_MODELS.get(
                self.wake_phrase,
                self.wake_phrase,  # Allow custom model paths
            )
            
            logger.info(f"Loading wake word model: {model_name}")
            logger.debug(f"Available models: {list(AVAILABLE_MODELS.keys())}")
            
            # Load the model
            self._model = Model(
                wakeword_models=[model_name],
                inference_framework="onnx",
            )
            
            logger.info(f"Wake word detector initialized for '{self.wake_phrase}'")
            logger.info(f"Detection threshold: {self.threshold}")
            logger.debug(f"Model loaded successfully: {self._model}")
            
        except ImportError as e:
            logger.error(f"Failed to import openwakeword: {e}")
            logger.error("Install with: pip install openwakeword")
            raise
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            logger.debug(f"Model load error details: {type(e).__name__}: {e}")
            raise
    
    def set_detection_callback(
        self,
        callback: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        """Set callback to run when wake word is detected.
        
        Args:
            callback: Async function to call on detection.
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
    
    def reset(self) -> None:
        """Reset the detector state."""
        if self._model:
            self._model.reset()
    
    async def process_audio(self, audio_data: bytes) -> bool:
        """Process audio chunk and check for wake word.
        
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
        
        # Log scores - verbose logs every chunk, otherwise every ~5 seconds
        # log_interval = 1 if self.verbose else 50
        # if self._process_count % log_interval == 0:
        #     scores_str = ", ".join([f"{k}={v:.3f}" for k, v in prediction.items()])
        #     if self.verbose:
        #         logger.debug(f"[WAKE] Scores: {scores_str}")
        #     else:
        #         logger.debug(f"Wake word scores (chunk #{self._process_count}): {scores_str}")
        
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
            await self._detection_callback()
        
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
    
    def get_last_scores(self) -> dict:
        """Get the last prediction scores for debugging.
        
        Returns:
            Dictionary of model names to scores.
        """
        return self._last_scores.copy()
    
    def get_available_models(self) -> list[str]:
        """Get list of available wake word models.
        
        Returns:
            List of model names.
        """
        return list(AVAILABLE_MODELS.keys())
