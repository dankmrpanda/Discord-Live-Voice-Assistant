"""Wake word detection using OpenWakeWord with per-user support."""

import asyncio
import concurrent.futures
import os
from pathlib import Path
import threading
import wave
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
        
        # Per-user locks to prevent race conditions between predict() and reset()
        self._user_locks: Dict[int, threading.Lock] = {}
        self._user_locks_lock = threading.Lock()  # Lock for the locks dict itself
        
        # Pre-warmed models ready for new users
        self._prewarmed_models: list = []
        self._prewarm_count = 2  # Number of models to keep pre-warmed
        
        self._is_enabled = True
        self._detection_callback: Optional[Callable[[int], Awaitable[None]]] = None  # Now takes user_id
        self._process_count = 0
        self._last_scores: dict = {}
        
        # Model name for creating instances
        self._model_name = None
        self._model_kwargs: dict = {}
        
        # === AGC (Automatic Gain Control) ===
        # Discord audio can be very loud (near full-scale). Other Discord users
        # benefit from client-side AGC, but the bot receives raw Opus-decoded PCM.
        # We apply our own AGC to normalize volume before wake word detection.
        self._agc_target_rms = 2000  # Target RMS level (~6% of full scale, typical speech)
        self._agc_running_rms: Dict[int, float] = {}  # Per-user running RMS estimate
        self._agc_alpha = 0.3  # Smoothing factor (0=slow adaptation, 1=instant)
        self._agc_min_gain = 0.05  # Minimum gain to avoid silence
        self._agc_max_gain = 10.0  # Maximum gain to avoid amplifying noise
        self._agc_enabled = True  # Can be disabled for debugging
        self._agc_logged_first: Dict[int, bool] = {}  # Track first AGC log per user
        
        # === Debug: Audio dump for offline analysis ===
        self._debug_wav_buffers: Dict[int, list] = {}  # user_id → list of int16 arrays
        self._debug_wav_samples_limit = 16000 * 30  # 30 seconds at 16kHz (extended)
        self._debug_wav_written: Dict[int, bool] = {}
        
        # === Debug: Raw Discord audio dump (pre-conversion) ===
        self._debug_raw_buffers: Dict[int, list] = {}  # user_id → list of raw bytes
        self._debug_raw_bytes_limit = 48000 * 2 * 2 * 10  # 10s of 48kHz stereo int16
        self._debug_raw_written: Dict[int, bool] = {}
        
        # Load the shared model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the OpenWakeWord model and pre-warm user models."""
        try:
            logger.debug("Importing openwakeword library")
            import openwakeword
            from openwakeword.model import Model
            
            # === ENVIRONMENT DIAGNOSTICS ===
            self._log_environment_info()
            
            # Get model name
            self._model_name = AVAILABLE_MODELS.get(
                self.wake_phrase,
                self.wake_phrase,  # Allow custom model paths
            )

            # Ensure model files exist in a writable cache directory and use
            # explicit file paths (wheels may not include bundled model assets).
            self._model_name, self._model_kwargs = self._prepare_model_assets(
                openwakeword,
                requested_model=self._model_name,
            )

            logger.info(f"Loading wake word model: {self._model_name}")
            logger.debug(f"Available models: {list(AVAILABLE_MODELS.keys())}")
            
            # Load the shared model
            self._model = Model(
                wakeword_models=[self._model_name],
                inference_framework="onnx",
                **self._model_kwargs,
            )
            
            # Log model file details
            self._log_model_details()
            
            logger.info(f"Wake word detector initialized for '{self.wake_phrase}'")
            logger.info(f"Detection threshold: {self.threshold}")
            logger.info("Per-user detection enabled for multi-user voice channels")
            
            # === Self-test: Verify model can actually detect wake words ===
            self._run_model_self_test()
            
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

    def _prepare_model_assets(self, openwakeword, requested_model: str) -> tuple[str, dict]:
        """Ensure required ONNX model files exist and return model path + kwargs."""
        model_cache_dir = Path(
            os.environ.get("OWW_MODEL_DIR", os.path.join("models", "openwakeword"))
        )
        if not model_cache_dir.is_absolute():
            model_cache_dir = (Path.cwd() / model_cache_dir).resolve()
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        from openwakeword import utils as oww_utils

        # If user provided an explicit model file path, use it as-is.
        requested_path = Path(requested_model)
        if requested_path.exists():
            logger.info(f"Using custom wake word model file: {requested_path}")
            # Still ensure feature extractor models exist in cache.
            self._ensure_feature_models(openwakeword, oww_utils, model_cache_dir)
            return str(requested_path), self._feature_model_kwargs(model_cache_dir)

        # Resolve requested built-in model from openwakeword metadata.
        model_url = self._resolve_builtin_model_url(openwakeword, requested_model)
        if model_url is None:
            available = ", ".join(sorted(openwakeword.MODELS.keys()))
            raise FileNotFoundError(
                f"Wake word model '{requested_model}' not found and is not a valid file path. "
                f"Available built-ins: {available}"
            )

        # Ensure shared feature extractor models.
        self._ensure_feature_models(openwakeword, oww_utils, model_cache_dir)

        # Ensure requested wake word model.
        wake_tflite_name = model_url.split("/")[-1]
        wake_onnx_name = wake_tflite_name.replace(".tflite", ".onnx")
        wake_onnx_path = model_cache_dir / wake_onnx_name
        if not wake_onnx_path.exists():
            logger.info(f"Downloading wake word model assets to {model_cache_dir}")
            oww_utils.download_file(model_url, str(model_cache_dir))
            oww_utils.download_file(model_url.replace(".tflite", ".onnx"), str(model_cache_dir))

        if not wake_onnx_path.exists():
            raise FileNotFoundError(f"Downloaded wake model not found: {wake_onnx_path}")

        logger.info(f"Wake word model path: {wake_onnx_path}")
        return str(wake_onnx_path), self._feature_model_kwargs(model_cache_dir)

    def _resolve_builtin_model_url(self, openwakeword, requested_model: str) -> Optional[str]:
        """Resolve a built-in model URL from openwakeword.MODELS."""
        # Direct key match first (e.g., "hey_jarvis")
        if requested_model in openwakeword.MODELS:
            return openwakeword.MODELS[requested_model]["download_url"]

        # Then try matching by filename stem/version string (e.g., "hey_jarvis_v0.1")
        for model_info in openwakeword.MODELS.values():
            model_file = model_info["download_url"].split("/")[-1]
            stem = model_file.replace(".tflite", "")
            if requested_model == stem:
                return model_info["download_url"]

        # Fallback: if wake phrase was alias key in AVAILABLE_MODELS, map it.
        alias_key = next((k for k, v in AVAILABLE_MODELS.items() if v == requested_model), None)
        if alias_key and alias_key in openwakeword.MODELS:
            return openwakeword.MODELS[alias_key]["download_url"]

        return None

    def _ensure_feature_models(self, openwakeword, oww_utils, model_cache_dir: Path) -> None:
        """Ensure embedding/melspectrogram ONNX models exist in cache dir."""
        required = ["melspectrogram.onnx", "embedding_model.onnx"]
        missing = [name for name in required if not (model_cache_dir / name).exists()]
        if not missing:
            return

        logger.info(f"Downloading OpenWakeWord feature models to {model_cache_dir}")
        for feature_model in openwakeword.FEATURE_MODELS.values():
            base_url = feature_model["download_url"]
            oww_utils.download_file(base_url, str(model_cache_dir))
            oww_utils.download_file(base_url.replace(".tflite", ".onnx"), str(model_cache_dir))

    def _feature_model_kwargs(self, model_cache_dir: Path) -> dict:
        """Build kwargs to point OpenWakeWord to local feature models."""
        melspec_path = model_cache_dir / "melspectrogram.onnx"
        embedding_path = model_cache_dir / "embedding_model.onnx"
        if not melspec_path.exists() or not embedding_path.exists():
            raise FileNotFoundError(
                f"Required feature models missing in {model_cache_dir} "
                f"(need melspectrogram.onnx and embedding_model.onnx)"
            )
        return {
            "melspec_model_path": str(melspec_path),
            "embedding_model_path": str(embedding_path),
        }
    
    def _log_environment_info(self) -> None:
        """Log package versions and environment details for debugging."""
        try:
            import openwakeword
            oww_version = getattr(openwakeword, '__version__', 'unknown')
            logger.info(f"📦 openwakeword version: {oww_version}")
        except Exception:
            logger.warning("Could not determine openwakeword version")
        
        try:
            import onnxruntime as ort
            logger.info(f"📦 onnxruntime version: {ort.__version__}")
            logger.info(f"   ONNX providers: {ort.get_available_providers()}")
        except Exception:
            logger.warning("Could not determine onnxruntime version")
        
        try:
            logger.info(f"📦 numpy version: {np.__version__}")
        except Exception:
            pass
        
        try:
            import scipy
            logger.info(f"📦 scipy version: {scipy.__version__}")
        except Exception:
            pass
    
    def _log_model_details(self) -> None:
        """Log details about loaded model files for debugging."""
        if self._model is None:
            return
        
        try:
            import pathlib
            
            # Log model file paths and sizes
            for mdl_name, mdl in self._model.models.items():
                # For ONNX models, get the model path
                model_path = getattr(mdl, '_model_path', None)
                if model_path is None:
                    # Try to find the model file from openwakeword resources
                    import openwakeword
                    oww_dir = pathlib.Path(openwakeword.__file__).parent
                    candidates = list(oww_dir.glob(f"**/*{mdl_name}*"))
                    logger.info(f"   Model '{mdl_name}' file candidates: {[str(c) for c in candidates]}")
                    for c in candidates:
                        logger.info(f"   -> {c}: {c.stat().st_size} bytes")
                else:
                    logger.info(f"   Model '{mdl_name}' path: {model_path}")
            
            # Log preprocessor model details
            prep = self._model.preprocessor
            if hasattr(prep, 'melspec_model'):
                melspec_path = getattr(prep, '_melspec_model_path', 'unknown')
                logger.info(f"   Melspec model: {melspec_path}")
            if hasattr(prep, 'embedding_model'):
                embed_path = getattr(prep, '_embedding_model_path', 'unknown')
                logger.info(f"   Embedding model: {embed_path}")
            
            # Log ONNX model input/output shapes
            for mdl_name in self._model.models.keys():
                onnx_model = self._model.models[mdl_name]
                if hasattr(onnx_model, 'get_inputs'):
                    inputs = onnx_model.get_inputs()
                    outputs = onnx_model.get_outputs()
                    logger.info(f"   ONNX '{mdl_name}' input: {[(i.name, i.shape, i.type) for i in inputs]}")
                    logger.info(f"   ONNX '{mdl_name}' output: {[(o.name, o.shape, o.type) for o in outputs]}")
            
        except Exception as e:
            logger.warning(f"Failed to log model details: {e}")
    
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
                        **self._model_kwargs,
                    )
                    self._prewarmed_models.append(model)
                logger.info(f"Pre-warmed {models_to_create} models (total pool: {len(self._prewarmed_models)})")
        except Exception as e:
            logger.warning(f"Failed to pre-warm models: {e}")
    
    def _run_model_self_test(self) -> None:
        """Run a self-test to verify the model can detect wake words.
        
        Generates a series of test predictions with silence and random noise
        to verify the model is responding correctly.
        """
        try:
            if self._model is None:
                logger.warning("Cannot run self-test: model is None")
                return
            
            logger.info("🧪 Running wake word model self-test...")
            
            # Test 1: Predict on silence — should return 0 or near 0
            silence = np.zeros(1280, dtype=np.int16)
            for i in range(10):  # need at least 5 frames for init period
                result_silence = self._model.predict(silence)
            scores_silence = {k: f"{v:.4f}" for k, v in result_silence.items()}
            logger.info(f"   Self-test silence:  {scores_silence}")
            
            # Test 2: Predict on random noise — should return 0 or low
            noise = np.random.randint(-3000, 3000, 1280, dtype=np.int16)
            for i in range(5):
                result_noise = self._model.predict(noise)
            scores_noise = {k: f"{v:.4f}" for k, v in result_noise.items()}
            logger.info(f"   Self-test noise:    {scores_noise}")
            
            # Test 3: Check model attributes
            logger.info(f"   Model models: {list(self._model.models.keys())}")
            logger.info(f"   Model inputs: {self._model.model_inputs}")
            logger.info(f"   Model outputs: {self._model.model_outputs}")
            logger.info(f"   VAD threshold: {self._model.vad_threshold}")
            logger.info(f"   Preprocessor feature_buffer shape: {self._model.preprocessor.feature_buffer.shape}")
            logger.info(f"   Prediction buffer keys: {list(self._model.prediction_buffer.keys())}")
            
            # Reset after self-test to start clean
            self._model.reset()
            logger.info("   Self-test complete ✓ (model reset for clean start)")
            
        except Exception as e:
            logger.warning(f"Model self-test failed: {e}")
    
    def _apply_agc(self, audio_np: np.ndarray, user_id: int) -> np.ndarray:
        """Apply Automatic Gain Control to normalize audio volume.
        
        Discord clients apply AGC when playing audio for human users,
        but bots receive raw Opus-decoded PCM. This simulates receive-side
        AGC to bring audio levels into the range the wake word model expects.
        
        Uses a running RMS estimate for smooth gain changes.
        
        Args:
            audio_np: Input audio as int16 numpy array.
            user_id: Discord user ID (for per-user gain tracking).
            
        Returns:
            Gain-adjusted int16 numpy array.
        """
        if not self._agc_enabled:
            return audio_np
        
        # Compute current chunk RMS
        current_rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        
        if current_rms < 10:  # Nearly silent, don't adjust
            return audio_np
        
        # Update running RMS estimate with exponential smoothing
        if user_id not in self._agc_running_rms:
            self._agc_running_rms[user_id] = current_rms
        else:
            self._agc_running_rms[user_id] = (
                self._agc_alpha * current_rms +
                (1 - self._agc_alpha) * self._agc_running_rms[user_id]
            )
        
        running_rms = self._agc_running_rms[user_id]
        
        # Calculate gain to reach target RMS
        gain = self._agc_target_rms / max(running_rms, 1.0)
        gain = np.clip(gain, self._agc_min_gain, self._agc_max_gain)
        
        # Log AGC info for first chunk per user and then periodically
        if not self._agc_logged_first.get(user_id, False):
            self._agc_logged_first[user_id] = True
            logger.info(f"🔊 AGC for user {user_id}: input_rms={current_rms:.0f}, "
                       f"target_rms={self._agc_target_rms}, gain={gain:.3f}")
            logger.info(f"   Input was {current_rms/self._agc_target_rms:.1f}x louder than target")
        elif self._user_process_counts.get(user_id, 0) % 100 == 0:
            logger.debug(f"AGC user {user_id}: rms={running_rms:.0f}, gain={gain:.3f}")
        
        # Apply gain with clipping protection
        adjusted = np.clip(
            audio_np.astype(np.float32) * gain,
            -32767, 32767
        ).astype(np.int16)
        
        return adjusted
    
    def _dump_raw_discord_audio(self, user_id: int, raw_data: bytes) -> None:
        """Save raw 48kHz stereo Discord audio to WAV for comparison.
        
        This lets the user hear exactly what Discord delivers to the bot
        (before any conversion/resampling), compared to the processed 16kHz WAV.
        """
        if self._debug_raw_written.get(user_id, False):
            return
        
        if user_id not in self._debug_raw_buffers:
            self._debug_raw_buffers[user_id] = []
        
        self._debug_raw_buffers[user_id].append(raw_data)
        total_bytes = sum(len(b) for b in self._debug_raw_buffers[user_id])
        
        if total_bytes >= self._debug_raw_bytes_limit:
            try:
                log_dir = os.environ.get("LOG_DIR", "logs")
                os.makedirs(log_dir, exist_ok=True)
                wav_path = os.path.join(log_dir, f"debug_raw_48k_user_{user_id}.wav")
                
                all_raw = b"".join(self._debug_raw_buffers[user_id])
                # Trim to limit
                all_raw = all_raw[:self._debug_raw_bytes_limit]
                
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(2)  # Stereo
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(48000)  # 48kHz
                    wf.writeframes(all_raw)
                
                n_frames = len(all_raw) // 4  # 2 channels * 2 bytes
                logger.info(f"📼 Raw Discord audio saved: {wav_path} ({n_frames} stereo frames, {n_frames/48000:.1f}s)")
                logger.info(f"   This is the audio BEFORE any conversion - what Discord delivers to the bot")
                
                self._debug_raw_written[user_id] = True
                del self._debug_raw_buffers[user_id]
            except Exception as e:
                logger.warning(f"Failed to write raw debug WAV for user {user_id}: {e}")
    
    def _dump_audio_to_wav(self, user_id: int, audio_np: np.ndarray) -> None:
        """Accumulate audio and dump to WAV file for offline diagnostic analysis.
        
        Only writes the first 30 seconds of audio per user (extended from 10s).
        File is saved to /app/logs/ (or ./logs/ on local) for later retrieval.
        """
        if self._debug_wav_written.get(user_id, False):
            return
        
        if user_id not in self._debug_wav_buffers:
            self._debug_wav_buffers[user_id] = []
        
        self._debug_wav_buffers[user_id].append(audio_np.copy())
        
        total_samples = sum(len(a) for a in self._debug_wav_buffers[user_id])
        
        if total_samples >= self._debug_wav_samples_limit:
            # Write WAV file
            try:
                log_dir = os.environ.get("LOG_DIR", "logs")
                os.makedirs(log_dir, exist_ok=True)
                wav_path = os.path.join(log_dir, f"debug_audio_user_{user_id}.wav")
                
                all_audio = np.concatenate(self._debug_wav_buffers[user_id])
                # Trim to exactly the limit
                all_audio = all_audio[:self._debug_wav_samples_limit]
                
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)
                    wf.writeframes(all_audio.tobytes())
                
                logger.info(f"📼 Debug audio WAV saved: {wav_path} ({len(all_audio)} samples, {len(all_audio)/16000:.1f}s)")
                logger.info(f"   RMS: {np.sqrt(np.mean(all_audio.astype(np.float32)**2)):.0f}, Peak: {np.max(np.abs(all_audio))}")
                logger.info(
                    f"   To test: python -c \"import openwakeword; "
                    f"m = openwakeword.Model(wakeword_models=['{self._model_name}'], "
                    f"inference_framework='onnx'); print(m.predict_clip('{wav_path}')[:3])\""
                )
                
                # === CRITICAL DIAGNOSTIC: Run predict_clip on the saved WAV ===
                # This compares streaming prediction (which yields 0.0000) against
                # clip-based prediction (fresh model, no state issues)
                try:
                    from openwakeword.model import Model as OWWModel
                    test_model = OWWModel(
                        wakeword_models=[self._model_name],
                        inference_framework="onnx",
                        **self._model_kwargs,
                    )
                    clip_predictions = test_model.predict_clip(wav_path)
                    max_scores = {}
                    for pred in clip_predictions:
                        for name, score in pred.items():
                            max_scores[name] = max(max_scores.get(name, 0.0), score)
                    logger.info(f"   🔬 predict_clip max scores: {', '.join(f'{k}={v:.6f}' for k, v in max_scores.items())}")
                    
                    any_detected = any(v >= self.threshold for v in max_scores.values())
                    if any_detected:
                        logger.info(f"   ✅ predict_clip DETECTS wake word! Streaming prediction has a bug.")
                    else:
                        logger.info(f"   ❌ predict_clip also fails to detect. Audio quality or model issue.")
                    
                    del test_model
                except Exception as clip_err:
                    logger.warning(f"   predict_clip test failed: {clip_err}")
                
                # === Also save AGC-normalized version for comparison ===
                try:
                    agc_wav_path = os.path.join(log_dir, f"debug_audio_agc_user_{user_id}.wav")
                    # Apply simple normalization: scale to target RMS
                    audio_f = all_audio.astype(np.float32)
                    audio_rms = np.sqrt(np.mean(audio_f ** 2))
                    if audio_rms > 10:
                        gain = self._agc_target_rms / audio_rms
                        audio_agc = np.clip(audio_f * gain, -32767, 32767).astype(np.int16)
                    else:
                        audio_agc = all_audio
                    
                    with wave.open(agc_wav_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_agc.tobytes())
                    
                    agc_rms = np.sqrt(np.mean(audio_agc.astype(np.float32) ** 2))
                    logger.info(f"📼 AGC-normalized WAV saved: {agc_wav_path} (gain={gain:.3f})")
                    logger.info(f"   Original RMS: {audio_rms:.0f}, Normalized RMS: {agc_rms:.0f}")
                    
                    # Run predict_clip on AGC-normalized audio
                    try:
                        from openwakeword.model import Model as OWWModel
                        test_model2 = OWWModel(
                            wakeword_models=[self._model_name],
                            inference_framework="onnx",
                            **self._model_kwargs,
                        )
                        agc_predictions = test_model2.predict_clip(agc_wav_path)
                        agc_max_scores = {}
                        for pred in agc_predictions:
                            for name, score in pred.items():
                                agc_max_scores[name] = max(agc_max_scores.get(name, 0.0), score)
                        logger.info(f"   🔬 AGC predict_clip max scores: {', '.join(f'{k}={v:.6f}' for k, v in agc_max_scores.items())}")
                        
                        if any(v >= self.threshold for v in agc_max_scores.values()):
                            logger.info(f"   ✅ AGC predict_clip DETECTS wake word! Volume was the issue.")
                        else:
                            logger.info(f"   ❌ AGC predict_clip also fails. Issue is NOT just volume.")
                        del test_model2
                    except Exception as agc_clip_err:
                        logger.warning(f"   AGC predict_clip test failed: {agc_clip_err}")
                    
                except Exception as agc_err:
                    logger.warning(f"Failed to save AGC WAV: {agc_err}")
                
                self._debug_wav_written[user_id] = True
                del self._debug_wav_buffers[user_id]  # Free memory
            except Exception as e:
                logger.warning(f"Failed to write debug WAV for user {user_id}: {e}")
    
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
                    asyncio.get_running_loop().call_soon(self._prewarm_models)
                else:
                    # Fall back to creating a new model (has load latency)
                    from openwakeword.model import Model
                    logger.debug(f"Creating new wake word model for user {user_id} (no pre-warmed models available)")
                    self._user_models[user_id] = Model(
                        wakeword_models=[self._model_name],
                        inference_framework="onnx",
                        **self._model_kwargs,
                    )
                    logger.info(f"Created wake word model for user {user_id}")
                
                self._user_process_counts[user_id] = 0
                self._user_last_scores[user_id] = {}
                
            except Exception as e:
                logger.error(f"Failed to create model for user {user_id}: {e}")
                # Fall back to shared model
                return self._model
        
        return self._user_models[user_id]
    
    def _get_user_lock(self, user_id: int) -> threading.Lock:
        """Get or create a lock for a specific user.
        
        Args:
            user_id: Discord user ID.
            
        Returns:
            threading.Lock for this user.
        """
        with self._user_locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = threading.Lock()
            return self._user_locks[user_id]
    
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
                # Use lock to prevent race with predict()
                lock = self._get_user_lock(user_id)
                with lock:
                    self._user_models[user_id].reset()
                    self._user_process_counts[user_id] = 0
                logger.debug(f"Reset wake word model for user {user_id}")
        else:
            # Reset all user models
            for uid, model in self._user_models.items():
                lock = self._get_user_lock(uid)
                with lock:
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
        if user_id in self._user_process_counts:
            del self._user_process_counts[user_id]
        if user_id in self._user_last_scores:
            del self._user_last_scores[user_id]
        # Clean up the lock too
        with self._user_locks_lock:
            if user_id in self._user_locks:
                del self._user_locks[user_id]
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
        
        # Validate audio data - OpenWakeWord ONNX model requires minimum 16 samples
        # Each sample is 2 bytes (16-bit), so minimum 32 bytes needed
        MIN_AUDIO_BYTES = 32  # 16 samples * 2 bytes per sample
        if not audio_data or len(audio_data) < MIN_AUDIO_BYTES:
            logger.debug(f"Skipping audio chunk for user {user_id}: too small ({len(audio_data) if audio_data else 0} bytes, need {MIN_AUDIO_BYTES})")
            return False
        
        # Get or create user-specific model
        model = self._get_or_create_user_model(user_id)
        if model is None:
            return False
        
        self._user_process_counts[user_id] = self._user_process_counts.get(user_id, 0) + 1
        chunk_num = self._user_process_counts[user_id]
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Verify we have enough samples after conversion
        # OpenWakeWord accumulates audio internally and processes in 1280-sample windows.
        # Small chunks are fine - they get buffered by the model. We only reject
        # near-empty chunks that would cause ONNX inference edge cases.
        # NOTE: Discord audio chunks vary in size (320-1280+ samples at 16kHz depending
        # on Discord decoder behavior). Keep this low to avoid silently dropping valid audio.
        MIN_SAMPLES = 16
        if len(audio_np) < MIN_SAMPLES:
            logger.debug(f"Skipping audio chunk for user {user_id}: insufficient samples ({len(audio_np)}, need {MIN_SAMPLES})")
            return False
        
        # === Debug: Dump audio to WAV for offline analysis ===
        self._dump_audio_to_wav(user_id, audio_np)
        
        # === AGC: Normalize volume before wake word detection ===
        # Discord bots receive raw Opus-decoded PCM without receive-side AGC.
        # Other Discord users hear AGC-normalized audio from their clients.
        # We apply our own AGC to match what a human listener would hear.
        audio_np = self._apply_agc(audio_np, user_id)
        
        # Log audio diagnostics: first 10 chunks in detail, then every 50th
        if self.verbose and (chunk_num <= 10 or chunk_num % 50 == 1):
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            peak = np.max(np.abs(audio_np))
            logger.debug(f"Audio stats for user {user_id} (chunk #{chunk_num}): "
                        f"samples={len(audio_np)}, rms={rms:.0f}, peak={peak}, "
                        f"rms_pct={rms/32767*100:.1f}%"
                        f"{f', first_5={audio_np[:5].tolist()}' if chunk_num <= 3 else ''}")
        
        # Run prediction in thread executor to avoid blocking event loop
        # This is CPU-bound work that can take several milliseconds
        # Use per-user lock to prevent race with reset()
        lock = self._get_user_lock(user_id)
        
        def predict_with_lock():
            with lock:
                # Get preprocessor state BEFORE prediction for diagnostics
                prep = model.preprocessor
                pre_accum = prep.accumulated_samples
                pre_remainder = prep.raw_data_remainder.shape[0] if hasattr(prep, 'raw_data_remainder') else -1
                pre_feat_shape = prep.feature_buffer.shape if hasattr(prep, 'feature_buffer') else None
                
                # Run prediction with timing info for detailed diagnostics
                result = model.predict(audio_np)
                
                # Get preprocessor state AFTER prediction
                post_accum = prep.accumulated_samples
                n_prepared = getattr(prep, '_last_n_prepared', None)
                
                return result, {
                    'pre_accum': pre_accum,
                    'pre_remainder': pre_remainder,
                    'pre_feat_shape': pre_feat_shape,
                    'post_accum': post_accum,
                    'feat_shape': prep.feature_buffer.shape if hasattr(prep, 'feature_buffer') else None,
                    'pred_buffer_len': len(model.prediction_buffer.get(list(model.models.keys())[0], [])) if model.prediction_buffer else 0,
                }
        
        loop = asyncio.get_running_loop()
        executor = _get_inference_executor()
        try:
            prediction, diag_info = await loop.run_in_executor(executor, predict_with_lock)
        except Exception as e:
            # Log but don't spam - ONNX dimension errors can occur on edge-case
            # audio chunks; the model will recover on the next valid chunk
            if chunk_num % 50 == 1:
                logger.warning(f"Wake word prediction error for user {user_id} (chunk #{chunk_num}): {e}")
            return False
        
        self._user_last_scores[user_id] = prediction
        
        # === ENHANCED LOGGING: Log every prediction for first 20 chunks, then every 5th ===
        if self.verbose and (chunk_num <= 20 or chunk_num % 5 == 0):
            scores_str = ", ".join(f"{name}={score:.6f}" for name, score in prediction.items())
            diag_str = (f"prep=[accum:{diag_info['pre_accum']}→{diag_info['post_accum']}, "
                       f"remainder:{diag_info['pre_remainder']}, "
                       f"feat:{diag_info['feat_shape']}, "
                       f"pred_buf_len:{diag_info['pred_buffer_len']}]")
            logger.debug(f"WW user {user_id} chunk#{chunk_num}: {scores_str} | {diag_str}")
        
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
        
        # Validate audio data - OpenWakeWord ONNX model requires minimum 16 samples
        MIN_AUDIO_BYTES = 32  # 16 samples * 2 bytes per sample
        if not audio_data or len(audio_data) < MIN_AUDIO_BYTES:
            return False
        
        self._process_count += 1
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Verify we have enough samples after conversion
        if len(audio_np) < 16:
            return False
        
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
        
        # Validate audio data - OpenWakeWord ONNX model requires minimum 16 samples
        MIN_AUDIO_BYTES = 32  # 16 samples * 2 bytes per sample
        if not audio_data or len(audio_data) < MIN_AUDIO_BYTES:
            return False
        
        self._process_count += 1
        
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Verify we have enough samples after conversion
        if len(audio_np) < 16:
            return False
        
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
