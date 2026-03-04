"""Audio debug logger for saving audio at each pipeline stage.

When enabled via config (log_audio_files: true), this saves WAV files of audio
at each step of the processing pipeline:

1. Raw Discord audio (48kHz stereo PCM)
2. After discord_to_gemini conversion (16kHz mono PCM)
3. Audio fed to wake word detector (16kHz mono PCM)
4. Audio sent to Gemini (16kHz mono PCM)

Files are saved in a timestamped session subdirectory under the logs folder.
All writes are done in a background thread to avoid blocking the event loop.
"""

import io
import os
import wave
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logger import get_logger

logger = get_logger("audio.debug_logger")


class AudioDebugLogger:
    """Saves audio data as WAV files at each pipeline processing stage.
    
    Audio is accumulated in memory buffers per-stage, then flushed to disk
    either periodically or when a pipeline interaction (wake word -> response) completes.
    
    Pipeline stages:
        1. raw_discord     — Raw 48kHz stereo PCM from Discord sink
        2. converted_16k   — After discord_to_gemini() (16kHz mono)
        3. wake_word_input  — Chunks fed to WakeWordDetector
        4. gemini_input    — Chunks streamed to Gemini Live API
    """

    # Stage names and their audio parameters
    STAGES = {
        "raw_discord":     {"sample_rate": 48000, "channels": 2, "sample_width": 2},
        "converted_16k":   {"sample_rate": 16000, "channels": 1, "sample_width": 2},
        "wake_word_input": {"sample_rate": 16000, "channels": 1, "sample_width": 2},
        "gemini_input":    {"sample_rate": 16000, "channels": 1, "sample_width": 2},
    }

    def __init__(self, log_directory: str = "logs", enabled: bool = False):
        """Initialize the audio debug logger.
        
        Args:
            log_directory: Base log directory path.
            enabled: Whether audio file logging is enabled.
        """
        self._enabled = enabled
        self._log_directory = Path(log_directory)
        self._session_dir: Optional[Path] = None
        self._interaction_count = 0
        
        # Per-interaction, per-stage audio buffers
        # Key: stage name, Value: list of bytes chunks
        self._buffers: dict[str, list[bytes]] = defaultdict(list)
        self._buffer_lock = threading.Lock()
        
        # Per-user tracking for wake word stage
        self._user_buffers: dict[int, dict[str, list[bytes]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Background writer thread
        self._write_queue: list[tuple[Path, bytes, dict]] = []
        self._write_lock = threading.Lock()
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_running = False
        
        if self._enabled:
            self._setup_session_directory()
            self._start_writer_thread()
            logger.info(f"Audio debug logger ENABLED - saving to {self._session_dir}")
        else:
            logger.debug("Audio debug logger disabled")

    @property
    def enabled(self) -> bool:
        """Check if audio debug logging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable audio debug logging."""
        if value and not self._enabled:
            self._enabled = True
            self._setup_session_directory()
            self._start_writer_thread()
            logger.info(f"Audio debug logger ENABLED - saving to {self._session_dir}")
        elif not value and self._enabled:
            self._enabled = False
            self._flush_all()
            self._stop_writer_thread()
            logger.info("Audio debug logger DISABLED")

    def _setup_session_directory(self) -> None:
        """Create a timestamped session directory for audio files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._log_directory / f"audio_debug_{timestamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Audio debug session directory: {self._session_dir}")

    def _start_writer_thread(self) -> None:
        """Start the background thread for writing WAV files."""
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return
        self._writer_running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="audio_debug_writer",
        )
        self._writer_thread.start()
        logger.debug("Audio debug writer thread started")

    def _stop_writer_thread(self) -> None:
        """Stop the background writer thread."""
        self._writer_running = False
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

    def _writer_loop(self) -> None:
        """Background loop that processes write queue."""
        while self._writer_running:
            items = []
            with self._write_lock:
                if self._write_queue:
                    items = self._write_queue.copy()
                    self._write_queue.clear()

            for filepath, audio_data, params in items:
                try:
                    self._write_wav(filepath, audio_data, params)
                except Exception as e:
                    logger.error(f"Error writing audio debug file {filepath}: {e}")

            time.sleep(0.1)  # Small sleep to avoid busy-waiting

        # Final drain on shutdown
        with self._write_lock:
            for filepath, audio_data, params in self._write_queue:
                try:
                    self._write_wav(filepath, audio_data, params)
                except Exception:
                    pass
            self._write_queue.clear()

    @staticmethod
    def _write_wav(filepath: Path, audio_data: bytes, params: dict) -> None:
        """Write PCM audio data to a WAV file.
        
        Args:
            filepath: Output WAV file path.
            audio_data: Raw PCM audio bytes.
            params: Dict with sample_rate, channels, sample_width.
        """
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(params["channels"])
            wf.setsampwidth(params["sample_width"])
            wf.setframerate(params["sample_rate"])
            wf.writeframes(audio_data)

    # =========================================================================
    # Public API — called from the audio pipeline
    # =========================================================================

    def log_raw_discord(self, pcm_data: bytes, user_id: int) -> None:
        """Log raw audio from Discord (48kHz stereo PCM).
        
        Called from WakeWordSink.write() or AudioCapture.process_discord_audio_per_user().
        
        Args:
            pcm_data: Raw PCM audio bytes from Discord.
            user_id: Discord user ID.
        """
        if not self._enabled:
            return
        with self._buffer_lock:
            self._user_buffers[user_id]["raw_discord"].append(pcm_data)

    def log_converted(self, pcm_data: bytes, user_id: int) -> None:
        """Log audio after discord_to_gemini conversion (16kHz mono PCM).
        
        Called from AudioCapture.process_discord_audio_per_user() after conversion.
        
        Args:
            pcm_data: Converted PCM audio bytes.
            user_id: Discord user ID.
        """
        if not self._enabled:
            return
        with self._buffer_lock:
            self._user_buffers[user_id]["converted_16k"].append(pcm_data)

    def log_wake_word_input(self, pcm_data: bytes, user_id: int) -> None:
        """Log audio chunk fed to the wake word detector.
        
        Called from VoiceHandler._on_audio_chunk_received() or WakeWordDetector.
        
        Args:
            pcm_data: PCM audio bytes (16kHz mono).
            user_id: Discord user ID.
        """
        if not self._enabled:
            return
        with self._buffer_lock:
            self._user_buffers[user_id]["wake_word_input"].append(pcm_data)

    def log_gemini_input(self, pcm_data: bytes) -> None:
        """Log audio chunk sent to Gemini.
        
        Called from VoiceHandler._send_audio_loop().
        
        Args:
            pcm_data: PCM audio bytes (16kHz mono) sent to Gemini.
        """
        if not self._enabled:
            return
        with self._buffer_lock:
            self._buffers["gemini_input"].append(pcm_data)

    def on_interaction_start(self, user_id: int) -> None:
        """Signal that a new interaction is starting (wake word or /ask detected).
        
        Flushes any previous interaction's audio and starts fresh buffers.
        
        Args:
            user_id: Discord user ID who triggered the interaction.
        """
        if not self._enabled:
            return
        
        # Flush previous interaction data if any
        self._flush_all()
        
        self._interaction_count += 1
        logger.info(f"Audio debug: Interaction #{self._interaction_count} started (user {user_id})")

    def on_interaction_end(self, user_id: int) -> None:
        """Signal that an interaction has completed.
        
        Flushes all buffered audio to disk as WAV files.
        
        Args:
            user_id: Discord user ID who triggered the interaction.
        """
        if not self._enabled:
            return
        
        self._flush_interaction(user_id)
        logger.info(f"Audio debug: Interaction #{self._interaction_count} audio saved")

    # =========================================================================
    # Flushing — write accumulated audio to disk
    # =========================================================================

    def _flush_interaction(self, user_id: int) -> None:
        """Flush all buffered audio for the current interaction to disk.
        
        Args:
            user_id: The user whose interaction audio to flush.
        """
        if self._session_dir is None:
            return

        interaction_dir = self._session_dir / f"interaction_{self._interaction_count:04d}_user_{user_id}"
        interaction_dir.mkdir(parents=True, exist_ok=True)

        with self._buffer_lock:
            # Flush per-user stage buffers
            if user_id in self._user_buffers:
                for stage_name, chunks in self._user_buffers[user_id].items():
                    if chunks:
                        audio_data = b"".join(chunks)
                        params = self.STAGES[stage_name]
                        filepath = interaction_dir / f"{stage_name}.wav"
                        
                        duration = len(audio_data) / (params["sample_rate"] * params["channels"] * params["sample_width"])
                        logger.debug(f"Audio debug: Saving {stage_name} ({len(audio_data)} bytes, {duration:.2f}s) -> {filepath.name}")
                        
                        with self._write_lock:
                            self._write_queue.append((filepath, audio_data, params))
                
                # Clear the user's buffers
                self._user_buffers[user_id].clear()

            # Flush global stage buffers (e.g., gemini_input)
            for stage_name, chunks in self._buffers.items():
                if chunks:
                    audio_data = b"".join(chunks)
                    params = self.STAGES[stage_name]
                    filepath = interaction_dir / f"{stage_name}.wav"
                    
                    duration = len(audio_data) / (params["sample_rate"] * params["channels"] * params["sample_width"])
                    logger.debug(f"Audio debug: Saving {stage_name} ({len(audio_data)} bytes, {duration:.2f}s) -> {filepath.name}")
                    
                    with self._write_lock:
                        self._write_queue.append((filepath, audio_data, params))
            
            self._buffers.clear()

    def _flush_all(self) -> None:
        """Flush all buffered audio across all users."""
        with self._buffer_lock:
            # Gather all user IDs with data
            user_ids = list(self._user_buffers.keys())
        
        for uid in user_ids:
            self._flush_interaction(uid)

    def cleanup(self) -> None:
        """Clean up resources. Call on shutdown."""
        self._flush_all()
        self._stop_writer_thread()
        logger.debug("Audio debug logger cleaned up")
