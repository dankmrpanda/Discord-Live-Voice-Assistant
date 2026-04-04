"""Audio processing utilities for format conversion with stateful resampling."""

import math
import numpy as np
from scipy import signal
from typing import Optional, Dict, Tuple

from ..utils.logger import get_logger

logger = get_logger("audio.processor")


class AudioProcessor:
    """Handles audio format conversion between Discord and Gemini formats.

    Uses stateful resampling to avoid boundary artifacts between chunks.
    Each user gets independent filter state so multi-user audio stays clean.
    """

    def __init__(
        self,
        discord_sample_rate: int = 48000,
        gemini_input_sample_rate: int = 16000,
        gemini_output_sample_rate: int = 24000,
    ):
        self.discord_sample_rate = discord_sample_rate
        self.gemini_input_sample_rate = gemini_input_sample_rate
        self.gemini_output_sample_rate = gemini_output_sample_rate

        # Pre-compute GCD-based resampling ratios for common conversions
        self._resample_ratios: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._setup_resampling_ratios()

        # --- Stateful resampling ---
        # Pre-design anti-aliasing filters for each conversion direction.
        # For downsampling 48kHz -> 16kHz: low-pass at 8kHz (Nyquist of target)
        # For upsampling 24kHz -> 48kHz: low-pass at 12kHz (Nyquist of source)
        self._aa_filters: Dict[Tuple[int, int], np.ndarray] = {}
        self._aa_filter_zi_templates: Dict[Tuple[int, int], np.ndarray] = {}
        self._setup_aa_filters()

        # Per-user filter states: keyed by (user_id, orig_sr, target_sr)
        self._user_filter_states: Dict[Tuple[int, int, int], np.ndarray] = {}

        logger.debug(
            f"AudioProcessor initialized: discord={discord_sample_rate}Hz, "
            f"gemini_in={gemini_input_sample_rate}Hz, gemini_out={gemini_output_sample_rate}Hz"
        )

    def _setup_resampling_ratios(self) -> None:
        """Pre-compute optimal resampling ratios using GCD."""
        conversions = [
            (self.discord_sample_rate, self.gemini_input_sample_rate),  # 48k -> 16k
            (self.gemini_output_sample_rate, self.discord_sample_rate),  # 24k -> 48k
        ]

        for orig_sr, target_sr in conversions:
            gcd = math.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            self._resample_ratios[(orig_sr, target_sr)] = (up, down)
            logger.debug(f"Resampling {orig_sr}Hz -> {target_sr}Hz: up={up}, down={down}")

    def _setup_aa_filters(self) -> None:
        """Pre-design anti-aliasing filters for known conversions."""
        conversions = [
            # (orig_sr, target_sr, cutoff_hz)
            # Downsample: cutoff at target Nyquist
            (self.discord_sample_rate, self.gemini_input_sample_rate,
             self.gemini_input_sample_rate / 2),
            # Upsample: cutoff at source Nyquist
            (self.gemini_output_sample_rate, self.discord_sample_rate,
             self.gemini_output_sample_rate / 2),
        ]

        for orig_sr, target_sr, cutoff_hz in conversions:
            # Normalize cutoff to Nyquist of the HIGHER sample rate
            process_sr = max(orig_sr, target_sr)
            normalized_cutoff = cutoff_hz / (process_sr / 2)
            # Clamp to valid range
            normalized_cutoff = min(normalized_cutoff, 0.99)

            # 8th-order Butterworth gives good stopband attenuation
            sos = signal.butter(8, normalized_cutoff, btype='low', output='sos')
            zi_template = signal.sosfilt_zi(sos)

            key = (orig_sr, target_sr)
            self._aa_filters[key] = sos
            self._aa_filter_zi_templates[key] = zi_template
            logger.debug(
                f"AA filter for {orig_sr}->{target_sr}Hz: "
                f"cutoff={cutoff_hz}Hz, normalized={normalized_cutoff:.4f}"
            )

    def pcm_to_numpy(self, pcm_data: bytes, sample_width: int = 2) -> np.ndarray:
        """Convert PCM bytes to numpy array normalized to [-1, 1]."""
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16

        audio = np.frombuffer(pcm_data, dtype=dtype)
        return audio.astype(np.float32) * (1.0 / np.iinfo(dtype).max)

    def numpy_to_pcm(self, audio: np.ndarray, sample_width: int = 2) -> bytes:
        """Convert numpy array to PCM bytes."""
        if sample_width == 2:
            dtype = np.int16
            scale = 32768.0
            min_val = -32768
            max_val = 32767
        elif sample_width == 4:
            dtype = np.int32
            scale = 2147483648.0
            min_val = -2147483648
            max_val = 2147483647
        else:
            dtype = np.int16
            scale = 32768.0
            min_val = -32768
            max_val = 32767

        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_scaled = np.rint(audio_clipped * scale).astype(np.int64)
        audio_int = np.clip(audio_scaled, min_val, max_val).astype(dtype)
        return audio_int.tobytes()

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio (stateless — use resample_stateful for streaming).

        Kept for non-streaming use cases (e.g., gemini_to_discord playback).
        """
        if orig_sr == target_sr:
            return audio

        ratio_key = (orig_sr, target_sr)
        if ratio_key in self._resample_ratios:
            up, down = self._resample_ratios[ratio_key]
            resampled = signal.resample_poly(audio, up, down)
            return resampled.astype(np.float32)

        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)

    def resample_stateful(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
        user_id: Optional[int] = None,
    ) -> np.ndarray:
        """Resample audio with per-user filter state for seamless chunk boundaries.

        Uses a pre-designed anti-aliasing filter with sosfilt() to maintain
        filter state between calls, then decimates/interpolates as needed.
        This eliminates clicks and discontinuities at chunk boundaries.

        Args:
            audio: Input audio as numpy array (float32).
            orig_sr: Original sample rate.
            target_sr: Target sample rate.
            user_id: User ID for per-user state tracking. If None, falls back to stateless.
        """
        if orig_sr == target_sr:
            return audio

        ratio_key = (orig_sr, target_sr)

        # Fall back to stateless if no user_id or no pre-designed filter
        if user_id is None or ratio_key not in self._aa_filters:
            return self.resample(audio, orig_sr, target_sr)

        sos = self._aa_filters[ratio_key]
        zi_template = self._aa_filter_zi_templates[ratio_key]

        # Get or initialize per-user filter state
        state_key = (user_id, orig_sr, target_sr)
        if state_key not in self._user_filter_states:
            # Initialize filter state scaled to the first sample
            initial_val = audio[0] if len(audio) > 0 else 0.0
            self._user_filter_states[state_key] = zi_template * initial_val

        zi = self._user_filter_states[state_key]

        if orig_sr > target_sr:
            # --- Downsampling (e.g., 48kHz -> 16kHz) ---
            # 1. Anti-alias filter (stateful)
            filtered, zi_out = signal.sosfilt(sos, audio, zi=zi)
            self._user_filter_states[state_key] = zi_out
            # 2. Decimate: take every Nth sample
            _, down = self._resample_ratios[ratio_key]
            return filtered[::down].astype(np.float32)
        else:
            # --- Upsampling (e.g., 24kHz -> 48kHz) ---
            up, _ = self._resample_ratios[ratio_key]
            # 1. Insert zeros between samples (zero-stuffing)
            upsampled = np.zeros(len(audio) * up, dtype=np.float32)
            upsampled[::up] = audio * up  # Scale to maintain energy
            # 2. Anti-alias/interpolation filter (stateful)
            filtered, zi_out = signal.sosfilt(sos, upsampled, zi=zi)
            self._user_filter_states[state_key] = zi_out
            return filtered.astype(np.float32)

    def reset_user_state(self, user_id: int) -> None:
        """Clear resampler state for a user (call when user leaves VC)."""
        keys_to_remove = [k for k in self._user_filter_states if k[0] == user_id]
        for key in keys_to_remove:
            del self._user_filter_states[key]
        if keys_to_remove:
            logger.debug(f"Reset resampler state for user {user_id}")

    def stereo_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono by averaging channels.

        Args:
            audio: Stereo audio array with shape (samples, 2) or interleaved.

        Returns:
            Mono audio array.
        """
        if audio.ndim == 1:
            if len(audio) % 2 == 0:
                stereo = audio.reshape(-1, 2)
                return stereo.mean(axis=1).astype(np.float32)
            return audio
        elif audio.ndim == 2:
            return audio.mean(axis=1).astype(np.float32)
        return audio

    def discord_to_gemini(
        self,
        pcm_data: bytes,
        is_stereo: bool = True,
        user_id: Optional[int] = None,
    ) -> bytes:
        """Convert Discord audio format to Gemini input format.

        Discord: 48kHz, 16-bit PCM, stereo
        Gemini: 16kHz, 16-bit PCM, mono

        Uses stateful resampling when user_id is provided to eliminate
        boundary artifacts between chunks.

        Args:
            pcm_data: Raw PCM audio from Discord.
            is_stereo: Whether the input is stereo (explicit, not guessed).
            user_id: User ID for per-user resampler state.

        Returns:
            PCM audio formatted for Gemini.
        """
        if not pcm_data:
            return b""

        # Validate PCM frame alignment (2 bytes per sample per channel)
        bytes_per_frame = 2 * (2 if is_stereo else 1)
        remainder = len(pcm_data) % bytes_per_frame
        if remainder != 0:
            logger.warning(
                f"PCM data not frame-aligned ({len(pcm_data)} bytes, "
                f"remainder={remainder}), trimming {remainder} bytes"
            )
            pcm_data = pcm_data[:len(pcm_data) - remainder]
            if not pcm_data:
                return b""

        # Convert to numpy
        audio = self.pcm_to_numpy(pcm_data)

        # Convert stereo to mono using explicit flag (don't guess from array length)
        if is_stereo:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.float32)

        # Resample 48kHz -> 16kHz with per-user state
        audio = self.resample_stateful(
            audio,
            self.discord_sample_rate,
            self.gemini_input_sample_rate,
            user_id=user_id,
        )

        return self.numpy_to_pcm(audio)

    def gemini_to_discord(self, pcm_data: bytes) -> bytes:
        """Convert Gemini output format to Discord playback format.

        Gemini: 24kHz, 16-bit PCM, mono
        Discord: 48kHz, 16-bit PCM, stereo

        Uses stateless resampling (playback doesn't need per-user state).
        """
        audio = self.pcm_to_numpy(pcm_data)

        # Resample 24kHz -> 48kHz (stateless is fine for playback)
        audio = self.resample(
            audio,
            self.gemini_output_sample_rate,
            self.discord_sample_rate,
        )

        # Duplicate mono to stereo
        stereo = np.repeat(audio, 2)
        return self.numpy_to_pcm(stereo)

    def get_audio_duration(self, pcm_data: bytes, sample_rate: int, channels: int = 1) -> float:
        """Calculate the duration of PCM audio in seconds."""
        bytes_per_sample = 2 * channels
        num_samples = len(pcm_data) // bytes_per_sample
        return num_samples / sample_rate
