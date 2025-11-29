"""Audio processing utilities for format conversion with optimized resampling."""

import io
import numpy as np
from scipy import signal
from typing import Optional, Dict, Tuple

from ..utils.logger import get_logger

logger = get_logger("audio.processor")


class AudioProcessor:
    """Handles audio format conversion between Discord and Gemini formats.
    
    Optimized with:
    - Pre-computed resampling filters for common rate conversions
    - Efficient polyphase resampling (scipy.signal.resample_poly)
    - Cached filter coefficients to avoid recomputation
    """
    
    def __init__(
        self,
        discord_sample_rate: int = 48000,
        gemini_input_sample_rate: int = 16000,
        gemini_output_sample_rate: int = 24000,
    ):
        """Initialize the audio processor.
        
        Args:
            discord_sample_rate: Discord's audio sample rate (48kHz).
            gemini_input_sample_rate: Sample rate expected by Gemini (16kHz).
            gemini_output_sample_rate: Sample rate output by Gemini (24kHz).
        """
        self.discord_sample_rate = discord_sample_rate
        self.gemini_input_sample_rate = gemini_input_sample_rate
        self.gemini_output_sample_rate = gemini_output_sample_rate
        
        # Pre-compute GCD-based resampling ratios for common conversions
        # Discord (48kHz) to Gemini input (16kHz): 48/16 = 3/1 (downsample by 3)
        # Gemini output (24kHz) to Discord (48kHz): 48/24 = 2/1 (upsample by 2)
        self._resample_ratios: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._setup_resampling_ratios()
        
        logger.debug(f"AudioProcessor initialized: discord={discord_sample_rate}Hz, "
                    f"gemini_in={gemini_input_sample_rate}Hz, gemini_out={gemini_output_sample_rate}Hz")
    
    def _setup_resampling_ratios(self) -> None:
        """Pre-compute optimal resampling ratios using GCD."""
        import math
        
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
    
    def pcm_to_numpy(self, pcm_data: bytes, sample_width: int = 2) -> np.ndarray:
        """Convert PCM bytes to numpy array.
        
        Args:
            pcm_data: Raw PCM audio bytes.
            sample_width: Bytes per sample (2 for 16-bit).
            
        Returns:
            Numpy array of audio samples (float32, normalized to [-1, 1]).
        """
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16
        
        audio = np.frombuffer(pcm_data, dtype=dtype)
        # Normalize to float32 [-1, 1] using multiplication (faster than division)
        return audio.astype(np.float32) * (1.0 / np.iinfo(dtype).max)
    
    def numpy_to_pcm(self, audio: np.ndarray, sample_width: int = 2) -> bytes:
        """Convert numpy array to PCM bytes.
        
        Args:
            audio: Numpy array of audio samples (expected [-1, 1] range).
            sample_width: Bytes per sample (2 for 16-bit).
            
        Returns:
            Raw PCM audio bytes.
        """
        if sample_width == 2:
            dtype = np.int16
            max_val = 32767
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483647
        else:
            dtype = np.int16
            max_val = 32767
        
        # Clip and scale to integer range
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int = (audio_clipped * max_val).astype(dtype)
        return audio_int.tobytes()
    
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to a different sample rate.
        
        Uses polyphase resampling for efficiency when ratios are known,
        falls back to scipy.signal.resample for arbitrary ratios.
        
        Args:
            audio: Input audio as numpy array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.
            
        Returns:
            Resampled audio as numpy array.
        """
        if orig_sr == target_sr:
            return audio
        
        # Use pre-computed ratios for polyphase resampling if available
        ratio_key = (orig_sr, target_sr)
        if ratio_key in self._resample_ratios:
            up, down = self._resample_ratios[ratio_key]
            # resample_poly is more efficient for integer ratios
            resampled = signal.resample_poly(audio, up, down)
            return resampled.astype(np.float32)
        
        # Fall back to general resampling for non-standard rates
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    
    def stereo_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono by averaging channels.
        
        Args:
            audio: Stereo audio array with shape (samples, 2) or interleaved.
            
        Returns:
            Mono audio array.
        """
        if audio.ndim == 1:
            # Check if it's interleaved stereo
            if len(audio) % 2 == 0:
                # Assume interleaved, reshape and average
                stereo = audio.reshape(-1, 2)
                return stereo.mean(axis=1).astype(np.float32)
            return audio
        elif audio.ndim == 2:
            return audio.mean(axis=1).astype(np.float32)
        return audio
    
    def discord_to_gemini(self, pcm_data: bytes, is_stereo: bool = True) -> bytes:
        """Convert Discord audio format to Gemini input format.
        
        Discord: 48kHz, 16-bit PCM, stereo
        Gemini: 16kHz, 16-bit PCM, mono
        
        Optimized path: stereo->mono first, then resample (fewer samples to process).
        
        Args:
            pcm_data: Raw PCM audio from Discord.
            is_stereo: Whether the input is stereo.
            
        Returns:
            PCM audio formatted for Gemini.
        """
        # Convert to numpy
        audio = self.pcm_to_numpy(pcm_data)
        
        # Convert stereo to mono FIRST (reduces samples by half before resampling)
        if is_stereo:
            audio = self.stereo_to_mono(audio)
        
        # Resample from 48kHz to 16kHz (uses optimized polyphase: down by 3)
        audio = self.resample(
            audio,
            self.discord_sample_rate,
            self.gemini_input_sample_rate,
        )
        
        # Convert back to PCM bytes
        return self.numpy_to_pcm(audio)
    
    def gemini_to_discord(self, pcm_data: bytes) -> bytes:
        """Convert Gemini output format to Discord playback format.
        
        Gemini: 24kHz, 16-bit PCM, mono
        Discord: 48kHz, 16-bit PCM, stereo
        
        Args:
            pcm_data: Raw PCM audio from Gemini.
            
        Returns:
            PCM audio formatted for Discord playback.
        """
        # Convert to numpy
        audio = self.pcm_to_numpy(pcm_data)
        
        # Resample from 24kHz to 48kHz (uses optimized polyphase: up by 2)
        audio = self.resample(
            audio,
            self.gemini_output_sample_rate,
            self.discord_sample_rate,
        )
        
        # Convert mono to stereo by duplicating the channel
        # Use np.repeat for better performance than column_stack + flatten
        stereo = np.repeat(audio, 2)
        
        # Convert back to PCM bytes
        return self.numpy_to_pcm(stereo)
    
    def get_audio_duration(self, pcm_data: bytes, sample_rate: int, channels: int = 1) -> float:
        """Calculate the duration of PCM audio in seconds.
        
        Args:
            pcm_data: Raw PCM audio bytes.
            sample_rate: Sample rate of the audio.
            channels: Number of audio channels.
            
        Returns:
            Duration in seconds.
        """
        # 16-bit = 2 bytes per sample
        bytes_per_sample = 2 * channels
        num_samples = len(pcm_data) // bytes_per_sample
        return num_samples / sample_rate
