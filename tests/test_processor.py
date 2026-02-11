"""Unit tests for AudioProcessor format conversions."""

import numpy as np
import pytest

from src.audio.processor import AudioProcessor


@pytest.fixture
def processor() -> AudioProcessor:
    """Return a default AudioProcessor."""
    return AudioProcessor()


# ── pcm_to_numpy / numpy_to_pcm round-trip ──────────────────────────

def test_pcm_roundtrip_16bit(processor: AudioProcessor) -> None:
    """PCM→numpy→PCM should preserve data within quantisation error."""
    original = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    pcm = original.tobytes()

    arr = processor.pcm_to_numpy(pcm, sample_width=2)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0
    assert arr.min() >= -1.0

    reconstructed = np.frombuffer(processor.numpy_to_pcm(arr), dtype=np.int16)
    np.testing.assert_array_equal(reconstructed, original)


def test_pcm_to_numpy_empty(processor: AudioProcessor) -> None:
    """Empty input should produce an empty array."""
    arr = processor.pcm_to_numpy(b"")
    assert len(arr) == 0


# ── stereo_to_mono ───────────────────────────────────────────────────

def test_stereo_to_mono_interleaved(processor: AudioProcessor) -> None:
    """Interleaved stereo [L, R, L, R, …] should average to mono."""
    stereo = np.array([0.5, -0.5, 0.8, -0.2], dtype=np.float32)
    mono = processor.stereo_to_mono(stereo)
    assert mono.shape == (2,)
    np.testing.assert_allclose(mono, [0.0, 0.3], atol=1e-6)


def test_stereo_to_mono_2d(processor: AudioProcessor) -> None:
    """2-D (N, 2) stereo should average to (N,) mono."""
    stereo = np.array([[0.5, -0.5], [0.8, -0.2]], dtype=np.float32)
    mono = processor.stereo_to_mono(stereo)
    assert mono.shape == (2,)
    np.testing.assert_allclose(mono, [0.0, 0.3], atol=1e-6)


def test_stereo_to_mono_already_mono(processor: AudioProcessor) -> None:
    """Odd-length 1-D array should be returned as-is."""
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = processor.stereo_to_mono(mono)
    np.testing.assert_array_equal(out, mono)


# ── resample ─────────────────────────────────────────────────────────

def test_resample_identity(processor: AudioProcessor) -> None:
    """Resampling to the same rate should return the same data."""
    audio = np.random.randn(480).astype(np.float32)
    out = processor.resample(audio, 48000, 48000)
    np.testing.assert_array_equal(out, audio)


def test_resample_48k_to_16k_length(processor: AudioProcessor) -> None:
    """48 kHz → 16 kHz should produce roughly 1/3 the samples."""
    audio = np.random.randn(4800).astype(np.float32)
    out = processor.resample(audio, 48000, 16000)
    assert len(out) == 1600


def test_resample_24k_to_48k_length(processor: AudioProcessor) -> None:
    """24 kHz → 48 kHz should produce exactly 2× the samples."""
    audio = np.random.randn(2400).astype(np.float32)
    out = processor.resample(audio, 24000, 48000)
    assert len(out) == 4800


# ── discord_to_gemini ────────────────────────────────────────────────

def test_discord_to_gemini_output_length(processor: AudioProcessor) -> None:
    """48 kHz stereo → 16 kHz mono: output should be 1/6 the samples."""
    # 960 stereo samples = 1920 int16 values = 3840 bytes (20 ms @ 48 kHz)
    pcm = np.zeros(960 * 2, dtype=np.int16).tobytes()
    out = processor.discord_to_gemini(pcm, is_stereo=True)
    # 960 stereo frames → 960 mono → 320 after 3× downsample → 640 bytes
    expected_bytes = 320 * 2  # 16-bit
    assert len(out) == expected_bytes


def test_discord_to_gemini_mono_input(processor: AudioProcessor) -> None:
    """Mono input (is_stereo=False) should skip stereo→mono conversion."""
    pcm = np.zeros(960, dtype=np.int16).tobytes()
    out = processor.discord_to_gemini(pcm, is_stereo=False)
    # 960 mono → 320 after 3× downsample → 640 bytes
    assert len(out) == 320 * 2


# ── gemini_to_discord ────────────────────────────────────────────────

def test_gemini_to_discord_output_length(processor: AudioProcessor) -> None:
    """24 kHz mono → 48 kHz stereo: output should be 4× the input bytes."""
    # 240 mono samples at 24 kHz = 480 bytes
    pcm = np.zeros(240, dtype=np.int16).tobytes()
    out = processor.gemini_to_discord(pcm)
    # 240 → 480 after 2× upsample → 960 stereo values → 1920 bytes
    assert len(out) == 240 * 2 * 2 * 2  # upsample*2, stereo*2, 16-bit*2


# ── get_audio_duration ───────────────────────────────────────────────

def test_audio_duration_mono(processor: AudioProcessor) -> None:
    """1 second of 16 kHz mono = 32000 bytes → 1.0 s."""
    pcm = b"\x00" * (16000 * 2)
    assert processor.get_audio_duration(pcm, 16000, channels=1) == pytest.approx(1.0)


def test_audio_duration_stereo(processor: AudioProcessor) -> None:
    """1 second of 48 kHz stereo = 192000 bytes → 1.0 s."""
    pcm = b"\x00" * (48000 * 2 * 2)
    assert processor.get_audio_duration(pcm, 48000, channels=2) == pytest.approx(1.0)
