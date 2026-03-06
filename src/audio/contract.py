"""Audio format contract enforcement for the pipeline.

Defines canonical audio formats at each pipeline stage and provides
validation functions that fail fast with actionable log messages.

Pipeline stages and their canonical formats:
  Discord recv  →  48 000 Hz, S16LE, stereo  (3 840 B per 20 ms frame)
  Gemini input  →  16 000 Hz, S16LE, mono    (  640 B per 20 ms frame)
  Gemini output →  24 000 Hz, S16LE, mono    (  960 B per 20 ms frame)
  Discord send  →  48 000 Hz, S16LE, stereo  (3 840 B per 20 ms frame)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger("audio.contract")


# ---------------------------------------------------------------------------
# Format definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFormat:
    """Immutable audio format specification."""
    sample_rate: int
    channels: int
    sample_width: int  # bytes per sample (2 = S16LE)
    label: str = ""

    @property
    def bytes_per_frame(self) -> int:
        return self.sample_width * self.channels

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.bytes_per_frame

    def expected_bytes(self, duration_ms: float) -> int:
        return int(self.bytes_per_second * duration_ms / 1000)

    def duration_ms(self, num_bytes: int) -> float:
        if self.bytes_per_frame == 0:
            return 0.0
        num_samples = num_bytes // self.bytes_per_frame
        return num_samples / self.sample_rate * 1000


# Canonical pipeline formats
DISCORD_FORMAT = AudioFormat(sample_rate=48000, channels=2, sample_width=2, label="discord")
GEMINI_INPUT_FORMAT = AudioFormat(sample_rate=16000, channels=1, sample_width=2, label="gemini_in")
GEMINI_OUTPUT_FORMAT = AudioFormat(sample_rate=24000, channels=1, sample_width=2, label="gemini_out")

# Expected frame size for Discord 20 ms frames
DISCORD_FRAME_BYTES = DISCORD_FORMAT.expected_bytes(20)        # 3840
GEMINI_IN_FRAME_BYTES = GEMINI_INPUT_FORMAT.expected_bytes(20)  # 640
GEMINI_OUT_FRAME_BYTES = GEMINI_OUTPUT_FORMAT.expected_bytes(20) # 960


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_pcm_chunk(
    data: bytes,
    expected_format: AudioFormat,
    *,
    label: str = "chunk",
    min_duration_ms: float = 1.0,
    max_duration_ms: float = 5000.0,
    warn_only: bool = True,
) -> bool:
    """Validate a PCM chunk against the expected format.

    Args:
        data: Raw PCM bytes.
        expected_format: The canonical format this stage expects.
        label: Human‐readable stage name for log lines.
        min_duration_ms: Minimum acceptable duration.
        max_duration_ms: Maximum acceptable duration.
        warn_only: If True, log warnings instead of raising.

    Returns:
        True if valid.
    """
    if not data:
        msg = f"[contract:{label}] empty chunk"
        if warn_only:
            logger.warning(msg)
            return False
        raise ValueError(msg)

    # Alignment check: bytes must be divisible by frame size
    if len(data) % expected_format.bytes_per_frame != 0:
        msg = (
            f"[contract:{label}] misaligned — {len(data)} B "
            f"not divisible by {expected_format.bytes_per_frame} "
            f"(expect {expected_format.sample_rate}Hz, "
            f"{expected_format.channels}ch, "
            f"{expected_format.sample_width * 8}bit)"
        )
        if warn_only:
            logger.error(msg)
            return False
        raise ValueError(msg)

    duration = expected_format.duration_ms(len(data))
    if duration < min_duration_ms:
        logger.warning(
            f"[contract:{label}] too short: {duration:.1f} ms < {min_duration_ms} ms"
        )
        return False
    if duration > max_duration_ms:
        logger.warning(
            f"[contract:{label}] too long: {duration:.1f} ms > {max_duration_ms} ms"
        )
        return False

    return True


def compute_audio_stats(data: bytes, sample_width: int = 2) -> dict:
    """Compute RMS, peak amplitude, and silence/clipping flags.

    Args:
        data: Raw PCM bytes (S16LE).
        sample_width: Bytes per sample.

    Returns:
        dict with keys: rms, peak, is_silent, is_clipping, duration_ms.
    """
    if not data or len(data) < sample_width:
        return {
            "rms": 0.0, "peak": 0.0,
            "is_silent": True, "is_clipping": False,
            "num_samples": 0,
        }

    audio = np.frombuffer(data, dtype=np.int16)
    if len(audio) == 0:
        return {
            "rms": 0.0, "peak": 0.0,
            "is_silent": True, "is_clipping": False,
            "num_samples": 0,
        }

    audio_f = audio.astype(np.float32) / 32767.0
    rms = float(np.sqrt(np.mean(audio_f ** 2)))
    peak = float(np.max(np.abs(audio_f)))

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "is_silent": rms < 0.005,
        "is_clipping": peak > 0.99,
        "num_samples": len(audio),
    }


def assert_chunk_contract(
    data: bytes,
    expected_format: AudioFormat,
    *,
    label: str = "chunk",
    seq: Optional[int] = None,
) -> None:
    """Hard assertion — raises on contract violation.

    Use in debug / dry-run paths where you want to fail fast.
    """
    if not data:
        raise ValueError(f"[contract:{label}] seq={seq} — empty chunk")

    if len(data) % expected_format.bytes_per_frame != 0:
        raise ValueError(
            f"[contract:{label}] seq={seq} — misaligned: {len(data)} B, "
            f"frame_size={expected_format.bytes_per_frame}"
        )

    stats = compute_audio_stats(data)
    if stats["is_clipping"]:
        logger.warning(
            f"[contract:{label}] seq={seq} — CLIPPING detected "
            f"(peak={stats['peak']:.4f}, rms={stats['rms']:.4f})"
        )
