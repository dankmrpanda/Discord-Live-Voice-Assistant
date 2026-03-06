"""Dry-run diagnostic harness for audio pipeline testing.

Modes
-----
WAV dry-run (requires Discord connection)::

    Set diagnostics.dry_run: true in config.yaml, then:
    python -m src.main
    # Bot joins configured VC, captures audio to the configured output path,
    # prints stats, then leaves.

Mock Gemini sink (standalone, no Discord needed)::

    python -m src.diagnostics.wav_harness --mock-sink
    # Generates 10 s of synthetic audio, feeds through mock sink,
    # validates format, reports drops/backpressure.

Analyse an existing WAV file::

    python -m src.diagnostics.wav_harness --analyze out/test.wav
"""

import asyncio
import argparse
import sys
import time
import wave
from pathlib import Path

import importlib
import importlib.util

import numpy as np

# ---------------------------------------------------------------------------
# Direct import of contract.py to avoid triggering the heavy __init__.py
# chain (Discord, dotenv, etc.) when running standalone diagnostics.
# ---------------------------------------------------------------------------
_contract_path = Path(__file__).resolve().parent.parent / "audio" / "contract.py"

def _load_contract():
    """Import contract.py without touching src.audio.__init__."""
    # We need the logger that contract.py depends on
    _project_root = _contract_path.resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    # Provide a lightweight stub for src.utils.logger so contract.py
    # doesn't pull in the full utils package (which needs dotenv).
    import logging
    _stub_module_name = "src.utils.logger"
    if _stub_module_name not in sys.modules:
        import types
        stub = types.ModuleType(_stub_module_name)
        stub.get_logger = logging.getLogger          # type: ignore[attr-defined]
        sys.modules[_stub_module_name] = stub
        # Also ensure the parent packages exist in sys.modules
        for parent in ("src", "src.utils"):
            if parent not in sys.modules:
                p = types.ModuleType(parent)
                p.__path__ = []                       # type: ignore[attr-defined]
                sys.modules[parent] = p

    spec = importlib.util.spec_from_file_location("src.audio.contract", _contract_path)
    mod = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    # Register parent packages so relative imports resolve
    import types as _t
    for pkg in ("src", "src.audio"):
        if pkg not in sys.modules:
            p = _t.ModuleType(pkg)
            p.__path__ = []                           # type: ignore[attr-defined]
            sys.modules[pkg] = p
    sys.modules["src.audio.contract"] = mod
    spec.loader.exec_module(mod)                      # type: ignore[union-attr]
    return mod

_contract = _load_contract()

DISCORD_FORMAT = _contract.DISCORD_FORMAT
GEMINI_INPUT_FORMAT = _contract.GEMINI_INPUT_FORMAT
GEMINI_OUTPUT_FORMAT = _contract.GEMINI_OUTPUT_FORMAT
validate_pcm_chunk = _contract.validate_pcm_chunk
compute_audio_stats = _contract.compute_audio_stats


# ---------------------------------------------------------------------------
# WAV file helpers
# ---------------------------------------------------------------------------

def write_wav(
    filepath: str,
    pcm_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Write raw PCM bytes to a WAV file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    duration = len(pcm_data) / (sample_rate * channels * sample_width)
    print(f"  ✓ Wrote {filepath}: {len(pcm_data):,} bytes, {duration:.2f} s")


def analyze_wav(filepath: str) -> dict:
    """Read a WAV file and print format + amplitude stats."""
    with wave.open(filepath, "rb") as wf:
        params = {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
            "num_frames": wf.getnframes(),
        }
        pcm_data = wf.readframes(wf.getnframes())

    duration = params["num_frames"] / params["sample_rate"]
    stats = compute_audio_stats(pcm_data)

    print(f"\n  WAV analysis: {filepath}")
    print(f"    Format  : {params['sample_rate']} Hz, {params['channels']} ch, "
          f"{params['sample_width'] * 8}-bit")
    print(f"    Duration: {duration:.3f} s ({params['num_frames']:,} frames)")
    print(f"    RMS     : {stats['rms']:.6f}")
    print(f"    Peak    : {stats['peak']:.6f}")
    print(f"    Silent  : {stats['is_silent']}")
    print(f"    Clipping: {stats['is_clipping']}")
    return {**params, "duration_s": round(duration, 3), **stats}


# ---------------------------------------------------------------------------
# Mock Gemini sink
# ---------------------------------------------------------------------------

class MockGeminiSink:
    """Validates and counts PCM chunks.  Simulates bounded backpressure."""

    def __init__(self, max_queue_seconds: float = 2.0):
        max_chunks = int(max_queue_seconds / 0.02)  # ~100 for 2 s of 20 ms frames
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_chunks)
        self.chunks_accepted = 0
        self.chunks_dropped = 0
        self.total_bytes = 0
        self.total_duration_ms = 0.0
        self.contract_violations = 0
        self._start_time = 0.0

    async def send_audio(self, chunk: bytes) -> bool:
        """Accept a PCM chunk; return False if dropped."""
        if self._start_time == 0:
            self._start_time = time.time()

        if not validate_pcm_chunk(
            chunk, GEMINI_INPUT_FORMAT, label="mock_sink", warn_only=True
        ):
            self.contract_violations += 1
            return False

        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self.chunks_dropped += 1
            return False

        self.chunks_accepted += 1
        self.total_bytes += len(chunk)
        self.total_duration_ms += GEMINI_INPUT_FORMAT.duration_ms(len(chunk))
        return True

    def report(self) -> None:
        elapsed = time.time() - self._start_time if self._start_time else 0
        total = self.chunks_accepted + self.chunks_dropped
        drop_pct = (
            f"{self.chunks_dropped / total * 100:.1f} %" if total else "0 %"
        )
        print(f"\n  Mock Gemini Sink Report")
        print(f"    Elapsed        : {elapsed:.2f} s")
        print(f"    Chunks accepted: {self.chunks_accepted}")
        print(f"    Chunks dropped : {self.chunks_dropped} ({drop_pct})")
        print(f"    Violations     : {self.contract_violations}")
        print(f"    Total bytes    : {self.total_bytes:,}")
        print(f"    Audio duration : {self.total_duration_ms / 1000:.2f} s")
        print(f"    Queue depth    : {self._queue.qsize()}")


# ---------------------------------------------------------------------------
# WAV capture helper (used by dry-run mode in voice_handler)
# ---------------------------------------------------------------------------

class WavCaptureCollector:
    """Collects 16 kHz mono PCM chunks and writes a WAV on flush.

    Used by the dry-run mode built into VoiceHandler.
    """

    def __init__(self, output_path: str = "out/test.wav", max_seconds: float = 10.0):
        self.output_path = output_path
        self.max_bytes = int(GEMINI_INPUT_FORMAT.bytes_per_second * max_seconds)
        self._chunks: list[bytes] = []
        self._total_bytes = 0
        self._start_time = 0.0
        self._packets = 0

    @property
    def is_full(self) -> bool:
        return self._total_bytes >= self.max_bytes

    def add_chunk(self, pcm: bytes) -> None:
        if self._start_time == 0:
            self._start_time = time.time()
        if self._total_bytes >= self.max_bytes:
            return
        self._chunks.append(pcm)
        self._total_bytes += len(pcm)
        self._packets += 1

    def flush(self) -> dict:
        """Write WAV and return stats dict."""
        pcm = b"".join(self._chunks)
        write_wav(
            self.output_path,
            pcm,
            sample_rate=GEMINI_INPUT_FORMAT.sample_rate,
            channels=GEMINI_INPUT_FORMAT.channels,
            sample_width=GEMINI_INPUT_FORMAT.sample_width,
        )
        elapsed = time.time() - self._start_time if self._start_time else 0
        stats = compute_audio_stats(pcm)
        result = {
            "packets": self._packets,
            "total_bytes": self._total_bytes,
            "elapsed_s": round(elapsed, 3),
            "packets_per_sec": round(self._packets / elapsed, 1) if elapsed else 0,
            "bytes_per_sec": round(self._total_bytes / elapsed, 1) if elapsed else 0,
            **stats,
        }
        print(f"\n  Dry-Run Capture Stats")
        print(f"    Packets       : {result['packets']}")
        print(f"    Elapsed       : {result['elapsed_s']} s")
        print(f"    Packets/sec   : {result['packets_per_sec']}")
        print(f"    Decoded B/sec : {result['bytes_per_sec']:,.0f}")
        print(f"    RMS           : {result['rms']:.6f}")
        print(f"    Peak          : {result['peak']:.6f}")
        print(f"    Silent        : {result['is_silent']}")
        print(f"    Clipping      : {result['is_clipping']}")
        self._chunks.clear()
        self._total_bytes = 0
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _run_mock_sink_test() -> None:
    """Generate synthetic 10 s of audio and validate through mock sink."""
    print("\n  Mock-sink test: 10 s of 440 Hz sine → bounded queue\n")
    sink = MockGeminiSink(max_queue_seconds=2.0)

    async def _test():
        sr = GEMINI_INPUT_FORMAT.sample_rate
        duration_s = 10.0
        t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)

        chunk_samples = int(sr * 0.02)
        total_chunks = len(audio) // chunk_samples

        for i in range(total_chunks):
            start = i * chunk_samples
            chunk = audio[start : start + chunk_samples].tobytes()
            await sink.send_audio(chunk)
            # Simulate real-time pace
            await asyncio.sleep(0.001)

        sink.report()

    asyncio.run(_test())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio pipeline diagnostic harness"
    )
    parser.add_argument(
        "--mock-sink", action="store_true",
        help="Run mock Gemini sink backpressure test",
    )
    parser.add_argument(
        "--analyze", type=str, metavar="FILE",
        help="Analyze an existing WAV file",
    )
    args = parser.parse_args()

    if args.analyze:
        analyze_wav(args.analyze)
    elif args.mock_sink:
        _run_mock_sink_test()
    else:
        parser.print_help()
        print("\n  Tip: For full Discord dry-run, set diagnostics.dry_run: true in config.yaml and run the bot.")
