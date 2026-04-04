"""Audio capture, playback, and processing modules."""

from .processor import AudioProcessor

__all__ = ["AudioCapture", "AudioPlayback", "AudioProcessor", "WakeWordSink"]


def __getattr__(name: str):
    """Lazy-load heavy audio modules so processor-only imports stay lightweight."""
    if name == "AudioCapture":
        from .capture import AudioCapture
        return AudioCapture
    if name == "AudioPlayback":
        from .playback import AudioPlayback
        return AudioPlayback
    if name == "WakeWordSink":
        from .sink import WakeWordSink
        return WakeWordSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
