"""Custom Discord audio sink for wake word detection with per-user audio processing."""

import asyncio
from typing import Optional, Callable, Awaitable, TYPE_CHECKING, Dict
import discord

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from .capture import AudioCapture

logger = get_logger("audio.sink")

# Resolve Sink base class — py-cord moved sinks between versions.
# Older py-cord: discord.sinks.Sink
# Newer py-cord (master/3.x): discord.voice.receive.AudioSink or similar
# We also handle the case where discord.py is installed instead of py-cord.
_SinkBase = None
_FiltersContainer = None

if hasattr(discord, 'sinks'):
    _SinkBase = discord.sinks.Sink
    _FiltersContainer = discord.sinks.Filters.container
else:
    # py-cord master may have restructured — try alternate import paths
    try:
        from discord.sinks import Sink as _SinkBase, Filters
        _FiltersContainer = Filters.container
    except ImportError:
        pass

if _SinkBase is None:
    # Fallback: define a minimal base class so the module can still load.
    # Audio recording won't work, but the bot won't crash on import.
    logger.error(
        "Could not find discord.sinks.Sink — py-cord[voice] may not be installed correctly. "
        "Audio recording will not work. Install with: "
        "pip install git+https://github.com/Pycord-Development/pycord.git@master#egg=py-cord[voice]"
    )

    class _FallbackSink:
        """Minimal stub so WakeWordSink can be defined without crashing."""
        def __init__(self, **kwargs):
            self.finished = False

    _SinkBase = _FallbackSink

if _FiltersContainer is None:
    # No-op decorator fallback
    def _FiltersContainer(func):
        return func


class WakeWordSink(_SinkBase):
    """Custom sink that receives Discord audio and processes per-user wake word detection.

    This sink receives raw PCM audio from Discord voice connections and
    forwards it to the AudioCapture system for per-user processing.
    Each user's audio is processed separately to enable proper wake word
    detection with 3+ users in the voice channel.
    """

    # Required by py-cord 2.8's SinkEventRouter
    __sink_listeners__: list = []

    def walk_children(self):
        """Yield child sinks (none for this sink)."""
        return iter([])

    def is_opus(self) -> bool:
        """We want decoded PCM, not raw Opus."""
        return False

    def __init__(
        self,
        *,
        capture: Optional["AudioCapture"] = None,
        audio_callback: Optional[Callable[[bytes, int], Awaitable[None]]] = None,
        filters=None,
    ):
        """Initialize the wake word sink.
        
        Args:
            capture: AudioCapture instance to receive audio.
            audio_callback: Optional async callback for raw audio data (data, user_id).
            filters: Optional filters for the sink.
        """
        try:
            super().__init__(filters=filters)
        except TypeError:
            # Fallback base class doesn't accept filters
            super().__init__()
        self._capture = capture
        self._audio_callback = audio_callback
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunk_count = 0
        self._per_user_chunk_count: Dict[int, int] = {}
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet, will be set later
            pass
        
        logger.info("WakeWordSink initialized (per-user audio processing enabled)")
    
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Explicitly set the event loop for cross-thread audio scheduling.

        Must be called from the asyncio thread before recording starts,
        since write() is called from Discord's audio thread where
        get_running_loop() will always fail.

        Args:
            loop: The asyncio event loop to schedule coroutines on.
        """
        self._loop = loop
        logger.debug("Event loop explicitly set on sink")

    def set_capture(self, capture: "AudioCapture") -> None:
        """Set the audio capture instance.
        
        Args:
            capture: AudioCapture instance to receive audio.
        """
        self._capture = capture
        logger.debug("AudioCapture set on sink")
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes, int], Awaitable[None]]],
    ) -> None:
        """Set callback for raw audio data.
        
        Args:
            callback: Async function to call with raw audio bytes and user ID.
        """
        self._audio_callback = callback

    @_FiltersContainer
    def write(self, data, user) -> None:
        """Receive audio data from Discord for a specific user.

        This method is called by Discord's voice system when audio is received.
        The @Filters.container decorator handles user filtering.
        Audio is processed PER-USER to enable proper wake word detection
        even with multiple users speaking.

        Note: py-cord master passes VoiceData objects (with .pcm attribute) and
        Member objects (with .id attribute), not raw bytes and int user IDs.
        We handle both old and new API formats for compatibility.

        Args:
            data: VoiceData object or raw PCM audio bytes (48kHz, 16-bit, stereo).
            user: Member/User object or integer user ID.
        """
        # --- Extract raw PCM bytes from data ---
        # py-cord master: data is a VoiceData object with .pcm attribute
        # py-cord older: data is raw bytes
        if hasattr(data, 'pcm'):
            pcm_data = data.pcm
        elif isinstance(data, (bytes, bytearray)):
            pcm_data = bytes(data)
        else:
            try:
                pcm_data = bytes(data)
            except (TypeError, ValueError):
                logger.error(f"Cannot extract PCM from data type {type(data).__name__}, skipping")
                return

        # --- Extract integer user ID ---
        # py-cord master: user is a Member/User object with .id attribute
        # py-cord older: user is an integer
        if hasattr(user, 'id'):
            user_id = user.id
        elif isinstance(user, int):
            user_id = user
        else:
            try:
                user_id = int(user)
            except (TypeError, ValueError):
                logger.error(f"Cannot extract user ID from type {type(user).__name__}, skipping")
                return

        self._chunk_count += 1

        # Log type info on first chunk for diagnostics
        if self._chunk_count == 1:
            logger.info(
                f"First audio chunk: data_type={type(data).__name__}, "
                f"pcm_size={len(pcm_data)}, user_type={type(user).__name__}, "
                f"user_id={user_id}"
            )

        # Track per-user chunk counts
        if user_id not in self._per_user_chunk_count:
            self._per_user_chunk_count[user_id] = 0
            logger.info(f"New user detected in voice: {user_id}")
        self._per_user_chunk_count[user_id] += 1

        # Log occasionally (per user)
        if self._per_user_chunk_count[user_id] % 500 == 1:
            logger.debug(f"Audio chunk #{self._per_user_chunk_count[user_id]} from user {user_id}: {len(pcm_data)} bytes")

        # Schedule async processing in the event loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No event loop available, cannot process audio")
                return

        # Process audio asynchronously - PER USER
        if self._capture is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._capture.process_discord_audio_per_user(pcm_data, user_id, is_stereo=True),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio processing for user {user_id}: {e}")

        # Call raw audio callback if set (now includes user ID)
        if self._audio_callback is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._audio_callback(pcm_data, user_id),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Error scheduling audio callback for user {user_id}: {e}")
    
    def cleanup(self) -> None:
        """Clean up the sink resources."""
        logger.info(f"WakeWordSink cleanup - processed {self._chunk_count} total audio chunks")
        for user_id, count in self._per_user_chunk_count.items():
            logger.debug(f"  User {user_id}: {count} chunks")
        self._per_user_chunk_count.clear()
        self.finished = True
    
    def get_all_audio(self):
        """Get all recorded audio (required by Sink interface).
        
        We don't store audio, so this returns empty list.
        """
        return []
    
    def format_audio(self, audio):
        """Format audio (required by Sink interface).
        
        We process audio in real-time, so this is a no-op.
        """
        pass
    
    def get_active_users(self) -> list:
        """Get list of users who have sent audio.
        
        Returns:
            List of user IDs that have sent audio.
        """
        return list(self._per_user_chunk_count.keys())
