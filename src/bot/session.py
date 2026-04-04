"""Voice session runtime state for VoiceHandler."""

from dataclasses import dataclass, field
from typing import Optional
import asyncio

import discord


@dataclass
class VoiceRuntimeSession:
    """Mutable runtime state for a single voice-handler session."""

    # Voice connection and channel tracking
    voice_client: Optional[discord.VoiceClient] = None
    target_channel: Optional[discord.VoiceChannel] = None

    # Connection lifecycle events
    connection_ready: asyncio.Event = field(default_factory=asyncio.Event)
    connection_failed: asyncio.Event = field(default_factory=asyncio.Event)

    # Background tasks
    audio_loop_task: Optional[asyncio.Task] = None
    capture_task: Optional[asyncio.Task] = None
    send_task: Optional[asyncio.Task] = None
    receive_task: Optional[asyncio.Task] = None
    pending_reconnect_task: Optional[asyncio.Task] = None

    # Current request/session metadata
    triggered_user_id: Optional[int] = None
    is_capturing_for_gemini: bool = False
    capture_start_time: Optional[float] = None
    audio_chunks_sent: int = 0

    # Runtime buffers/events
    speech_buffer: list[bytes] = field(default_factory=list)
    streaming_complete: asyncio.Event = field(default_factory=asyncio.Event)
    gemini_reconnect_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def cancel_task(self, attr_name: str) -> None:
        """Cancel a task attribute if present, then clear it."""
        task = getattr(self, attr_name)
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        setattr(self, attr_name, None)
