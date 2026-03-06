"""Voice handler with state machine for managing voice interactions."""

import asyncio
from enum import Enum
from typing import Optional, TYPE_CHECKING

import discord
import numpy as np
from discord.ext import voice_recv

from ..utils.logger import get_logger
from ..audio.capture import AudioCapture
from ..audio.playback import AudioPlayback
from ..audio.processor import AudioProcessor
from ..audio.sink import WakeWordSink
from ..wake_word.detector import WakeWordDetector
from ..ai.gemini_client import GeminiLiveClient

if TYPE_CHECKING:
    from ..utils.config import Config

logger = get_logger("bot.voice_handler")


class BotState(Enum):
    """State machine states for voice interaction."""
    IDLE = "idle"  # Not in a voice channel
    CONNECTING = "connecting"  # Currently joining a voice channel
    LISTENING = "listening"  # In channel, listening for wake word
    PROCESSING = "processing"  # Wake word detected, capturing user speech
    SPEAKING = "speaking"  # Playing response audio


class VoiceHandler:
    """Manages voice channel connections and the voice interaction pipeline.
    
    Implements a state machine:
    IDLE -> CONNECTING (on join request)
    CONNECTING -> LISTENING (on successful connection)
    LISTENING -> PROCESSING (on wake word detection)
    PROCESSING -> SPEAKING (on response ready)
    SPEAKING -> LISTENING (on playback complete)
    """
    
    def __init__(self, config: "Config"):
        """Initialize the voice handler.
        
        Args:
            config: Application configuration.
        """
        logger.debug("Initializing VoiceHandler")
        logger.debug(f"Config: wake_phrase={config.wake_phrase}, voice={config.gemini_voice}, threshold={config.wake_word_threshold}")
        
        self.config = config
        
        # Capture settings from config
        self.capture_duration = getattr(config, 'capture_duration', 5.0)
        self.silence_threshold = getattr(config, 'silence_threshold', 0.5)
        self.log_audio = getattr(config, 'log_audio', False)
        self._state = BotState.IDLE
        self._state_lock = asyncio.Lock()
        
        # Voice client
        self._voice_client: Optional[voice_recv.VoiceRecvClient] = None
        self._target_channel: Optional[discord.VoiceChannel] = None
        
        # Connection management
        self._connection_ready = asyncio.Event()
        self._connection_failed = asyncio.Event()
        self._audio_loop_task: Optional[asyncio.Task] = None
        
        # Components
        logger.debug("Creating AudioProcessor")
        self._processor = AudioProcessor(
            discord_sample_rate=config.discord_sample_rate,
            gemini_input_sample_rate=config.gemini_input_sample_rate,
            gemini_output_sample_rate=config.gemini_output_sample_rate,
            input_gain=getattr(config, 'input_gain', 0.5),
        )
        
        self._capture = AudioCapture(
            self._processor,
            silence_threshold=self.silence_threshold,
        )
        self._capture.contract_validation = config.diag_audio_contract_validation
        self._playback = AudioPlayback(
            self._processor,
            buffer_ms=config.playback_buffer_ms,
        )
        
        # Store event loop for run_coroutine_threadsafe (used by BasicSink callback)
        self._event_loop = asyncio.get_running_loop()

        # Listen recovery tracking
        self._last_relisten_time: float = 0.0
        self._relisten_count: int = 0

        # Create sink helper for audio stats and diagnostics
        self._sink = WakeWordSink(
            dump_raw_audio=config.diag_dump_raw_audio,
            dump_audio_dir=config.diag_dump_audio_dir,
        )
        
        self._wake_detector = WakeWordDetector(
            wake_phrase=config.wake_phrase,
            threshold=config.wake_word_threshold,
            sample_rate=config.gemini_input_sample_rate,
            verbose=self.log_audio,
        )
        
        logger.debug("Creating GeminiLiveClient")
        self._gemini = GeminiLiveClient(
            api_key=config.gemini_api_key,
            voice=config.gemini_voice,
            system_instruction=getattr(config, 'system_prompt', None),
            model=config.gemini_model,
            thinking=config.gemini_thinking,
            google_search=config.gemini_google_search,
            function_calling=config.gemini_function_calling,
            automatic_function_response=config.gemini_automatic_function_response,
        )
        
        # Audio buffer for post-wake word capture
        self._speech_buffer: list[bytes] = []
        self._capture_task: Optional[asyncio.Task] = None
        
        # Track which user triggered wake word
        self._triggered_user_id: Optional[int] = None
        
        # Real-time streaming state
        self._is_capturing_for_gemini = False
        self._capture_start_time: Optional[float] = None
        self._audio_chunks_sent = 0
        
        # Streaming tasks
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._streaming_complete = asyncio.Event()

        # Health reporter task (logs 1/sec pipeline stats)
        self._health_reporter_task: Optional[asyncio.Task] = None

        # Session tracking for structured logs
        self._session_id: str = ""
        self._guild_id: Optional[int] = None
        self._channel_id: Optional[int] = None

        # Dry-run mode (WAV capture instead of Gemini)
        self._dry_run = False
        self._dry_run_task: Optional[asyncio.Task] = None
        self._wav_collector = None

        # Fire-and-forget task for config reconnects
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Queue for /ask commands (prompt, user_id)
        self._ask_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=20)
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Register for config changes
        self.config.add_change_listener(self._on_config_changed)
        
        logger.debug("VoiceHandler initialization complete")

    def __del__(self) -> None:
        try:
            self.config.remove_change_listener(self._on_config_changed)
        except Exception:
            pass

    def _on_config_changed(self, config: "Config", changed_fields: list) -> None:
        """Handle configuration changes.
        
        Args:
            config: The updated config object.
            changed_fields: List of field names that changed.
        """
        logger.info(f"🔄 Config changed: {', '.join(changed_fields)}")
        
        # Update local cached values
        if "capture_duration" in changed_fields:
            self.capture_duration = config.capture_duration
            logger.info(f"  → Capture duration: {self.capture_duration}s")
        
        if "silence_threshold" in changed_fields:
            self.silence_threshold = config.silence_threshold
            # Also update the AudioCapture's VAD threshold
            self._capture.set_silence_threshold(config.silence_threshold)
            logger.info(f"  → Silence threshold: {self.silence_threshold}s")
        
        if "playback_buffer_ms" in changed_fields:
            self._playback.buffer_ms = config.playback_buffer_ms
            logger.info(f"  → Playback buffer: {config.playback_buffer_ms}ms")
        
        if "input_gain" in changed_fields:
            self._processor.input_gain = np.clip(config.input_gain, 0.01, 2.0)
            logger.info(f"  → Input gain: {self._processor.input_gain}")
        
        if "log_audio" in changed_fields:
            self.log_audio = config.log_audio
            self._wake_detector.verbose = config.log_audio
            logger.info(f"  → Log audio: {self.log_audio}")
        
        # Update wake word detector
        if "wake_phrase" in changed_fields or "wake_word_threshold" in changed_fields:
            logger.info(f"  → Wake phrase: '{config.wake_phrase_display}' (threshold: {config.wake_word_threshold})")
            # Recreate wake word detector with new settings
            self._wake_detector = WakeWordDetector(
                wake_phrase=config.wake_phrase,
                threshold=config.wake_word_threshold,
                sample_rate=config.gemini_input_sample_rate,
                verbose=self.log_audio,
            )
            self._wake_detector.set_detection_callback(self._on_wake_word_detected)
            logger.info("  → Wake word detector recreated")
        
        # Update Gemini client if voice, model, thinking, google_search, function_calling, automatic_function_response, or system prompt changed
        gemini_changed = any(f in changed_fields for f in [
            "gemini_voice", "gemini_model", "gemini_thinking", 
            "gemini_google_search", "gemini_function_calling", 
            "gemini_automatic_function_response", "system_prompt"
        ])
        if gemini_changed:
            if "gemini_voice" in changed_fields:
                logger.info(f"  → Gemini voice: {config.gemini_voice}")
            if "gemini_model" in changed_fields:
                logger.info(f"  → Gemini model: {config.gemini_model}")
            if "gemini_thinking" in changed_fields:
                logger.info(f"  → Gemini thinking: {config.gemini_thinking}")
            if "gemini_google_search" in changed_fields:
                logger.info(f"  → Gemini Google Search: {config.gemini_google_search}")
            if "gemini_function_calling" in changed_fields:
                logger.info(f"  → Gemini function calling: {config.gemini_function_calling}")
            if "gemini_automatic_function_response" in changed_fields:
                logger.info(f"  → Gemini automatic function response: {config.gemini_automatic_function_response}")
            # Need to reconnect Gemini with new settings
            self._reconnect_task = asyncio.create_task(self._reconnect_gemini_with_new_config())
            self._reconnect_task.add_done_callback(self._task_exception_handler)

        # Diagnostics settings
        if "diag_audio_contract_validation" in changed_fields:
            self._capture.contract_validation = config.diag_audio_contract_validation
            logger.info(f"  -> Audio contract validation: {config.diag_audio_contract_validation}")
    
    async def _reconnect_gemini_with_new_config(self) -> None:
        """Reconnect to Gemini with updated configuration."""
        try:
            logger.info("🔄 Reconnecting to Gemini with new config...")
            
            # Disconnect existing session
            if self._gemini.is_connected:
                await self._gemini.disconnect()
            
            # Create new client with updated settings
            self._gemini = GeminiLiveClient(
                api_key=self.config.gemini_api_key,
                voice=self.config.gemini_voice,
                system_instruction=getattr(self.config, 'system_prompt', None),
                model=self.config.gemini_model,
                thinking=self.config.gemini_thinking,
                google_search=self.config.gemini_google_search,
                function_calling=self.config.gemini_function_calling,
                automatic_function_response=self.config.gemini_automatic_function_response,
            )
            
            # Reconnect if we're in a voice channel
            if self._state != BotState.IDLE:
                if await self._gemini.connect():
                    logger.info("✓ Reconnected to Gemini with new settings")
                else:
                    logger.error("Failed to reconnect to Gemini")
            else:
                logger.info("✓ Gemini client updated (will connect when joining voice)")
                
        except Exception as e:
            logger.error(f"Error reconnecting Gemini: {e}")
    
    def _task_exception_handler(self, task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Unhandled exception in background task {task.get_name()}: {exc}")

    def _setup_callbacks(self) -> None:
        """Set up callbacks between components."""
        # Wake word detection callback (now receives user_id)
        self._wake_detector.set_detection_callback(self._on_wake_word_detected)
        
        # Playback completion callback
        self._playback.set_after_callback(self._on_playback_complete_sync)
        
        # Audio capture callback (receives audio and user_id)
        self._capture.set_audio_callback(self._on_audio_chunk_received)
    
    async def _on_audio_chunk_received(self, audio_data: bytes, user_id: int) -> None:
        """Callback when audio chunk is received from a user.

        This is called for each processed audio chunk and handles per-user
        wake word detection.

        Args:
            audio_data: PCM audio bytes (16kHz, mono).
            user_id: Discord user ID this audio came from.
        """
        if self._state == BotState.LISTENING and user_id != 0:
            # Process wake word detection for this specific user
            await self._wake_detector.process_audio_for_user(audio_data, user_id)

    def _on_voice_audio_received(self, user, data) -> None:
        """Synchronous callback from voice_recv.BasicSink.

        Called on the PacketRouter thread when decoded audio arrives from
        Discord. Must be fast and thread-safe.

        Args:
            user: discord.User who sent the audio, or None.
            data: voice_recv.VoiceData with .pcm attribute (48kHz stereo).
        """
        try:
            pcm_data = data.pcm
            user_id = user.id if user else 0

            # Forward to sink helper for stats and raw audio dump
            self._sink.handle_audio(pcm_data, user_id)

            # Schedule async audio processing in the event loop
            if self._event_loop and not self._event_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._capture.process_discord_audio_per_user(pcm_data, user_id, is_stereo=True),
                    self._event_loop,
                )
        except Exception as e:
            logger.error(f"Error in voice audio callback: {e}")

    def _start_listening(self) -> None:
        """Create a BasicSink and start listening on the voice client.

        This is idempotent -- if the voice client is already listening it
        will stop first, then re-register.  Used both for initial join and
        for automatic recovery when the reader dies.
        """
        if not self._voice_client:
            logger.warning("Cannot start listening: no voice client")
            return

        # Stop existing listener if any
        try:
            if self._voice_client.is_listening():
                self._voice_client.stop_listening()
        except Exception:
            pass

        sink = voice_recv.BasicSink(self._on_voice_audio_received)
        self._voice_client.listen(sink)
        logger.info("Voice listening started (BasicSink) - now receiving audio from Discord")
    
    @property
    def state(self) -> BotState:
        """Get current bot state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to a voice channel."""
        return self._voice_client is not None and self._voice_client.is_connected()
    
    @property
    def is_connecting(self) -> bool:
        """Check if currently connecting to a voice channel."""
        return self._state == BotState.CONNECTING
    
    async def _set_state(self, new_state: BotState) -> None:
        """Set bot state with logging."""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            logger.info(
                f"State transition: {old_state.value} -> {new_state.value} "
                f"[session={self._session_id}]"
            )
    
    async def _wait_for_voice_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the voice connection to be fully ready.

        With discord.py 2.7+, the connect() method should return a properly
        connected voice client. This method provides a small grace period
        and verifies the connection is stable.
        
        Args:
            timeout: Maximum time to wait for connection (seconds).
            
        Returns:
            True if connection is ready, False if timed out or failed.
        """
        if not self._voice_client:
            return False
        
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.5
        
        logger.debug(f"Verifying voice connection (timeout={timeout}s)")
        
        # Give the connection a moment to settle after connect() returns
        await asyncio.sleep(0.5)
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                if self._voice_client is None:
                    logger.warning("Voice client was destroyed during verification")
                    return False
                
                # Check if the voice client is connected
                if self._voice_client.is_connected():
                    logger.info("Voice connection verified successfully")
                    return True
                
                logger.debug("Waiting for voice connection...")
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error checking voice connection status: {e}")
                await asyncio.sleep(check_interval)
        
        logger.warning(f"Voice connection verification timed out after {timeout}s")
        return False
    
    async def join_channel(
        self,
        channel: discord.VoiceChannel,
    ) -> bool:
        """Join a voice channel and start listening.
        
        Args:
            channel: Discord voice channel to join.
            
        Returns:
            True if successfully joined, False otherwise.
        """
        try:
            logger.info(f"Joining voice channel: {channel.name} (ID: {channel.id})")
            logger.debug(f"Channel details: guild={channel.guild.name}, members={len(channel.members)}")
            
            # Set state to connecting
            await self._set_state(BotState.CONNECTING)
            self._target_channel = channel
            self._connection_ready.clear()
            self._connection_failed.clear()
            
            # Connect to voice channel
            # Use reconnect=False for initial connection to fail fast with the
            # actual error instead of retrying in a loop for 20s and swallowing
            # the error. After successful connection, we enable reconnect for
            # ongoing stability.
            logger.debug("Connecting to voice channel...")
            max_attempts = 2
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    self._voice_client = await asyncio.wait_for(
                        channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30.0, reconnect=False),
                        timeout=35.0
                    )
                    last_error = None
                    break
                except asyncio.TimeoutError:
                    last_error = "Voice connection timed out"
                    logger.error(f"Voice connection attempt {attempt}/{max_attempts} timed out")
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    logger.error(f"Voice connection attempt {attempt}/{max_attempts} failed: {last_error}")

                # Check if the bot connected despite the error
                existing_vc = channel.guild.voice_client
                if existing_vc and existing_vc.is_connected():
                    logger.info("Bot connected to voice despite error, using existing connection")
                    self._voice_client = existing_vc
                    last_error = None
                    break

                # Clean up stale state before retry
                if existing_vc:
                    try:
                        await existing_vc.disconnect(force=True)
                    except Exception:
                        pass

                if attempt < max_attempts:
                    logger.info(f"Retrying voice connection in 1s...")
                    await asyncio.sleep(1.0)

            if last_error:
                raise Exception(
                    f"Failed to connect to voice channel after {max_attempts} attempts: {last_error}"
                )

            # Enable reconnect for ongoing stability (handles 4017 Reconnect Required)
            if hasattr(self._voice_client, 'reconnect'):
                self._voice_client.reconnect = True
            
            logger.debug(f"Voice client obtained: {self._voice_client}")
            
            # Verify the connection is ready
            if not await self._wait_for_voice_ready(timeout=10.0):
                raise Exception("Voice connection verification failed")
            
            logger.info("Voice connection is ready")
            
            # Set up playback with voice client
            logger.debug("Setting up playback with voice client")
            self._playback.set_voice_client(self._voice_client)
            
            # Start audio capture
            logger.debug("Starting audio capture")
            await self._capture.start()
            
            # Enable wake word detection
            logger.debug("Enabling wake word detection")
            self._wake_detector.enable()
            
            # Start listening with BasicSink to receive audio from Discord
            logger.debug("Starting voice listening with BasicSink")
            self._start_listening()
            
            # Start the audio processing loop (for wake word detection)
            logger.debug("Starting audio receive loop")
            self._audio_loop_task = asyncio.create_task(self._audio_receive_loop())

            # Track session metadata for structured logging
            import uuid
            self._session_id = uuid.uuid4().hex[:12]
            self._guild_id = channel.guild.id
            self._channel_id = channel.id
            logger.info(f"[session={self._session_id}] guild={self._guild_id} channel={self._channel_id}")

            # Start health reporter (1/sec diagnostic line)
            self._health_reporter_task = asyncio.create_task(self._health_reporter_loop())
            
            # Connect to Gemini
            logger.debug("Connecting to Gemini Live API")
            await self._gemini.connect()
            logger.debug("Gemini connection established")
            
            # Start Gemini health check / keep-alive task
            logger.debug("Starting Gemini health check background task")
            await self._gemini.start_health_check()
            
            # Check for dry-run mode
            if self.config.diag_dry_run:
                logger.info(
                    f"DRY_RUN mode -- capturing {self.config.diag_dry_run_duration}s "
                    f"of audio to {self.config.diag_dry_run_output}, then leaving"
                )
                self._dry_run = True
                self._dry_run_task = asyncio.create_task(self._run_dry_run_capture())
                self._dry_run_task.add_done_callback(self._task_exception_handler)
            
            # Transition to listening state
            await self._set_state(BotState.LISTENING)
            
            logger.info(f"Joined channel '{channel.name}', listening for '{self.config.wake_phrase_display}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            await self.leave_channel()
            return False
    
    async def leave_channel(self) -> None:
        """Leave the current voice channel and clean up resources."""
        logger.info("Leaving voice channel")
        
        # Clear the /ask queue on leave
        queue_size = self._ask_queue.qsize()
        if queue_size > 0:
            logger.info(f"Clearing /ask queue ({queue_size} items)")
            while not self._ask_queue.empty():
                try:
                    self._ask_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Stop listening first (before disabling other components)
        if self._voice_client:
            try:
                if self._voice_client.is_listening():
                    logger.debug("Stopping voice listening")
                    self._voice_client.stop_listening()
            except Exception as e:
                logger.warning(f"Error stopping listening: {e}")

        # Clean up sink
        if self._sink:
            try:
                self._sink.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up sink: {e}")
        
        # Stop components first
        self._wake_detector.disable()
        self._capture.stop_streaming_to_gemini()
        await self._capture.stop()
        self._playback.stop()
        
        # Cancel audio loop task
        if self._audio_loop_task:
            self._audio_loop_task.cancel()
            try:
                await self._audio_loop_task
            except asyncio.CancelledError:
                pass
            self._audio_loop_task = None

        # Cancel health reporter
        if self._health_reporter_task:
            self._health_reporter_task.cancel()
            try:
                await self._health_reporter_task
            except asyncio.CancelledError:
                pass
            self._health_reporter_task = None
        
        # Cancel capture task
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        
        # Cancel send/receive tasks
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Cancel fire-and-forget tasks
        for attr in ("_reconnect_task", "_dry_run_task"):
            task = getattr(self, attr, None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            setattr(self, attr, None)

        # Stop Gemini health check and disconnect
        await self._gemini.stop_health_check()
        await self._gemini.disconnect()
        
        # Remove config change listener to prevent memory leak
        self.config.remove_change_listener(self._on_config_changed)
        logger.debug("Removed config change listener")
        
        # Clean up all user resources
        self._cleanup_all_users()
        
        # Disconnect from voice
        if self._voice_client:
            try:
                if self._voice_client.is_connected():
                    await self._voice_client.disconnect(force=True)
            except Exception as e:
                logger.warning(f"Error disconnecting voice client: {e}")
            self._voice_client = None
        
        # Clear target channel
        self._target_channel = None
        
        # Reset connection events
        self._connection_ready.clear()
        self._connection_failed.clear()
        
        await self._set_state(BotState.IDLE)
    
    async def _audio_receive_loop(self) -> None:
        """Main loop for receiving and processing audio from Discord.
        
        This loop monitors connection state and performs periodic housekeeping.
        Wake word detection is now handled per-user in _on_audio_chunk_received.
        """
        logger.debug("Audio receive loop started")
        loop_count = 0
        
        while self.is_connected:
            try:
                loop_count += 1
                if loop_count % 200 == 0:  # Log every 10 seconds (200 * 50ms)
                    active_users = self._capture.get_active_users()
                    detector_users = self._wake_detector.get_active_users()
                    logger.debug(f"Audio loop heartbeat: state={self._state.value}, loops={loop_count}, "
                                f"capture_users={len(active_users)}, detector_users={len(detector_users)}")
                
                # Note: Wake word detection is now handled in _on_audio_chunk_received
                # which is called per-user when audio is processed
                
                await asyncio.sleep(0.05)  # 50ms polling interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio receive loop: {e}")
                await asyncio.sleep(0.1)

    async def _health_reporter_loop(self) -> None:
        """Log a structured health line every second.

        Also acts as a watchdog: if the voice_recv listener dies (e.g. due
        to a voice gateway reconnect or PacketRouter crash), this loop
        detects it and re-establishes listening automatically.
        """
        import time as _time

        logger.debug("[health] Health reporter started")
        last_pushed = 0
        last_consumed = 0
        ticks_without_audio = 0
        RELISTEN_COOLDOWN = 5.0  # seconds between re-listen attempts

        while self.is_connected:
            try:
                h = self._capture.get_pipeline_health()
                active_users = self._capture.get_active_users()
                gemini_state = self._gemini.state.value if self._gemini else "none"

                # Compute throughput delta since last tick
                pushed_delta = h["total_pushed"] - last_pushed
                consumed_delta = h["total_consumed"] - last_consumed
                last_pushed = h["total_pushed"]
                last_consumed = h["total_consumed"]

                # Check if voice_recv listener is alive
                listening = False
                try:
                    listening = bool(self._voice_client and self._voice_client.is_listening())
                except Exception:
                    pass

                logger.debug(
                    f"[health] session={self._session_id} "
                    f"state={self._state.value} "
                    f"users={len(active_users)} "
                    f"sink_chunks={self._sink._chunk_count if self._sink else '?'} "
                    f"listening={listening} "
                    f"relistens={self._relisten_count} "
                    f"buf={h['buffer_depth']}/{h['buffer_max']} "
                    f"pushed/s={pushed_delta} consumed/s={consumed_delta} "
                    f"drops={h['total_drops']} "
                    f"gemini={gemini_state} "
                    f"speech={h['speech_detected']} "
                    f"ask_q={self._ask_queue.qsize()}"
                )

                # --- Listen recovery watchdog ---
                if (
                    not listening
                    and self._voice_client
                    and self._voice_client.is_connected()
                    and self._state in (BotState.LISTENING, BotState.PROCESSING)
                ):
                    now = _time.monotonic()
                    if (now - self._last_relisten_time) >= RELISTEN_COOLDOWN:
                        self._relisten_count += 1
                        self._last_relisten_time = now
                        logger.warning(
                            f"[health] Listener is dead! Re-establishing listening "
                            f"(attempt #{self._relisten_count}) [session={self._session_id}]"
                        )
                        try:
                            self._start_listening()
                        except Exception as e:
                            logger.error(f"[health] Failed to re-establish listening: {e}")

                # Warn if no audio received after several seconds
                sink_chunks = self._sink._chunk_count if self._sink else 0
                if sink_chunks == 0:
                    ticks_without_audio += 1
                    if ticks_without_audio == 5:
                        logger.warning(
                            "[health] No audio received from Discord after 5s. "
                            "Check: is voice_recv delivering packets? "
                            "Is another user actually in the channel and unmuted?"
                        )
                    elif ticks_without_audio == 15:
                        logger.warning(
                            "[health] Still no audio after 15s. "
                            "Discord voice_recv may not be working. "
                            "sink.handle_audio() has never been called."
                        )
                else:
                    ticks_without_audio = 0

                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[health] reporter error: {e}")
                await asyncio.sleep(1.0)

        logger.debug("[health] Health reporter stopped")

    async def _run_dry_run_capture(self) -> None:
        """Dry-run mode: capture audio to WAV, print stats, leave.

        Activated by diagnostics.dry_run in config.yaml.
        """
        from ..diagnostics.wav_harness import WavCaptureCollector
        import time

        collector = WavCaptureCollector(
            output_path=self.config.diag_dry_run_output,
            max_seconds=self.config.diag_dry_run_duration,
        )
        original_cb = self._capture._audio_callback

        # Intercept the per-user audio callback to also feed the collector
        async def _dry_run_cb(audio_data: bytes, user_id: int) -> None:
            if user_id != 0:
                collector.add_chunk(audio_data)
            if original_cb:
                await original_cb(audio_data, user_id)

        self._capture.set_audio_callback(_dry_run_cb)
        duration = self.config.diag_dry_run_duration
        logger.info(f"Dry-run: recording for {duration} seconds...")

        start = time.time()
        timeout = duration + 2.0
        while not collector.is_full and (time.time() - start) < timeout:
            await asyncio.sleep(0.1)

        # Restore original callback
        self._capture.set_audio_callback(original_cb)

        stats = collector.flush()
        logger.info(f"Dry-run capture complete: {stats}")

        # Leave voice after dry-run
        logger.info("Dry-run finished -- disconnecting")
        await self.leave_channel()

    async def _on_wake_word_detected(self, user_id: int) -> None:
        """Handle wake word detection from a specific user.
        
        Args:
            user_id: Discord user ID who triggered the wake word.
        """
        if self._state != BotState.LISTENING:
            logger.debug("Wake word detected but not in LISTENING state, ignoring")
            return
        
        logger.info(f"🎤 Wake word detected from user {user_id}! Starting low-latency streaming..."
                     f" [session={self._session_id}]")
        
        # Store which user triggered the wake word
        self._triggered_user_id = user_id
        
        # Set this user as the active user for audio capture
        # This means only their audio will be captured for the prompt
        self._capture.set_active_user(user_id)
        
        # Transition to processing state
        await self._set_state(BotState.PROCESSING)
        
        # Disable wake word detection during processing
        self._wake_detector.disable()
        
        # Clear any old buffer data (but keep user-specific buffer for their speech)
        self._speech_buffer.clear()
        self._capture.clear_buffer()  # Clear shared buffer, user buffer preserved
        
        # Start the streaming pipeline (concurrent send/receive)
        self._capture_task = asyncio.create_task(self._run_streaming_pipeline())
    
    async def _run_streaming_pipeline(self) -> None:
        """Run the full streaming pipeline with sequential send then receive.
        
        This method:
        1. Captures and sends user audio to Gemini until speech ends
        2. Signals end_turn() to tell Gemini user is done speaking
        3. Receives and plays Gemini's audio response
        
        IMPORTANT: We use sequential (not concurrent) send/receive because
        Gemini may send turn_complete prematurely if we start receiving
        before sending end_turn(). This caused 0 audio chunks to be received.
        """
        try:
            import time
            
            logger.info("🚀 Starting low-latency streaming pipeline")
            
            # Check if Gemini needs reconnection
            if not self._gemini.is_connected:
                logger.warning(f"Gemini not connected (state={self._gemini.state}), attempting reconnect...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini")
                    await self._reset_to_listening()
                    return
                logger.info("✓ Successfully reconnected to Gemini")
            
            # Initialize streaming state
            self._is_capturing_for_gemini = True
            self._capture_start_time = time.time()
            self._audio_chunks_sent = 0
            self._streaming_complete.clear()
            
            # Start streaming mode in capture (enables direct queue pushing)
            self._capture.start_streaming_to_gemini()
            
            # Transition to processing state while capturing user speech
            await self._set_state(BotState.PROCESSING)
            
            # IMPORTANT: Sequential pipeline - send first, then receive
            # Gemini sends turn_complete when it thinks user is done speaking.
            # If we start receiving before calling end_turn(), Gemini may send
            # turn_complete prematurely with 0 audio chunks.
            
            # Phase 1: Capture and send all user audio
            self._send_task = asyncio.create_task(self._send_audio_loop())
            try:
                await self._send_task
            except Exception as e:
                logger.error(f"Error in send loop: {e}")
            
            # Phase 2: Start streaming playback (will buffer until chunks arrive)
            playback_started = self._playback.start_streaming_playback()
            if not playback_started:
                logger.error("Failed to start streaming playback")
                await self._reset_to_listening()
                return
            
            # Transition to speaking state now that we're receiving
            await self._set_state(BotState.SPEAKING)
            
            # Phase 3: Receive and play Gemini's response
            self._receive_task = asyncio.create_task(self._receive_response_loop())
            try:
                await self._receive_task
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
            
            # Cleanup
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
            
            # Wait for playback to complete
            logger.debug("Waiting for playback to complete...")
            playback_finished = await self._playback.wait_for_playback(timeout=60.0)
            if not playback_finished:
                logger.warning("Playback timed out after 60s, forcing stop")
                self._playback.stop()
            logger.info("✓ Streaming pipeline complete")
            
            await self._reset_to_listening()
            
        except asyncio.CancelledError:
            logger.debug("Streaming pipeline cancelled")
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
        except Exception as e:
            logger.error(f"❌ Error in streaming pipeline: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
            await self._reset_to_listening()
    
    async def _send_audio_loop(self) -> None:
        """Send audio to Gemini in real-time from the streaming queue.
        
        Captures user audio and sends to Gemini until one of:
        1. VAD detects silence after speech
        2. No audio chunks received for 300ms (user stopped)
        3. Max capture duration reached
        
        After completion, calls end_turn() to signal Gemini that
        the user has finished speaking and a response is expected.
        """
        import time
        
        capture_start = time.time()
        total_audio_bytes = 0
        max_duration = self.capture_duration  # Max capture time as safety limit
        
        # No-chunk timeout - if Discord stops sending frames, user has stopped speaking
        # This is more reliable than VAD alone since Discord stops delivering audio on silence
        no_chunk_timeout = 0.3  # 300ms without chunks = user stopped
        last_chunk_time = time.time()
        received_at_least_one_chunk = False
        
        logger.debug(f"📤 Starting audio send loop (max {max_duration}s, no-chunk timeout {no_chunk_timeout}s, min speech 1.0s)")
        
        try:
            while self._is_capturing_for_gemini:
                current_time = time.time()
                elapsed = current_time - capture_start
                
                # Check for max duration timeout
                if elapsed >= max_duration:
                    logger.info(f"📤 Max capture duration reached ({max_duration}s)")
                    break
                
                # Check for VAD silence detection
                if self._capture.is_silence_detected():
                    logger.info(f"📤 VAD silence detected after speech ({elapsed:.1f}s), ending capture")
                    break
                
                # Check for no-chunk timeout (Discord stopped sending = user stopped talking)
                if received_at_least_one_chunk:
                    time_since_last_chunk = current_time - last_chunk_time
                    if time_since_last_chunk >= no_chunk_timeout:
                        logger.info(f"📤 No chunks for {time_since_last_chunk:.2f}s - user stopped speaking, ending capture")
                        break
                
                # Get audio chunk from streaming queue (non-blocking with short timeout)
                chunk = await self._capture.get_streaming_chunk(timeout=0.05)
                
                if chunk and len(chunk) > 0:
                    received_at_least_one_chunk = True
                    last_chunk_time = current_time
                    total_audio_bytes += len(chunk)
                    self._audio_chunks_sent += 1
                    
                    # Send audio to Gemini immediately
                    await self._gemini.send_audio(chunk)
                    
                    # Log progress periodically
                    if self._audio_chunks_sent % 25 == 0:
                        logger.debug(f"📤 Sent {self._audio_chunks_sent} chunks ({total_audio_bytes} bytes, {elapsed:.1f}s)")
        
        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
        except Exception as e:
            logger.error(f"Error in send loop: {e}")
        finally:
            # Signal end of audio input
            elapsed = time.time() - capture_start
            h = self._capture.get_pipeline_health()
            logger.info(
                f"📤 Send complete: {self._audio_chunks_sent} chunks, "
                f"{total_audio_bytes} bytes in {elapsed:.2f}s "
                f"[session={self._session_id} drops={h['total_drops']} "
                f"pushed={h['total_pushed']} consumed={h['total_consumed']}]"
            )
            
            if total_audio_bytes > 0:
                await self._gemini.end_turn()
            else:
                logger.warning("⚠️ No audio captured after wake word")
                self._streaming_complete.set()
    
    async def _receive_response_loop(self) -> None:
        """Receive audio from Gemini and stream to playback immediately.
        
        Includes a timeout to prevent indefinite blocking if Gemini
        doesn't respond.
        """
        import time
        
        logger.debug("🔊 Starting response receive loop")
        receive_start = time.time()
        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None
        
        # Timeout for receiving first chunk (Gemini should respond within 30s)
        receive_timeout = 30.0
        
        try:
            # Wrap the receive in a timeout
            async def receive_with_activity_check():
                nonlocal chunk_count, total_bytes, first_chunk_time
                last_activity = time.time()
                
                async for audio_chunk in self._gemini.receive_responses():
                    last_activity = time.time()
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - receive_start
                        logger.info(
                            f"🔊 First audio chunk received in {latency:.3f}s "
                            f"[session={self._session_id}] — starting playback!"
                        )
                    
                    chunk_count += 1
                    total_bytes += len(audio_chunk)
                    
                    # Add chunk to streaming playback immediately
                    self._playback.add_streaming_chunk(audio_chunk)
                    
                    if chunk_count % 10 == 0:
                        logger.debug(f"🔊 Received {chunk_count} chunks ({total_bytes} bytes)")
            
            await asyncio.wait_for(receive_with_activity_check(), timeout=receive_timeout)
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Receive loop timed out after {receive_timeout}s (received {chunk_count} chunks)")
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            # Mark streaming as complete
            self._playback.finish_streaming()
            self._streaming_complete.set()
            
            elapsed = time.time() - receive_start
            logger.info(
                f"🔊 Receive complete: {chunk_count} chunks, "
                f"{total_bytes} bytes in {elapsed:.2f}s "
                f"[session={self._session_id}]"
            )
    
    def _on_playback_complete_sync(self) -> None:
        """Synchronous callback when playback completes.

        Note: This is called by Discord's audio system. The streaming pipeline
        handles state transitions explicitly.
        """
        logger.debug("Playback complete callback triggered")
    
    async def _reset_to_listening(self) -> None:
        """Reset to listening state after processing/speaking.
        
        Checks the /ask queue first (higher priority than wake word).
        If there are queued prompts, processes them before resuming wake word detection.
        """
        if self._state == BotState.IDLE:
            return
        
        # Clear triggered user tracking
        self._triggered_user_id = None
        self._capture.set_active_user(None)
        
        # Stop streaming mode and reset VAD
        self._capture.stop_streaming_to_gemini()
        self._capture.reset_vad_state()
        self._is_capturing_for_gemini = False
        
        # Clear buffers
        self._speech_buffer.clear()
        self._capture.clear_buffer()
        
        # Check /ask queue first (higher priority than wake word)
        if not self._ask_queue.empty():
            try:
                prompt, user_id = self._ask_queue.get_nowait()
                remaining = self._ask_queue.qsize()
                logger.info(f"📋 Processing queued /ask prompt from user {user_id} ({remaining} remaining in queue)")
                
                # Process the queued prompt directly (don't call process_text_prompt to avoid redundant state checks)
                self._triggered_user_id = user_id
                await self._set_state(BotState.PROCESSING)
                self._wake_detector.disable()
                self._speech_buffer.clear()
                self._capture.clear_buffer()
                self._capture_task = asyncio.create_task(self._run_text_prompt_pipeline(prompt))
                return
            except asyncio.QueueEmpty:
                pass  # Queue was emptied between check and get, continue to wake word
        
        # No queued prompts, re-enable wake word detection
        self._wake_detector.enable()
        self._wake_detector.reset()
        
        # Transition back to listening
        await self._set_state(BotState.LISTENING)
        
        logger.info(f"Ready for next wake word '{self.config.wake_phrase_display}'")
    
    async def process_text_prompt(self, prompt: str, user_id: int) -> bool:
        """Process a text prompt from the /ask command and respond via voice.
        
        Args:
            prompt: The text prompt to send to Gemini.
            user_id: Discord user ID who sent the prompt.
            
        Returns:
            True if processing started successfully, False otherwise.
        """
        if self._state != BotState.LISTENING:
            logger.warning(f"Cannot process text prompt: not in LISTENING state (current: {self._state})")
            return False
        
        logger.info(f"📝 Processing text prompt from user {user_id}: {prompt[:50]}...")
        
        try:
            # Store which user triggered the command
            self._triggered_user_id = user_id
            
            # Transition to processing state
            await self._set_state(BotState.PROCESSING)
            
            # Disable wake word detection during processing
            self._wake_detector.disable()
            
            # Clear any old buffer data
            self._speech_buffer.clear()
            self._capture.clear_buffer()
            
            # Start the text prompt processing pipeline
            self._capture_task = asyncio.create_task(self._run_text_prompt_pipeline(prompt))
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting text prompt processing: {e}")
            await self._reset_to_listening()
            return False
    
    def queue_text_prompt(self, prompt: str, user_id: int) -> int:
        """Queue a text prompt to be processed after the current request.
        
        This is called when the bot is busy processing another request.
        The queued prompt will be processed with higher priority than wake word detection.
        
        Args:
            prompt: The text prompt to queue.
            user_id: Discord user ID who sent the prompt.
            
        Returns:
            Position in queue (1-indexed).
        """
        try:
            self._ask_queue.put_nowait((prompt, user_id))
        except asyncio.QueueFull:
            logger.warning(f"Ask queue full, rejecting prompt from user {user_id}")
            return -1
        position = self._ask_queue.qsize()
        logger.info(f"📋 Queued /ask prompt from user {user_id} at position #{position}: {prompt[:50]}...")
        return position
    
    def get_queue_size(self) -> int:
        """Get the number of items in the /ask queue.
        
        Returns:
            Number of queued prompts.
        """
        return self._ask_queue.qsize()
    
    def get_queue_items(self) -> list[tuple[str, int]]:
        """Get all items in the /ask queue without removing them.
        
        Returns:
            List of (prompt, user_id) tuples in queue order.
        """
        # Access the internal deque to peek at items without consuming them
        return list(self._ask_queue._queue)
    
    async def stop_response(self) -> bool:
        """Stop the current request and reset to listening state.
        
        This cancels all active tasks (capture, send, receive) and stops
        playback immediately. Then processes the next queued prompt or
        returns to listening for wake word.
        
        Returns:
            True if stopped successfully, False if not processing or speaking.
        """
        if self._state not in (BotState.PROCESSING, BotState.SPEAKING):
            logger.warning(f"Cannot stop response: not in PROCESSING or SPEAKING state (current: {self._state})")
            return False
        
        logger.info("🛑 Stopping current request")
        
        # Stop streaming mode and reset capture state
        self._is_capturing_for_gemini = False
        self._capture.stop_streaming_to_gemini()
        
        # Cancel the capture task if running (manages the streaming pipeline)
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        
        # Cancel the send task if running
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        
        # Cancel the receive task if running
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        # Stop the playback
        self._playback.stop()
        
        # Mark streaming as complete
        self._streaming_complete.set()
        
        # Reset to listening state (will process next queued prompt if any)
        await self._reset_to_listening()
        
        logger.info("✓ Request stopped, moving to next prompt or listening")
        return True
    
    def pause_response(self) -> bool:
        """Pause the current response playback.
        
        Returns:
            True if paused successfully, False if not speaking or already paused.
        """
        if self._state != BotState.SPEAKING:
            logger.warning(f"Cannot pause response: not in SPEAKING state (current: {self._state})")
            return False
        
        if self._playback.is_paused:
            logger.debug("Response already paused")
            return False
        
        return self._playback.pause()
    
    def resume_response(self) -> bool:
        """Resume the paused response playback.
        
        Returns:
            True if resumed successfully, False if not speaking or not paused.
        """
        if self._state != BotState.SPEAKING:
            logger.warning(f"Cannot resume response: not in SPEAKING state (current: {self._state})")
            return False
        
        if not self._playback.is_paused:
            logger.debug("Response is not paused")
            return False
        
        return self._playback.resume()
    
    @property
    def is_response_paused(self) -> bool:
        """Check if the current response is paused."""
        return self._state == BotState.SPEAKING and self._playback.is_paused
    
    async def _run_text_prompt_pipeline(self, prompt: str) -> None:
        """Run the text prompt processing pipeline.
        
        This method:
        1. Sends the text prompt to Gemini
        2. Receives and plays the audio response
        
        Args:
            prompt: The text prompt to send.
        """
        try:
            logger.info("🚀 Starting text prompt pipeline")
            
            # Check if Gemini needs reconnection
            if not self._gemini.is_connected:
                logger.warning(f"Gemini not connected (state={self._gemini.state}), attempting reconnect...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini")
                    await self._reset_to_listening()
                    return
                logger.info("✓ Successfully reconnected to Gemini")
            
            # Initialize streaming state
            self._streaming_complete.clear()
            
            # Transition to speaking state
            await self._set_state(BotState.SPEAKING)
            
            # Start streaming playback (will play silence until chunks arrive)
            playback_started = self._playback.start_streaming_playback()
            if not playback_started:
                logger.error("Failed to start streaming playback")
                await self._reset_to_listening()
                return
            
            # Send the text prompt and receive response concurrently
            self._send_task = asyncio.create_task(self._send_text_prompt(prompt))
            self._receive_task = asyncio.create_task(self._receive_response_loop())
            
            # Wait for both to complete
            try:
                await asyncio.gather(self._send_task, self._receive_task)
            except Exception as e:
                logger.error(f"Error in text prompt pipeline: {e}")
            
            # Wait for playback to complete
            logger.debug("Waiting for playback to complete...")
            playback_finished = await self._playback.wait_for_playback(timeout=60.0)
            if not playback_finished:
                logger.warning("Playback timed out after 60s, forcing stop")
                self._playback.stop()
            logger.info("✓ Text prompt pipeline complete")
            
            await self._reset_to_listening()
            
        except asyncio.CancelledError:
            logger.debug("Text prompt pipeline cancelled")
        except Exception as e:
            logger.error(f"❌ Error in text prompt pipeline: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            await self._reset_to_listening()
    
    async def _send_text_prompt(self, prompt: str) -> None:
        """Send the text prompt to Gemini.
        
        Args:
            prompt: The text prompt to send.
        """
        try:
            logger.debug(f"📤 Sending text prompt to Gemini: {prompt[:100]}...")
            await self._gemini.send_text(prompt)
            logger.info("📤 Text prompt sent successfully")
        except Exception as e:
            logger.error(f"Error sending text prompt: {e}")
            self._streaming_complete.set()
    
    def cleanup_user(self, user_id: int) -> None:
        """Clean up resources for a user who left the voice channel.
        
        This frees audio buffers and wake word models for the user.
        
        Args:
            user_id: Discord user ID to clean up.
        """
        logger.info(f"Cleaning up resources for user {user_id}")
        
        # Clean up audio capture buffers
        self._capture.cleanup_user(user_id)
        
        # Clean up wake word detector models
        self._wake_detector.cleanup_user(user_id)
        
        # Clean up sink tracking if available
        if self._sink and hasattr(self._sink, '_per_user_chunk_count'):
            if user_id in self._sink._per_user_chunk_count:
                del self._sink._per_user_chunk_count[user_id]
        
        logger.debug(f"Completed cleanup for user {user_id}")
    
    def _cleanup_all_users(self) -> None:
        """Clean up resources for all users.
        
        Called when leaving the voice channel to free all user resources.
        """
        # Get list of all users from capture
        user_ids = self._capture.get_active_users()
        detector_users = self._wake_detector.get_active_users()
        
        # Combine and deduplicate
        all_users = set(user_ids) | set(detector_users)
        
        for user_id in all_users:
            self.cleanup_user(user_id)
        
        logger.info(f"Cleaned up resources for {len(all_users)} users")
