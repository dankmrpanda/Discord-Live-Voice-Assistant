"""Voice handler with state machine for managing voice interactions."""

import asyncio
import logging
from enum import Enum
from typing import Optional, TYPE_CHECKING

import discord
from discord.ext import voice_recv

from ..utils.logger import get_logger, log_exception
from ..audio.capture import AudioCapture
from ..audio.playback import AudioPlayback
from ..audio.processor import AudioProcessor
from ..audio.sink import WakeWordSink
from ..wake_word.detector import WakeWordDetector
from ..ai.gemini_client import GeminiLiveClient, GeminiSessionState

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


@dataclass
class SendAudioResult:
    """Result of streaming user audio to Gemini."""

    chunks_sent: int
    total_bytes: int
    ended_turn: bool
    reason: str


@dataclass
class ReceiveAudioResult:
    """Result of receiving Gemini audio."""

    chunks_received: int
    total_bytes: int
    reason: str
    timed_out: bool
    had_error: bool


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
        self.gemini_first_chunk_timeout = getattr(config, 'gemini_first_chunk_timeout', 30.0)
        self.gemini_chunk_idle_timeout = getattr(config, 'gemini_chunk_idle_timeout', 8.0)
        self.gemini_max_turn_duration = getattr(config, 'gemini_max_turn_duration', 90.0)
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
        self._listener_restart_task: Optional[asyncio.Task] = None
        self._listener_restart_lock = asyncio.Lock()
        self._listener_restart_count = 0
        self._listener_last_error: Optional[str] = None
        self._is_leaving_channel = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # VoiceHandler can be created outside a running loop in tests.
            pass
        
        # Components
        logger.debug("Creating AudioProcessor")
        self._processor = AudioProcessor(
            discord_sample_rate=config.discord_sample_rate,
            gemini_input_sample_rate=config.gemini_input_sample_rate,
            gemini_output_sample_rate=config.gemini_output_sample_rate,
        )
        
        self._capture = AudioCapture(
            self._processor,
            silence_threshold=self.silence_threshold,
        )
        self._playback = AudioPlayback(
            self._processor,
            buffer_ms=config.playback_buffer_ms,
        )
        
        # Create sink for receiving Discord audio
        self._sink = WakeWordSink(capture=self._capture)
        
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
        
        # Queue for /ask commands (prompt, user_id)
        self._ask_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Register for config changes
        self.config.add_change_listener(self._on_config_changed)
        
        logger.debug("VoiceHandler initialization complete")
    
    def _on_config_changed(self, config: "Config", changed_fields: list) -> None:
        """Handle configuration changes.
        
        Args:
            config: The updated config object.
            changed_fields: List of field names that changed.
        """
        logger.info(f"Config changed: {', '.join(changed_fields)}")
        
        # Update local cached values
        if "capture_duration" in changed_fields:
            self.capture_duration = config.capture_duration
            logger.info(f"  -> Capture duration: {self.capture_duration}s")
        
        if "silence_threshold" in changed_fields:
            self.silence_threshold = config.silence_threshold
            # Also update the AudioCapture's VAD threshold
            self._capture.set_silence_threshold(config.silence_threshold)
            logger.info(f"  -> Silence threshold: {self.silence_threshold}s")

        if "gemini_first_chunk_timeout" in changed_fields:
            self.gemini_first_chunk_timeout = config.gemini_first_chunk_timeout
            logger.info(f"  -> Gemini first chunk timeout: {self.gemini_first_chunk_timeout}s")

        if "gemini_chunk_idle_timeout" in changed_fields:
            self.gemini_chunk_idle_timeout = config.gemini_chunk_idle_timeout
            logger.info(f"  -> Gemini chunk idle timeout: {self.gemini_chunk_idle_timeout}s")

        if "gemini_max_turn_duration" in changed_fields:
            self.gemini_max_turn_duration = config.gemini_max_turn_duration
            logger.info(f"  -> Gemini max turn duration: {self.gemini_max_turn_duration}s")
        
        if "playback_buffer_ms" in changed_fields:
            self._playback.buffer_ms = config.playback_buffer_ms
            logger.info(f"  -> Playback buffer: {config.playback_buffer_ms}ms")
        
        if "log_audio" in changed_fields:
            self.log_audio = config.log_audio
            self._wake_detector.verbose = config.log_audio
            logger.info(f"  -> Log audio: {self.log_audio}")
        
        # Update wake word detector
        if "wake_phrase" in changed_fields or "wake_word_threshold" in changed_fields:
            logger.info(f"  -> Wake phrase: '{config.wake_phrase_display}' (threshold: {config.wake_word_threshold})")
            # Recreate wake word detector with new settings
            self._wake_detector = WakeWordDetector(
                wake_phrase=config.wake_phrase,
                threshold=config.wake_word_threshold,
                sample_rate=config.gemini_input_sample_rate,
                verbose=self.log_audio,
            )
            self._wake_detector.set_detection_callback(self._on_wake_word_detected)
            logger.info("  -> Wake word detector recreated")
        
        # Update Gemini client if voice, model, thinking, google_search, function_calling, automatic_function_response, or system prompt changed
        gemini_changed = any(f in changed_fields for f in [
            "gemini_voice", "gemini_model", "gemini_thinking", 
            "gemini_google_search", "gemini_function_calling", 
            "gemini_automatic_function_response", "system_prompt"
        ])
        if gemini_changed:
            if "gemini_voice" in changed_fields:
                logger.info(f"  -> Gemini voice: {config.gemini_voice}")
            if "gemini_model" in changed_fields:
                logger.info(f"  -> Gemini model: {config.gemini_model}")
            if "gemini_thinking" in changed_fields:
                logger.info(f"  -> Gemini thinking: {config.gemini_thinking}")
            if "gemini_google_search" in changed_fields:
                logger.info(f"  -> Gemini Google Search: {config.gemini_google_search}")
            if "gemini_function_calling" in changed_fields:
                logger.info(f"  -> Gemini function calling: {config.gemini_function_calling}")
            if "gemini_automatic_function_response" in changed_fields:
                logger.info(f"  -> Gemini automatic function response: {config.gemini_automatic_function_response}")
            # Need to reconnect Gemini with new settings
            asyncio.create_task(self._reconnect_gemini_with_new_config())
    
    async def _reconnect_gemini_with_new_config(self) -> None:
        """Reconnect to Gemini with updated configuration."""
        try:
            logger.info("Reconnecting to Gemini with new config...")
            
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
                    logger.info("Reconnected to Gemini with new settings")
                else:
                    logger.error("Failed to reconnect to Gemini")
            else:
                logger.info("Gemini client updated (will connect when joining voice)")
                
        except Exception as e:
            log_exception(logger, "Error reconnecting Gemini", e)
    
    def _setup_callbacks(self) -> None:
        """Set up callbacks between components."""
        # Wake word detection callback (now receives user_id)
        self._wake_detector.set_detection_callback(self._on_wake_word_detected)
        
        # Playback completion callback
        self._playback.set_after_callback(self._on_playback_complete_sync)
        
        # Audio capture callback (receives audio and user_id)
        self._capture.set_audio_callback(self._on_audio_chunk_received)
    
    async def _on_audio_chunk_received(self, audio_data: bytes, user_id: int, raw_discord_data: bytes = b"") -> None:
        """Callback when audio chunk is received from a user.
        
        This is called for each processed audio chunk and handles per-user
        wake word detection.
        
        Args:
            audio_data: PCM audio bytes (16kHz, mono).
            user_id: Discord user ID this audio came from.
            raw_discord_data: Original raw Discord audio (48kHz stereo) for debug saving.
        """
        if self._state == BotState.LISTENING and user_id != 0:
            # Save raw Discord audio for debugging (before any conversion)
            if raw_discord_data:
                self._wake_detector._dump_raw_discord_audio(user_id, raw_discord_data)
            # Process wake word detection for this specific user
            await self._wake_detector.process_audio_for_user(audio_data, user_id)
    
    def _on_recording_finished(self, exception: Exception = None) -> None:
        """Handle recording finished event.

        This is called when stop_recording() is called or the bot disconnects.
        py-cord 2.8 passes a single exception parameter (or None on clean stop).

        Args:
            exception: Exception that caused recording to stop, or None.
        """
        if exception:
            logger.warning(f"Recording finished with error: {exception}")
        else:
            logger.info("Recording finished")
        self._sink.cleanup()
    
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

    def is_connected_to_channel(self, channel: discord.abc.Connectable) -> bool:
        """Check if connected to the given voice channel."""
        if not self._voice_client or not self._voice_client.channel:
            return False
        return self._voice_client.channel.id == channel.id
    
    async def _set_state(self, new_state: BotState) -> None:
        """Set bot state with logging."""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
    
    async def _cleanup_partial_voice_client(self, guild: discord.Guild) -> None:
        """Disconnect and clean up a partially-connected voice client."""
        try:
            if self._voice_client is not None:
                await self._voice_client.disconnect(force=True)
        except Exception:
            pass
        # Also remove any lingering voice client on the guild
        try:
            if guild.voice_client is not None:
                await guild.voice_client.disconnect(force=True)
        except Exception:
            pass
        self._voice_client = None

    async def _wait_for_voice_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the voice connection to be fully ready.

        Uses py-cord 2.8+'s built-in wait_until_connected() when available,
        falling back to polling is_connected() for older versions.

        Args:
            timeout: Maximum time to wait for connection (seconds).

        Returns:
            True if connection is ready, False if timed out or failed.
        """
        if not self._voice_client:
            return False

        logger.debug(f"Verifying voice connection (timeout={timeout}s)")

        # py-cord 2.8+ provides wait_until_connected()
        if hasattr(self._voice_client, 'wait_until_connected'):
            try:
                result = self._voice_client.wait_until_connected(timeout=timeout)
                # Handle both sync and async versions
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    ready = await result
                else:
                    ready = result
                if ready:
                    logger.info("Voice connection verified via wait_until_connected()")
                else:
                    logger.warning(f"Voice connection not ready after {timeout}s")
                return ready
            except Exception as e:
                log_exception(logger, "Error waiting for voice connection", e, level=logging.WARNING)
                return False

        # Fallback for older py-cord versions
        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(1.0)
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if self._voice_client is None:
                return False
            if self._voice_client.is_connected():
                logger.info("Voice connection verified successfully")
                return True
            await asyncio.sleep(0.5)

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
            self._event_loop = asyncio.get_running_loop()
            self._is_leaving_channel = False
            self._listener_restart_count = 0
            self._listener_last_error = None
            
            # Set state to connecting
            await self._set_state(BotState.CONNECTING)
            self._target_channel = channel
            self._connection_ready.clear()
            self._connection_failed.clear()
            
            # Connect to voice channel with retry logic.
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                logger.debug(f"Connecting to voice channel (attempt {attempt}/{max_attempts})...")
                try:
                    self._voice_client = await asyncio.wait_for(
                        channel.connect(timeout=30.0, reconnect=True),
                        timeout=35.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Voice connection attempt {attempt} timed out")
                    await self._cleanup_partial_voice_client(channel.guild)
                    if attempt == max_attempts:
                        raise Exception("Voice channel connection timed out after all retries")
                    await asyncio.sleep(2.0)
                    continue
                except Exception as e:
                    logger.warning(f"Voice connection attempt {attempt} failed: {e}")
                    await self._cleanup_partial_voice_client(channel.guild)
                    if attempt == max_attempts:
                        raise
                    await asyncio.sleep(2.0)
                    continue

                logger.debug(f"Voice client obtained: {self._voice_client}")

                # Verify the connection is actually ready
                if await self._wait_for_voice_ready(timeout=10.0):
                    break

                # Verification failed — clean up and retry
                logger.warning(f"Voice connection verification failed on attempt {attempt}")
                await self._cleanup_partial_voice_client(channel.guild)
                self._voice_client = None
                if attempt == max_attempts:
                    raise Exception("Voice connection verification failed after all retries")
                await asyncio.sleep(2.0)

            if not self._voice_client or not self._voice_client.is_connected():
                raise Exception("Voice connection not established")
            
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
            
            # Ensure the sink has the event loop for cross-thread scheduling
            self._sink.set_loop(asyncio.get_running_loop())

            # Start recording with our custom sink to receive audio
            logger.debug("Starting voice recording with WakeWordSink")
            self._voice_client.start_recording(
                self._sink,
                self._on_recording_finished,
            )
            
            # Start the audio processing loop (for wake word detection)
            logger.debug("Starting audio receive loop")
            self._audio_loop_task = asyncio.create_task(self._audio_receive_loop())
            
            # Connect to Gemini (with timeout to prevent /join from hanging)
            logger.debug("Connecting to Gemini Live API")
            try:
                gemini_connected = await asyncio.wait_for(
                    self._gemini.connect(),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                raise Exception("Gemini connection timed out")
            if not gemini_connected:
                raise Exception("Failed to connect to Gemini API")
            logger.debug("Gemini connection established")
            
            # Start Gemini health check / keep-alive task
            logger.debug("Starting Gemini health check background task")
            await self._gemini.start_health_check()
            
            # Transition to listening state
            await self._set_state(BotState.LISTENING)
            
            logger.info(f"Joined channel '{channel.name}', listening for '{self.config.wake_phrase_display}'")
            return True
            
        except Exception as e:
            log_exception(logger, "Failed to join voice channel", e)
            await self.leave_channel()
            return False
    
    async def leave_channel(self) -> None:
        """Leave the current voice channel and clean up resources."""
        logger.info("Leaving voice channel")
        self._is_leaving_channel = True
        
        # Clear the /ask queue on leave
        queue_size = self._ask_queue.qsize()
        if queue_size > 0:
            logger.info(f"Clearing /ask queue ({queue_size} items)")
            while not self._ask_queue.empty():
                try:
                    self._ask_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Stop recording first (before disabling other components)
        is_recording = (self._voice_client and
                        (self._voice_client.is_recording() if hasattr(self._voice_client, 'is_recording')
                         else getattr(self._voice_client, 'recording', False)))
        if is_recording:
            try:
                logger.debug("Stopping voice receive listener")
                self._voice_client.stop_listening()
            except Exception as e:
                log_exception(logger, "Error stopping recording", e, level=logging.WARNING)
        
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
                log_exception(logger, "Error disconnecting voice client", e, level=logging.WARNING)
            self._voice_client = None
        
        # Clear target channel
        self._target_channel = None
        
        # Reset connection events
        self._connection_ready.clear()
        self._connection_failed.clear()
        
        await self._set_state(BotState.IDLE)
        self._is_leaving_channel = False
    
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
                    sink_chunks = self._sink._chunk_count if self._sink else -1
                    logger.debug(f"Audio loop heartbeat: state={self._state.value}, loops={loop_count}, "
                                f"capture_users={len(active_users)}, detector_users={len(detector_users)}, "
                                f"sink_chunks={sink_chunks}")
                    # Warn if listening is active but no audio has been received
                    if sink_chunks == 0 and loop_count >= 400:
                        logger.warning("No audio received from Discord after 20+ seconds of listening. "
                                      "Ensure at least one user is speaking in the voice channel.")
                
                # Note: Wake word detection is now handled in _on_audio_chunk_received
                # which is called per-user when audio is processed
                
                await asyncio.sleep(0.05)  # 50ms polling interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_exception(logger, "Error in audio receive loop", e)
                await asyncio.sleep(0.1)
    
    async def _on_wake_word_detected(self, user_id: int) -> None:
        """Handle wake word detection from a specific user.
        
        Args:
            user_id: Discord user ID who triggered the wake word.
        """
        if self._state != BotState.LISTENING:
            logger.debug("Wake word detected but not in LISTENING state, ignoring")
            return
        
        logger.info(f"Wake word detected from user {user_id}. Starting low-latency streaming.")
        
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
        
        # Start the streaming pipeline (sequential send -> receive)
        self._capture_task = asyncio.create_task(self._run_streaming_pipeline())
    
    async def _run_streaming_pipeline(self) -> None:
        """Run the full streaming pipeline with sequential send then receive.

        This method:
        1. Captures and sends user audio to Gemini until speech ends
        2. Signals end_turn() to tell Gemini user is done speaking
        3. Receives and plays Gemini's audio response
        """
        try:
            logger.info("Starting low-latency streaming pipeline")

            # Check if Gemini needs reconnection
            if not self._gemini.is_connected:
                logger.warning(f"Gemini not connected (state={self._gemini.state}), attempting reconnect...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini")
                    await self._reset_to_listening()
                    return
                logger.info("Successfully reconnected to Gemini")

            # Initialize streaming state
            self._is_capturing_for_gemini = True
            self._capture_start_time = time.time()
            self._audio_chunks_sent = 0
            self._streaming_complete.clear()

            # Start streaming mode in capture (enables direct queue pushing)
            self._capture.start_streaming_to_gemini()

            # Transition to processing state while capturing user speech
            await self._set_state(BotState.PROCESSING)

            # Phase 1: Capture and send all user audio
            self._send_task = asyncio.create_task(self._send_audio_loop())
            try:
                await self._send_task
            except Exception as e:
                log_exception(logger, "Error in send loop", e)

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
                log_exception(logger, "Error in receive loop", e)

            # Cleanup
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()

            # Wait for playback to complete
            logger.debug("Waiting for playback to complete...")
            playback_finished = await self._playback.wait_for_playback(timeout=60.0)
            if not playback_finished:
                logger.warning("Playback wait timed out; forcing playback stop")
                self._playback.stop()
            logger.info("Streaming pipeline complete")

            await self._reset_to_listening()

        except asyncio.CancelledError:
            logger.debug("Streaming pipeline cancelled")
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
            self._playback.stop()
            raise
        except Exception as e:
            log_exception(logger, "Error in streaming pipeline", e)
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
            self._playback.stop()
            await self._reset_to_listening()

    async def _send_audio_loop(self) -> SendAudioResult:
        """Send audio to Gemini in real-time from the streaming queue."""
        capture_start = time.time()
        total_audio_bytes = 0
        chunks_sent = 0
        ended_turn = False
        reason = "completed"
        max_duration = self.capture_duration
        silence_threshold = float(getattr(self, "silence_threshold", 1.0))

        # No-chunk timeout: Discord may stop sending packets during short pauses.
        # Use a more forgiving timeout derived from silence_threshold to avoid
        # cutting off mid-request when users pause briefly.
        no_chunk_timeout = max(0.8, min(silence_threshold, 1.5))
        min_capture_before_no_chunk = 1.0
        last_chunk_time = time.time()
        received_at_least_one_chunk = False

        logger.debug(
            "Starting audio send loop "
            f"(max {max_duration}s, no-chunk timeout {no_chunk_timeout}s, "
            f"min-capture-before-timeout {min_capture_before_no_chunk}s, min speech 1.0s)"
        )

        try:
            while self._is_capturing_for_gemini:
                current_time = time.time()
                elapsed = current_time - capture_start

                # Check for max duration timeout
                if elapsed >= max_duration:
                    logger.info(f"Max capture duration reached ({max_duration}s)")
                    reason = "max_capture_duration"
                    break

                # Check for VAD silence detection
                if self._capture.is_silence_detected():
                    logger.info(f"VAD silence detected after speech ({elapsed:.1f}s), ending capture")
                    reason = "vad_silence"
                    break

                # Check for no-chunk timeout (Discord stopped sending = user stopped talking)
                if received_at_least_one_chunk and elapsed >= min_capture_before_no_chunk:
                    time_since_last_chunk = current_time - last_chunk_time
                    if time_since_last_chunk >= no_chunk_timeout:
                        logger.info(
                            f"No chunks for {time_since_last_chunk:.2f}s - user stopped speaking, ending capture"
                        )
                        reason = "no_chunk_timeout"
                        break

                # Get audio chunk from streaming queue (non-blocking with short timeout)
                chunk = await self._capture.get_streaming_chunk(timeout=0.02)

                if chunk and len(chunk) > 0:
                    received_at_least_one_chunk = True
                    last_chunk_time = current_time
                    total_audio_bytes += len(chunk)
                    self._audio_chunks_sent += 1
                    chunks_sent += 1

                    # Send audio to Gemini immediately
                    if not await self._gemini.send_audio(chunk):
                        reason = "send_audio_failed"
                        break

                    if self._audio_chunks_sent % 25 == 0:
                        logger.debug(
                            f"Sent {self._audio_chunks_sent} chunks ({total_audio_bytes} bytes, {elapsed:.1f}s)"
                        )

        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
            raise
        except Exception as e:
            log_exception(logger, "Error in send loop", e)
        finally:
            # Signal end of audio input
            elapsed = time.time() - capture_start
            logger.info(f"📤 Send complete: {self._audio_chunks_sent} chunks, {total_audio_bytes} bytes in {elapsed:.2f}s")
            
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
        timed_out = False
        had_error = False
        reason = "turn_complete"
        response_stream = self._gemini.receive_responses()

        try:
            while True:
                elapsed = time.time() - receive_start
                remaining_turn_time: Optional[float] = None
                if self.gemini_max_turn_duration > 0:
                    remaining_turn_time = self.gemini_max_turn_duration - elapsed
                    if remaining_turn_time <= 0:
                        timed_out = True
                        reason = "max_turn_timeout"
                        break

                timeout_for_next_chunk = (
                    self.gemini_first_chunk_timeout
                    if chunk_count == 0
                    else self.gemini_chunk_idle_timeout
                )
                if remaining_turn_time is not None:
                    timeout_for_next_chunk = min(timeout_for_next_chunk, remaining_turn_time)

                try:
                    audio_chunk = await asyncio.wait_for(
                        anext(response_stream),
                        timeout=timeout_for_next_chunk,
                    )
                except StopAsyncIteration:
                    reason = "turn_complete"
                    break
                except asyncio.TimeoutError:
                    timed_out = True
                    reason = "first_chunk_timeout" if chunk_count == 0 else "chunk_idle_timeout"
                    break

                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - receive_start
                    logger.info(f"First audio chunk received in {latency:.3f}s - starting playback")

                chunk_count += 1
                total_bytes += len(audio_chunk)

                if not self._playback.add_streaming_chunk(audio_chunk):
                    had_error = True
                    reason = "playback_not_streaming"
                    break

                if chunk_count % 10 == 0:
                    logger.debug(f"Received {chunk_count} chunks ({total_bytes} bytes)")

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
            raise
        except Exception as e:
            log_exception(logger, "Error in receive loop", e)
        finally:
            # Close async generator if supported
            aclose = getattr(response_stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass

            # Mark streaming as complete
            self._playback.finish_streaming()
            self._streaming_complete.set()

            elapsed = time.time() - receive_start
            logger.info(f"Receive complete: {chunk_count} chunks, {total_bytes} bytes in {elapsed:.2f}s")

        return ReceiveAudioResult(
            chunks_received=chunk_count,
            total_bytes=total_bytes,
            reason=reason,
            timed_out=timed_out,
            had_error=had_error,
        )

    async def _get_gemini_response_and_play(self) -> None:
        """Legacy method - now handled by streaming pipeline."""
        logger.debug("_get_gemini_response_and_play called - redirecting to streaming pipeline")
        # This is now handled by _receive_response_loop
        pass
    
    async def _stream_audio_to_gemini(self) -> None:
        """Legacy method - now redirects to streaming pipeline."""
        await self._run_streaming_pipeline()
    
    async def _capture_user_speech(self) -> None:
        """Legacy method - now redirects to streaming approach."""
        await self._run_streaming_pipeline()
    
    def _on_playback_complete_sync(self) -> None:
        """Synchronous callback when playback completes.
        
        Note: This is called by Discord's audio system. We don't reset to listening
        here since _get_gemini_response_and_play handles that explicitly.
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
                logger.info(f"Processing queued /ask prompt from user {user_id} ({remaining} remaining in queue)")
                
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
        
        logger.info(f"Processing text prompt from user {user_id}: {prompt[:50]}...")
        
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
            log_exception(logger, "Error starting text prompt processing", e)
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
        self._ask_queue.put_nowait((prompt, user_id))
        position = self._ask_queue.qsize()
        logger.info(f"Queued /ask prompt from user {user_id} at position #{position}: {prompt[:50]}...")
        return position
    
    def get_queue_size(self) -> int:
        """Get the number of items in the /ask queue.
        
        Returns:
            Number of queued prompts.
        """
        return self._ask_queue.qsize()

    def clear_queue(self) -> int:
        """Clear all queued text prompts.

        Returns:
            Number of removed queued prompts.
        """
        removed = 0
        while not self._ask_queue.empty():
            try:
                self._ask_queue.get_nowait()
                removed += 1
            except asyncio.QueueEmpty:
                break
        if removed > 0:
            logger.info(f"Cleared {removed} queued /ask prompt(s)")
        return removed
    
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
        
        logger.info("Stopping current request")
        
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
        
        logger.info("Request stopped, moving to next prompt or listening")
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
            logger.info("Starting text prompt pipeline")
            
            # Check if Gemini needs reconnection
            if not self._gemini.is_connected:
                logger.warning(f"Gemini not connected (state={self._gemini.state}), attempting reconnect...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini")
                    await self._reset_to_listening()
                    return
                logger.info("Successfully reconnected to Gemini")
            
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
            
            # Send text prompt first, then receive model response.
            self._send_task = asyncio.create_task(self._send_text_prompt(prompt))
            send_ok = await self._send_task
            if not send_ok:
                logger.error("Text prompt send failed, resetting to listening")
                self._playback.stop()
                await self._reset_to_listening()
                return

            self._receive_task = asyncio.create_task(self._receive_response_loop())
            
            # Wait for both to complete
            results = await asyncio.gather(self._send_task, self._receive_task, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = "send" if i == 0 else "receive"
                    log_exception(logger, f"Error in {task_name} task", result)
            
            # Wait for playback to complete
            logger.debug("Waiting for playback to complete...")
            playback_finished = await self._playback.wait_for_playback(timeout=60.0)
            if not playback_finished:
                logger.warning("Text playback wait timed out; forcing playback stop")
                self._playback.stop()
            logger.info("Text prompt pipeline complete")
            
            await self._reset_to_listening()
            
        except asyncio.CancelledError:
            logger.debug("Text prompt pipeline cancelled")
            self._playback.stop()
            raise
        except Exception as e:
            log_exception(logger, "Error in text prompt pipeline", e)
            await self._reset_to_listening()
    
    async def _send_text_prompt(self, prompt: str) -> bool:
        """Send the text prompt to Gemini.
        
        Args:
            prompt: The text prompt to send.
        """
        try:
            logger.debug(f"Sending text prompt to Gemini: {prompt[:100]}...")
            ok = await self._gemini.send_text(prompt)
            if not ok:
                logger.error("Gemini send_text failed")
                return False
            logger.info("Text prompt sent successfully")
            return True
        except Exception as e:
            log_exception(logger, "Error sending text prompt", e)
            self._streaming_complete.set()
            return False
    
    async def handle_audio_packet(
        self,
        user: discord.User,
        audio_data: bytes,
    ) -> None:
        """Handle incoming audio packet from a user.
        
        This method should be called by a voice receive sink.
        
        Args:
            user: The Discord user who sent the audio.
            audio_data: Raw audio data from Discord.
        """
        logger.debug(f"Received audio packet from {user.name}: {len(audio_data)} bytes")
        # Process audio through capture pipeline
        await self._capture.process_discord_audio_per_user(audio_data, user.id, is_stereo=True)
    
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

