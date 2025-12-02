"""Voice handler with state machine for managing voice interactions."""

import asyncio
from enum import Enum
from typing import Optional, TYPE_CHECKING

import discord

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
        self._voice_client: Optional[discord.VoiceClient] = None
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
        )
        
        self._capture = AudioCapture(
            self._processor,
            silence_threshold=self.silence_threshold,
        )
        self._playback = AudioPlayback(self._processor)
        
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
        
        # Update Gemini client if voice, model, or system prompt changed
        if "gemini_voice" in changed_fields or "gemini_model" in changed_fields or "system_prompt" in changed_fields:
            if "gemini_voice" in changed_fields:
                logger.info(f"  → Gemini voice: {config.gemini_voice}")
            if "gemini_model" in changed_fields:
                logger.info(f"  → Gemini model: {config.gemini_model}")
            # Need to reconnect Gemini with new settings
            asyncio.create_task(self._reconnect_gemini_with_new_config())
    
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
    
    async def _on_recording_finished(self, sink, channel, *args) -> None:
        """Handle recording finished event.
        
        This is called when stop_recording() is called or the bot disconnects.
        
        Args:
            sink: The sink that was recording.
            channel: The channel that was being recorded.
        """
        logger.info(f"Recording finished for channel: {channel}")
        sink.cleanup()
    
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
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
    
    async def _wait_for_voice_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the voice connection to be fully ready.
        
        With py-cord 2.7+, the connect() method should return a properly
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
            # py-cord 2.7+ has fixed the voice connection issues
            logger.debug("Connecting to voice channel...")
            try:
                self._voice_client = await asyncio.wait_for(
                    channel.connect(timeout=60.0, reconnect=True),
                    timeout=65.0
                )
            except asyncio.TimeoutError:
                logger.error("Voice channel connection timed out")
                raise Exception("Connection to voice channel timed out")
            
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
            
            # Start recording with our custom sink to receive audio
            logger.debug("Starting voice recording with WakeWordSink")
            self._voice_client.start_recording(
                self._sink,
                self._on_recording_finished,
                channel,
            )
            logger.info("Voice recording started - now receiving audio from Discord")
            
            # Start the audio processing loop (for wake word detection)
            logger.debug("Starting audio receive loop")
            self._audio_loop_task = asyncio.create_task(self._audio_receive_loop())
            
            # Connect to Gemini
            logger.debug("Connecting to Gemini Live API")
            await self._gemini.connect()
            logger.debug("Gemini connection established")
            
            # Start Gemini health check / keep-alive task
            logger.debug("Starting Gemini health check background task")
            await self._gemini.start_health_check()
            
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
        
        # Stop recording first (before disabling other components)
        if self._voice_client and self._voice_client.recording:
            try:
                logger.debug("Stopping voice recording")
                self._voice_client.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording: {e}")
        
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
    
    async def _on_wake_word_detected(self, user_id: int) -> None:
        """Handle wake word detection from a specific user.
        
        Args:
            user_id: Discord user ID who triggered the wake word.
        """
        if self._state != BotState.LISTENING:
            logger.debug("Wake word detected but not in LISTENING state, ignoring")
            return
        
        logger.info(f"🎤 Wake word detected from user {user_id}! Starting low-latency streaming...")
        
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
        """Run the full streaming pipeline with concurrent send/receive.
        
        This method:
        1. Starts streaming audio to Gemini immediately
        2. Starts receiving and playing responses concurrently
        3. Uses VAD to detect end of speech instead of fixed duration
        4. Plays audio as soon as first chunk arrives from Gemini
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
            
            # Transition to speaking state early - we'll start playing as chunks arrive
            await self._set_state(BotState.SPEAKING)
            
            # Start streaming playback (will play silence until chunks arrive)
            playback_started = self._playback.start_streaming_playback()
            if not playback_started:
                logger.error("Failed to start streaming playback")
                await self._reset_to_listening()
                return
            
            # Start concurrent tasks
            self._send_task = asyncio.create_task(self._send_audio_loop())
            self._receive_task = asyncio.create_task(self._receive_response_loop())
            
            # Wait for both to complete
            try:
                await asyncio.gather(self._send_task, self._receive_task)
            except Exception as e:
                logger.error(f"Error in streaming pipeline: {e}")
            
            # Cleanup
            self._is_capturing_for_gemini = False
            self._capture.stop_streaming_to_gemini()
            
            # Wait for playback to complete
            logger.debug("Waiting for playback to complete...")
            await self._playback.wait_for_playback(timeout=60.0)
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
        
        This runs concurrently with receive, so response can start playing
        while user is still speaking.
        
        Uses multiple end-of-speech detection strategies:
        1. VAD silence detection (speech followed by silence frames)
        2. No-chunk timeout (Discord stops sending when user is quiet)
        3. Max capture duration (safety limit)
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
        
        logger.debug(f"📤 Starting audio send loop (max {max_duration}s, no-chunk timeout {no_chunk_timeout}s)")
        
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
                chunk = await self._capture.get_streaming_chunk(timeout=0.02)
                
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
            logger.info(f"📤 Send complete: {self._audio_chunks_sent} chunks, {total_audio_bytes} bytes in {elapsed:.2f}s")
            
            if total_audio_bytes > 0:
                await self._gemini.end_turn()
            else:
                logger.warning("⚠️ No audio captured after wake word")
                self._streaming_complete.set()
    
    async def _receive_response_loop(self) -> None:
        """Receive audio from Gemini and stream to playback immediately.
        
        This runs concurrently with send, so response starts playing
        as soon as the first chunk arrives.
        """
        import time
        
        logger.debug("🔊 Starting response receive loop")
        receive_start = time.time()
        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None
        
        try:
            async for audio_chunk in self._gemini.receive_responses():
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - receive_start
                    logger.info(f"🔊 First audio chunk received in {latency:.3f}s - starting playback!")
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Add chunk to streaming playback immediately
                self._playback.add_streaming_chunk(audio_chunk)
                
                if chunk_count % 10 == 0:
                    logger.debug(f"🔊 Received {chunk_count} chunks ({total_bytes} bytes)")
        
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            # Mark streaming as complete
            self._playback.finish_streaming()
            self._streaming_complete.set()
            
            elapsed = time.time() - receive_start
            logger.info(f"🔊 Receive complete: {chunk_count} chunks, {total_bytes} bytes in {elapsed:.2f}s")
    
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
        """Reset to listening state after processing/speaking."""
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
        
        # Re-enable wake word detection
        self._wake_detector.enable()
        self._wake_detector.reset()
        
        # Transition back to listening
        await self._set_state(BotState.LISTENING)
        
        logger.info(f"Ready for next wake word '{self.config.wake_phrase_display}'")
    
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
        await self._capture.process_discord_audio(audio_data, is_stereo=True)
