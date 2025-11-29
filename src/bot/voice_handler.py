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
        
        self._capture = AudioCapture(self._processor)
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
        
        # Disconnect from Gemini
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
        
        logger.info(f"🎤 Wake word detected from user {user_id}! Starting real-time speech capture...")
        
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
        
        # Start real-time streaming to Gemini
        self._capture_task = asyncio.create_task(self._stream_audio_to_gemini())
    
    async def _stream_audio_to_gemini(self) -> None:
        """Stream audio in real-time to Gemini after wake word detection.
        
        This method captures audio and streams it to Gemini Live API in real-time,
        then receives and plays the response audio immediately.
        """
        try:
            import time
            
            logger.info(f"🎙️ Starting real-time audio stream for {self.capture_duration} seconds")
            
            # Check if Gemini needs reconnection
            if not self._gemini.is_connected:
                logger.warning(f"Gemini not connected (state={self._gemini.state}), attempting reconnect...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini")
                    await self._reset_to_listening()
                    return
                logger.info("✓ Successfully reconnected to Gemini")
            
            # Initialize capture state
            self._is_capturing_for_gemini = True
            self._capture_start_time = time.time()
            self._audio_chunks_sent = 0
            self._speech_buffer.clear()
            
            # Stream audio in small chunks for low latency
            capture_start = time.time()
            chunk_interval = 0.05  # 50ms interval for responsive streaming
            total_audio_bytes = 0
            
            logger.debug("📤 Starting to stream audio chunks to Gemini...")
            
            while (time.time() - capture_start) < self.capture_duration:
                # Get and consume the latest audio chunk (avoids re-sending same audio)
                chunk = await self._capture.get_and_consume_chunk()
                
                if chunk is not None and len(chunk) > 0:
                    audio_bytes = self._processor.numpy_to_pcm(chunk)
                    
                    # Only send if we have valid audio data
                    if audio_bytes and len(audio_bytes) > 0:
                        self._speech_buffer.append(audio_bytes)
                        total_audio_bytes += len(audio_bytes)
                        self._audio_chunks_sent += 1
                        
                        # Send audio to Gemini in real-time
                        await self._gemini.send_audio(audio_bytes)
                    
                    # Log progress periodically
                    if self._audio_chunks_sent % 10 == 0:
                        elapsed = time.time() - capture_start
                        logger.debug(f"📤 Streamed {self._audio_chunks_sent} chunks ({total_audio_bytes} bytes, {elapsed:.1f}s)")
                
                await asyncio.sleep(chunk_interval)
            
            # End of capture
            self._is_capturing_for_gemini = False
            elapsed = time.time() - capture_start
            
            logger.info(f"✓ Capture complete: {self._audio_chunks_sent} chunks, {total_audio_bytes} bytes in {elapsed:.2f}s")
            
            if total_audio_bytes > 0:
                # Signal end of turn and get response
                await self._get_gemini_response_and_play()
            else:
                logger.warning("⚠️ No audio captured after wake word")
                await self._reset_to_listening()
                
        except asyncio.CancelledError:
            logger.debug("Audio streaming cancelled")
            self._is_capturing_for_gemini = False
        except Exception as e:
            logger.error(f"❌ Error streaming audio to Gemini: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._is_capturing_for_gemini = False
            await self._reset_to_listening()
    
    async def _get_gemini_response_and_play(self) -> None:
        """Request response from Gemini and play audio in real-time."""
        try:
            import time
            from ..ai.gemini_client import GeminiSessionState
            
            logger.info("📨 Requesting response from Gemini...")
            
            # Check if Gemini is in error state and needs reconnection
            if self._gemini.state == GeminiSessionState.ERROR:
                logger.warning("Gemini in error state, attempting reconnection before response...")
                await self._gemini.disconnect()
                if not await self._gemini.connect():
                    logger.error("Failed to reconnect to Gemini after error")
                    await self._reset_to_listening()
                    return
                logger.info("✓ Reconnected to Gemini after error state")
            
            # Signal end of user input
            await self._gemini.end_turn()
            
            # Transition to speaking state
            await self._set_state(BotState.SPEAKING)
            
            # Collect and play response in real-time
            logger.debug("🔊 Waiting for Gemini audio response...")
            response_start = time.time()
            response_audio = await self._gemini.get_full_audio_response()
            response_time = time.time() - response_start
            
            if response_audio:
                audio_duration = len(response_audio) / 48000  # Approximate duration at 24kHz (will be converted to 48kHz)
                logger.info(f"🔊 Playing response: {len(response_audio)} bytes (~{audio_duration:.1f}s) received in {response_time:.2f}s")
                
                await self._playback.play_gemini_audio(response_audio)
                
                # Wait for playback to complete
                await self._playback.wait_for_playback(timeout=60.0)
                logger.info("✓ Playback complete")
            else:
                logger.warning("⚠️ No audio response from Gemini")
            
            await self._reset_to_listening()
                
        except Exception as e:
            logger.error(f"❌ Error getting Gemini response: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            await self._reset_to_listening()
    
    async def _capture_user_speech(self) -> None:
        """Legacy method - now redirects to streaming approach."""
        await self._stream_audio_to_gemini()
    
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
