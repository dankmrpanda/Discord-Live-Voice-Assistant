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
        )
        
        # Audio buffer for post-wake word capture
        self._speech_buffer: list[bytes] = []
        self._capture_task: Optional[asyncio.Task] = None
        
        # Set up callbacks
        self._setup_callbacks()
        logger.debug("VoiceHandler initialization complete")
    
    def _setup_callbacks(self) -> None:
        """Set up callbacks between components."""
        # Wake word detection callback
        self._wake_detector.set_detection_callback(self._on_wake_word_detected)
        
        # Playback completion callback
        self._playback.set_after_callback(self._on_playback_complete_sync)
    
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
        
        Note: Discord's voice receive is limited. This implementation
        uses a sink approach where available.
        """
        # Discord.py voice receive is complex and somewhat undocumented
        # For now, we'll use a polling approach with the voice client
        # In production, consider using discord.py's listen() or a sink
        
        logger.debug("Audio receive loop started")
        loop_count = 0
        
        while self.is_connected:
            try:
                loop_count += 1
                if loop_count % 200 == 0:  # Log every 10 seconds (200 * 50ms)
                    logger.debug(f"Audio loop heartbeat: state={self._state.value}, loops={loop_count}")
                
                # Check if we should process audio
                if self._state == BotState.LISTENING:
                    # Get audio chunk for wake word detection
                    chunk = await self._capture.get_chunk_for_wake_word()
                    if chunk is not None:
                        audio_bytes = self._processor.numpy_to_pcm(chunk)
                        
                        # Log audio details if enabled
                        if self.log_audio:
                            import numpy as np
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                            rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
                            peak = np.max(np.abs(audio_array))
                            logger.info(f"[AUDIO] Chunk {loop_count}: {len(audio_bytes)} bytes, RMS={rms:.0f}, Peak={peak}, samples={len(audio_array)}")
                        elif loop_count % 100 == 0:  # Log occasionally when log_audio is off
                            logger.debug(f"Processing audio chunk: {len(audio_bytes)} bytes")
                        
                        await self._wake_detector.process_audio(audio_bytes)
                
                await asyncio.sleep(0.05)  # 50ms polling interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio receive loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _on_wake_word_detected(self) -> None:
        """Handle wake word detection."""
        if self._state != BotState.LISTENING:
            logger.debug("Wake word detected but not in LISTENING state, ignoring")
            return
        
        logger.info("Wake word detected! Starting speech capture...")
        
        # Transition to processing state
        await self._set_state(BotState.PROCESSING)
        
        # Disable wake word detection during processing
        self._wake_detector.disable()
        
        # Start capturing user speech
        self._capture_task = asyncio.create_task(self._capture_user_speech())
    
    async def _capture_user_speech(self) -> None:
        """Capture user speech after wake word detection."""
        try:
            logger.debug(f"Starting speech capture for {self.capture_duration} seconds")
            self._speech_buffer.clear()
            
            # Capture audio for the specified duration
            # In a real implementation, use VAD for smarter end-of-speech detection
            capture_start = asyncio.get_event_loop().time()
            chunks_captured = 0
            
            while (asyncio.get_event_loop().time() - capture_start) < self.capture_duration:
                # Get recent audio
                chunk = await self._capture.get_chunk_for_wake_word()
                if chunk is not None:
                    audio_bytes = self._processor.numpy_to_pcm(chunk)
                    self._speech_buffer.append(audio_bytes)
                    chunks_captured += 1
                    if chunks_captured % 10 == 0:
                        elapsed = asyncio.get_event_loop().time() - capture_start
                        logger.debug(f"Capturing speech: {chunks_captured} chunks, {elapsed:.1f}s elapsed")
                
                await asyncio.sleep(0.1)
            
            # Combine captured audio
            logger.debug(f"Speech capture complete: {chunks_captured} total chunks")
            if self._speech_buffer:
                full_audio = b"".join(self._speech_buffer)
                logger.info(f"Captured {len(full_audio)} bytes of speech ({len(full_audio)/32000:.2f}s at 16kHz)")
                
                # Send to Gemini and get response
                await self._process_with_gemini(full_audio)
            else:
                logger.warning("No speech captured after wake word")
                await self._reset_to_listening()
                
        except asyncio.CancelledError:
            logger.debug("Speech capture cancelled")
        except Exception as e:
            logger.error(f"Error capturing speech: {e}")
            await self._reset_to_listening()
    
    async def _process_with_gemini(self, audio_data: bytes) -> None:
        """Process captured audio with Gemini and play response.
        
        Args:
            audio_data: Captured speech audio (16kHz mono PCM).
        """
        try:
            logger.info(f"Sending {len(audio_data)} bytes of audio to Gemini...")
            logger.debug(f"Audio duration: {len(audio_data)/32000:.2f}s at 16kHz")
            
            # Send audio to Gemini
            import time
            send_start = time.time()
            await self._gemini.send_audio(audio_data)
            logger.debug(f"Audio sent to Gemini in {time.time() - send_start:.3f}s")
            
            # Request response
            logger.debug("Requesting response from Gemini (end_turn)")
            await self._gemini.end_turn()
            
            # Transition to speaking state
            await self._set_state(BotState.SPEAKING)
            
            # Collect and play response
            logger.debug("Waiting for Gemini audio response...")
            response_start = time.time()
            response_audio = await self._gemini.get_full_audio_response()
            logger.debug(f"Received response in {time.time() - response_start:.3f}s")
            
            if response_audio:
                logger.info(f"Playing {len(response_audio)} bytes of response audio ({len(response_audio)/48000:.2f}s at 24kHz)")
                await self._playback.play_gemini_audio(response_audio)
                
                # Wait for playback to complete
                await self._playback.wait_for_playback(timeout=60.0)
            else:
                logger.warning("No audio response from Gemini")
                await self._reset_to_listening()
                
        except Exception as e:
            logger.error(f"Error processing with Gemini: {e}")
            await self._reset_to_listening()
    
    def _on_playback_complete_sync(self) -> None:
        """Synchronous callback when playback completes."""
        # Schedule the async reset
        asyncio.create_task(self._reset_to_listening())
    
    async def _reset_to_listening(self) -> None:
        """Reset to listening state after processing/speaking."""
        if self._state == BotState.IDLE:
            return
        
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
