"""Gemini Live API client for real-time voice interaction."""

import asyncio
import base64
from typing import Optional, Callable, Awaitable, AsyncIterator
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger("ai.gemini")


class GeminiSessionState(Enum):
    """State of the Gemini Live API session."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class GeminiLiveClient:
    """Client for Gemini Live API with real-time audio streaming.
    
    Uses the Gemini Live API for bidirectional audio streaming,
    providing low-latency speech-to-speech interaction.
    """
    
    # Gemini Live API model for native audio
    MODEL = "gemini-2.0-flash-live-001"
    
    def __init__(
        self,
        api_key: str,
        voice: str = "Puck",
        system_instruction: Optional[str] = None,
    ):
        """Initialize the Gemini Live API client.
        
        Args:
            api_key: Google Gemini API key.
            voice: Voice to use for TTS (Puck, Charon, Kore, etc.).
            system_instruction: Optional system prompt for the model.
        """
        logger.debug(f"Initializing GeminiLiveClient with voice={voice}, model={self.MODEL}")
        
        self.api_key = api_key
        self.voice = voice
        self.system_instruction = system_instruction or self._default_system_instruction()
        
        logger.debug(f"System instruction length: {len(self.system_instruction)} chars")
        
        self._client = None
        self._session = None
        self._session_manager = None
        self._state = GeminiSessionState.DISCONNECTED
        
        # Callbacks
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._text_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._completion_callback: Optional[Callable[[], Awaitable[None]]] = None
        
        # Audio buffer for collecting response
        self._audio_buffer: list[bytes] = []
        
        self._initialize_client()
    
    def _default_system_instruction(self) -> str:
        """Get default system instruction for the assistant."""
        return """You are a helpful voice assistant in a Discord voice channel. 
        Keep your responses concise and conversational since they will be spoken aloud.
        Be friendly, helpful, and natural in your responses.
        If you don't understand something, ask for clarification.
        Avoid using markdown formatting, bullet points, or numbered lists since this is voice output."""
    
    def _initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            logger.debug("Importing google-genai library")
            from google import genai
            
            logger.debug("Creating Gemini client with API key")
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized successfully (model: {self.MODEL})")
            
        except ImportError as e:
            logger.error(f"Failed to import google-genai: {e}")
            logger.error("Install with: pip install google-genai")
            raise
    
    @property
    def state(self) -> GeminiSessionState:
        """Get current session state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self._state in (GeminiSessionState.CONNECTED, GeminiSessionState.STREAMING)
    
    def set_audio_callback(
        self,
        callback: Optional[Callable[[bytes], Awaitable[None]]],
    ) -> None:
        """Set callback for receiving audio chunks.
        
        Args:
            callback: Async function called with PCM audio bytes.
        """
        self._audio_callback = callback
    
    def set_text_callback(
        self,
        callback: Optional[Callable[[str], Awaitable[None]]],
    ) -> None:
        """Set callback for receiving text transcriptions.
        
        Args:
            callback: Async function called with text strings.
        """
        self._text_callback = callback
    
    def set_completion_callback(
        self,
        callback: Optional[Callable[[], Awaitable[None]]],
    ) -> None:
        """Set callback for when response is complete.
        
        Args:
            callback: Async function called when response finishes.
        """
        self._completion_callback = callback
    
    async def connect(self) -> bool:
        """Connect to the Gemini Live API.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if self._state not in (GeminiSessionState.DISCONNECTED, GeminiSessionState.ERROR):
            logger.warning(f"Cannot connect: state is {self._state}")
            return False
        
        # If in ERROR state, clean up first
        if self._state == GeminiSessionState.ERROR:
            logger.info("Reconnecting from ERROR state, cleaning up...")
            self._session = None
            self._session_manager = None
        
        try:
            from google.genai import types
            import time
            
            self._state = GeminiSessionState.CONNECTING
            logger.info("Connecting to Gemini Live API...")
            logger.debug(f"Model: {self.MODEL}, Voice: {self.voice}")
            
            # Configure the session
            logger.debug("Creating LiveConnectConfig")
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice,
                        )
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(text=self.system_instruction)]
                ),
            )
            
            # Connect to the Live API
            # Note: aio.live.connect() returns an async context manager
            # We use __aenter__ to get the session object
            logger.debug("Establishing WebSocket connection...")
            connect_start = time.time()
            self._session_manager = self._client.aio.live.connect(
                model=self.MODEL,
                config=config,
            )
            self._session = await self._session_manager.__aenter__()
            connect_time = time.time() - connect_start
            
            self._state = GeminiSessionState.CONNECTED
            logger.info(f"Connected to Gemini Live API in {connect_time:.3f}s (voice: {self.voice})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            logger.debug(f"Connection error details: {type(e).__name__}: {e}")
            self._state = GeminiSessionState.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the Gemini Live API."""
        if self._session_manager:
            try:
                await self._session_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        
        self._session = None
        self._session_manager = None
        self._state = GeminiSessionState.DISCONNECTED
        self._audio_buffer.clear()
        logger.info("Disconnected from Gemini Live API")
    
    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to Gemini.
        
        Args:
            audio_data: PCM audio bytes (16kHz, 16-bit, mono).
        """
        if not self._session or self._state not in (
            GeminiSessionState.CONNECTED,
            GeminiSessionState.STREAMING,
        ):
            logger.warning(f"Cannot send audio: session={self._session is not None}, state={self._state}")
            return
        
        # Validate audio data
        if not audio_data or len(audio_data) == 0:
            logger.debug("Skipping empty audio chunk")
            return
        
        # Ensure audio data has even number of bytes (16-bit PCM = 2 bytes per sample)
        if len(audio_data) % 2 != 0:
            logger.warning(f"Audio data has odd number of bytes ({len(audio_data)}), trimming")
            audio_data = audio_data[:-1]
        
        try:
            from google.genai import types
            
            logger.debug(f"Sending audio: {len(audio_data)} bytes ({len(audio_data)/32000:.2f}s at 16kHz)")
            
            # Send as realtime input with raw bytes
            # Note: types.Blob expects raw bytes, NOT base64-encoded string
            # The library handles base64 encoding internally
            # Using "audio/pcm" without rate parameter as per SDK tests
            await self._session.send_realtime_input(
                audio=types.Blob(
                    mime_type="audio/pcm",
                    data=audio_data,  # Raw bytes, not base64
                )
            )
            
            self._state = GeminiSessionState.STREAMING
            logger.debug("Audio sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            logger.debug(f"Send audio error details: {type(e).__name__}: {e}")
    
    async def send_text(self, text: str) -> None:
        """Send text input to Gemini.
        
        Args:
            text: Text message to send.
        """
        if not self._session or not self.is_connected:
            return
        
        try:
            from google.genai import types
            
            await self._session.send_client_content(
                turns=[
                    types.Content(
                        parts=[types.Part(text=text)],
                        role="user",
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error sending text: {e}")
    
    async def end_turn(self) -> None:
        """Signal end of user input and request response."""
        if not self._session or not self.is_connected:
            logger.warning(f"Cannot end turn: session={self._session is not None}, connected={self.is_connected}")
            return
        
        try:
            logger.debug("Sending end of turn signal to Gemini")
            # Send end of turn signal
            await self._session.send_client_content(
                turn_complete=True,
            )
            logger.debug("End of turn signal sent successfully")
            
        except Exception as e:
            logger.error(f"Error ending turn: {e}")
            logger.debug(f"End turn error details: {type(e).__name__}: {e}")
    
    async def receive_responses(self) -> AsyncIterator[bytes]:
        """Receive audio responses from Gemini.
        
        Yields:
            PCM audio bytes (24kHz, 16-bit, mono).
        """
        if not self._session:
            logger.warning("Cannot receive responses: no session")
            return
        
        try:
            import time
            self._audio_buffer.clear()
            logger.debug("Starting to receive responses from Gemini")
            receive_start = time.time()
            chunk_count = 0
            total_bytes = 0
            
            async for response in self._session.receive():
                # Check for audio data
                if response.data:
                    # Handle both raw bytes and base64-encoded responses
                    # The google-genai SDK may return either depending on version
                    if isinstance(response.data, bytes):
                        audio_bytes = response.data
                    else:
                        # Assume base64-encoded string
                        audio_bytes = base64.b64decode(response.data)
                    
                    chunk_count += 1
                    total_bytes += len(audio_bytes)
                    self._audio_buffer.append(audio_bytes)
                    
                    if chunk_count % 10 == 0:
                        logger.debug(f"Received {chunk_count} audio chunks ({total_bytes} bytes total)")
                    
                    if self._audio_callback:
                        await self._audio_callback(audio_bytes)
                    
                    yield audio_bytes
                
                # Check for text (transcription or response text)
                if response.text:
                    logger.debug(f"Received text from Gemini: {response.text[:100]}...")
                    if self._text_callback:
                        await self._text_callback(response.text)
                
                # Check for turn completion
                if response.server_content and response.server_content.turn_complete:
                    elapsed = time.time() - receive_start
                    logger.info(f"Gemini response complete: {chunk_count} chunks, {total_bytes} bytes in {elapsed:.3f}s")
                    if self._completion_callback:
                        await self._completion_callback()
                    break
                    
        except Exception as e:
            logger.error(f"Error receiving response: {e}")
            logger.debug(f"Receive error details: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._state = GeminiSessionState.ERROR
    
    async def get_full_audio_response(self) -> bytes:
        """Collect full audio response.
        
        Returns:
            Complete audio response as PCM bytes.
        """
        logger.debug("Collecting full audio response")
        audio_chunks = []
        async for chunk in self.receive_responses():
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)
    
    async def process_voice_request(
        self,
        audio_data: bytes,
    ) -> Optional[bytes]:
        """Process a complete voice request and get audio response.
        
        This is a convenience method that sends audio, ends the turn,
        and collects the full response.
        
        Args:
            audio_data: User's voice input (16kHz, 16-bit, mono PCM).
            
        Returns:
            Audio response bytes, or None if error.
        """
        try:
            # Ensure connected
            if not self.is_connected:
                if not await self.connect():
                    return None
            
            # Send the audio
            await self.send_audio(audio_data)
            
            # End the turn to request response
            await self.end_turn()
            
            # Collect response
            return await self.get_full_audio_response()
            
        except Exception as e:
            logger.error(f"Error processing voice request: {e}")
            return None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
