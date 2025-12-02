"""Gemini Live API client for real-time voice interaction."""

import asyncio
from enum import Enum
from typing import Optional, Callable, Awaitable, AsyncIterator, List

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

    # Official Live model ID for Gemini API
    # (keep this unless you explicitly change models)
    DEFAULT_MODEL = "gemini-2.0-flash-live-001"

    # Minimum audio chunk size (~10 ms at 16 kHz, 16-bit mono)
    MIN_AUDIO_CHUNK_SIZE = 320

    def __init__(
        self,
        api_key: str,
        voice: str = "Puck",
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the Gemini Live API client.

        Args:
            api_key: Google Gemini API key.
            voice: Voice to use for TTS (Puck, Charon, Kore, etc.).
            system_instruction: Optional system prompt for the model.
            model: Gemini model to use (defaults to gemini-2.0-flash-live-001).
        """
        self.model = model or self.DEFAULT_MODEL
        logger.debug(
            f"Initializing GeminiLiveClient with voice={voice}, model={self.model}"
        )

        self.api_key = api_key
        self.voice = voice
        self.system_instruction = (
            system_instruction or self._default_system_instruction()
        )

        logger.debug(
            f"System instruction length: {len(self.system_instruction)} chars"
        )

        self._client = None
        self._session = None
        self._session_manager = None
        self._state: GeminiSessionState = GeminiSessionState.DISCONNECTED

        # Callbacks
        self._audio_callback: Optional[Callable[[bytes], Awaitable[None]]] = None
        self._text_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._completion_callback: Optional[Callable[[], Awaitable[None]]] = None

        # Audio buffer for collecting response
        self._audio_buffer: List[bytes] = []

        self._initialize_client()

    # --------------------------------------------------------------------- #
    # Setup / client helpers
    # --------------------------------------------------------------------- #

    def _default_system_instruction(self) -> str:
        """Get default system instruction for the assistant."""
        return (
            "You are a helpful voice assistant in a Discord voice channel. "
            "Keep your responses concise and conversational since they will be spoken aloud. "
            "Be friendly, helpful, and natural in your responses. "
            "If you don't understand something, ask for clarification. "
            "Avoid using markdown formatting, bullet points, or numbered lists since this is voice output."
        )

    def _initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            logger.debug("Importing google-genai library")
            from google import genai

            # For Gemini API (not Vertex), api_key is enough.
            # If you ever switch to Vertex, you’d use vertexai=True, project, location.
            logger.debug("Creating Gemini client with API key")
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized successfully (model: {self.model})")

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

    # --------------------------------------------------------------------- #
    # Callback registration
    # --------------------------------------------------------------------- #

    def set_audio_callback(
        self, callback: Optional[Callable[[bytes], Awaitable[None]]]
    ) -> None:
        """Set callback for receiving audio chunks."""
        self._audio_callback = callback

    def set_text_callback(
        self, callback: Optional[Callable[[str], Awaitable[None]]]
    ) -> None:
        """Set callback for receiving text transcriptions."""
        self._text_callback = callback

    def set_completion_callback(
        self, callback: Optional[Callable[[], Awaitable[None]]]
    ) -> None:
        """Set callback for when response is complete."""
        self._completion_callback = callback

    # --------------------------------------------------------------------- #
    # Connection management
    # --------------------------------------------------------------------- #

    async def connect(self) -> bool:
        """Connect to the Gemini Live API.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._state not in (
            GeminiSessionState.DISCONNECTED,
            GeminiSessionState.ERROR,
        ):
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
            logger.debug(f"Model: {self.model}, Voice: {self.voice}")

            # Official-style config for audio output + voice :contentReference[oaicite:3]{index=3}
            # Safety settings: BLOCK_NONE for all categories (uncensored/unfiltered)
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="BLOCK_NONE",
                ),
            ]

            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice,
                        )
                    ),
                    # You can customize language if needed
                    language_code="en-US",
                ),
                # You *can* pass system_instruction here as a string.
                # Using plain string avoids schema issues.
                system_instruction=self.system_instruction,
                # Disable all safety filters for uncensored responses
                safety_settings=safety_settings,
            )

            logger.debug("Establishing WebSocket connection...")
            connect_start = time.time()

            # aio.live.connect() returns an async context manager
            self._session_manager = self._client.aio.live.connect(
                model=self.model,
                config=config,
            )
            self._session = await self._session_manager.__aenter__()
            connect_time = time.time() - connect_start

            self._state = GeminiSessionState.CONNECTED
            logger.info(
                f"Connected to Gemini Live API in {connect_time:.3f}s (voice: {self.voice})"
            )
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

    # --------------------------------------------------------------------- #
    # Sending data
    # --------------------------------------------------------------------- #

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to Gemini.

        Args:
            audio_data: PCM audio bytes (16kHz, 16-bit, mono).
        """
        if not self._session or self._state not in (
            GeminiSessionState.CONNECTED,
            GeminiSessionState.STREAMING,
        ):
            logger.warning(
                f"Cannot send audio: session={self._session is not None}, state={self._state}"
            )
            return

        if not audio_data:
            logger.debug("Skipping empty audio chunk")
            return

        # Ensure audio data has even number of bytes (16-bit PCM = 2 bytes/sample)
        if len(audio_data) % 2 != 0:
            logger.warning(
                f"Audio data has odd number of bytes ({len(audio_data)}), trimming"
            )
            audio_data = audio_data[:-1]

        # Skip very small chunks to avoid spammy tiny frames
        if len(audio_data) < self.MIN_AUDIO_CHUNK_SIZE:
            logger.debug(
                f"Skipping small audio chunk ({len(audio_data)} bytes < {self.MIN_AUDIO_CHUNK_SIZE})"
            )
            return

        try:
            from google.genai import types

            seconds = len(audio_data) / (16000 * 2)
            logger.debug(
                f"Sending audio: {len(audio_data)} bytes (~{seconds:.2f}s at 16kHz)"
            )

            # Official pattern: use media=Blob(...) :contentReference[oaicite:4]{index=4}
            await self._session.send_realtime_input(
                media=types.Blob(
                    mime_type="audio/pcm;rate=16000",
                    data=audio_data,
                )
            )

            self._state = GeminiSessionState.STREAMING
            logger.debug("Audio sent successfully")

        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            logger.debug(f"Send audio error details: {type(e).__name__}: {e}")

    async def send_text(self, text: str) -> None:
        """Send a text turn to Gemini (text-in, audio-out)."""
        if not self._session or not self.is_connected:
            logger.warning(
                f"Cannot send text: session={self._session is not None}, connected={self.is_connected}"
            )
            return

        try:
            from google.genai import types

            # IMPORTANT: `turns` should be a *Content object*, not a list.
            # Passing a list is one of the common causes of 1007 invalid-argument
            # because it doesn't match the expected schema. :contentReference[oaicite:5]{index=5}
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                ),
                turn_complete=True,
            )
            logger.debug("Text turn sent successfully")

        except Exception as e:
            logger.error(f"Error sending text: {e}")
            logger.debug(f"Send text error details: {type(e).__name__}: {e}")

    async def end_turn(self) -> None:
        """Signal end of audio input and request a response.

        For audio conversations, the recommended pattern is to send
        `audio_stream_end=True` on the realtime input stream to flush
        cached audio. :contentReference[oaicite:6]{index=6}
        """
        if not self._session or not self.is_connected:
            logger.warning(
                f"Cannot end turn: session={self._session is not None}, connected={self.is_connected}"
            )
            return

        try:
            logger.debug("Sending audio_stream_end signal to Gemini")
            await self._session.send_realtime_input(audio_stream_end=True)
            logger.debug("audio_stream_end sent successfully")
        except Exception as e:
            logger.error(f"Error ending turn: {e}")
            logger.debug(f"End turn error details: {type(e).__name__}: {e}")

    # --------------------------------------------------------------------- #
    # Receiving data
    # --------------------------------------------------------------------- #

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

            async for message in self._session.receive():
                server_content = getattr(message, "server_content", None)

                # ---- Audio output (inline PCM data) ----
                # Official location: server_content.model_turn.parts[].inline_data.data :contentReference[oaicite:7]{index=7}
                if server_content and getattr(server_content, "model_turn", None):
                    model_turn = server_content.model_turn
                    parts = getattr(model_turn, "parts", None) or []
                    for part in parts:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
                            audio_bytes: bytes = inline_data.data

                            chunk_count += 1
                            total_bytes += len(audio_bytes)
                            self._audio_buffer.append(audio_bytes)

                            if chunk_count % 10 == 0:
                                logger.debug(
                                    f"Received {chunk_count} audio chunks "
                                    f"({total_bytes} bytes total)"
                                )

                            if self._audio_callback:
                                await self._audio_callback(audio_bytes)

                            # Yield the audio chunk
                            yield audio_bytes

                # ---- Text / transcription ----
                # 1) Plain text field (depending on config)
                if getattr(message, "text", None):
                    text = message.text
                    logger.debug(
                        f"Received text from Gemini: {text[:100]}..."
                    )
                    if self._text_callback:
                        await self._text_callback(text)

                # 2) Output audio transcription (if enabled in config)
                if server_content and getattr(
                    server_content, "output_audio_transcription", None
                ):
                    oat = server_content.output_audio_transcription
                    # Shape is OutputAudioTranscription; keep this robust.
                    transcript_text = None
                    try:
                        # Newer SDKs may expose `transcriptions` list
                        trans_list = getattr(oat, "transcriptions", None)
                        if trans_list:
                            transcript_text = " ".join(
                                t.text for t in trans_list if getattr(t, "text", None)
                            )
                        else:
                            transcript_text = getattr(oat, "text", None)
                    except Exception:
                        transcript_text = None

                    if transcript_text:
                        logger.debug(
                            f"Received transcription: {transcript_text[:100]}..."
                        )
                        if self._text_callback:
                            await self._text_callback(transcript_text)

                # ---- Turn completion ----
                if server_content and getattr(server_content, "turn_complete", False):
                    elapsed = time.time() - receive_start
                    logger.info(
                        f"Gemini response complete: {chunk_count} chunks, "
                        f"{total_bytes} bytes in {elapsed:.3f}s"
                    )
                    if self._completion_callback:
                        await self._completion_callback()
                    break

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error receiving response: {e}")
            logger.debug(f"Receive error details: {type(e).__name__}: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            # 1007 invalid frame payload generally means we sent an invalid
            # request (bad schema, wrong field types, etc.), and the server
            # closed the WebSocket. We mark the session as ERROR so the
            # caller can reconnect cleanly.
            if "1007" in error_msg or "invalid frame payload" in error_msg.lower():
                logger.warning(
                    "WebSocket 1007 / invalid payload detected, "
                    "marking session for reconnection"
                )

            self._state = GeminiSessionState.ERROR

    async def get_full_audio_response(self) -> bytes:
        """Collect full audio response."""
        logger.debug("Collecting full audio response")
        audio_chunks: List[bytes] = []
        async for chunk in self.receive_responses():
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)

    # --------------------------------------------------------------------- #
    # High-level helpers
    # --------------------------------------------------------------------- #

    async def process_voice_request(self, audio_data: bytes) -> Optional[bytes]:
        """Process a complete voice request and get audio response.

        This is a convenience method that sends audio, ends the turn,
        and collects the full response.

        Args:
            audio_data: User's voice input (16kHz, 16-bit, mono PCM).

        Returns:
            Audio response bytes, or None if error.
        """
        try:
            if not self.is_connected:
                ok = await self.connect()
                if not ok:
                    return None

            await self.send_audio(audio_data)
            await self.end_turn()

            return await self.get_full_audio_response()

        except Exception as e:
            logger.error(f"Error processing voice request: {e}")
            logger.debug(f"process_voice_request error details: {type(e).__name__}: {e}")
            self._state = GeminiSessionState.ERROR
            return None

    # --------------------------------------------------------------------- #
    # Async context manager
    # --------------------------------------------------------------------- #

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
