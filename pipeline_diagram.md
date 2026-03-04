# Discord Live VC Bot: End-to-End Pipeline

This diagram illustrates the flow of audio data and control signals from the moment a user speaks in a Discord voice channel to the moment the bot responds with AI-generated audio.

```mermaid
graph TD
    subgraph discord_vc [Discord Voice Channel]
        user[User Speaking]
        bot_voice[Bot Audio Output]
    end

    subgraph pycord [Py-Cord Library]
        op_in[Opus Receiver]
        op_out[Opus Transmitter]
    end

    subgraph audio_input [Audio Input Pipeline]
        sink[WakeWordSink]
        processor_in[AudioProcessor]
        capture[AudioCapture & VAD]
        wake_detector[WakeWordDetector]
    end

    subgraph orchestration [VoiceHandler State Machine]
        state{Bot State}
    end

    subgraph gemini_ai [Google Gemini Live API]
        gemini_client[GeminiLiveClient]
        gemini_cloud((Gemini 2.0 Flash))
    end

    subgraph audio_output [Audio Output Pipeline]
        processor_out[AudioProcessor]
        playback[AudioPlayback]
        source[StreamingPCMSource]
    end

    %% Flow - Input Path
    user -- "Opus Audio" --> op_in
    op_in -- "48kHz Stereo PCM" --> sink
    sink -- "Per-User Process" --> processor_in
    processor_in -- "16kHz Mono PCM" --> capture
    capture -- "Audio Chunks" --> wake_detector
    
    %% Trigger
    wake_detector -- "Wake Word Detected" --> state
    state -- "Transition to PROCESSING" --> gemini_client

    %% Flow - AI Interaction
    capture -- "Real-time Stream" --> gemini_client
    gemini_client -- "Websocket" --> gemini_cloud
    gemini_cloud -- "AI Audio (24kHz Mono)" --> gemini_client
    
    %% Flow - Output Path
    gemini_client -- "Audio Chunks" --> playback
    playback -- "Add to Stream" --> source
    source -- "Retrieve 20ms Frames" --> processor_out
    processor_out -- "48kHz Stereo PCM" --> op_out
    op_out -- "Opus Audio" --> bot_voice

    %% Styling
    style discord_vc fill:#7289da,stroke:#fff,color:#fff
    style gemini_ai fill:#4285f4,stroke:#fff,color:#fff
    style orchestration fill:#f39c12,stroke:#fff,color:#fff
    style state fill:#f1c40f,stroke:#333
```

### Text-Based Pipeline Diagram (ASCII)

```text
    [ DISCORD VOICE CHANNEL ]
      |                 ^
      | (Opus Audio)     | (Opus Audio)
      v                 |
    [ PY-CORD ] <-------|
      | (Decrypted 48kHz Stereo)
      v
    [ WAKE WORD SINK ] (Separates users)
      |
      v
    [ AUDIO PROCESSOR ] (Downsamples 48kHz -> 16kHz Mono)
      |
      v
    [ WAKE WORD DETECTOR ] (openwakeword - "Hey Jarvis")
      |
      +----- (DETECTED!) ----> [ VOICE HANDLER ] (Bot State Machine)
                                    |
                                    v (Start Streaming)
    [ AUDIO CAPTURE ] -------------+
      |                             |
      v (16kHz Mono Stream)         |
    [ GEMINI LIVE CLIENT ] <--------+
      |
      v (WebSocket Connection)
    [ GOOGLE GEMINI 2.0 AI ]
      |
      v (AI Returns 24kHz Mono Audio)
    [ GEMINI LIVE CLIENT ]
      |
      v (Audio Chunks)
    [ AUDIO PLAYBACK ] (Streams to Discord)
      |
      v (Upsamples 24kHz -> 48kHz Stereo)
    [ PY-CORD AUDIO OUTPUT ]
```


### Component Breakdown

1.  **Discord & Py-Cord**: Receives encrypted Opus packets, decrypts them into raw 48kHz stereo PCM.
2.  **WakeWordSink**: A custom Discord sink that separates audio by user to ensure the wake word can be detected even if multiple people are talking.
3.  **AudioProcessor**: Handles sampling rate conversion (48kHz <-> 16kHz/24kHz) and channel mixing (Stereo <-> Mono).
4.  **AudioCapture & VAD**: Buffers audio for detection and uses WebRTC VAD (Voice Activity Detection) to detect when a user stops speaking.
5.  **WakeWordDetector**: Uses `openwakeword` to monitor 16kHz mono audio for the specific wake phrase (e.g., "Hey Jarvis").
6.  **VoiceHandler**: The central "brain" or state machine. It coordinates when to start/stop listening and when to switch to responding.
7.  **GeminiLiveClient**: Manages the persistent WebSocket connection to Google's Gemini 2.0 Flash. It streams the user's speech up and receives the AI's response down in real-time.
8.  **StreamingPCMSource**: A custom Discord AudioSource that acts as a low-latency buffer, playing audio chunks from Gemini as they arrive rather than waiting for the full response.
