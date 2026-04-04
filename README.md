# 🤖 J.A.R.V.I.S.

<div align="center">

### *"Just A Rather Very Intelligent System"*

**A real-time AI voice assistant for Discord — inspired by Tony Stark's legendary AI companion**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord-py--cord%202.8.0rc1-5865F2.svg)](https://pycord.dev/)
[![Gemini](https://img.shields.io/badge/AI-Gemini%20Live-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DAVE Status](https://img.shields.io/badge/DAVE%20send-✅%20working-brightgreen.svg)](https://github.com/Pycord-Development/pycord/pull/3143)
[![DAVE Receive](https://img.shields.io/badge/DAVE%20receive-⚠️%20pending-orange.svg)](https://github.com/Pycord-Development/pycord/issues/3139)

*"Good evening, sir. I've prepared the voice channel for your arrival."*

</div>

---

## ⚠️ Current Status — Discord DAVE Encryption

On **March 2, 2026**, Discord enforced end-to-end encryption (DAVE protocol) on all voice calls.
This requires **py-cord 2.8.0rc1+**, which is now in use.

| Feature | Status |
|---|---|
| Connect to voice channel | ✅ Working |
| Play audio (Gemini responses via `/ask`) | ✅ Working |
| Wake word detection (voice receive) | ⚠️ Pending DAVE receive support in py-cord |

**Wake word detection is temporarily unavailable.** Voice receive through sinks is not yet
implemented for DAVE-encrypted channels. Track progress at
[pycord#3139](https://github.com/Pycord-Development/pycord/issues/3139).
Until then, use `/ask` to interact with Jarvis via text.

---

## ✨ Features

### 🎤 Voice Interaction
- **"Hey Jarvis" Wake Word** — Activate with the iconic wake phrase, powered by [OpenWakeWord](https://github.com/dscripka/openWakeWord) *(temporarily unavailable — see status above)*
- **Real-Time Speech-to-Speech** — Low-latency voice conversations using Gemini Live API's bidirectional audio streaming
- **Multi-User Support** — Per-user audio processing and wake word detection, even with 3+ users in the channel
- **Voice Activity Detection (VAD)** — Intelligent speech detection with configurable silence thresholds

### 💬 Text & Queue System
- **Text Prompts** — Send prompts directly via `/ask` command without wake word activation
- **Smart Queue** — Automatic queuing when busy, with text prompts taking priority over wake word detection
- **Queue Management** — View pending prompts with `/queue`

### 🎛️ Playback Controls
- **Stop** — Cancel current response and move to next in queue
- **Pause/Resume** — Pause and continue responses mid-playback
- **Streaming Playback** — Audio plays as it arrives with configurable buffer delay

### ⚡ Advanced Capabilities
- **Thinking Mode** — Enhanced reasoning with Gemini's internal thought process
- **Google Search Grounding** — Real-time web search for current information
- **Function Calling** — Extensible tool support (framework ready for your custom tools)
- **Hot Reload** — Configuration changes apply without restart

### 🐳 Deployment Options
- **Docker** — Production-ready containerization
- **Local Development** — Run directly with Python

---

## 📋 Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| FFmpeg | Latest |
| Docker & Docker Compose | Latest (optional) |

### API Keys Required
- **Discord Bot Token** — [Discord Developer Portal](https://discord.com/developers/applications)
- **Google Gemini API Key** — [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/discord-live-vc-bot.git
cd discord-live-vc-bot
```

### 2. Create Environment File

```bash
# Create .env file with your API keys
cat > .env << EOF
DISCORD_BOT_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_gemini_api_key
EOF
```

### 3. Run with Docker (Recommended)

```bash
cd docker
docker-compose up --build
```

### 4. Run Locally (Development)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Jarvis
python -m src.main
```

---

## 🤖 Discord Bot Setup

1. **Create Application**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application" and name it "Jarvis"

2. **Configure Bot**
   - Navigate to "Bot" section
   - Click "Add Bot"
   - Enable **Privileged Gateway Intents**:
     - ✅ Message Content Intent
     - ✅ Server Members Intent (optional)

3. **Get Bot Token**
   - Under "Bot" section, click "Reset Token"
   - Copy the token to your `.env` file

4. **Generate Invite URL**
   - Go to OAuth2 → URL Generator
   - **Scopes**: `bot`, `applications.commands`
   - **Bot Permissions**:
     - ✅ Connect
     - ✅ Speak
     - ✅ Use Voice Activity

5. **Invite Jarvis**
   - Use the generated URL to invite Jarvis to your server

---

## 🎮 Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `/join` | 🔊 Jarvis joins your current voice channel |
| `/leave` | 👋 Jarvis leaves the voice channel |
| `/status` | 📊 Show Jarvis status, settings, and queue size |
| `/ask <prompt>` | 💬 Send a text prompt (queued if busy) |
| `/queue` | 📋 View pending prompts |

### Playback Controls

| Command | Description |
|---------|-------------|
| `/stop` | 🛑 Stop current response and process next in queue |
| `/pause` | ⏸️ Pause the current response |
| `/continue` | ▶️ Resume a paused response |

---

## ⚙️ Configuration

Jarvis uses **two configuration sources**:

| File | Purpose | Reload |
|------|---------|--------|
| `.env` | API keys and secrets | Requires restart |
| `config.yaml` | All bot settings | **Auto-reloads** |

### `.env` — Secrets Only

```env
DISCORD_BOT_TOKEN=your_discord_token
GEMINI_API_KEY=your_gemini_key
DISCORD_APPLICATION_ID=optional_app_id
```

### `config.yaml` — Jarvis Settings

```yaml
# Wake Word Configuration
wake_word:
  phrase: "hey_jarvis"     # The iconic wake phrase
  threshold: 0.3           # Detection sensitivity (0.0-1.0)

# Voice Settings
voice:
  name: "Sulafat"          # Gemini voice for responses

# Gemini Model Settings
gemini:
  model: "gemini-2.5-flash-native-audio-preview-09-2025"
  thinking: true           # Enable enhanced reasoning
  google_search: true      # Enable web search grounding
  function_calling: true   # Enable tool calling
  automatic_function_response: true

# Bot Behavior
behavior:
  capture_duration: 7.0    # Max seconds to record after wake word
  silence_threshold: 1.0   # Seconds of silence to end capture

# Customize Jarvis's personality
system_prompt: |
  You are Jarvis, an AI assistant inspired by Tony Stark's J.A.R.V.I.S.
  You are helpful, witty, and slightly formal in your responses.
  Keep responses concise since they will be spoken aloud.
  Occasionally use dry humor when appropriate.
  Avoid markdown formatting since this is voice output.

# Audio Settings
audio:
  playback_buffer_ms: 200  # Buffer before playback starts

# Logging
logging:
  level: "INFO"            # DEBUG for troubleshooting
  log_audio: false         # Log audio processing details
```

### Available Wake Words

| Wake Word | Model Name |
|-----------|------------|
| `hey_jarvis` | **Default** — *"Hey Jarvis"* |
| `alexa` | Amazon Alexa |
| `hey_mycroft` | Mycroft assistant |
| `timer` | Keyword detection |
| `weather` | Keyword detection |

### Available Voices

#### Stable Model (`gemini-2.0-flash-live-001`)
| Voice | Description |
|-------|-------------|
| `Puck` | Energetic, youthful |
| `Charon` | Deep, authoritative |
| `Kore` | Warm, friendly |
| `Fenrir` | Bold, confident |
| `Aoede` | Melodic, expressive |
| `Leda` | Calm, soothing |
| `Orus` | Clear, professional |
| `Zephyr` | Light, airy |

#### Preview Model (`gemini-2.5-flash-native-audio-preview-09-2025`)
All above voices plus:
| Voice | Description |
|-------|-------------|
| `Sulafat` | Warm, confident, persuasive |
| `Despina` | Warm, inviting, smooth |
| `Vindemiatrix` | Calm, mature, reassuring |

> 💡 **Tip**: For the most Jarvis-like experience, try `Charon` (authoritative) or `Orus` (professional)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Discord Voice Channel                         │
│                 (48kHz Stereo Opus Audio)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WakeWordSink                                  │
│            Per-user audio capture & routing                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AudioProcessor                                │
│         48kHz Stereo → 16kHz Mono (Polyphase Resampling)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   WakeWordDetector   │   │          AudioCapture                │
│    (OpenWakeWord)    │   │   Per-user buffers + VAD detection   │
│   Per-user models    │   │    + Streaming ring buffer           │
└──────────┬───────────┘   └──────────────────┬───────────────────┘
           │                                   │
           │ "Hey Jarvis" detected             │ Audio stream
           ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VoiceHandler                                  │
│   State Machine: IDLE → CONNECTING → LISTENING →                │
│                  PROCESSING → SPEAKING → LISTENING              │
│   + /ask Queue Management + Response Controls                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GeminiLiveClient                                │
│          Bidirectional WebSocket Audio Streaming                 │
│    + Health Check + Auto-Reconnect + Thinking Mode               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AudioPlayback                                 │
│         StreamingPCMSource with buffered playback               │
│    24kHz Mono → 48kHz Stereo (Polyphase Resampling)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Discord Voice Channel                          │
│                    (Jarvis Responds)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
discord-live-vc-bot/
├── docker/
│   ├── Dockerfile           # Production container
│   └── docker-compose.yml   # Container orchestration
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point & startup
│   ├── bot/
│   │   ├── client.py        # Discord bot & slash commands
│   │   └── voice_handler.py # Voice state machine & streaming
│   ├── audio/
│   │   ├── capture.py       # Audio input + VAD + per-user buffers
│   │   ├── playback.py      # Streaming audio output
│   │   ├── processor.py     # Format conversion & resampling
│   │   └── sink.py          # Discord audio receiver
│   ├── ai/
│   │   └── gemini_client.py # Gemini Live API client
│   ├── wake_word/
│   │   └── detector.py      # Wake word detection (per-user)
│   └── utils/
│       ├── config.py        # Config loading & hot reload
│       └── logger.py        # Logging setup
├── .env                     # API keys (create this)
├── .gitignore
├── config.yaml              # Jarvis configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🔧 Dependencies

### Core Libraries
| Package | Purpose |
|---------|---------|
| [py-cord 2.8.0rc1](https://pycord.dev/) | Discord API — required for DAVE voice encryption ([PR #3143](https://github.com/Pycord-Development/pycord/pull/3143)) |
| [davey](https://pypi.org/project/davey/) | Discord DAVE/OpenMLS encryption (auto-installed with py-cord[voice]) |
| [google-genai](https://pypi.org/project/google-genai/) | Gemini Live API client |
| [openwakeword](https://github.com/dscripka/openWakeWord) | Wake word detection |
| [webrtcvad](https://pypi.org/project/webrtcvad/) | Voice activity detection |

### Audio Processing
| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Polyphase resampling |
| librosa | Audio analysis |
| soundfile | Audio I/O |

### Utilities
| Package | Purpose |
|---------|---------|
| PyYAML | Config file parsing |
| python-dotenv | Environment variables |
| aiohttp | Async HTTP |
| PyNaCl 1.6.x | Voice encryption (1.6+ required for py-cord 2.8) |

---

## 🔍 Troubleshooting

### Jarvis Can't Hear Users / Wake Word Not Working

> **Note:** As of py-cord 2.8.0rc1, voice receive is broken due to Discord's DAVE
> encryption enforcement (March 2, 2026). Wake word detection will not work until
> py-cord adds DAVE receive support. Use `/ask` in the meantime.
> Track: [pycord#3139](https://github.com/Pycord-Development/pycord/issues/3139)

Once DAVE receive is fixed:

1. ✅ Ensure Jarvis has **"Use Voice Activity"** permission
2. ✅ Check that users aren't server-muted or self-deafened
3. ✅ Verify Jarvis is properly connected (check `/status`)
4. ⬇️ Lower the `threshold` in `config.yaml` (try `0.3` or `0.2`)
5. 📊 Enable `log_audio: true` in config for debugging

### High Latency or Stuttering

1. 📶 Check your network connection
2. ⬆️ Increase `playback_buffer_ms` (try `300` or `400`)
3. 🌍 Gemini API latency varies by region and load

### Jarvis Disconnects Unexpectedly

1. 🔄 Jarvis auto-reconnects to Gemini on errors
2. 📋 Check logs for specific error messages
3. ✅ Verify API keys are valid and have quota

### No Audio Response

1. 🔇 Check Jarvis isn't muted in Discord
2. ⚙️ Verify Gemini model and voice compatibility
3. 📊 Enable `DEBUG` logging to trace the pipeline

---

## 🔒 Security Notes

- **Never commit `.env`** — It's in `.gitignore` by default
- API keys are only stored in `.env` (not in `config.yaml`)
- Docker container runs as non-root user (`botuser`)
- Jarvis only accesses voice channels it's invited to

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — Wake word detection
- [Pycord](https://pycord.dev/) — Discord API library
- [Google Gemini](https://ai.google.dev/) — AI and voice synthesis
- **Marvel/Iron Man** — For the inspiration behind J.A.R.V.I.S.

---

<div align="center">

*"I do anything and everything that Mr. Stark requires — including occasionally taking out the trash."*

**Built with ❤️ for Discord voice communities**

🔵 J.A.R.V.I.S. 🔵

</div>
