# Discord Live Voice Assistant

<div align="center">

**A real-time AI voice assistant for Discord voice channels powered by Google Gemini Live API**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord-py--cord-5865F2.svg)](https://pycord.dev/)
[![Gemini](https://img.shields.io/badge/AI-Gemini%20Live-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ✨ Features

### 🎤 Voice Interaction
- **Wake Word Detection** — Activate the bot hands-free using customizable wake phrases powered by [OpenWakeWord](https://github.com/dscripka/openWakeWord)
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
- **Function Calling** — Extensible tool support (framework ready)
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

# Run the bot
python -m src.main
```

---

## 🤖 Discord Bot Setup

1. **Create Application**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application" and name your bot

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

5. **Invite Bot**
   - Use the generated URL to invite the bot to your server

---

## 🎮 Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `/join` | 🔊 Join your current voice channel |
| `/leave` | 👋 Leave the voice channel |
| `/status` | 📊 Show bot status, settings, and queue size |
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

The bot uses **two configuration sources**:

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

### `config.yaml` — Bot Settings

```yaml
# Wake Word Configuration
wake_word:
  phrase: "hey_jarvis"     # Wake phrase to listen for
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

# Customize personality
system_prompt: |
  You are a helpful voice assistant in a Discord voice channel.
  Keep your responses concise and conversational.
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
| `hey_jarvis` | Default, recommended |
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
           │ Wake word detected                │ Audio stream
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
│                        (Audio Out)                               │
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
├── config.yaml              # Bot configuration
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🔧 Dependencies

### Core Libraries
| Package | Purpose |
|---------|---------|
| [py-cord](https://pycord.dev/) | Discord API (development branch for voice fixes) |
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
| PyNaCl | Voice encryption |

---

## 🔍 Troubleshooting

### Bot Can't Hear Users

1. ✅ Ensure bot has **"Use Voice Activity"** permission
2. ✅ Check that users aren't server-muted or self-deafened
3. ✅ Verify the bot is properly connected (check `/status`)

### Wake Word Not Detecting

1. ⬇️ Lower the `threshold` in `config.yaml` (try `0.3` or `0.2`)
2. 🎤 Speak clearly and at normal volume
3. 🔊 Ensure your microphone is working in Discord
4. 📊 Enable `log_audio: true` in config for debugging

### High Latency or Stuttering

1. 📶 Check your network connection
2. ⬆️ Increase `playback_buffer_ms` (try `300` or `400`)
3. 🌍 Gemini API latency varies by region and load

### Bot Disconnects Unexpectedly

1. 🔄 The bot auto-reconnects to Gemini on errors
2. 📋 Check logs for specific error messages
3. ✅ Verify API keys are valid and have quota

### No Audio Response

1. 🔇 Check bot isn't muted in Discord
2. ⚙️ Verify Gemini model and voice compatibility
3. 📊 Enable `DEBUG` logging to trace the pipeline

---

## 🔒 Security Notes

- **Never commit `.env`** — It's in `.gitignore` by default
- API keys are only stored in `.env` (not in `config.yaml`)
- Docker container runs as non-root user (`botuser`)
- Bot only accesses voice channels it's invited to

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — Wake word detection
- [Pycord](https://pycord.dev/) — Discord API library
- [Google Gemini](https://ai.google.dev/) — AI and voice synthesis

---

<div align="center">

**Built with ❤️ for Discord voice communities**

</div>
