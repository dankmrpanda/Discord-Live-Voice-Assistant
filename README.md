# Discord Live VC Bot

A Discord-based AI voice assistant that joins voice channels and responds to voice commands using Google's Gemini Live API.

## Features

- **Wake Word Detection**: Listens for configurable wake phrase (default: "hey jarvis") using OpenWakeWord
- **Real-time Voice AI**: Uses Gemini Live API for low-latency speech-to-speech interaction
- **Modular Design**: Easily swap wake words, voices, and AI models
- **Single Request Processing**: Handles one user at a time to prevent overlapping responses

## Requirements

- Python 3.11+
- Docker & Docker Compose
- Discord Bot Token
- Google Gemini API Key
- FFmpeg

## Quick Start

### 1. Clone and Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

### 2. Run with Docker

```bash
cd docker
docker-compose up --build
```

### 3. Run Locally (Development)

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the bot
python -m src.main
```

## Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section and create a bot
4. Enable these Privileged Gateway Intents:
   - Message Content Intent
   - Server Members Intent (optional)
5. Copy the bot token to your `.env` file
6. Go to OAuth2 > URL Generator:
   - Select scopes: `bot`, `applications.commands`
   - Select permissions: `Connect`, `Speak`, `Use Voice Activity`
7. Use the generated URL to invite the bot to your server

## Bot Commands

- `/join` - Bot joins your current voice channel
- `/leave` - Bot leaves the voice channel
- `/status` - Check bot status and current settings

## Configuration

The bot uses two configuration files:

### 1. `.env` - API Keys and Secrets

Copy `.env.example` to `.env` and add your API keys:

```env
DISCORD_BOT_TOKEN=your_discord_token
GEMINI_API_KEY=your_gemini_key
```

### 2. `config.yaml` - Bot Settings

Edit `config.yaml` to customize the bot behavior:

```yaml
# Wake word settings
wake_word:
  phrase: "hey_jarvis"    # Wake phrase to listen for
  threshold: 0.5          # Detection sensitivity (0.0-1.0)

# Voice settings
voice:
  name: "Puck"            # Gemini voice for responses

# Bot behavior
behavior:
  capture_duration: 5.0   # How long to listen after wake word
  silence_threshold: 0.5  # Silence detection threshold

# Customize the bot's personality
system_prompt: |
  You are a helpful voice assistant...

# Logging
logging:
  level: "INFO"           # DEBUG for troubleshooting
```

After editing `config.yaml`, restart the bot:
```bash
docker compose restart
```

### Wake Word Options
- `hey_jarvis` (default)
- `alexa`
- `hey_mycroft`
- `hey_siri` (check licensing)

### Voice Options

Available Gemini voices (set `GEMINI_VOICE` in `.env`):
- `Puck` (default)
- `Charon`
- `Kore`
- `Fenrir`
- `Aoede`
- `Leda`
- `Orus`
- `Zephyr`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Discord Voice Channel                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Audio Capture (Opus → 16kHz PCM)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Wake Word Detection (OpenWakeWord)                         │
│  State: LISTENING → PROCESSING (on detection)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemini Live API (WebSocket)                                │
│  - Streams audio in                                          │
│  - Receives audio response                                   │
│  State: PROCESSING → SPEAKING                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Audio Playback (24kHz PCM → Opus)                          │
│  State: SPEAKING → LISTENING (after playback)               │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
discord-live-vc-bot/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── voice_handler.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py
│   │   ├── playback.py
│   │   └── processor.py
│   ├── ai/
│   │   ├── __init__.py
│   │   └── gemini_client.py
│   ├── wake_word/
│   │   ├── __init__.py
│   │   └── detector.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Troubleshooting

### Bot can't hear users
- Ensure the bot has "Use Voice Activity" permission
- Check that users aren't server-muted

### Wake word not detecting
- Try lowering `WAKE_WORD_THRESHOLD` in `.env`
- Speak clearly and directly

### High latency
- Check your network connection
- Gemini Live API is optimized for low latency but network conditions affect performance

## License

MIT License
