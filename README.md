# J.A.R.V.I.S.

Just A Rather Very Intelligent System.

A real-time AI voice assistant for Discord voice channels, powered by Gemini Live and OpenWakeWord.

## Features

- Wake word activation (`hey_jarvis` by default) with OpenWakeWord
- Real-time speech-to-speech responses using Gemini Live API
- Text prompt command (`/ask`) with queueing while busy
- Queue controls (`/queue`, `/clearqueue`)
- Playback controls (`/stop`, `/pause`, `/continue`)
- Config hot reload from `config.yaml` (no bot restart needed)
- Per-user audio handling and VAD-based capture end detection
- Docker workflow for reliable ONNX/OpenWakeWord runtime

## Requirements

- Python 3.11+
- FFmpeg
- A Discord bot token
- A Gemini API key

## Required Secrets (`.env`)

Create a `.env` file in the project root:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_gemini_api_key
# optional
DISCORD_APPLICATION_ID=your_discord_application_id
```

`DISCORD_BOT_TOKEN` and `GEMINI_API_KEY` are required.

## Quick Start (Docker)

From the repo root:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

View logs:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

Stop:

```bash
docker compose -f docker/docker-compose.yml down
```

Notes:

- Logs are stored in `./logs` (mounted to `/app/logs` in container)
- OpenWakeWord model assets are cached in `./openwakeword_models`
- Compose uses `network_mode: host` and mounts `src` directly for faster iteration

## Quick Start (Local Python)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure FFmpeg is installed and available on `PATH`.

3. Run the bot:

```bash
python -m src.main
```

## Discord Bot Setup

In Discord Developer Portal:

1. Create an application and add a bot
2. Enable intents:
   - `MESSAGE CONTENT INTENT`
   - `SERVER MEMBERS INTENT` (optional)
3. Generate an OAuth URL with scopes:
   - `bot`
   - `applications.commands`
4. Grant bot permissions:
   - Connect
   - Speak
   - Use Voice Activity

## Slash Commands

| Command | Description |
|---|---|
| `/join` | Join your current voice channel |
| `/leave` | Leave the voice channel |
| `/status` | Show connection/model/status details |
| `/ask <prompt>` | Send a text prompt (queued if busy) |
| `/queue` | Show queued prompts |
| `/clearqueue` | Remove all queued prompts |
| `/stop` | Stop current response and continue with next queued prompt |
| `/pause` | Pause current spoken response |
| `/continue` | Resume paused response |
| `/help` | Show help in Discord |

## Configuration

Two sources are used:

- `.env`: secrets only (requires restart after changes)
- `config.yaml`: runtime settings (auto-reloads while bot is running)

### `config.yaml` sections

- `wake_word`: phrase/model and detection threshold
- `voice`: Gemini response voice
- `gemini`: model and feature toggles (`thinking`, `google_search`, `function_calling`, `automatic_function_response`)
- `behavior`: capture duration, silence threshold, response timeout guards
- `audio`: sample rates and playback buffer
- `logging`: level, directory, audio debug logging
- `system_prompt`: assistant behavior/personality

### Wake word options

Built-in values include:

- `hey_jarvis` (default)
- `alexa`
- `hey_mycroft`
- `timer`
- `weather`

You can also provide a path to a custom `.onnx` wake-word model.

## Project Structure

```text
src/
  ai/              Gemini client and streaming parsing
  audio/           capture, processing, sink, playback
  bot/             Discord client, voice handler, queue/session logic
  utils/           config + logging
  wake_word/       OpenWakeWord integration
  main.py          application entrypoint

docker/
  Dockerfile
  docker-compose.yml

tests/
  unit tests for config, queue, session, sink, processing, pipeline
```

## Development

Run tests:

```bash
pytest -q
```

If your environment does not have all native audio/runtime deps, run tests in your containerized environment.

## Troubleshooting

- Bot starts but cannot join/speak:
  - Verify bot voice permissions in server/channel
  - Verify FFmpeg is installed
- Configuration errors at startup:
  - Check `.env` has required keys
- Wake word not triggering reliably:
  - Lower `wake_word.threshold` (for example `0.3`)
  - Enable `logging.log_audio: true` for diagnosis
- Choppy playback:
  - Increase `audio.playback_buffer_ms` (for example `250-350`)

## License

MIT. See `LICENSE.txt`.

