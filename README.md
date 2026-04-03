# Discord Live VC Bot (Node.js)

A Discord voice assistant built with Node.js, `discord.js`, `@discordjs/voice`, and Gemini Live API.

## Requirements
- Node.js 22+
- FFmpeg in PATH (recommended for voice playback stability)
- Discord bot token
- Gemini API key
- Picovoice AccessKey (required for wake-word detection)

## Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Create `.env` from `.env.example` and add real secrets.
3. Update `config.yaml` as needed.

## Run
- Standard local run (installs/checks/builds/starts):
  ```bash
  npm run bot
  ```
- Dev mode:
  ```bash
  npm run bot:dev
  ```
- PowerShell wrapper:
  ```powershell
  .\run-bot.ps1
  ```
- Bash wrapper:
  ```bash
  ./run-bot.sh
  ```

## Commands
- `/join`
- `/leave`
- `/status`
- `/ask <prompt>`
- `/queue`
- `/stop`
- `/pause`
- `/continue`

## Utility Scripts
- Register slash commands:
  ```bash
  npm run register-commands
  ```
- Voice dependency report:
  ```bash
  npm run healthcheck
  ```

## Project Layout
```text
src-node/
  main.ts
  bot/
  ai/
  audio/
  config/
  utils/
  wakeword/
scripts/
  run-local.mjs
  register-commands.ts
  healthcheck.ts
```

## Notes
- Wake-word detection uses Picovoice Porcupine in the Node runtime.
- Set `PICOVOICE_ACCESS_KEY` in `.env` to enable wake-word detection.
- Supported built-in wake phrases: `hey_jarvis` (mapped to `jarvis`), `jarvis`, `alexa`, `hey_google`, `ok_google`, `hey_siri`, `computer`, `picovoice`, `porcupine`.
- You can also use an absolute or relative path to a custom `.ppn` keyword model file.
