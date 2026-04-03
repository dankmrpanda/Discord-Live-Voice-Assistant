# Discord Live VC Bot (Node.js)

A Discord voice assistant built with Node.js, `discord.js`, `@discordjs/voice`, and Gemini Live API.

## Requirements
- Node.js 22+
- FFmpeg in PATH (recommended for voice playback stability)
- Discord bot token
- Gemini API key

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
- Wake-word detection is currently disabled in the Node runtime.
- Use `/ask` for interaction.
