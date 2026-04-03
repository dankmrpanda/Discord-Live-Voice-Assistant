import process from "node:process";

import { ConfigManager } from "./config/config.js";
import { ConfigWatcher } from "./config/watcher.js";
import { createLogger, getLogger, logException } from "./utils/logger.js";
import { DiscordVoiceBot } from "./bot/client.js";

async function main(): Promise<number> {
  const configManager = new ConfigManager();
  const config = configManager.config;

  createLogger(config.logLevel, config.logDirectory);
  const logger = getLogger("main");

  logger.info({ model: config.geminiModel, voice: config.geminiVoice }, "Starting Discord voice bot");

  const watcher = new ConfigWatcher(configManager);
  const bot = new DiscordVoiceBot(configManager);

  const shutdown = async (signal: string) => {
    logger.info({ signal }, "Shutting down");
    await watcher.stop();
    await bot.stop();
    process.exit(0);
  };

  process.on("SIGINT", () => {
    void shutdown("SIGINT");
  });
  process.on("SIGTERM", () => {
    void shutdown("SIGTERM");
  });

  try {
    await bot.start();
    await watcher.start();
    return 0;
  } catch (error) {
    logException(logger, "Fatal startup error", error);
    await watcher.stop();
    await bot.stop();
    return 1;
  }
}

main()
  .then((code) => {
    if (code !== 0) {
      process.exitCode = code;
    }
  })
  .catch((error) => {
    const logger = getLogger("main");
    logException(logger, "Unhandled fatal error", error);
    process.exitCode = 1;
  });
