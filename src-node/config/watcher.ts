import chokidar, { type FSWatcher } from "chokidar";

import type { ConfigManager } from "./config.js";
import { getLogger, logException } from "../utils/logger.js";

const logger = getLogger("config.watcher");

export class ConfigWatcher {
  private watcher: FSWatcher | null = null;

  public constructor(private readonly manager: ConfigManager) {}

  public async start(): Promise<void> {
    if (this.watcher) {
      return;
    }

    this.watcher = chokidar.watch(this.manager.getConfigPath(), {
      ignoreInitial: true,
      awaitWriteFinish: {
        stabilityThreshold: 300,
        pollInterval: 100,
      },
    });

    this.watcher.on("change", () => {
      try {
        const changed = this.manager.reload();
        if (changed.length > 0) {
          logger.info({ changed }, "Configuration reloaded");
        }
      } catch (error) {
        logException(logger, "Failed to reload config", error);
      }
    });

    logger.info({ file: this.manager.getConfigPath() }, "Config watcher started");
  }

  public async stop(): Promise<void> {
    if (!this.watcher) {
      return;
    }
    await this.watcher.close();
    this.watcher = null;
    logger.info("Config watcher stopped");
  }
}
