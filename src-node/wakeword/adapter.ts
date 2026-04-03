import { getLogger } from "../utils/logger.js";

const logger = getLogger("wakeword.adapter");

export type DetectionCallback = (userId: string) => Promise<void> | void;

export interface WakeWordAdapter {
  isAvailable(): boolean;
  enable(): void;
  disable(): void;
  reset(): void;
  close(): Promise<void>;
  setDetectionCallback(callback: DetectionCallback): void;
  processAudioForUser(userId: string, pcm16kMono: Buffer): Promise<boolean>;
}

class DisabledWakeWordAdapter implements WakeWordAdapter {
  private callback?: DetectionCallback;

  public isAvailable(): boolean {
    return false;
  }

  public enable(): void {}
  public disable(): void {}
  public reset(): void {}
  public async close(): Promise<void> {}
  public setDetectionCallback(callback: DetectionCallback): void {
    this.callback = callback;
  }

  public async processAudioForUser(): Promise<boolean> {
    return false;
  }
}

export function createWakeWordAdapter(): WakeWordAdapter {
  logger.info("Wake-word detection disabled; use /ask for interaction");
  return new DisabledWakeWordAdapter();
}
