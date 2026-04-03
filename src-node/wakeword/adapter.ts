import fs from "node:fs";
import path from "node:path";
import {
  BuiltinKeyword,
  Porcupine,
  getBuiltinKeywordPath,
} from "@picovoice/porcupine-node";

import { getLogger, logException } from "../utils/logger.js";

const logger = getLogger("wakeword.adapter");

const DETECTION_COOLDOWN_MS = 1_500;

const BUILTIN_KEYWORDS: Record<string, BuiltinKeyword> = {
  alexa: BuiltinKeyword.ALEXA,
  computer: BuiltinKeyword.COMPUTER,
  hey_google: BuiltinKeyword.HEY_GOOGLE,
  hey_siri: BuiltinKeyword.HEY_SIRI,
  jarvis: BuiltinKeyword.JARVIS,
  hey_jarvis: BuiltinKeyword.JARVIS,
  ok_google: BuiltinKeyword.OK_GOOGLE,
  picovoice: BuiltinKeyword.PICOVOICE,
  porcupine: BuiltinKeyword.PORCUPINE,
};

export type DetectionCallback = (userId: string) => Promise<void> | void;

export interface WakeWordAdapterConfig {
  wakePhrase: string;
  wakeWordThreshold: number;
  picovoiceAccessKey?: string;
}

export interface WakeWordAdapter {
  isAvailable(): boolean;
  updateConfig(config: WakeWordAdapterConfig): void;
  enable(): void;
  disable(): void;
  reset(): void;
  cleanupUser(userId: string): void;
  close(): Promise<void>;
  setDetectionCallback(callback: DetectionCallback): void;
  processAudioForUser(userId: string, pcm16kMono: Buffer): Promise<boolean>;
}

class DisabledWakeWordAdapter implements WakeWordAdapter {
  private callback?: DetectionCallback;

  public isAvailable(): boolean {
    return false;
  }

  public updateConfig(): void {}
  public enable(): void {}
  public disable(): void {}
  public reset(): void {}
  public cleanupUser(): void {}
  public async close(): Promise<void> {}

  public setDetectionCallback(callback: DetectionCallback): void {
    this.callback = callback;
  }

  public async processAudioForUser(): Promise<boolean> {
    return false;
  }
}

interface UserDetector {
  engine: Porcupine;
  pending: Int16Array;
  lastDetectedAt: number;
  framesProcessed: number;
}

class PorcupineWakeWordAdapter implements WakeWordAdapter {
  private callback?: DetectionCallback;
  private enabled = true;
  private userDetectors = new Map<string, UserDetector>();
  private keywordPath: string;
  private keywordLabel: string;
  private sensitivity: number;

  public constructor(private config: WakeWordAdapterConfig) {
    if (!config.picovoiceAccessKey) {
      throw new Error("Missing PICOVOICE_ACCESS_KEY");
    }

    const resolvedKeyword = resolveKeyword(config.wakePhrase);
    this.keywordPath = resolvedKeyword.path;
    this.keywordLabel = resolvedKeyword.label;
    this.sensitivity = thresholdToSensitivity(config.wakeWordThreshold);
    logger.info(
      {
        wakePhrase: config.wakePhrase,
        keywordLabel: this.keywordLabel,
        sensitivity: this.sensitivity,
      },
      "Wake-word detection enabled (Porcupine)",
    );
  }

  public isAvailable(): boolean {
    return true;
  }

  public updateConfig(config: WakeWordAdapterConfig): void {
    const nextKeyword = resolveKeyword(config.wakePhrase);
    const nextSensitivity = thresholdToSensitivity(config.wakeWordThreshold);
    const changed =
      nextKeyword.path !== this.keywordPath ||
      nextSensitivity !== this.sensitivity ||
      config.picovoiceAccessKey !== this.config.picovoiceAccessKey;

    this.config = config;
    this.keywordPath = nextKeyword.path;
    this.keywordLabel = nextKeyword.label;
    this.sensitivity = nextSensitivity;

    if (!changed) {
      return;
    }

    logger.info(
      {
        wakePhrase: config.wakePhrase,
        keywordLabel: this.keywordLabel,
        sensitivity: this.sensitivity,
      },
      "Wake-word config changed; resetting user detectors",
    );
    this.releaseAllDetectors();
  }

  public enable(): void {
    this.enabled = true;
  }

  public disable(): void {
    this.enabled = false;
  }

  public reset(): void {
    for (const detector of this.userDetectors.values()) {
      detector.pending = new Int16Array(0);
      detector.framesProcessed = 0;
    }
  }

  public cleanupUser(userId: string): void {
    const detector = this.userDetectors.get(userId);
    if (!detector) {
      return;
    }
    detector.engine.release();
    this.userDetectors.delete(userId);
    logger.debug({ userId, activeDetectors: this.userDetectors.size }, "Cleaned up user wake-word detector");
  }

  public async close(): Promise<void> {
    this.releaseAllDetectors();
  }

  public setDetectionCallback(callback: DetectionCallback): void {
    this.callback = callback;
  }

  public async processAudioForUser(userId: string, pcm16kMono: Buffer): Promise<boolean> {
    if (!this.enabled || pcm16kMono.length < 2) {
      return false;
    }

    const detector = this.getOrCreateUserDetector(userId);
    if (!detector) {
      return false;
    }

    const incoming = bufferToInt16(pcm16kMono);
    if (incoming.length === 0) {
      return false;
    }

    detector.pending = concatInt16(detector.pending, incoming);
    const frameLength = detector.engine.frameLength;

    while (detector.pending.length >= frameLength) {
      const frame = detector.pending.subarray(0, frameLength);
      const keywordIndex = detector.engine.process(frame);
      detector.framesProcessed += 1;
      detector.pending = detector.pending.subarray(frameLength);

      if (keywordIndex < 0) {
        continue;
      }

      const now = Date.now();
      if (now - detector.lastDetectedAt < DETECTION_COOLDOWN_MS) {
        logger.debug({ userId, keywordIndex }, "Wake-word detection throttled by cooldown");
        continue;
      }

      detector.lastDetectedAt = now;
      detector.pending = new Int16Array(0);
      logger.info(
        {
          userId,
          keywordIndex,
          keywordLabel: this.keywordLabel,
          framesProcessed: detector.framesProcessed,
        },
        "Wake word detected",
      );

      if (this.callback) {
        await this.callback(userId);
      }
      return true;
    }

    detector.pending = Int16Array.from(detector.pending);
    return false;
  }

  private getOrCreateUserDetector(userId: string): UserDetector | null {
    const existing = this.userDetectors.get(userId);
    if (existing) {
      return existing;
    }

    if (!this.config.picovoiceAccessKey) {
      return null;
    }

    try {
      const engine = new Porcupine(
        this.config.picovoiceAccessKey,
        [this.keywordPath],
        [this.sensitivity],
      );
      const detector: UserDetector = {
        engine,
        pending: new Int16Array(0),
        lastDetectedAt: 0,
        framesProcessed: 0,
      };
      this.userDetectors.set(userId, detector);
      logger.debug(
        {
          userId,
          frameLength: engine.frameLength,
          sampleRate: engine.sampleRate,
          activeDetectors: this.userDetectors.size,
        },
        "Created wake-word detector for user",
      );
      return detector;
    } catch (error) {
      logException(logger, "Failed to initialize per-user wake-word detector", error, { userId });
      return null;
    }
  }

  private releaseAllDetectors(): void {
    for (const detector of this.userDetectors.values()) {
      detector.engine.release();
    }
    this.userDetectors.clear();
  }
}

export function createWakeWordAdapter(config: WakeWordAdapterConfig): WakeWordAdapter {
  try {
    return new PorcupineWakeWordAdapter(config);
  } catch (error) {
    logException(logger, "Wake-word detection disabled; falling back to /ask only", error, {
      wakePhrase: config.wakePhrase,
    });
    return new DisabledWakeWordAdapter();
  }
}

function thresholdToSensitivity(threshold: number): number {
  const clamped = clamp(threshold, 0, 1);
  // Preserve previous config semantics:
  // lower threshold => more sensitive detection.
  return clamp(1 - clamped, 0, 1);
}

function resolveKeyword(phrase: string): { path: string; label: string } {
  const normalized = phrase.trim().toLowerCase().replace(/\s+/g, "_");
  const builtin = BUILTIN_KEYWORDS[normalized];
  if (builtin) {
    return { path: getBuiltinKeywordPath(builtin), label: normalized };
  }

  const resolved = path.isAbsolute(phrase) ? phrase : path.resolve(phrase);
  if (fs.existsSync(resolved) && path.extname(resolved).toLowerCase() === ".ppn") {
    return { path: resolved, label: path.basename(resolved) };
  }

  if (fs.existsSync(resolved) && path.extname(resolved).toLowerCase() === ".onnx") {
    throw new Error(`Unsupported wake-word model format (.onnx): ${resolved}. Use a Porcupine .ppn keyword file.`);
  }

  throw new Error(
    `Unsupported wake phrase '${phrase}'. Use one of: ${Object.keys(BUILTIN_KEYWORDS).join(", ")} or a .ppn keyword path.`,
  );
}

function bufferToInt16(buffer: Buffer): Int16Array {
  const sampleCount = Math.floor(buffer.length / 2);
  if (sampleCount <= 0) {
    return new Int16Array(0);
  }

  const out = new Int16Array(sampleCount);
  for (let i = 0; i < sampleCount; i += 1) {
    out[i] = buffer.readInt16LE(i * 2);
  }
  return out;
}

function concatInt16(a: Int16Array, b: Int16Array): Int16Array {
  if (a.length === 0) {
    return b;
  }
  if (b.length === 0) {
    return a;
  }

  const out = new Int16Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

