import { getLogger } from "../utils/logger.js";
import { AudioProcessor } from "./processor.js";

const logger = getLogger("audio.capture");

const VAD_ENERGY_THRESHOLD = 400;
const FRAME_DURATION_MS = 20;
const MIN_SPEECH_DURATION_MS = 1000;
const WAKE_WORD_GRACE_MS = 500;

export class AudioCapture {
  private activeUserId: string | null = null;
  private streaming = false;
  private streamingQueue: Buffer[] = [];
  private dataResolver: ((chunk: Buffer | null) => void) | null = null;

  private silenceFrames = 0;
  private silenceThresholdFrames: number;
  private speechDetected = false;
  private streamStartedAt = 0;
  private speechStartedAt = 0;

  public constructor(
    private readonly processor: AudioProcessor,
    private silenceThresholdSeconds: number,
  ) {
    this.silenceThresholdFrames = Math.max(1, Math.floor(silenceThresholdSeconds / (FRAME_DURATION_MS / 1000)));
  }

  public setSilenceThreshold(seconds: number): void {
    this.silenceThresholdSeconds = seconds;
    this.silenceThresholdFrames = Math.max(1, Math.floor(seconds / (FRAME_DURATION_MS / 1000)));
  }

  public setActiveUser(userId: string | null): void {
    this.activeUserId = userId;
  }

  public resetVadState(): void {
    this.silenceFrames = 0;
    this.speechDetected = false;
    this.streamStartedAt = 0;
    this.speechStartedAt = 0;
  }

  public startStreaming(): void {
    this.streaming = true;
    this.streamingQueue.length = 0;
    this.resetVadState();
    this.streamStartedAt = Date.now();
  }

  public stopStreaming(): void {
    this.streaming = false;
    this.streamingQueue.length = 0;
    this.resolveWaitingChunk(null);
    this.resetVadState();
  }

  public processUserPcm(pcm48kStereo: Buffer, userId: string): Buffer {
    const geminiPcm = this.processor.discordToGemini(pcm48kStereo);
    if (!this.streaming || this.activeUserId === null || this.activeUserId !== userId) {
      return geminiPcm;
    }

    const now = Date.now();
    const inGrace = now - this.streamStartedAt < WAKE_WORD_GRACE_MS;

    if (!inGrace) {
      this.updateVad(geminiPcm, now);
    }

    if (this.dataResolver) {
      const resolve = this.dataResolver;
      this.dataResolver = null;
      resolve(geminiPcm);
    } else {
      this.streamingQueue.push(geminiPcm);
      if (this.streamingQueue.length > 250) {
        this.streamingQueue.shift();
      }
    }

    return geminiPcm;
  }

  public async getStreamingChunk(timeoutMs = 100): Promise<Buffer | null> {
    if (this.streamingQueue.length > 0) {
      return this.streamingQueue.shift() ?? null;
    }

    return new Promise<Buffer | null>((resolve) => {
      const timeout = setTimeout(() => {
        if (this.dataResolver === resolve) {
          this.dataResolver = null;
        }
        resolve(null);
      }, timeoutMs);

      this.dataResolver = (chunk) => {
        clearTimeout(timeout);
        resolve(chunk);
      };
    });
  }

  public isSilenceDetected(): boolean {
    if (!this.speechDetected) {
      return false;
    }
    return this.silenceFrames >= this.silenceThresholdFrames;
  }

  private updateVad(geminiPcm: Buffer, now: number): void {
    if (geminiPcm.length < 2) {
      return;
    }

    const samples = new Int16Array(geminiPcm.buffer, geminiPcm.byteOffset, Math.floor(geminiPcm.length / 2));
    let absSum = 0;
    for (let i = 0; i < samples.length; i += 1) {
      absSum += Math.abs(samples[i] ?? 0);
    }
    const meanAbs = absSum / Math.max(1, samples.length);

    if (meanAbs > VAD_ENERGY_THRESHOLD) {
      if (!this.speechDetected) {
        this.speechDetected = true;
        this.speechStartedAt = now;
      }
      this.silenceFrames = 0;
      return;
    }

    if (!this.speechDetected) {
      return;
    }

    const speechMs = now - this.speechStartedAt;
    if (speechMs < MIN_SPEECH_DURATION_MS) {
      return;
    }

    this.silenceFrames += 1;
    if (this.silenceFrames === this.silenceThresholdFrames) {
      logger.debug({ silenceFrames: this.silenceFrames }, "VAD silence threshold reached");
    }
  }

  private resolveWaitingChunk(chunk: Buffer | null): void {
    if (!this.dataResolver) {
      return;
    }
    const resolve = this.dataResolver;
    this.dataResolver = null;
    resolve(chunk);
  }
}
