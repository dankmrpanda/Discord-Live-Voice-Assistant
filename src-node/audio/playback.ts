import { PassThrough } from "node:stream";
import {
  AudioPlayer,
  AudioPlayerStatus,
  StreamType,
  VoiceConnection,
  createAudioPlayer,
  createAudioResource,
} from "@discordjs/voice";

import { getLogger } from "../utils/logger.js";
import { AudioProcessor } from "./processor.js";

const logger = getLogger("audio.playback");

const DISCORD_PCM_BYTES_PER_MS = (48_000 * 2 * 2) / 1000;

export class AudioPlayback {
  private connection: VoiceConnection | null = null;
  private player: AudioPlayer;
  private stream: PassThrough | null = null;
  private buffering = true;
  private pendingBuffer = Buffer.alloc(0);

  public constructor(
    private readonly processor: AudioProcessor,
    private bufferMs: number,
  ) {
    this.player = createAudioPlayer();
  }

  public setBufferMs(ms: number): void {
    this.bufferMs = Math.max(0, Math.floor(ms));
  }

  public attachConnection(connection: VoiceConnection): void {
    this.connection = connection;
    connection.subscribe(this.player);
  }

  public startStreamingSession(): boolean {
    if (!this.connection) {
      logger.warn("Cannot start playback without voice connection");
      return false;
    }

    this.stream = new PassThrough();
    const resource = createAudioResource(this.stream, {
      inputType: StreamType.Raw,
      inlineVolume: false,
    });

    this.buffering = this.bufferMs > 0;
    this.pendingBuffer = Buffer.alloc(0);
    this.player.play(resource);
    logger.debug({ bufferMs: this.bufferMs }, "Streaming playback started");
    return true;
  }

  public addGeminiChunk(geminiChunk: Buffer): void {
    if (!this.stream) {
      return;
    }

    const discordPcm = this.processor.geminiToDiscord(geminiChunk);

    if (!this.buffering) {
      this.stream.write(discordPcm);
      return;
    }

    this.pendingBuffer = Buffer.concat([this.pendingBuffer, discordPcm]);
    const threshold = Math.floor(this.bufferMs * DISCORD_PCM_BYTES_PER_MS);
    if (this.pendingBuffer.length >= threshold) {
      this.stream.write(this.pendingBuffer);
      this.pendingBuffer = Buffer.alloc(0);
      this.buffering = false;
      logger.debug({ thresholdBytes: threshold }, "Playback buffer threshold reached");
    }
  }

  public finishStreaming(): void {
    if (!this.stream) {
      return;
    }

    if (this.pendingBuffer.length > 0) {
      this.stream.write(this.pendingBuffer);
      this.pendingBuffer = Buffer.alloc(0);
    }

    this.stream.end();
    this.stream = null;
  }

  public async waitForIdle(timeoutMs = 60_000): Promise<boolean> {
    if (this.player.state.status === AudioPlayerStatus.Idle) {
      return true;
    }

    return new Promise<boolean>((resolve) => {
      const onIdle = () => {
        cleanup();
        resolve(true);
      };
      const timeout = setTimeout(() => {
        cleanup();
        resolve(false);
      }, timeoutMs);

      const cleanup = () => {
        clearTimeout(timeout);
        this.player.off(AudioPlayerStatus.Idle, onIdle);
      };

      this.player.on(AudioPlayerStatus.Idle, onIdle);
    });
  }

  public stop(): void {
    this.stream?.destroy();
    this.stream = null;
    this.player.stop(true);
    this.pendingBuffer = Buffer.alloc(0);
    this.buffering = true;
  }

  public pause(): boolean {
    return this.player.pause(true);
  }

  public resume(): boolean {
    return this.player.unpause();
  }

  public get isPaused(): boolean {
    return this.player.state.status === AudioPlayerStatus.Paused;
  }

  public get isPlaying(): boolean {
    return this.player.state.status === AudioPlayerStatus.Playing;
  }
}
