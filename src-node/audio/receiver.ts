import { EndBehaviorType, VoiceConnection } from "@discordjs/voice";
import prism from "prism-media";

import { getLogger, logException } from "../utils/logger.js";

const logger = getLogger("audio.receiver");

type UserPcmHandler = (userId: string, pcm48kStereo: Buffer) => Promise<void> | void;

export class DiscordAudioReceiver {
  private connection: VoiceConnection | null = null;
  private readonly activeStreams = new Map<string, { opus: NodeJS.ReadableStream; decoder: NodeJS.ReadWriteStream }>();

  public constructor(private readonly onPcm: UserPcmHandler) {}

  public attach(connection: VoiceConnection): void {
    this.connection = connection;
    const receiver = connection.receiver;

    receiver.speaking.on("start", (userId) => {
      this.subscribeUser(userId);
    });

    receiver.speaking.on("end", (userId) => {
      this.cleanupUser(userId);
    });

    logger.info({ activeStreams: this.activeStreams.size }, "Audio receiver attached");
  }

  public stop(): void {
    logger.info({ activeStreams: this.activeStreams.size }, "Stopping audio receiver");
    for (const userId of this.activeStreams.keys()) {
      this.cleanupUser(userId);
    }
    this.activeStreams.clear();
    this.connection = null;
  }

  private subscribeUser(userId: string): void {
    if (!this.connection || this.activeStreams.has(userId)) {
      logger.debug(
        { userId, hasConnection: Boolean(this.connection), alreadySubscribed: this.activeStreams.has(userId) },
        "Skipping user audio subscribe",
      );
      return;
    }

    try {
      const receiver = this.connection.receiver;
      const opusStream = receiver.subscribe(userId, {
        end: {
          behavior: EndBehaviorType.AfterSilence,
          duration: 200,
        },
      });

      const decoder = new prism.opus.Decoder({
        frameSize: 960,
        channels: 2,
        rate: 48_000,
      });

      opusStream.pipe(decoder);

      decoder.on("data", (chunk: Buffer) => {
        void this.onPcm(userId, chunk);
      });

      const onClose = () => this.cleanupUser(userId);
      opusStream.once("close", onClose);
      opusStream.once("end", onClose);
      opusStream.once("error", (error) => {
        logException(logger, `Opus stream error for user ${userId}`, error);
        this.cleanupUser(userId);
      });
      decoder.once("error", (error) => {
        logException(logger, `Decoder error for user ${userId}`, error);
        this.cleanupUser(userId);
      });

      this.activeStreams.set(userId, {
        opus: opusStream,
        decoder,
      });
      logger.debug({ userId, activeStreams: this.activeStreams.size }, "Subscribed user audio stream");
    } catch (error) {
      logException(logger, `Failed to subscribe user ${userId}`, error);
    }
  }

  private cleanupUser(userId: string): void {
    const streams = this.activeStreams.get(userId);
    if (!streams) {
      return;
    }

    (streams.opus as { destroy?: () => void }).destroy?.();
    (streams.decoder as { destroy?: () => void }).destroy?.();
    this.activeStreams.delete(userId);
    logger.debug({ userId, activeStreams: this.activeStreams.size }, "Cleaned up user audio stream");
  }
}

