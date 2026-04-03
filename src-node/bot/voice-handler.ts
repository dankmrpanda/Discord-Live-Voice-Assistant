import {
  DiscordAPIError,
  type Guild,
  type GuildMember,
  type VoiceBasedChannel,
} from "discord.js";
import type { Logger } from "pino";
import {
  VoiceConnection,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
} from "@discordjs/voice";

import type { AppConfig, AskQueueItem } from "../types/index.js";
import { BotState } from "../types/index.js";
import { getLogger, logException } from "../utils/logger.js";
import { AudioProcessor } from "../audio/processor.js";
import { AudioPlayback } from "../audio/playback.js";
import { AudioCapture } from "../audio/capture.js";
import { DiscordAudioReceiver } from "../audio/receiver.js";
import { GeminiLiveClient } from "../ai/gemini-live-client.js";
import { createWakeWordAdapter, type WakeWordAdapter } from "../wakeword/adapter.js";

const logger = getLogger("bot.voice_handler");

export class VoiceHandler {
  private config: AppConfig;
  private readonly processor: AudioProcessor;
  private readonly playback: AudioPlayback;
  private readonly capture: AudioCapture;
  private readonly receiver: DiscordAudioReceiver;
  private readonly gemini: GeminiLiveClient;
  private readonly wakeWord: WakeWordAdapter;

  private state: BotState = BotState.IDLE;
  private connection: VoiceConnection | null = null;
  private channel: VoiceBasedChannel | null = null;

  private askQueue: AskQueueItem[] = [];
  private currentAbort: AbortController | null = null;
  private currentPipeline: Promise<void> | null = null;
  private manualStopInProgress = false;
  private activeUserId: string | null = null;
  private pipelineCounter = 0;

  public constructor(config: AppConfig) {
    this.config = config;
    this.processor = new AudioProcessor(
      config.discordSampleRate,
      config.geminiInputSampleRate,
      config.geminiOutputSampleRate,
    );
    this.playback = new AudioPlayback(this.processor, config.playbackBufferMs);
    this.capture = new AudioCapture(this.processor, config.silenceThreshold);
    this.receiver = new DiscordAudioReceiver(async (userId, pcm48kStereo) => {
      await this.onPcmChunk(userId, pcm48kStereo);
    });
    this.gemini = new GeminiLiveClient(config);
    this.wakeWord = createWakeWordAdapter();
    this.wakeWord.setDetectionCallback(async (userId) => {
      await this.onWakeWordDetected(userId);
    });
  }

  public updateConfig(config: AppConfig): void {
    this.config = config;
    this.capture.setSilenceThreshold(config.silenceThreshold);
    this.playback.setBufferMs(config.playbackBufferMs);
    logger.info(
      {
        silenceThreshold: config.silenceThreshold,
        playbackBufferMs: config.playbackBufferMs,
        captureDurationSec: config.captureDuration,
      },
      "Voice handler config updated",
    );
    void this.gemini.reconnectWithConfig(config);
  }

  public getState(): BotState {
    return this.state;
  }

  public getQueueItems(): AskQueueItem[] {
    return [...this.askQueue];
  }

  public getQueueSize(): number {
    return this.askQueue.length;
  }

  public get isConnected(): boolean {
    return this.connection?.state.status === VoiceConnectionStatus.Ready;
  }

  public get isResponsePaused(): boolean {
    return this.state === BotState.SPEAKING && this.playback.isPaused;
  }

  public get isWakeWordAvailable(): boolean {
    return this.wakeWord.isAvailable();
  }

  public async join(channel: VoiceBasedChannel): Promise<{ success: boolean; message: string }> {
    if (this.state !== BotState.IDLE) {
      logger.warn({ state: this.state, channelId: channel.id, guildId: channel.guild.id }, "Join rejected");
      return { success: false, message: "Already connected or connecting in this server." };
    }

    logger.info(
      {
        channelId: channel.id,
        channelName: channel.name,
        guildId: channel.guild.id,
      },
      "Joining voice channel",
    );
    this.setState(BotState.CONNECTING);
    this.channel = channel;

    try {
      this.connection = joinVoiceChannel({
        channelId: channel.id,
        guildId: channel.guild.id,
        adapterCreator: channel.guild.voiceAdapterCreator,
        selfDeaf: false,
        selfMute: false,
      });

      await entersState(this.connection, VoiceConnectionStatus.Ready, 25_000);

      this.playback.attachConnection(this.connection);
      this.receiver.attach(this.connection);

      const connected = await this.gemini.connect();
      if (!connected) {
        throw new Error("Failed to connect to Gemini Live API");
      }

      this.gemini.startHealthCheck();
      this.wakeWord.enable();
      this.setState(BotState.LISTENING);

      logger.info({ channelId: channel.id, guildId: channel.guild.id }, "Joined voice channel");
      const wakeWordMessage = this.isWakeWordAvailable
        ? `Say '${this.config.wakePhrase.replaceAll("_", " ")}' or use /ask.`
        : "Wake-word detection is disabled in this build; use /ask.";
      return {
        success: true,
        message: `Joined **${channel.name}**. ${wakeWordMessage}`,
      };
    } catch (error) {
      logException(logger, "Failed to join voice channel", error);
      await this.leave();
      return { success: false, message: "Failed to join voice channel." };
    }
  }

  public async leave(): Promise<{ success: boolean; message: string }> {
    logger.info(
      {
        channelId: this.channel?.id ?? null,
        guildId: this.channel?.guild.id ?? null,
        queueSize: this.askQueue.length,
      },
      "Leaving voice channel",
    );
    this.askQueue = [];

    this.manualStopInProgress = true;
    await this.stopCurrentPipeline();
    this.manualStopInProgress = false;

    this.receiver.stop();
    this.capture.stopStreaming();
    this.wakeWord.disable();
    await this.wakeWord.close();

    await this.gemini.disconnect();

    if (this.connection) {
      this.connection.destroy();
      this.connection = null;
    }

    this.channel = null;
    this.activeUserId = null;
    this.setState(BotState.IDLE);

    logger.info("Voice handler cleanup complete");
    return { success: true, message: "Left the voice channel." };
  }

  public async processTextPrompt(prompt: string, userId: string): Promise<boolean> {
    if (this.state !== BotState.LISTENING) {
      logger.debug({ state: this.state, userId }, "Text prompt rejected due to state");
      return false;
    }

    logger.info({ userId, promptLength: prompt.length }, "Accepted immediate text prompt");
    this.startTextPipeline(prompt, userId);
    return true;
  }

  public queueTextPrompt(prompt: string, userId: string): number {
    this.askQueue.push({
      prompt,
      userId,
      createdAt: Date.now(),
    });
    logger.info(
      { userId, queueSize: this.askQueue.length, promptLength: prompt.length },
      "Queued text prompt",
    );
    return this.askQueue.length;
  }

  public async stopResponse(): Promise<boolean> {
    if (this.state !== BotState.PROCESSING && this.state !== BotState.SPEAKING) {
      logger.debug({ state: this.state }, "Stop ignored because bot is not active");
      return false;
    }

    logger.info({ state: this.state, queueSize: this.askQueue.length }, "Stopping active response");
    this.manualStopInProgress = true;
    await this.stopCurrentPipeline();
    this.manualStopInProgress = false;
    this.playback.stop();
    await this.resetToListening();
    return true;
  }

  public pauseResponse(): boolean {
    if (this.state !== BotState.SPEAKING) {
      logger.debug({ state: this.state }, "Pause ignored because bot is not speaking");
      return false;
    }
    const paused = this.playback.pause();
    logger.info({ paused }, "Pause response request processed");
    return paused;
  }

  public resumeResponse(): boolean {
    if (this.state !== BotState.SPEAKING) {
      logger.debug({ state: this.state }, "Resume ignored because bot is not speaking");
      return false;
    }
    const resumed = this.playback.resume();
    logger.info({ resumed }, "Resume response request processed");
    return resumed;
  }

  public cleanupUser(member: GuildMember): void {
    if (this.activeUserId === member.id) {
      this.activeUserId = null;
      this.capture.setActiveUser(null);
    }
  }

  private async onPcmChunk(userId: string, pcm48kStereo: Buffer): Promise<void> {
    const geminiPcm = this.capture.processUserPcm(pcm48kStereo, userId);

    if (this.state === BotState.LISTENING) {
      await this.wakeWord.processAudioForUser(userId, geminiPcm);
    }
  }

  private async onWakeWordDetected(userId: string): Promise<void> {
    if (this.state !== BotState.LISTENING) {
      logger.debug({ userId, state: this.state }, "Wake word ignored due to current state");
      return;
    }

    logger.info({ userId }, "Wake word detected");
    this.startVoicePipeline(userId);
  }

  private startVoicePipeline(userId: string): void {
    if (this.currentPipeline) {
      logger.debug({ userId }, "Voice pipeline start skipped because another pipeline is active");
      return;
    }
    const pipeline = this.runVoicePipeline(userId);
    this.currentPipeline = pipeline;
    void pipeline;
  }

  private startTextPipeline(prompt: string, userId: string): void {
    if (this.currentPipeline) {
      logger.debug({ userId }, "Text pipeline start skipped because another pipeline is active");
      return;
    }
    this.wakeWord.disable();
    this.setState(BotState.PROCESSING);
    const pipeline = this.runTextPipeline(prompt, userId);
    this.currentPipeline = pipeline;
    void pipeline;
  }

  private async runVoicePipeline(userId: string): Promise<void> {
    if (this.state !== BotState.LISTENING) {
      return;
    }

    const pipelineId = this.nextPipelineId("voice");
    const pipelineLogger = this.getPipelineLogger("voice", pipelineId, userId);
    this.currentAbort = new AbortController();
    const signal = this.currentAbort.signal;

    this.activeUserId = userId;
    this.capture.setActiveUser(userId);

    this.wakeWord.disable();
    this.capture.startStreaming();
    this.setState(BotState.PROCESSING);

    let sentAudioBytes = 0;
    let sentAudioChunks = 0;
    let responseBytes = 0;
    let responseChunks = 0;
    let exitReason = "capture_timeout";
    let lastChunkAt = Date.now();
    const startedAt = Date.now();
    pipelineLogger.info(
      {
        captureDurationSec: this.config.captureDuration,
        silenceThresholdSec: this.config.silenceThreshold,
      },
      "Voice pipeline started",
    );

    try {
      if (!this.gemini.isConnected) {
        pipelineLogger.info("Gemini session not connected; reconnecting before voice turn");
        const ok = await this.gemini.connect();
        if (!ok) {
          throw new Error("Gemini reconnect failed");
        }
      }

      while (!signal.aborted) {
        const elapsed = Date.now() - startedAt;
        if (elapsed >= this.config.captureDuration * 1000) {
          exitReason = "capture_timeout";
          break;
        }

        if (this.capture.isSilenceDetected()) {
          exitReason = "silence_detected";
          break;
        }

        if (sentAudioBytes > 0 && Date.now() - lastChunkAt >= 300) {
          exitReason = "post_speech_quiet_gap";
          break;
        }

        const chunk = await this.capture.getStreamingChunk(25);
        if (!chunk || chunk.length === 0) {
          continue;
        }

        lastChunkAt = Date.now();
        sentAudioBytes += chunk.length;
        sentAudioChunks += 1;
        await this.gemini.sendAudio(chunk);
      }

      if (sentAudioBytes > 0) {
        pipelineLogger.info(
          { sentAudioBytes, sentAudioChunks, captureMs: Date.now() - startedAt, exitReason },
          "Finished voice capture; ending Gemini turn",
        );
        await this.gemini.endTurn();
      } else {
        exitReason = signal.aborted ? "aborted" : "no_audio";
        pipelineLogger.info({ captureMs: Date.now() - startedAt, exitReason }, "Voice capture produced no audio");
      }

      if (signal.aborted) {
        exitReason = "aborted";
        pipelineLogger.info("Voice pipeline aborted before playback");
        return;
      }

      this.setState(BotState.SPEAKING);
      if (!this.playback.startStreamingSession()) {
        throw new Error("Failed to start playback");
      }
      pipelineLogger.debug("Streaming playback session started");

      for await (const responseChunk of this.gemini.receiveResponses()) {
        if (signal.aborted) {
          exitReason = "aborted";
          break;
        }
        responseChunks += 1;
        responseBytes += responseChunk.length;
        this.playback.addGeminiChunk(responseChunk);
      }

      this.playback.finishStreaming();
      const playbackIdle = await this.playback.waitForIdle(60_000);
      pipelineLogger.info(
        {
          sentAudioBytes,
          sentAudioChunks,
          responseBytes,
          responseChunks,
          playbackIdle,
          durationMs: Date.now() - startedAt,
          exitReason,
        },
        "Voice pipeline completed",
      );
    } catch (error) {
      if (!signal.aborted) {
        logException(pipelineLogger, "Voice pipeline error", error);
      }
    } finally {
      this.capture.stopStreaming();
      this.playback.finishStreaming();
      this.currentAbort = null;
      this.currentPipeline = null;
      pipelineLogger.debug(
        {
          manualStopInProgress: this.manualStopInProgress,
          queueSize: this.askQueue.length,
          finalState: this.state,
        },
        "Voice pipeline cleanup finished",
      );
      if (!this.manualStopInProgress) {
        await this.resetToListening();
      }
    }
  }

  private async runTextPipeline(prompt: string, userId: string): Promise<void> {
    const pipelineId = this.nextPipelineId("text");
    const pipelineLogger = this.getPipelineLogger("text", pipelineId, userId);
    this.currentAbort = new AbortController();
    const signal = this.currentAbort.signal;

    this.activeUserId = userId;
    this.capture.setActiveUser(null);

    this.wakeWord.disable();
    this.setState(BotState.SPEAKING);
    const startedAt = Date.now();
    let responseBytes = 0;
    let responseChunks = 0;
    pipelineLogger.info(
      {
        promptLength: prompt.length,
        promptPreview: this.previewPrompt(prompt),
      },
      "Text pipeline started",
    );

    try {
      if (!this.gemini.isConnected) {
        pipelineLogger.info("Gemini session not connected; reconnecting before text turn");
        const ok = await this.gemini.connect();
        if (!ok) {
          throw new Error("Gemini reconnect failed");
        }
      }

      if (!this.playback.startStreamingSession()) {
        throw new Error("Failed to start playback");
      }

      await this.gemini.sendText(prompt);
      pipelineLogger.debug("Prompt sent to Gemini");

      for await (const responseChunk of this.gemini.receiveResponses()) {
        if (signal.aborted) {
          pipelineLogger.info("Text pipeline aborted during response streaming");
          break;
        }
        responseChunks += 1;
        responseBytes += responseChunk.length;
        this.playback.addGeminiChunk(responseChunk);
      }

      this.playback.finishStreaming();
      const playbackIdle = await this.playback.waitForIdle(60_000);
      pipelineLogger.info(
        {
          responseBytes,
          responseChunks,
          playbackIdle,
          durationMs: Date.now() - startedAt,
        },
        "Text pipeline completed",
      );
    } catch (error) {
      if (!signal.aborted) {
        logException(pipelineLogger, "Text prompt pipeline error", error);
      }
    } finally {
      this.playback.finishStreaming();
      this.currentAbort = null;
      this.currentPipeline = null;
      pipelineLogger.debug(
        {
          manualStopInProgress: this.manualStopInProgress,
          queueSize: this.askQueue.length,
          finalState: this.state,
        },
        "Text pipeline cleanup finished",
      );
      if (!this.manualStopInProgress) {
        await this.resetToListening();
      }
    }
  }

  private async resetToListening(): Promise<void> {
    if (this.state === BotState.IDLE) {
      return;
    }

    this.activeUserId = null;
    this.capture.setActiveUser(null);
    this.capture.resetVadState();

    if (this.askQueue.length > 0) {
      const next = this.askQueue.shift();
      if (next) {
        logger.info(
          {
            userId: next.userId,
            queueRemaining: this.askQueue.length,
            promptLength: next.prompt.length,
          },
          "Dequeued next text prompt",
        );
        this.startTextPipeline(next.prompt, next.userId);
        return;
      }
    }

    this.wakeWord.reset();
    this.wakeWord.enable();
    this.setState(BotState.LISTENING);
    logger.info({ queueSize: this.askQueue.length }, "Reset to listening state");
  }

  private async stopCurrentPipeline(): Promise<void> {
    if (this.currentAbort) {
      logger.info("Aborting active pipeline");
      this.currentAbort.abort();
      this.currentAbort = null;
    }

    if (this.currentPipeline) {
      try {
        logger.debug("Waiting for active pipeline shutdown");
        await this.currentPipeline;
      } catch {
        // Errors are already logged inside pipeline handlers.
      }
    }
  }

  private setState(state: BotState): void {
    if (this.state !== state) {
      logger.info(
        {
          from: this.state,
          to: state,
          activeUserId: this.activeUserId,
          queueSize: this.askQueue.length,
        },
        "State transition",
      );
    }
    this.state = state;
  }

  private nextPipelineId(kind: "voice" | "text"): string {
    this.pipelineCounter += 1;
    return `${kind}-${Date.now()}-${this.pipelineCounter}`;
  }

  private getPipelineLogger(kind: "voice" | "text", pipelineId: string, userId: string): Logger {
    return logger.child({
      pipelineType: kind,
      pipelineId,
      userId,
      channelId: this.channel?.id ?? null,
      guildId: this.channel?.guild.id ?? null,
    });
  }

  private previewPrompt(prompt: string): string {
    const compact = prompt.replace(/\s+/g, " ").trim();
    if (compact.length <= 160) {
      return compact;
    }
    return `${compact.slice(0, 157)}...`;
  }
}

export function isBotUserDisconnect(beforeGuild: Guild | null, afterGuild: Guild | null): boolean {
  return Boolean(beforeGuild && !afterGuild);
}

export function isDiscordApiError(error: unknown): error is DiscordAPIError {
  return error instanceof DiscordAPIError;
}


