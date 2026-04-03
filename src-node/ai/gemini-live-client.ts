import { GoogleGenAI, Modality, type LiveServerMessage } from "@google/genai";

import type { AppConfig } from "../types/index.js";
import { getLogger, logException } from "../utils/logger.js";
import { wait } from "../utils/errors.js";

const logger = getLogger("ai.gemini");

enum GeminiSessionState {
  DISCONNECTED = "disconnected",
  CONNECTING = "connecting",
  CONNECTED = "connected",
  STREAMING = "streaming",
  ERROR = "error",
}

const HEALTH_CHECK_INTERVAL_MS = 30_000;
const CONNECT_TIMEOUT_MS = 15_000;
const RECEIVE_TIMEOUT_MS = 20_000;

export class GeminiLiveClient {
  private readonly ai: GoogleGenAI;
  private session: any = null;
  private state: GeminiSessionState = GeminiSessionState.DISCONNECTED;
  private healthTimer: NodeJS.Timeout | null = null;
  private lastActivityAt = 0;
  private messageQueue: LiveServerMessage[] = [];
  private pendingMessageResolvers: Array<(message: LiveServerMessage | null) => void> = [];
  private turnCounter = 0;
  private activeTurnId: string | null = null;
  private activeTurnKind: "audio" | "text" | null = null;
  private sentAudioBytesForTurn = 0;
  private sentAudioChunksForTurn = 0;

  public constructor(private config: AppConfig) {
    this.ai = new GoogleGenAI({ apiKey: config.geminiApiKey });
  }

  public updateConfig(config: AppConfig): void {
    this.config = config;
  }

  public async reconnectWithConfig(config: AppConfig): Promise<void> {
    this.config = config;
    if (!this.isConnected) {
      return;
    }

    await this.disconnect();
    await this.connect();
  }

  public get isConnected(): boolean {
    return this.state === GeminiSessionState.CONNECTED || this.state === GeminiSessionState.STREAMING;
  }

  public async connect(): Promise<boolean> {
    if (this.isConnected || this.state === GeminiSessionState.CONNECTING) {
      logger.debug({ state: this.state }, "Gemini connect request ignored; already connected/connecting");
      return true;
    }

    this.state = GeminiSessionState.CONNECTING;
    this.clearMessageQueue();
    logger.info({ model: this.config.geminiModel, voice: this.config.geminiVoice }, "Connecting to Gemini Live");

    try {
      const connectPromise = this.ai.live.connect({
        model: this.config.geminiModel,
        callbacks: {
          onopen: () => logger.debug("Gemini Live socket opened"),
          onmessage: (message: LiveServerMessage) => {
            this.enqueueServerMessage(message);
          },
          onerror: (error: unknown) => logger.warn({ error }, "Gemini Live socket error"),
          onclose: () => {
            logger.debug("Gemini Live socket closed");
            this.clearMessageQueue();
            if (this.state !== GeminiSessionState.DISCONNECTED) {
              this.state = GeminiSessionState.ERROR;
            }
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: this.config.geminiVoice,
              },
            },
            languageCode: "en-US",
          },
          systemInstruction: this.config.systemPrompt,
          tools: this.buildTools(),
          thinkingConfig: this.config.geminiThinking
            ? {
                includeThoughts: true,
              }
            : undefined,
        },
      });

      this.session = await Promise.race([
        connectPromise,
        timeout(CONNECT_TIMEOUT_MS, "Gemini connect timeout"),
      ]);

      this.state = GeminiSessionState.CONNECTED;
      this.lastActivityAt = Date.now();
      logger.info({ model: this.config.geminiModel, voice: this.config.geminiVoice }, "Connected to Gemini Live");
      return true;
    } catch (error) {
      this.state = GeminiSessionState.ERROR;
      logException(logger, "Failed to connect to Gemini Live", error);
      return false;
    }
  }

  public async disconnect(): Promise<void> {
    this.stopHealthCheck();
    this.clearMessageQueue();
    this.finishActiveTurn("disconnect");

    if (!this.session) {
      this.state = GeminiSessionState.DISCONNECTED;
      logger.debug("Gemini disconnect requested without active session");
      return;
    }

    logger.info("Disconnecting Gemini session");
    try {
      if (typeof this.session.close === "function") {
        await this.session.close();
      }
    } catch (error) {
      logException(logger, "Failed to close Gemini session", error);
    }

    this.session = null;
    this.state = GeminiSessionState.DISCONNECTED;
    logger.info("Gemini session disconnected");
  }

  public startHealthCheck(): void {
    if (this.healthTimer) {
      return;
    }

    this.healthTimer = setInterval(() => {
      void this.performHealthCheck();
    }, HEALTH_CHECK_INTERVAL_MS);
  }

  public stopHealthCheck(): void {
    if (!this.healthTimer) {
      return;
    }
    clearInterval(this.healthTimer);
    this.healthTimer = null;
  }

  public async sendAudio(audio: Buffer): Promise<void> {
    if (!this.session || !this.isConnected || audio.length === 0) {
      return;
    }

    try {
      const turnId = this.beginTurn("audio");
      await this.session.sendRealtimeInput({
        media: {
          mimeType: "audio/pcm;rate=16000",
          data: audio,
        },
      });
      this.sentAudioBytesForTurn += audio.length;
      this.sentAudioChunksForTurn += 1;
      if (this.sentAudioChunksForTurn === 1 || this.sentAudioChunksForTurn % 20 === 0) {
        logger.debug(
          {
            turnId,
            chunkBytes: audio.length,
            sentAudioChunks: this.sentAudioChunksForTurn,
            sentAudioBytes: this.sentAudioBytesForTurn,
          },
          "Sent realtime audio chunk to Gemini",
        );
      }
      this.state = GeminiSessionState.STREAMING;
      this.lastActivityAt = Date.now();
    } catch (error) {
      logException(logger, "Failed to send audio to Gemini", error);
      this.state = GeminiSessionState.ERROR;
      this.finishActiveTurn("send_audio_error");
    }
  }

  public async sendText(text: string): Promise<void> {
    if (!this.session || !this.isConnected) {
      return;
    }

    try {
      const turnId = this.beginTurn("text");
      await this.session.sendClientContent({
        turns: {
          role: "user",
          parts: [{ text }],
        },
        turnComplete: true,
      });
      logger.info(
        {
          turnId,
          textLength: text.length,
          textPreview: previewText(text),
        },
        "Sent text prompt to Gemini",
      );
      this.lastActivityAt = Date.now();
    } catch (error) {
      logException(logger, "Failed to send text to Gemini", error);
      this.state = GeminiSessionState.ERROR;
      this.finishActiveTurn("send_text_error");
    }
  }

  public async endTurn(): Promise<void> {
    if (!this.session || !this.isConnected) {
      return;
    }

    try {
      await this.session.sendRealtimeInput({ audioStreamEnd: true });
      logger.info(
        {
          turnId: this.activeTurnId,
          sentAudioChunks: this.sentAudioChunksForTurn,
          sentAudioBytes: this.sentAudioBytesForTurn,
        },
        "Sent audio stream end to Gemini",
      );
      this.lastActivityAt = Date.now();
    } catch (error) {
      logException(logger, "Failed to end Gemini turn", error);
      this.state = GeminiSessionState.ERROR;
      this.finishActiveTurn("end_turn_error");
    }
  }

  public async *receiveResponses(): AsyncGenerator<Buffer, void, void> {
    if (!this.session) {
      return;
    }

    const startedAt = Date.now();
    let messageCount = 0;
    let responseChunks = 0;
    let responseBytes = 0;

    try {
      while (this.session && this.isConnected) {
        const message = await this.nextServerMessage(RECEIVE_TIMEOUT_MS);
        if (!message) {
          if (this.state === GeminiSessionState.ERROR || !this.session) {
            this.finishActiveTurn("receive_stopped_in_error");
            return;
          }
          logger.warn({ timeoutMs: RECEIVE_TIMEOUT_MS }, "Timed out waiting for Gemini response");
          this.state = GeminiSessionState.CONNECTED;
          this.finishActiveTurn("receive_timeout");
          return;
        }
        messageCount += 1;
        logger.debug(
          {
            turnId: this.activeTurnId,
            messageCount,
            messageKind: describeLiveMessage(message),
            queueDepth: this.messageQueue.length,
          },
          "Received Gemini live message",
        );

        if (message.goAway) {
          logger.warn({ goAway: message.goAway }, "Gemini server requested session shutdown");
          this.state = GeminiSessionState.ERROR;
          this.finishActiveTurn("server_go_away");
          return;
        }

        const serverContent = message?.serverContent;
        const modelTurn = serverContent?.modelTurn;
        const parts = Array.isArray(modelTurn?.parts) ? modelTurn.parts : [];

        for (const part of parts) {
          const data = part?.inlineData?.data;
          if (!data) {
            continue;
          }

          const buffer = toBuffer(data);
          if (buffer.length > 0) {
            this.lastActivityAt = Date.now();
            responseChunks += 1;
            responseBytes += buffer.length;
            yield buffer;
          }
        }

        if (serverContent?.turnComplete) {
          this.state = GeminiSessionState.CONNECTED;
          logger.info(
            {
              turnId: this.activeTurnId,
              turnKind: this.activeTurnKind,
              responseChunks,
              responseBytes,
              messageCount,
              durationMs: Date.now() - startedAt,
            },
            "Gemini turn completed",
          );
          this.finishActiveTurn("turn_complete");
          return;
        }
      }
    } catch (error) {
      logException(logger, "Error receiving Gemini response", error);
      this.state = GeminiSessionState.ERROR;
      this.finishActiveTurn("receive_error");
    }
  }

  private async performHealthCheck(): Promise<void> {
    if (!this.session) {
      return;
    }

    if (this.state === GeminiSessionState.ERROR) {
      logger.warn("Gemini session in error state; attempting reconnect");
      await this.reconnect();
      return;
    }

    const idleMs = Date.now() - this.lastActivityAt;
    if (idleMs > 5 * HEALTH_CHECK_INTERVAL_MS) {
      logger.debug({ idleMs }, "Gemini session idle");
    }
  }

  private async reconnect(): Promise<void> {
    await this.disconnect();
    await wait(500);
    await this.connect();
  }

  private buildTools(): Array<Record<string, unknown>> | undefined {
    const tools: Array<Record<string, unknown>> = [];

    if (this.config.geminiGoogleSearch) {
      tools.push({ googleSearch: {} });
    }

    if (this.config.geminiFunctionCalling) {
      tools.push({ functionDeclarations: [] });
    }

    return tools.length > 0 ? tools : undefined;
  }

  private enqueueServerMessage(message: LiveServerMessage): void {
    this.lastActivityAt = Date.now();
    const resolver = this.pendingMessageResolvers.shift();
    if (resolver) {
      resolver(message);
      return;
    }
    this.messageQueue.push(message);
    logger.debug(
      {
        turnId: this.activeTurnId,
        queueDepth: this.messageQueue.length,
        messageKind: describeLiveMessage(message),
      },
      "Queued Gemini live message",
    );
  }

  private async nextServerMessage(timeoutMs: number): Promise<LiveServerMessage | null> {
    const queued = this.messageQueue.shift();
    if (queued) {
      return queued;
    }

    return new Promise((resolve) => {
      const resolver = (message: LiveServerMessage | null): void => {
        clearTimeout(timer);
        resolve(message);
      };

      const timer = setTimeout(() => {
        const index = this.pendingMessageResolvers.indexOf(resolver);
        if (index >= 0) {
          this.pendingMessageResolvers.splice(index, 1);
        }
        resolve(null);
      }, timeoutMs);

      this.pendingMessageResolvers.push(resolver);
    });
  }

  private clearMessageQueue(): void {
    this.messageQueue = [];

    if (this.pendingMessageResolvers.length === 0) {
      return;
    }

    const resolvers = this.pendingMessageResolvers.splice(0);
    for (const resolve of resolvers) {
      resolve(null);
    }
  }

  private beginTurn(kind: "audio" | "text"): string {
    if (this.activeTurnId) {
      if (this.activeTurnKind !== kind) {
        logger.debug(
          {
            previousTurnId: this.activeTurnId,
            previousTurnKind: this.activeTurnKind,
            nextTurnKind: kind,
          },
          "Replacing active Gemini turn context",
        );
      }
      return this.activeTurnId;
    }

    this.turnCounter += 1;
    this.activeTurnId = `${kind}-${Date.now()}-${this.turnCounter}`;
    this.activeTurnKind = kind;
    this.sentAudioBytesForTurn = 0;
    this.sentAudioChunksForTurn = 0;
    logger.info({ turnId: this.activeTurnId, turnKind: kind }, "Gemini turn started");
    return this.activeTurnId;
  }

  private finishActiveTurn(reason: string): void {
    if (!this.activeTurnId) {
      return;
    }

    logger.debug(
      {
        turnId: this.activeTurnId,
        turnKind: this.activeTurnKind,
        reason,
        sentAudioChunks: this.sentAudioChunksForTurn,
        sentAudioBytes: this.sentAudioBytesForTurn,
      },
      "Gemini turn context cleared",
    );
    this.activeTurnId = null;
    this.activeTurnKind = null;
    this.sentAudioBytesForTurn = 0;
    this.sentAudioChunksForTurn = 0;
  }
}

function toBuffer(data: unknown): Buffer {
  if (Buffer.isBuffer(data)) {
    return data;
  }

  if (data instanceof Uint8Array) {
    return Buffer.from(data);
  }

  if (typeof data === "string") {
    try {
      return Buffer.from(data, "base64");
    } catch {
      return Buffer.alloc(0);
    }
  }

  return Buffer.alloc(0);
}

async function timeout(ms: number, message: string): Promise<never> {
  await wait(ms);
  throw new Error(message);
}

function previewText(text: string): string {
  const compact = text.replace(/\s+/g, " ").trim();
  if (compact.length <= 160) {
    return compact;
  }
  return `${compact.slice(0, 157)}...`;
}

function describeLiveMessage(message: LiveServerMessage): string {
  if (message.goAway) {
    return "goAway";
  }
  if (message.serverContent?.turnComplete) {
    return "serverContent.turnComplete";
  }
  if (message.serverContent?.modelTurn?.parts?.length) {
    return "serverContent.modelTurn";
  }
  if (message.toolCall) {
    return "toolCall";
  }
  if (message.toolCallCancellation) {
    return "toolCallCancellation";
  }
  if (message.setupComplete) {
    return "setupComplete";
  }
  if (message.sessionResumptionUpdate) {
    return "sessionResumptionUpdate";
  }
  return "other";
}



