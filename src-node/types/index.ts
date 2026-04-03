export enum BotState {
  IDLE = "idle",
  CONNECTING = "connecting",
  LISTENING = "listening",
  PROCESSING = "processing",
  SPEAKING = "speaking",
}

export interface AskQueueItem {
  prompt: string;
  userId: string;
  createdAt: number;
}

export interface AppConfig {
  discordBotToken: string;
  discordApplicationId?: string;
  discordGuildId?: string;
  geminiApiKey: string;

  wakePhrase: string;
  wakeWordThreshold: number;

  geminiVoice: string;
  geminiModel: string;
  geminiThinking: boolean;
  geminiGoogleSearch: boolean;
  geminiFunctionCalling: boolean;
  geminiAutomaticFunctionResponse: boolean;

  captureDuration: number;
  silenceThreshold: number;

  systemPrompt: string;

  logLevel: "trace" | "debug" | "info" | "warn" | "error" | "fatal";
  logDirectory: string;
  logAudio: boolean;

  discordSampleRate: number;
  geminiInputSampleRate: number;
  geminiOutputSampleRate: number;
  playbackBufferMs: number;
}
