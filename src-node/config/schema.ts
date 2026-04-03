import { z } from "zod";

const boolish = z
  .union([z.boolean(), z.string()])
  .transform((value) => {
    if (typeof value === "boolean") {
      return value;
    }
    return value.toLowerCase() === "true";
  });

export const appConfigSchema = z.object({
  discordBotToken: z.string().min(1),
  discordApplicationId: z.string().optional(),
  discordGuildId: z.string().optional(),

  geminiApiKey: z.string().min(1),
  picovoiceAccessKey: z.string().min(1).optional(),
  geminiVoice: z.string().default("Puck"),
  geminiModel: z.string().default("gemini-2.0-flash-live-001"),
  geminiThinking: boolish.default(false),
  geminiGoogleSearch: boolish.default(false),
  geminiFunctionCalling: boolish.default(false),
  geminiAutomaticFunctionResponse: boolish.default(false),

  wakePhrase: z.string().default("hey_jarvis"),
  wakeWordThreshold: z.number().min(0).max(1).default(0.5),

  captureDuration: z.number().positive().default(7),
  silenceThreshold: z.number().positive().default(1),

  systemPrompt: z.string().default(
    "You are a helpful voice assistant in a Discord voice channel. Keep responses concise and conversational for spoken output.",
  ),

  logLevel: z.enum(["trace", "debug", "info", "warn", "error", "fatal"]).default("info"),
  logDirectory: z.string().default("logs"),
  logAudio: boolish.default(false),

  discordSampleRate: z.number().int().positive().default(48000),
  geminiInputSampleRate: z.number().int().positive().default(16000),
  geminiOutputSampleRate: z.number().int().positive().default(24000),
  playbackBufferMs: z.number().int().min(0).default(200),
});

export type AppConfigSchema = z.infer<typeof appConfigSchema>;
