import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import YAML from "yaml";

import { appConfigSchema } from "./schema.js";
import type { AppConfig } from "../types/index.js";

export type ConfigChangeListener = (config: AppConfig, changedFields: string[]) => void;

interface RawYaml {
  wake_word?: { phrase?: string; threshold?: number };
  voice?: { name?: string };
  gemini?: {
    model?: string;
    thinking?: boolean;
    google_search?: boolean;
    function_calling?: boolean;
    automatic_function_response?: boolean;
  };
  behavior?: { capture_duration?: number; silence_threshold?: number };
  audio?: {
    discord_sample_rate?: number;
    gemini_input_sample_rate?: number;
    gemini_output_sample_rate?: number;
    playback_buffer_ms?: number;
  };
  logging?: { level?: "debug" | "info" | "warn" | "error"; directory?: string; log_audio?: boolean };
  system_prompt?: string;
}

function defaultSystemPrompt(): string {
  return [
    "You are a helpful voice assistant in a Discord voice channel.",
    "Keep your responses concise and conversational since they will be spoken aloud.",
    "Avoid markdown formatting and lists in spoken output.",
  ].join(" ");
}

function readYamlConfig(configPath: string): RawYaml {
  if (!fs.existsSync(configPath)) {
    return {};
  }
  const raw = fs.readFileSync(configPath, "utf8");
  const parsed = YAML.parse(raw);
  if (parsed && typeof parsed === "object") {
    return parsed as RawYaml;
  }
  return {};
}

export class ConfigManager {
  private readonly configPath: string;
  private currentConfig!: AppConfig;
  private readonly listeners = new Set<ConfigChangeListener>();

  public constructor(configPath = "config.yaml", envPath = ".env") {
    const resolvedEnv = path.resolve(envPath);
    dotenv.config({ path: resolvedEnv, quiet: true });
    this.configPath = path.resolve(configPath);
    this.currentConfig = this.loadConfigInternal();
  }

  public get config(): AppConfig {
    return this.currentConfig;
  }

  public addChangeListener(listener: ConfigChangeListener): void {
    this.listeners.add(listener);
  }

  public removeChangeListener(listener: ConfigChangeListener): void {
    this.listeners.delete(listener);
  }

  public reload(): string[] {
    const previous = this.currentConfig;
    const next = this.loadConfigInternal();

    const changedFields: string[] = [];
    for (const [key, value] of Object.entries(next)) {
      const typedKey = key as keyof AppConfig;
      if (previous[typedKey] !== value) {
        changedFields.push(key);
      }
    }

    this.currentConfig = next;

    if (changedFields.length > 0) {
      for (const listener of this.listeners) {
        listener(this.currentConfig, changedFields);
      }
    }

    return changedFields;
  }

  public getConfigPath(): string {
    return this.configPath;
  }

  private loadConfigInternal(): AppConfig {
    const yaml = readYamlConfig(this.configPath);

    const raw = {
      discordBotToken: process.env.DISCORD_BOT_TOKEN,
      discordApplicationId: process.env.DISCORD_APPLICATION_ID,
      discordGuildId: process.env.DISCORD_GUILD_ID,
      geminiApiKey: process.env.GEMINI_API_KEY,

      wakePhrase: yaml.wake_word?.phrase ?? "hey_jarvis",
      wakeWordThreshold: yaml.wake_word?.threshold ?? 0.5,

      geminiVoice: yaml.voice?.name ?? "Puck",
      geminiModel: yaml.gemini?.model ?? "gemini-2.0-flash-live-001",
      geminiThinking: yaml.gemini?.thinking ?? false,
      geminiGoogleSearch: yaml.gemini?.google_search ?? false,
      geminiFunctionCalling: yaml.gemini?.function_calling ?? false,
      geminiAutomaticFunctionResponse: yaml.gemini?.automatic_function_response ?? false,

      captureDuration: yaml.behavior?.capture_duration ?? 7,
      silenceThreshold: yaml.behavior?.silence_threshold ?? 1,

      systemPrompt: yaml.system_prompt ?? defaultSystemPrompt(),

      logLevel: (yaml.logging?.level ?? "info").toLowerCase(),
      logDirectory: yaml.logging?.directory ?? "logs",
      logAudio: yaml.logging?.log_audio ?? false,

      discordSampleRate: yaml.audio?.discord_sample_rate ?? 48000,
      geminiInputSampleRate: yaml.audio?.gemini_input_sample_rate ?? 16000,
      geminiOutputSampleRate: yaml.audio?.gemini_output_sample_rate ?? 24000,
      playbackBufferMs: yaml.audio?.playback_buffer_ms ?? 200,
    };

    const parsed = appConfigSchema.safeParse(raw);
    if (!parsed.success) {
      const details = parsed.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; ");
      throw new Error(`Invalid configuration: ${details}`);
    }

    return parsed.data;
  }
}

