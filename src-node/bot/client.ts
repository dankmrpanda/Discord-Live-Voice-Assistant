import {
  Client,
  EmbedBuilder,
  GatewayIntentBits,
  Interaction,
  REST,
  Routes,
  type ChatInputCommandInteraction,
  type GuildMember,
  type VoiceBasedChannel,
} from "discord.js";

import type { ConfigManager } from "../config/config.js";
import type { AppConfig } from "../types/index.js";
import { BotState } from "../types/index.js";
import { getLogger, logException } from "../utils/logger.js";
import { VoiceHandler } from "./voice-handler.js";
import { buildSlashCommands } from "./command-registry.js";

const logger = getLogger("bot.client");

export class DiscordVoiceBot {
  private readonly client: Client;
  private readonly handlers = new Map<string, VoiceHandler>();
  private readonly joiningGuilds = new Set<string>();
  private currentConfig: AppConfig;

  public constructor(private readonly configManager: ConfigManager) {
    this.currentConfig = configManager.config;

    this.client = new Client({
      intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildVoiceStates,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
      ],
    });

    this.client.on("clientReady", () => {
      void this.onReady();
    });

    this.client.on("interactionCreate", (interaction) => {
      void this.onInteraction(interaction);
    });

    this.client.on("voiceStateUpdate", (before, after) => {
      if (!this.client.user) {
        return;
      }

      const handler = this.handlers.get(before.guild.id);
      if (!handler) {
        return;
      }

      if (before.member?.id !== this.client.user.id && before.channelId && !after.channelId) {
        handler.cleanupUser(before.member as GuildMember);
      }

      if (before.member?.id === this.client.user.id && before.channelId && !after.channelId) {
        if (this.joiningGuilds.has(before.guild.id)) {
          return;
        }

        if (handler.getState() === BotState.CONNECTING) {
          return;
        }

        void this.removeHandler(before.guild.id);
      }
    });

    this.configManager.addChangeListener((config) => {
      this.currentConfig = config;
      for (const handler of this.handlers.values()) {
        handler.updateConfig(config);
      }
    });
  }

  public async start(): Promise<void> {
    await this.client.login(this.currentConfig.discordBotToken);
  }

  public async stop(): Promise<void> {
    for (const guildId of this.handlers.keys()) {
      await this.removeHandler(guildId);
    }
    this.client.destroy();
  }

  private async onReady(): Promise<void> {
    if (!this.client.user) {
      return;
    }

    logger.info({ user: this.client.user.tag, guilds: this.client.guilds.cache.size }, "Bot ready");
    await this.registerCommands();
  }

  private async registerCommands(): Promise<void> {
    const rest = new REST({ version: "10" }).setToken(this.currentConfig.discordBotToken);
    const commands = buildSlashCommands();
    const applicationId = this.currentConfig.discordApplicationId ?? this.client.application?.id;

    if (!applicationId) {
      logger.warn("Skipping command registration: application ID is unavailable");
      return;
    }

    try {
      if (this.currentConfig.discordGuildId) {
        await rest.put(
          Routes.applicationGuildCommands(
            applicationId,
            this.currentConfig.discordGuildId,
          ),
          { body: commands },
        );
        logger.info({ guildId: this.currentConfig.discordGuildId }, "Registered guild slash commands");
        return;
      }

      await rest.put(
        Routes.applicationCommands(applicationId),
        { body: commands },
      );
      logger.info("Registered global slash commands");
    } catch (error) {
      logException(logger, "Failed to register slash commands", error);
    }
  }

  private async onInteraction(interaction: Interaction): Promise<void> {
    if (!interaction.isChatInputCommand()) {
      return;
    }

    const context = {
      command: interaction.commandName,
      guildId: interaction.guildId,
      channelId: interaction.channelId,
      userId: interaction.user.id,
    };
    logger.info(context, "Received slash command");

    try {
      switch (interaction.commandName) {
        case "join":
          await this.handleJoin(interaction);
          break;
        case "leave":
          await this.handleLeave(interaction);
          break;
        case "status":
          await this.handleStatus(interaction);
          break;
        case "ask":
          await this.handleAsk(interaction);
          break;
        case "queue":
          await this.handleQueue(interaction);
          break;
        case "stop":
          await this.handleStop(interaction);
          break;
        case "pause":
          await this.handlePause(interaction);
          break;
        case "continue":
          await this.handleContinue(interaction);
          break;
        default:
          await interaction.reply({ content: "Unknown command.", ephemeral: true });
      }
      logger.info(context, "Slash command handled successfully");
    } catch (error) {
      logException(logger, `Command handler failed: ${interaction.commandName}`, error, context);
      const content = "An error occurred while running this command.";
      if (interaction.deferred || interaction.replied) {
        await interaction.followUp({ content, ephemeral: true });
      } else {
        await interaction.reply({ content, ephemeral: true });
      }
    }
  }

  private async handleJoin(interaction: ChatInputCommandInteraction): Promise<void> {
    const member = interaction.member as GuildMember;
    const channel = member.voice.channel as VoiceBasedChannel | null;

    if (!channel) {
      await interaction.reply({ content: "You need to be in a voice channel first.", ephemeral: true });
      return;
    }

    await interaction.deferReply();

    const guildId = interaction.guildId ?? "";
    this.joiningGuilds.add(guildId);

    try {
      const handler = this.getOrCreateHandler(guildId);
      const result = await handler.join(channel);
      logger.info(
        {
          guildId,
          channelId: channel.id,
          channelName: channel.name,
          success: result.success,
        },
        "Join command completed",
      );
      await interaction.editReply(result.message);

      if (!result.success) {
        this.handlers.delete(guildId);
      }
    } finally {
      this.joiningGuilds.delete(guildId);
    }
  }

  private async handleLeave(interaction: ChatInputCommandInteraction): Promise<void> {
    const guildId = interaction.guildId;
    if (!guildId || !this.handlers.has(guildId)) {
      await interaction.reply({ content: "I am not in a voice channel.", ephemeral: true });
      return;
    }

    await interaction.deferReply();
    const handler = this.handlers.get(guildId);
    if (!handler) {
      await interaction.editReply("I am not in a voice channel.");
      return;
    }

    const result = await handler.leave();
    logger.info({ guildId, success: result.success }, "Leave command completed");
    this.handlers.delete(guildId);
    await interaction.editReply(result.message);
  }

  private async handleStatus(interaction: ChatInputCommandInteraction): Promise<void> {
    const handler = interaction.guildId ? this.handlers.get(interaction.guildId) : null;

    const state = handler?.getState() ?? BotState.IDLE;
    const wakeWordEnabled = handler?.isWakeWordAvailable ?? false;
    const wakeWordValue = wakeWordEnabled
      ? this.currentConfig.wakePhrase.replaceAll("_", " ")
      : "Disabled in this build (use /ask)";
    const statusText =
      state === BotState.IDLE
        ? "Not connected to a voice channel."
        : state === BotState.CONNECTING
          ? "Connecting to voice channel..."
          : state === BotState.LISTENING
            ? wakeWordEnabled
              ? `Listening for '${this.currentConfig.wakePhrase.replaceAll("_", " ")}'`
              : "Listening for /ask prompts"
            : state === BotState.PROCESSING
              ? "Processing request..."
              : `Speaking response${handler?.isResponsePaused ? " (paused)" : ""}`;

    const embed = new EmbedBuilder()
      .setTitle("Voice Assistant Status")
      .setDescription(statusText)
      .addFields(
        { name: "Wake Phrase", value: wakeWordValue, inline: true },
        { name: "Voice", value: this.currentConfig.geminiVoice, inline: true },
        { name: "Queue", value: `${handler?.getQueueSize() ?? 0} prompt(s)`, inline: true },
      );

    await interaction.reply({ embeds: [embed] });
  }

  private async handleAsk(interaction: ChatInputCommandInteraction): Promise<void> {
    const guildId = interaction.guildId;
    const prompt = interaction.options.getString("prompt", true).trim();
    const maxPromptLength = 2000;

    if (!guildId || !prompt) {
      await interaction.reply({ content: "Please provide a prompt.", ephemeral: true });
      return;
    }

    if (prompt.length > maxPromptLength) {
      await interaction.reply({
        content: `Prompt is too long. Maximum length is ${maxPromptLength} characters.`,
        ephemeral: true,
      });
      return;
    }

    const handler = this.handlers.get(guildId);
    if (!handler || handler.getState() === BotState.IDLE) {
      await interaction.reply({ content: "I am not connected to voice. Use /join first.", ephemeral: true });
      return;
    }

    const member = interaction.member as GuildMember;
    logger.info(
      {
        guildId,
        userId: member.id,
        promptLength: prompt.length,
        promptPreview: previewPrompt(prompt),
        handlerState: handler.getState(),
      },
      "Ask command received",
    );

    if (handler.getState() === BotState.PROCESSING || handler.getState() === BotState.SPEAKING) {
      const position = handler.queueTextPrompt(prompt, member.id);
      logger.info({ guildId, userId: member.id, queuePosition: position }, "Ask command queued");
      await interaction.reply({
        content: `I am busy. Prompt queued at position #${position}.`,
        ephemeral: false,
      });
      return;
    }

    if (handler.getState() === BotState.CONNECTING) {
      await interaction.reply({ content: "Still connecting. Try again in a moment.", ephemeral: true });
      return;
    }

    await interaction.deferReply();
    const started = await handler.processTextPrompt(prompt, member.id);
    if (!started) {
      await interaction.editReply("Failed to process prompt.");
      return;
    }
    logger.info({ guildId, userId: member.id }, "Ask command started processing");
    await interaction.editReply(`Processing: \"${prompt.slice(0, 100)}${prompt.length > 100 ? "..." : ""}\"`);
  }

  private async handleQueue(interaction: ChatInputCommandInteraction): Promise<void> {
    const handler = interaction.guildId ? this.handlers.get(interaction.guildId) : null;
    if (!handler || handler.getState() === BotState.IDLE) {
      await interaction.reply({ content: "I am not connected to a voice channel.", ephemeral: true });
      return;
    }

    const queue = handler.getQueueItems();
    if (queue.length === 0) {
      await interaction.reply({ content: "Queue is empty.", ephemeral: true });
      return;
    }

    const embed = new EmbedBuilder().setTitle("Prompt Queue").setDescription(`${queue.length} prompt(s) queued`);
    queue.slice(0, 10).forEach((item, index) => {
      const text = item.prompt.length > 80 ? `${item.prompt.slice(0, 80)}...` : item.prompt;
      embed.addFields({ name: `#${index + 1}`, value: `<@${item.userId}>: ${text}` });
    });

    await interaction.reply({ embeds: [embed], ephemeral: false });
  }

  private async handleStop(interaction: ChatInputCommandInteraction): Promise<void> {
    const handler = interaction.guildId ? this.handlers.get(interaction.guildId) : null;
    if (!handler) {
      await interaction.reply({ content: "I am not connected to voice.", ephemeral: true });
      return;
    }

    const stopped = await handler.stopResponse();
    if (!stopped) {
      await interaction.reply({ content: "I am not currently processing or speaking.", ephemeral: true });
      return;
    }

    const queueRemaining = handler.getQueueSize();
    if (queueRemaining > 0) {
      await interaction.reply({
        content: `Stopped current request. Processing next queued prompt (${queueRemaining} remaining).`,
        ephemeral: false,
      });
      return;
    }

    await interaction.reply({ content: "Stopped current request. Listening for wake word.", ephemeral: false });
  }

  private async handlePause(interaction: ChatInputCommandInteraction): Promise<void> {
    const handler = interaction.guildId ? this.handlers.get(interaction.guildId) : null;
    if (!handler) {
      await interaction.reply({ content: "I am not connected to voice.", ephemeral: true });
      return;
    }

    if (handler.getState() !== BotState.SPEAKING) {
      await interaction.reply({ content: "I am not currently speaking.", ephemeral: true });
      return;
    }

    if (handler.isResponsePaused) {
      await interaction.reply({ content: "Response is already paused. Use /continue to resume.", ephemeral: true });
      return;
    }

    const paused = handler.pauseResponse();
    if (!paused) {
      await interaction.reply({ content: "Unable to pause right now.", ephemeral: true });
      return;
    }

    await interaction.reply("Paused response.");
  }

  private async handleContinue(interaction: ChatInputCommandInteraction): Promise<void> {
    const handler = interaction.guildId ? this.handlers.get(interaction.guildId) : null;
    if (!handler) {
      await interaction.reply({ content: "I am not connected to voice.", ephemeral: true });
      return;
    }

    if (handler.getState() !== BotState.SPEAKING) {
      await interaction.reply({ content: "I am not currently speaking.", ephemeral: true });
      return;
    }

    if (!handler.isResponsePaused) {
      await interaction.reply({ content: "Response is not paused.", ephemeral: true });
      return;
    }

    const resumed = handler.resumeResponse();
    if (!resumed) {
      await interaction.reply({ content: "Unable to resume right now.", ephemeral: true });
      return;
    }

    await interaction.reply("Resumed response.");
  }

  private getOrCreateHandler(guildId: string): VoiceHandler {
    const existing = this.handlers.get(guildId);
    if (existing) {
      return existing;
    }

    const handler = new VoiceHandler(this.currentConfig);
    this.handlers.set(guildId, handler);
    return handler;
  }

  private async removeHandler(guildId: string): Promise<void> {
    const handler = this.handlers.get(guildId);
    if (!handler) {
      return;
    }

    logger.info({ guildId }, "Removing voice handler");
    await handler.leave();
    this.handlers.delete(guildId);
  }
}

function previewPrompt(prompt: string): string {
  const compact = prompt.replace(/\s+/g, " ").trim();
  if (compact.length <= 160) {
    return compact;
  }
  return `${compact.slice(0, 157)}...`;
}
