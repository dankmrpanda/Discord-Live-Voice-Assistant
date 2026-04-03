import { SlashCommandBuilder } from "discord.js";

export function buildSlashCommands(): ReturnType<SlashCommandBuilder["toJSON"]>[] {
  return [
    new SlashCommandBuilder().setName("join").setDescription("Join your current voice channel").toJSON(),
    new SlashCommandBuilder().setName("leave").setDescription("Leave the voice channel").toJSON(),
    new SlashCommandBuilder().setName("status").setDescription("Check the bot status").toJSON(),
    new SlashCommandBuilder()
      .setName("ask")
      .setDescription("Send a text prompt to Jarvis")
      .addStringOption((option) => option.setName("prompt").setDescription("Your prompt").setRequired(true))
      .toJSON(),
    new SlashCommandBuilder().setName("queue").setDescription("View queued prompts").toJSON(),
    new SlashCommandBuilder().setName("stop").setDescription("Stop current request and process next").toJSON(),
    new SlashCommandBuilder().setName("pause").setDescription("Pause current response playback").toJSON(),
    new SlashCommandBuilder().setName("continue").setDescription("Resume paused response playback").toJSON(),
  ];
}
