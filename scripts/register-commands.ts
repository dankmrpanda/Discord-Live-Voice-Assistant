import { REST, Routes } from "discord.js";

import { ConfigManager } from "../src-node/config/config.js";
import { buildSlashCommands } from "../src-node/bot/command-registry.js";

async function run(): Promise<void> {
  const config = new ConfigManager().config;
  const rest = new REST({ version: "10" }).setToken(config.discordBotToken);
  const appId = config.discordApplicationId;

  if (!appId) {
    throw new Error("DISCORD_APPLICATION_ID is required for command registration.");
  }

  const commands = buildSlashCommands();

  if (config.discordGuildId) {
    await rest.put(Routes.applicationGuildCommands(appId, config.discordGuildId), { body: commands });
    // eslint-disable-next-line no-console
    console.log(`Registered ${commands.length} guild commands for ${config.discordGuildId}`);
    return;
  }

  await rest.put(Routes.applicationCommands(appId), { body: commands });
  // eslint-disable-next-line no-console
  console.log(`Registered ${commands.length} global commands`);
}

run().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exitCode = 1;
});
