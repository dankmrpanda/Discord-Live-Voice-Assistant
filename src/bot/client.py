"""Discord bot client with app commands (discord.py version)."""

import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional

from ..utils.logger import get_logger
from ..utils.config import Config, ConfigWatcher
from .voice_handler import VoiceHandler, BotState

logger = get_logger("bot.client")


class DiscordBot(commands.Bot):
    """Discord bot with voice channel support and slash commands."""
    
    def __init__(self, config: Config):
        """Initialize the Discord bot.
        
        Args:
            config: Application configuration.
        """
        logger.debug("Initializing DiscordBot")
        
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        logger.debug(f"Intents configured: message_content={intents.message_content}, voice_states={intents.voice_states}, guilds={intents.guilds}")
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            description="AI Voice Assistant powered by Gemini",
        )
        
        self.config = config
        self._voice_handlers: dict[int, VoiceHandler] = {}  # guild_id -> handler
        self._joining_guilds: set[int] = set()  # guilds currently in join process
        self._leaving_guilds: set[int] = set()  # guilds currently in explicit leave flow
        self._config_watcher: Optional[ConfigWatcher] = None
        logger.debug("DiscordBot initialization complete")
        
        # Register slash commands
        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register slash commands with discord.py app command tree."""

        @self.tree.command(name="join", description="Join your current voice channel")
        async def join_command(interaction: discord.Interaction) -> None:
            member = interaction.user if isinstance(interaction.user, discord.Member) else None
            if member is None or member.voice is None or member.voice.channel is None:
                await interaction.response.send_message(
                    "You need to be in a voice channel first!",
                    ephemeral=True,
                )
                return

            channel = member.voice.channel
            await interaction.response.defer(thinking=True)
            success, message = await self.join_voice(channel)
            await interaction.followup.send(message, ephemeral=not success)

        @self.tree.command(name="leave", description="Leave the voice channel")
        async def leave_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            await interaction.response.defer(thinking=True)
            success, message = await self.leave_voice(interaction.guild_id)
            await interaction.followup.send(message, ephemeral=not success)

        @self.tree.command(name="status", description="Check the bot's current status")
        async def status_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)

            if not handler or handler.state == BotState.IDLE:
                status = "Not connected to a voice channel."
            else:
                state_messages = {
                    BotState.IDLE: "Idle",
                    BotState.CONNECTING: "Connecting to voice channel...",
                    BotState.LISTENING: f"Listening for '{self.config.wake_phrase_display}'",
                    BotState.PROCESSING: "Processing your request...",
                    BotState.SPEAKING: "Speaking response",
                }
                state_msg = state_messages.get(handler.state, "Unknown")

                if handler.state == BotState.SPEAKING and handler.is_response_paused:
                    state_msg += " (paused)"

                status = f"**Status:** {state_msg}"

            embed = discord.Embed(
                title="Voice Assistant Status",
                description=status,
                color=discord.Color.blue(),
            )
            embed.add_field(name="Wake Phrase", value=f"`{self.config.wake_phrase_display}`", inline=True)
            embed.add_field(name="Voice", value=self.config.gemini_voice, inline=True)
            embed.add_field(name="Model", value=self.config.gemini_model, inline=True)
            embed.add_field(
                name="Features",
                value=(
                    f"Capture: {self.config.capture_duration}s • "
                    f"Silence: {self.config.silence_threshold}s • "
                    f"Buffer: {self.config.playback_buffer_ms}ms"
                ),
                inline=False,
            )

            if handler and handler.state != BotState.IDLE:
                queue_size = handler.get_queue_size()
                embed.add_field(name="Queue", value=f"{queue_size} prompt(s)", inline=True)

            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="ask", description="Send a text prompt to the bot (no wake word needed)")
        @app_commands.describe(prompt="Your question or prompt for the AI")
        async def ask_command(interaction: discord.Interaction, prompt: str) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)

            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel. Use `/join` first!",
                    ephemeral=True,
                )
                return

            if handler.state in (BotState.PROCESSING, BotState.SPEAKING):
                position = handler.queue_text_prompt(prompt, interaction.user.id)
                if position == -1:
                    await interaction.response.send_message(
                        "The prompt queue is full. Please wait for some prompts to be processed first.",
                        ephemeral=True,
                    )
                    return
                await interaction.response.send_message(
                    f"I'm currently busy. Your prompt has been queued at position #{position}. "
                    "It will be processed automatically after the current request finishes.",
                    ephemeral=True,
                )
                return

            if handler.state == BotState.CONNECTING:
                await interaction.response.send_message(
                    "I'm still connecting to the voice channel. Please wait a moment.",
                    ephemeral=True,
                )
                return

            prompt = prompt.strip()
            if not prompt:
                await interaction.response.send_message(
                    "Please provide a non-empty prompt.",
                    ephemeral=True,
                )
                return

            max_prompt_length = 2000
            if len(prompt) > max_prompt_length:
                await interaction.response.send_message(
                    f"Prompt is too long. Maximum length is {max_prompt_length} characters.",
                    ephemeral=True,
                )
                return

            await interaction.response.defer(thinking=True)

            try:
                success = await handler.process_text_prompt(prompt, interaction.user.id)
                if success:
                    preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
                    await interaction.followup.send(f"Processing: \"{preview}\"")
                else:
                    await interaction.followup.send(
                        "Failed to process your prompt. Please try again.",
                        ephemeral=True,
                    )
            except Exception as e:
                logger.error(f"Error processing /ask command: {e}")
                await interaction.followup.send(
                    "An error occurred while processing your request. Please try again.",
                    ephemeral=True,
                )

        @self.tree.command(name="queue", description="View the current /ask prompt queue")
        async def queue_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)
            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel.",
                    ephemeral=True,
                )
                return

            queue_items = handler.get_queue_items()
            if not queue_items:
                await interaction.response.send_message(
                    "The prompt queue is empty.",
                    ephemeral=True,
                )
                return

            embed = discord.Embed(
                title="Prompt Queue",
                description=f"**{len(queue_items)}** prompt(s) waiting to be processed:",
                color=discord.Color.blue(),
            )

            max_items_to_show = 10
            visible_items = queue_items[:max_items_to_show]
            for i, (item_prompt, user_id) in enumerate(visible_items, 1):
                display_prompt = item_prompt[:80] + "..." if len(item_prompt) > 80 else item_prompt
                embed.add_field(
                    name=f"#{i}",
                    value=f"<@{user_id}>: *{display_prompt}*",
                    inline=False,
                )

            hidden_items = len(queue_items) - len(visible_items)
            if hidden_items > 0:
                embed.set_footer(text=f"...and {hidden_items} more prompt(s) not shown.")

            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="stop", description="Stop the current request and move to the next in queue")
        async def stop_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)
            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel.",
                    ephemeral=True,
                )
                return

            if handler.state not in (BotState.PROCESSING, BotState.SPEAKING):
                await interaction.response.send_message(
                    "I'm not currently processing or speaking.",
                    ephemeral=True,
                )
                return

            success = await handler.stop_response()
            if success:
                queue_size = handler.get_queue_size()
                if queue_size > 0:
                    await interaction.response.send_message(
                        f"Request stopped. Processing next prompt in queue ({queue_size} remaining)."
                    )
                else:
                    await interaction.response.send_message("Request stopped. Listening for wake word.")
            else:
                await interaction.response.send_message(
                    "Failed to stop the request.",
                    ephemeral=True,
                )

        @self.tree.command(name="pause", description="Pause the bot's current response")
        async def pause_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)
            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel.",
                    ephemeral=True,
                )
                return

            if handler.state != BotState.SPEAKING:
                await interaction.response.send_message(
                    "I'm not currently speaking.",
                    ephemeral=True,
                )
                return

            if handler.is_response_paused:
                await interaction.response.send_message(
                    "Response is already paused. Use `/continue` to resume.",
                    ephemeral=True,
                )
                return

            success = handler.pause_response()
            if success:
                await interaction.response.send_message("Response paused. Use `/continue` to resume.")
            else:
                await interaction.response.send_message(
                    "Failed to pause the response.",
                    ephemeral=True,
                )

        @self.tree.command(name="continue", description="Continue the bot's paused response")
        async def continue_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)
            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel.",
                    ephemeral=True,
                )
                return

            if handler.state != BotState.SPEAKING:
                await interaction.response.send_message(
                    "I'm not currently speaking.",
                    ephemeral=True,
                )
                return

            if not handler.is_response_paused:
                await interaction.response.send_message(
                    "Response is not paused.",
                    ephemeral=True,
                )
                return

            success = handler.resume_response()
            if success:
                await interaction.response.send_message("Response resumed.")
            else:
                await interaction.response.send_message(
                    "Failed to resume the response.",
                    ephemeral=True,
                )

        @self.tree.command(name="help", description="Show all available commands")
        async def help_command(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(embed=self._build_help_embed(), ephemeral=True)

        @self.tree.command(name="clearqueue", description="Clear all queued prompts")
        async def clearqueue_command(interaction: discord.Interaction) -> None:
            if interaction.guild_id is None:
                await interaction.response.send_message(
                    "This command can only be used in a server.",
                    ephemeral=True,
                )
                return

            handler = self.get_voice_handler(interaction.guild_id)
            if not handler or handler.state == BotState.IDLE:
                await interaction.response.send_message(
                    "I'm not connected to a voice channel.",
                    ephemeral=True,
                )
                return

            removed = handler.clear_queue()
            if removed == 0:
                await interaction.response.send_message("The queue is already empty.", ephemeral=True)
            else:
                await interaction.response.send_message(f"Cleared **{removed}** prompt(s) from the queue.")

        logger.debug(
            "App commands registered: /join, /leave, /status, /ask, /queue, /stop, /pause, /continue, /help, /clearqueue"
        )

    def _build_help_embed(self) -> discord.Embed:
        """Build a help embed listing all slash commands."""
        embed = discord.Embed(
            title="Voice Assistant - Commands",
            description="Here are all available slash commands:",
            color=discord.Color.green(),
        )
        cmds = [
            ("/join", "Join your current voice channel."),
            ("/leave", "Leave the voice channel."),
            ("/status", "Show the bot's current state, model, and settings."),
            ("/ask <prompt>", "Send a text prompt (queued if busy)."),
            ("/queue", "View the current prompt queue."),
            ("/clearqueue", "Clear all queued prompts."),
            ("/stop", "Stop the current response and move on."),
            ("/pause", "Pause the bot's spoken response."),
            ("/continue", "Resume a paused response."),
            ("/help", "Show this help message."),
        ]
        for name, desc in cmds:
            embed.add_field(name=name, value=desc, inline=False)
        return embed

    async def setup_hook(self) -> None:
        """Sync application commands after login, before ready."""
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} app command(s)")
        except Exception as exc:
            logger.error(f"Failed to sync app commands: {exc}")
    
    async def on_ready(self) -> None:
        """Handle bot ready event."""
        logger.info("=" * 40)
        logger.info(f"Bot is ready! Logged in as {self.user}")
        logger.info(f"Bot ID: {self.user.id}")
        logger.info(f"Wake phrase: '{self.config.wake_phrase_display}'")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")
        for guild in self.guilds:
            logger.debug(f"  - {guild.name} (ID: {guild.id}, Members: {guild.member_count})")
        logger.info("=" * 40)
        
        # Start config file watcher
        if self._config_watcher is None:
            self._config_watcher = ConfigWatcher(self.config, check_interval=2.0)
            await self._config_watcher.start()
            logger.info("Config file watcher started (changes auto-reload)")
    
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Handle voice state changes."""
        # Handle other users leaving the voice channel
        if member.id != self.user.id and before.channel and not after.channel:
            guild_id = before.channel.guild.id
            handler = self.get_voice_handler(guild_id)
            
            # Check if bot is in the same channel the user left from
            if handler and handler.is_connected:
                if handler.is_connected_to_channel(before.channel):
                    logger.debug(f"User {member.name} ({member.id}) left voice channel, cleaning up resources")
                    handler.cleanup_user(member.id)
        
        # Only log our own voice state changes at debug level
        if member.id == self.user.id:
            logger.debug(f"Voice state update: {member.name} - before={before.channel}, after={after.channel}")
            
            # Handle disconnection
            if before.channel and not after.channel:
                guild_id = before.channel.guild.id
                
                # Ignore disconnects during join process (voice handshake can cause brief disconnects)
                if guild_id in self._joining_guilds:
                    logger.debug(f"Ignoring disconnect during join process for guild {guild_id}")
                    return
                
                # Ignore disconnects during explicit leave command flow
                if guild_id in self._leaving_guilds:
                    logger.debug(f"Ignoring disconnect during leave process for guild {guild_id}")
                    return
                
                # Check if we have a handler that's still connecting
                if guild_id in self._voice_handlers:
                    handler = self._voice_handlers[guild_id]
                    
                    # Ignore if handler is in connecting state
                    if handler.is_connecting:
                        logger.debug(f"Ignoring disconnect - handler is still connecting")
                        return
                    
                    # Ignore if the voice client is still connected (spurious event)
                    if handler.is_connected:
                        logger.debug(f"Ignoring spurious disconnect event - voice client still connected")
                        return
                
                # Real disconnection - clean up
                logger.info(f"Bot disconnected from voice channel '{before.channel.name}' in guild {guild_id}")
                if guild_id in self._voice_handlers:
                    handler = self._voice_handlers.get(guild_id)
                    if handler is not None:
                        await handler.leave_channel()
                    self._voice_handlers.pop(guild_id, None)
                    logger.debug(f"Voice handler removed for guild {guild_id}")
    
    def get_voice_handler(self, guild_id: int) -> Optional[VoiceHandler]:
        """Get the voice handler for a guild."""
        return self._voice_handlers.get(guild_id)
    
    async def join_voice(self, channel: discord.VoiceChannel) -> tuple[bool, str]:
        """Join a voice channel.
        
        Args:
            channel: The voice channel to join.
            
        Returns:
            Tuple of (success, message).
        """
        guild_id = channel.guild.id
        logger.info(f"Join voice request: channel='{channel.name}', guild_id={guild_id}")
        
        # Check if already connected
        if guild_id in self._voice_handlers:
            handler = self._voice_handlers[guild_id]
            if handler.is_connected:
                logger.debug(f"Already connected to voice in guild {guild_id}")
                return False, "Already connected to a voice channel in this server."
            if handler.is_connecting:
                logger.debug(f"Already connecting to voice in guild {guild_id}")
                return False, "Already connecting to a voice channel. Please wait..."
            # Clean up old disconnected handler
            logger.debug("Cleaning up old voice handler")
            await handler.leave_channel()
            del self._voice_handlers[guild_id]
        
        # Mark guild as joining to prevent voice state handler cleanup during handshake
        self._joining_guilds.add(guild_id)
        
        try:
            logger.debug("Creating new VoiceHandler")
            handler = VoiceHandler(self.config)
            
            # Store handler immediately so we can track its state
            self._voice_handlers[guild_id] = handler
            
            success = await handler.join_channel(channel)
            
            if success:
                logger.info(f"Successfully joined voice channel '{channel.name}'")
                return True, f"Joined **{channel.name}**! Say '{self.config.wake_phrase_display}' to talk to me."
            else:
                # Clean up failed handler
                if guild_id in self._voice_handlers:
                    del self._voice_handlers[guild_id]
                logger.error(f"Failed to join voice channel '{channel.name}'")
                return False, "Failed to join voice channel. Please try again."
        except Exception as e:
            # Clean up on exception
            if guild_id in self._voice_handlers:
                try:
                    await self._voice_handlers[guild_id].leave_channel()
                except Exception:
                    pass
                del self._voice_handlers[guild_id]
            logger.error(f"Exception joining voice channel: {e}")
            return False, f"Error joining voice channel: {str(e)}"
        finally:
            # Always remove from joining set when done
            self._joining_guilds.discard(guild_id)
    
    async def leave_voice(self, guild_id: int) -> tuple[bool, str]:
        """Leave voice channel in a guild."""
        logger.info(f"Leave voice request: guild_id={guild_id}")
        
        if guild_id not in self._voice_handlers:
            logger.debug("Not connected to any voice channel")
            return False, "Not connected to a voice channel."

        self._leaving_guilds.add(guild_id)
        try:
            handler = self._voice_handlers.get(guild_id)
            if handler is not None:
                await handler.leave_channel()
            self._voice_handlers.pop(guild_id, None)
            logger.info(f"Left voice channel in guild {guild_id}")
            return True, "Left the voice channel. Goodbye!"
        finally:
            self._leaving_guilds.discard(guild_id)
    
    async def close(self) -> None:
        """Clean up resources when bot is closing."""
        logger.info("Bot closing, cleaning up resources...")
        
        # Stop config watcher
        if self._config_watcher:
            await self._config_watcher.stop()
            logger.debug("Config watcher stopped")
        
        # Leave all voice channels
        for guild_id in list(self._voice_handlers.keys()):
            try:
                await self.leave_voice(guild_id)
            except Exception as e:
                logger.debug(f"Error leaving voice in guild {guild_id}: {e}")
        
        # Call parent close
        await super().close()
        logger.info("Bot cleanup complete")


# Global bot reference (kept for compatibility)
_bot: Optional[DiscordBot] = None


def set_bot(bot: DiscordBot) -> None:
    """Set the global bot reference."""
    global _bot
    _bot = bot

