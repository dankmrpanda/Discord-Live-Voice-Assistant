"""Discord bot client with slash commands (py-cord version)."""

import discord
from discord.ext import commands
from typing import Optional

from ..utils.logger import get_logger
from ..utils.config import Config
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
        logger.debug("DiscordBot initialization complete")
        
        # Register slash commands
        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register slash commands with the bot."""
        
        @self.slash_command(name="join", description="Join your current voice channel")
        async def join_command(ctx: discord.ApplicationContext) -> None:
            """Slash command to join a voice channel."""
            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.respond("You need to be in a voice channel first!", ephemeral=True)
                return
            
            channel = ctx.author.voice.channel
            await ctx.defer()
            success, message = await self.join_voice(channel)
            await ctx.followup.send(message)
        
        @self.slash_command(name="leave", description="Leave the voice channel")
        async def leave_command(ctx: discord.ApplicationContext) -> None:
            """Slash command to leave voice channel."""
            await ctx.defer()
            success, message = await self.leave_voice(ctx.guild_id)
            await ctx.followup.send(message)
        
        @self.slash_command(name="status", description="Check the bot's current status")
        async def status_command(ctx: discord.ApplicationContext) -> None:
            """Slash command to check bot status."""
            handler = self.get_voice_handler(ctx.guild_id)
            
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
                status = f"**Status:** {state_msg}"
            
            embed = discord.Embed(
                title="🎤 Voice Assistant Status",
                description=status,
                color=discord.Color.blue(),
            )
            embed.add_field(name="Wake Phrase", value=f"`{self.config.wake_phrase_display}`", inline=True)
            embed.add_field(name="Voice", value=self.config.gemini_voice, inline=True)
            
            await ctx.respond(embed=embed)
        
        logger.debug("Slash commands registered: /join, /leave, /status")
    
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
    
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Handle voice state changes."""
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
                
                # Check if we have a handler that's still connecting
                if guild_id in self._voice_handlers:
                    handler = self._voice_handlers[guild_id]
                    
                    # Ignore if handler is in connecting state
                    if handler.is_connecting:
                        logger.debug(f"Ignoring disconnect - handler is still connecting")
                        return
                    
                    # Ignore if the voice client is still connected (spurious event)
                    if handler._voice_client and handler._voice_client.is_connected():
                        logger.debug(f"Ignoring spurious disconnect event - voice client still connected")
                        return
                
                # Real disconnection - clean up
                logger.info(f"Bot disconnected from voice channel '{before.channel.name}' in guild {guild_id}")
                if guild_id in self._voice_handlers:
                    await self._voice_handlers[guild_id].leave_channel()
                    del self._voice_handlers[guild_id]
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
        
        handler = self._voice_handlers[guild_id]
        await handler.leave_channel()
        del self._voice_handlers[guild_id]
        logger.info(f"Left voice channel in guild {guild_id}")
        
        return True, "Left the voice channel. Goodbye!"


# Global bot reference (kept for compatibility)
_bot: Optional[DiscordBot] = None


def set_bot(bot: DiscordBot) -> None:
    """Set the global bot reference."""
    global _bot
    _bot = bot
