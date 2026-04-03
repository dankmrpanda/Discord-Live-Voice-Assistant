import { BotState } from "../types/index.js";

export const STATE_LABELS: Record<BotState, string> = {
  [BotState.IDLE]: "Idle",
  [BotState.CONNECTING]: "Connecting to voice channel...",
  [BotState.LISTENING]: "Listening for wake word",
  [BotState.PROCESSING]: "Processing request...",
  [BotState.SPEAKING]: "Speaking response",
};
