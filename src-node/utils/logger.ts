import fs from "node:fs";
import path from "node:path";
import pino, { type Logger, type LoggerOptions } from "pino";

let rootLogger: Logger | undefined;

const LEVELS = new Set(["trace", "debug", "info", "warn", "error", "fatal"]);
const SERVICE_NAME = "discord-live-vc-bot";
const REDACT_PATHS = [
  "discordBotToken",
  "discordApplicationId",
  "geminiApiKey",
  "*.discordBotToken",
  "*.discordApplicationId",
  "*.geminiApiKey",
];

export function createLogger(level: string, directory: string): Logger {
  const normalizedLevel = LEVELS.has(level) ? level : "info";

  fs.mkdirSync(directory, { recursive: true });
  const sessionName = `session_${new Date().toISOString().replace(/[:.]/g, "-")}.log`;
  const sessionPath = path.join(directory, sessionName);

  const options: LoggerOptions = {
    name: SERVICE_NAME,
    base: { service: SERVICE_NAME },
    level: normalizedLevel,
    timestamp: pino.stdTimeFunctions.isoTime,
    serializers: {
      err: pino.stdSerializers.err,
    },
    redact: {
      paths: REDACT_PATHS,
      censor: "[REDACTED]",
    },
    formatters: {
      level: (label) => ({ level: label }),
    },
  };

  const streams = [
    { stream: pino.destination({ sync: true, fd: 1 }) },
    { stream: pino.destination({ sync: true, dest: sessionPath }) },
  ];

  rootLogger = pino(options, pino.multistream(streams));
  rootLogger.info({ sessionPath }, "Logger initialized");
  return rootLogger;
}

export function getLogger(name?: string): Logger {
  if (!rootLogger) {
    rootLogger = pino({
      name: SERVICE_NAME,
      base: { service: SERVICE_NAME },
      level: "info",
      timestamp: pino.stdTimeFunctions.isoTime,
      serializers: {
        err: pino.stdSerializers.err,
      },
      redact: {
        paths: REDACT_PATHS,
        censor: "[REDACTED]",
      },
    });
  }
  return name ? rootLogger.child({ module: name }) : rootLogger;
}

export function logException(
  logger: Logger,
  message: string,
  error: unknown,
  context?: Record<string, unknown>,
): void {
  if (error instanceof Error) {
    logger.error({ ...context, err: error }, message);
    return;
  }
  logger.error({ ...context, error }, message);
}

