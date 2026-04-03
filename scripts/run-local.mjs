#!/usr/bin/env node
import { copyFileSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const args = new Set(process.argv.slice(2));
const devMode = args.has("--dev");
const noInstall = args.has("--no-install");
const skipChecks = args.has("--skip-checks");

const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";

function runStep(label, command, commandArgs) {
  console.log(`\n==> ${label}`);
  const result = spawnSync(command, commandArgs, {
    stdio: "inherit",
    env: process.env,
    shell: process.platform === "win32",
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function ensureEnvFile() {
  if (existsSync(".env")) {
    return;
  }

  if (existsSync(".env.example")) {
    copyFileSync(".env.example", ".env");
    console.log("Created .env from .env.example.");
    console.log("Fill in your real secrets in .env, then re-run this script.");
    process.exit(1);
  }

  console.error("Missing .env and .env.example.");
  process.exit(1);
}

function checkFfmpeg() {
  const result = spawnSync("ffmpeg", ["-version"], {
    stdio: "ignore",
    env: process.env,
  });

  if (result.status !== 0) {
    console.warn("Warning: ffmpeg not found in PATH. Voice playback may fail.");
  }
}

ensureEnvFile();
checkFfmpeg();

if (!noInstall) {
  runStep("Installing dependencies", npmCmd, ["install"]);
}

if (!skipChecks) {
  runStep("Type checking", npmCmd, ["run", "typecheck"]);
  runStep("Building", npmCmd, ["run", "build"]);
  runStep("Voice dependency healthcheck", npmCmd, ["run", "healthcheck"]);
}

if (devMode) {
  runStep("Starting bot in dev mode", npmCmd, ["run", "dev"]);
} else {
  runStep("Starting bot", npmCmd, ["run", "start"]);
}
