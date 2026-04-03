#!/usr/bin/env sh
set -eu

for arg in "$@"; do
  case "$arg" in
    --dev|--no-install|--skip-checks)
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

node scripts/run-local.mjs "$@"
