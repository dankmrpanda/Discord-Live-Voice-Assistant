#!/usr/bin/env sh
set -eu

DEV=0
NO_INSTALL=0
SKIP_CHECKS=0

for arg in "$@"; do
  case "$arg" in
    --dev)
      DEV=1
      ;;
    --no-install)
      NO_INSTALL=1
      ;;
    --skip-checks)
      SKIP_CHECKS=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if [ "$NO_INSTALL" -eq 0 ]; then
  echo "==> Installing dependencies"
  npm install
fi

NODE_ARGS=""
[ "$DEV" -eq 1 ] && NODE_ARGS="$NODE_ARGS --dev"
[ "$SKIP_CHECKS" -eq 1 ] && NODE_ARGS="$NODE_ARGS --skip-checks"

node scripts/run-local.mjs --no-install $NODE_ARGS
