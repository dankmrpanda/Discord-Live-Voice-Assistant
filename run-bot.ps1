param(
  [switch]$Dev,
  [switch]$NoInstall,
  [switch]$SkipChecks
)

$ErrorActionPreference = 'Stop'

$ArgsList = @()
if ($Dev) { $ArgsList += '--dev' }
if ($NoInstall) { $ArgsList += '--no-install' }
if ($SkipChecks) { $ArgsList += '--skip-checks' }

node scripts/run-local.mjs @ArgsList
exit $LASTEXITCODE
