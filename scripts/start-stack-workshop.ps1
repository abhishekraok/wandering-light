param(
    [int]$Port = 8018,
    [switch]$NoOpen,
    [switch]$Rebuild,
    [switch]$InstallPolicy,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$workshopRoot = Split-Path -Parent $PSScriptRoot
$workshopUrl = "http://127.0.0.1:$Port/"

# Reuse the app if it is already running; do not terminate any existing process.
$runningApp = $null
try {
    $runningApp = Invoke-RestMethod -Uri ($workshopUrl + 'api/context') -TimeoutSec 2
    if ($runningApp.lessons[0].id -eq 'first-light' -and -not $Rebuild -and -not $InstallPolicy) {
        if ($Checkpoint) {
            $modelResult = Invoke-RestMethod -Uri ($workshopUrl + 'api/model/load') -Method Post -ContentType 'application/json' -Body (@{checkpoint=$Checkpoint} | ConvertTo-Json)
            if (-not $modelResult.ready) { Write-Warning $modelResult.error }
        }
        if (-not $NoOpen) { Start-Process $workshopUrl }
        Write-Host "Signal House is already running at $workshopUrl"
        return
    }
} catch { }

$workshopPython = $null
$workshopVenv = Join-Path $workshopRoot '.venv-workshop312'
$workshopVenvPython = Join-Path $workshopVenv 'Scripts\python.exe'
if (Test-Path -LiteralPath $workshopVenvPython) { $workshopPython = $workshopVenvPython }
if (-not $workshopPython) {
    $pythonCandidates = @()
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $pythonCandidates += $pythonCommand.Source }
    $pythonCandidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    foreach ($candidate in $pythonCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $workshopPython = $candidate; break }
        }
    }
}
if (-not $workshopPython) { throw 'Install Python 3.12 or newer, then run this launcher again.' }

Push-Location $workshopRoot
try {
    if ($InstallPolicy) {
        if (-not (Test-Path -LiteralPath $workshopVenvPython)) {
            & $workshopPython -m venv $workshopVenv
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated policy environment.' }
        }
        $workshopPython = $workshopVenvPython
        & $workshopPython -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the CPU model runtime.' }
        & $workshopPython -m pip install -r wandering_light/stack_workshop/requirements-policy.txt
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the policy dependencies.' }
    }
    $staticEntry = Join-Path $workshopRoot 'wandering_light\stack_workshop\static\index.html'
    if ($Rebuild -or -not (Test-Path -LiteralPath $staticEntry)) {
        $nodeCandidates = @()
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        if ($nodeCommand) { $nodeCandidates += $nodeCommand.Source }
        $nodeCandidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        $workshopNode = $null
        foreach ($candidate in $nodeCandidates) {
            if (Test-Path -LiteralPath $candidate) {
                & $candidate -e 'const [a,b]=process.versions.node.split(String.fromCharCode(46)).map(Number);process.exit(a>22||(a===22&&b>=12)?0:1)' 2>$null
                if ($LASTEXITCODE -eq 0) { $workshopNode = $candidate; break }
            }
        }
        if (-not $workshopNode) { throw 'Install Node.js 22.12 or newer to build the editor.' }
        Push-Location (Join-Path $workshopRoot 'web\stack-workshop')
        try {
            if (-not (Test-Path -LiteralPath 'node_modules\vite\bin\vite.js')) {
                $npmCommand = Get-Command npm.cmd -ErrorAction Stop
                & $npmCommand.Source ci
                if ($LASTEXITCODE -ne 0) { throw 'Could not install the editor dependencies.' }
            }
            & $workshopNode node_modules/typescript/bin/tsc --noEmit
            if ($LASTEXITCODE -ne 0) { throw 'The editor type check failed.' }
            & $workshopNode node_modules/vite/bin/vite.js build
            if ($LASTEXITCODE -ne 0) { throw 'The editor build failed.' }
        } finally { Pop-Location }
    }
    if ($runningApp -and $runningApp.lessons[0].id -eq 'first-light') {
        Write-Host "Updated the local files. Reload $workshopUrl to see frontend changes."
        if ($InstallPolicy) { Write-Host 'Restart the existing workshop terminal to use newly installed policy dependencies.' }
        if (-not $NoOpen) { Start-Process $workshopUrl }
        return
    }
    $launchArgs = @('-m', 'wandering_light.stack_workshop.server', '--port', "$Port")
    if (-not $NoOpen) { $launchArgs += '--open' }
    if ($Checkpoint) { $launchArgs += @('--checkpoint', $Checkpoint) }
    Write-Host "Signal House: $workshopUrl"
    Write-Host 'Keep this terminal open. Press Ctrl+C to stop this app.'
    & $workshopPython @launchArgs
    if ($LASTEXITCODE -ne 0) { throw 'The workshop service stopped with an error.' }
} finally { Pop-Location }
