# OpenMark Startup Script
# Run from PowerShell: .\start_openmark.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location -LiteralPath $root

$port = if ($env:OPENMARK_PORT) { [int]$env:OPENMARK_PORT } else { 7860 }
$url = "http://127.0.0.1:$port"
$logDir = Join-Path $root "data"
$stdoutLog = Join-Path $logDir "stdout.log"
$stderrLog = Join-Path $logDir "stderr.log"

function Write-Step($text) {
    Write-Host "  $text" -ForegroundColor Gray
}

function Test-TcpPort($hostName, $portNumber) {
    try {
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $tcp.Connect($hostName, $portNumber)
        $tcp.Close()
        return $true
    } catch {
        return $false
    }
}

function Test-HttpReady($targetUrl) {
    try {
        $response = Invoke-WebRequest -Uri $targetUrl -TimeoutSec 2 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Resolve-Python {
    $candidates = @(
        (Join-Path $root ".venv\Scripts\python.exe"),
        (Join-Path $root "venv\Scripts\python.exe"),
        "python",
        "py"
    )

    foreach ($candidate in $candidates) {
        try {
            $cmd = Get-Command $candidate -ErrorAction Stop
            return $cmd.Source
        } catch {}
    }

    throw "Could not find Python. Install Python or create .venv, then run this again."
}

Write-Host ""
Write-Host "  OpenMark Personal Knowledge Graph" -ForegroundColor Cyan
Write-Host "  ---------------------------------" -ForegroundColor DarkGray
Write-Host ""

Write-Step "[1/4] Checking Neo4j..."
if (-not (Test-TcpPort "127.0.0.1" 7687)) {
    Write-Host ""
    Write-Host "  Neo4j is not running." -ForegroundColor Yellow
    Write-Host "  Start the OpenMark database in Neo4j Desktop, then run this again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-Host "        Neo4j bolt://127.0.0.1:7687 is running" -ForegroundColor Green

Write-Step "[2/4] Checking existing UI..."
if (Test-HttpReady $url) {
    Write-Host "        UI already running at $url" -ForegroundColor Green
    Start-Process $url
    exit 0
}

Write-Step "[3/4] Resolving Python..."
$python = Resolve-Python
Write-Host "        $python" -ForegroundColor Green

if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Write-Step "[4/4] Starting UI..."
Write-Host "        Logs: $stdoutLog and $stderrLog" -ForegroundColor DarkGray

$process = Start-Process -FilePath $python `
    -ArgumentList "-m", "openmark.ui.app" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$ready = $false
for ($i = 1; $i -le 45; $i++) {
    Start-Sleep -Seconds 2
    if ($process.HasExited) {
        Write-Host ""
        Write-Host "  UI process exited early with code $($process.ExitCode)." -ForegroundColor Red
        Write-Host "  Check logs:" -ForegroundColor Yellow
        Write-Host "    $stdoutLog"
        Write-Host "    $stderrLog"
        exit $process.ExitCode
    }

    if (Test-HttpReady $url) {
        $ready = $true
        break
    }

    Write-Host "        Waiting... ($($i * 2)s)" -ForegroundColor DarkGray
}

if (-not $ready) {
    Write-Host ""
    Write-Host "  UI did not become ready in time, but process $($process.Id) is still running." -ForegroundColor Yellow
    Write-Host "  Open $url manually or check logs." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  OpenMark is ready." -ForegroundColor Green
Write-Host "  $url" -ForegroundColor Cyan
Write-Host ""
Start-Process $url
