# OpenMark Startup Script
# Run this to start the full stack: checks Neo4j, starts UI, opens browser
# Usage: Right-click -> Run with PowerShell
#        Or: .\start_openmark.ps1

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "  OpenMark — Personal Knowledge Graph" -ForegroundColor Cyan
Write-Host "  ─────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""

# ── 1. Check Neo4j is running ──────────────────────────────────────────────
Write-Host "  [1/3] Checking Neo4j..." -ForegroundColor Gray
$neo4jUp = $false
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect("127.0.0.1", 7687)
    $tcp.Close()
    $neo4jUp = $true
    Write-Host "        Neo4j bolt://127.0.0.1:7687 — RUNNING" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "  ⚠  Neo4j is NOT running." -ForegroundColor Yellow
    Write-Host "     Open Neo4j Desktop → start the 'openmark' instance." -ForegroundColor Yellow
    Write-Host "     Then run this script again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to exit"
    exit 1
}

# ── 2. Check if UI already running ────────────────────────────────────────
Write-Host "  [2/3] Checking for existing UI..." -ForegroundColor Gray
$portInUse = $false
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:7860/" -TimeoutSec 2 -ErrorAction Stop
    $portInUse = $true
} catch {}

if ($portInUse) {
    Write-Host "        UI already running at http://127.0.0.1:7860" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Opening browser..." -ForegroundColor Cyan
    Start-Process "http://127.0.0.1:7860"
    exit 0
}

# ── 3. Start the UI ────────────────────────────────────────────────────────
Write-Host "  [3/3] Starting OpenMark UI..." -ForegroundColor Gray
Write-Host "        (loading pplx model + Neo4j connection — ~20 seconds)" -ForegroundColor DarkGray

$uiProcess = Start-Process -FilePath "C:\Python314\python.exe" `
    -ArgumentList "-m", "openmark.ui.app" `
    -WorkingDirectory $PSScriptRoot `
    -PassThru `
    -WindowStyle Hidden

# Wait for UI to be ready (poll port 7860)
$ready = $false
$tries = 0
while (-not $ready -and $tries -lt 30) {
    Start-Sleep -Seconds 2
    $tries++
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860/" -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true }
    } catch {}
    Write-Host "        Waiting... ($($tries * 2)s)" -ForegroundColor DarkGray -NoNewline
    Write-Host "`r" -NoNewline
}

if ($ready) {
    Write-Host ""
    Write-Host "  ✅  OpenMark is ready!" -ForegroundColor Green
    Write-Host "      http://127.0.0.1:7860" -ForegroundColor Cyan
    Write-Host ""
    Start-Process "http://127.0.0.1:7860"
} else {
    Write-Host ""
    Write-Host "  ⚠  UI didn't start in time. Check for errors." -ForegroundColor Yellow
    Write-Host "     Try: python -m openmark.ui.app" -ForegroundColor Gray
}
