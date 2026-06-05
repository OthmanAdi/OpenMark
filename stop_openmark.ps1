# OpenMark Stop Script
# Run from PowerShell: .\stop_openmark.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$port = if ($env:OPENMARK_PORT) { [int]$env:OPENMARK_PORT } else { 7860 }

Write-Host ""
Write-Host "  Stopping OpenMark UI on port $port" -ForegroundColor Cyan
Write-Host ""

$connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
$processIds = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)

if ($processIds.Count -eq 0) {
    Write-Host "  No OpenMark UI listener found." -ForegroundColor Green
    exit 0
}

foreach ($processId in $processIds) {
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        Write-Host "  Stopping PID $processId ($($process.ProcessName))..." -ForegroundColor Gray
        Stop-Process -Id $processId -Force
    } catch {
        Write-Host "  Could not stop PID ${processId}: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Start-Sleep -Seconds 1
$remaining = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "  Some listener is still active on port $port." -ForegroundColor Yellow
    exit 1
}

Write-Host "  OpenMark UI stopped." -ForegroundColor Green
