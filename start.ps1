<#
.SYNOPSIS
    One-click startup script for WorkspaceRAG.
.DESCRIPTION
    1. Verifies/starts Docker Desktop.
    2. Runs docker compose up -d for PostgreSQL (pgvector) and FastAPI backend.
    3. Polls the backend health check endpoint until ready.
    4. Launches the Streamlit UI.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       WorkspaceRAG Local Launcher       " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check Docker Daemon
Write-Host "[1/4] Checking Docker daemon..." -ForegroundColor Yellow
$dockerRunning = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerRunning = $true }
} catch {
    $dockerRunning = $false
}

if (-not $dockerRunning) {
    Write-Host "Docker is not running. Launching Docker Desktop..." -ForegroundColor Yellow
    $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerPath) {
        Start-Process $dockerPath
    } else {
        Write-Error "Docker Desktop executable not found. Please start Docker manually."
    }

    Write-Host "Waiting for Docker daemon to become ready (up to 60s)..." -NoNewline
    $timeout = 60
    $elapsed = 0
    while (-not $dockerRunning -and $elapsed -lt $timeout) {
        Start-Sleep -Seconds 3
        $elapsed += 3
        Write-Host "." -NoNewline
        try {
            $null = docker info 2>&1
            if ($LASTEXITCODE -eq 0) { $dockerRunning = $true }
        } catch { }
    }
    Write-Host ""
    if (-not $dockerRunning) {
        Write-Error "Docker daemon did not start in time. Please verify Docker Desktop and retry."
    }
}
Write-Host "Docker daemon is ready." -ForegroundColor Green

# 2. Start Containers
Write-Host "[2/4] Starting PostgreSQL & FastAPI containers..." -ForegroundColor Yellow
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start docker containers."
}

# 3. Wait for Health Check
Write-Host "[3/4] Waiting for backend API to become healthy..." -ForegroundColor Yellow
$apiHealthy = $false
$healthUrl = "http://localhost:8000/api/v1/query/health"
$retries = 30
for ($i = 0; $i -lt $retries; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp.status -eq "ok") {
            $apiHealthy = $true
            break
        }
    } catch { }
    Start-Sleep -Seconds 1
}

if ($apiHealthy) {
    Write-Host "Backend API is online and healthy (http://localhost:8000)." -ForegroundColor Green
} else {
    Write-Warning "Backend API did not respond with 'ok' within 30s. Check logs with: docker compose logs api"
}

# 4. Launch Streamlit UI
Write-Host "[4/4] Opening Streamlit UI in default browser..." -ForegroundColor Yellow
Write-Host "UI is running at http://localhost:8501" -ForegroundColor Cyan
try {
    Start-Process "http://localhost:8501"
} catch {
    Write-Host "Please open http://localhost:8501 in your browser." -ForegroundColor Green
}

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "  WorkspaceRAG is live at http://localhost:8501" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "To view container logs:  docker compose logs -f"
Write-Host "To stop all services:    docker compose down"
Write-Host "Press Enter to exit this launcher window..."
$null = Read-Host
