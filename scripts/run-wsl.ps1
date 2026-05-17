# Run the Flask app inside WSL2 with GPU + matching TF version.
# Usage:  .\scripts\run-wsl.ps1
# Then open http://127.0.0.1:5000 in a Windows browser.

param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$VenvPath = '~/venvs/emotion',
    [string]$AppHost = '127.0.0.1',
    [int]$Port = 5000
)

$projectPath = (Resolve-Path "$PSScriptRoot/..").Path
$wslPath = ($projectPath -replace '^([A-Za-z]):', { '/mnt/' + $_.Groups[1].Value.ToLower() }) -replace '\\', '/'

$cmd = @"
cd $wslPath && \
TF_USE_LEGACY_KERAS=1 \
TF_CPP_MIN_LOG_LEVEL=2 \
HOST=$AppHost \
PORT=$Port \
$VenvPath/bin/python app.py
"@

Write-Host "Starting Flask app in WSL distro: $Distro" -ForegroundColor Cyan
Write-Host "Project path (WSL): $wslPath" -ForegroundColor Cyan
Write-Host "Open browser at:    http://${AppHost}:$Port" -ForegroundColor Green
Write-Host ""

wsl -d $Distro -e bash -c $cmd
