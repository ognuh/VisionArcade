@echo off
title VisionArcade - NVIDIA RTX 3050
cd /d "%~dp0"

echo Dang mo Chrome tach biet chay tren NVIDIA GeForce RTX 3050...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --user-data-dir="%TEMP%\VisionArcadeRTX3050" ^
  --force_high_performance_gpu ^
  --use-angle=d3d11 ^
  --ignore-gpu-blocklist ^
  --no-first-run ^
  --app=http://localhost:5173
