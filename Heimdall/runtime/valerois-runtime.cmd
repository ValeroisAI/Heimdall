@echo off
rem Valerois Runtime (Windows): double-click friendly wrapper around valerois-runtime.ps1
rem   valerois-runtime.cmd model.gguf [-Port 1234] [-Ctx 131072]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0valerois-runtime.ps1" %*
