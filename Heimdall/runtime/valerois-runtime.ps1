# Valerois Runtime: starts the Bifrost-patched llama-server as a local OpenAI-compatible endpoint (Windows).
#
#   .\valerois-runtime.ps1 model.gguf                     # http://127.0.0.1:1234/v1  (Nova: Settings > Models > Local)
#   .\valerois-runtime.ps1 model.gguf -Port 8080 -Ctx 65536
#   .\valerois-runtime.ps1 model.gguf -Extra '--temp','0.6'   # passed to llama-server unchanged
#
# Environment: VALEROIS_MODEL, VALEROIS_PORT, VALEROIS_CTX, GGML_VK_VISIBLE_DEVICES (pick a GPU).
param(
    [Parameter(Position = 0)] [string] $Model = $env:VALEROIS_MODEL,
    [int]      $Port = $(if ($env:VALEROIS_PORT) { [int]$env:VALEROIS_PORT } else { 1234 }),
    [string]   $HostName = "127.0.0.1",
    [int]      $Ctx = $(if ($env:VALEROIS_CTX) { [int]$env:VALEROIS_CTX } else { 0 }),
    [int]      $Checkpoints = 4,
    [string]   $ApiKey = "",
    [string]   $Bin = "",
    [string[]] $Extra = @()
)
$ErrorActionPreference = "Stop"

if (-not $Model) { Write-Error "give a .gguf model: .\valerois-runtime.ps1 model.gguf" }
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model" }

if (-not $Bin) {
    foreach ($d in @($PSScriptRoot, (Join-Path $PSScriptRoot "bin"))) {
        if (Test-Path (Join-Path $d "llama-server.exe")) { $Bin = $d; break }
    }
}
$server = Join-Path $Bin "llama-server.exe"
if (-not (Test-Path $server)) { Write-Error "llama-server.exe not found (use -Bin DIR)" }

# Bifrost layers keep a constant state, so a long context costs no extra memory. Other GGUFs grow a KV cache.
$name = [System.IO.Path]::GetFileName($Model)
if ($Ctx -le 0) { $Ctx = $(if ($name.ToLower().Contains("bifrost")) { 131072 } else { 32768 }) }

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Error "port $Port is in use (LM Studio?). Pick another one: -Port 1235" }

$args_ = @("-m", $Model, "-ngl", "99", "-c", "$Ctx", "-np", "1", "--jinja", "--host", $HostName, "--port", "$Port",
           "--alias", [System.IO.Path]::GetFileNameWithoutExtension($name), "--ctx-checkpoints", "$Checkpoints")
if ($ApiKey) { $args_ += @("--api-key", $ApiKey) }
$args_ += $Extra

Write-Host "Valerois Runtime: $name | context $Ctx | http://${HostName}:$Port/v1"
Write-Host "  Nova: Settings > Models > Local (port 1234), or 'Your own server' with the URL above."
& $server @args_
exit $LASTEXITCODE
