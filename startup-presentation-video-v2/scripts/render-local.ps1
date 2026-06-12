$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$ffmpegDir = Join-Path $projectRoot "node_modules\ffmpeg-static"
$ffprobeDir = Join-Path $projectRoot "node_modules\ffprobe-static\bin\win32\x64"

$env:PATH = "$ffmpegDir;$ffprobeDir;$env:PATH"

npx --yes hyperframes@0.6.90 render --quality standard --output renders/startup-profit-presentation-v2.mp4
