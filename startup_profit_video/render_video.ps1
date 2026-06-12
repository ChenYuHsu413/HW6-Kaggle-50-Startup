$ErrorActionPreference = "Stop"

# 使用已經存在於 V2 專案中的 ffmpeg 和 ffprobe 靜態二進位檔案
$ffmpegDir = "d:\AI Class ChenYu\AIClass\hw6\startup-presentation-video-v2\node_modules\ffmpeg-static"
$ffprobeDir = "d:\AI Class ChenYu\AIClass\hw6\startup-presentation-video-v2\node_modules\ffprobe-static\bin\win32\x64"

$env:PATH = "$ffmpegDir;$ffprobeDir;$env:PATH"

# 確保輸出路徑資料夾存在
New-Item -ItemType Directory -Force -Path "renders" | Out-Null

Write-Output "開始執行 HyperFrames 影像渲染與合成..."
npx --yes hyperframes@0.6.90 render --quality standard --output renders/startup_profit_presentation.mp4
Write-Output "影片渲染成功！檔案已輸出至: renders/startup_profit_presentation.mp4"
