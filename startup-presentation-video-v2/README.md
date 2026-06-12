# Startup Presentation Video V2

50 Startups CRISP-DM 簡報影片第二版。保留手繪風 slide 素材，改用動態運鏡與精簡敘事。

與 V1（`../startup-presentation-video-pptx/`）的差異：

| | V1 | V2 |
|---|---|---|
| 長度 | 165 秒（11 段） | 76.5 秒（6 段） |
| 畫面 | 固定卡片 + 區塊漸次揭示 | 全幅 slide + Ken Burns 運鏡 |
| 轉場 | 卡片翻入 | 擦入 / 推移 / zoom-through / whip-pan 快切 |
| 旁白 | 舊 TTS（zhiwei） | Edge TTS `zh-TW-YunJheNeural`（台灣中文男聲） |
| 講稿 | 11 段逐 slide 解說 | 6 段故事線：鉤子 → 問題 → 兩條路 → 實驗 → 收斂 → 結論 |

## 結構

- `index.html` — 主 composition（6 景、運鏡、字幕、印章動畫）
- `assets/slides/` — 手繪 slide PNG（取自 V1 的 8 張）
- `assets/narration/` — Edge TTS 生成的 6 段 mp3
- `assets/narration-script.md` — 講稿與字幕全文
- `renders/` — 輸出 MP4

## 重新生成旁白

```powershell
pip install edge-tts
edge-tts --voice zh-TW-YunJheNeural --rate=+8% --text "講稿內容" --write-media assets/narration/seg-01.mp3
```

## 指令

需要 Node >= 22（`nvm use 24.14.0`）。

```bash
npm install          # ffmpeg-static / ffprobe-static
npm run check        # lint + validate + inspect
npm run dev          # 預覽（背景長駐）
npm run render:local # 渲染 MP4 到 renders/
```

## 場景時間軸

| 景 | 時間 | Slide | 運鏡 |
|---|---|---|---|
| 1 鉤子 | 0–13.2s | 01 火箭標題 | 高空落下 + 緩推 |
| 2 問題 | 12.7–24.5s | 02 獲利機器 | 右向左橫搖 |
| 3 兩條路 | 24.0–37.3s | 05 路標 | 拉遠揭示 |
| 4 實驗 | 36.8–47.6s | 06→07 | 左推入，42.2s whip-pan 快切 |
| 5 收斂 | 47.1–61.4s | 08 獎盃 | zoom-through + 紅印章 SLAM（53.6s） |
| 6 結論 | 60.9–76.5s | 10→11 | 柔焦溶接 + 淡出 |
