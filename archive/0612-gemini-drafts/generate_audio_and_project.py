import os
import json
import subprocess
import shutil
from tinytag import TinyTag

# 定義 12 頁投影片的旁白講稿 (繁體中文，適合輕快女聲，節奏分明)
NARRATIONS = {
    1: "歡迎收看。今天我們將以五十家新創公司的資料為基礎，利用機器學習預測企業獲利，探討如何做出更好的商業決策。",
    2: "我們的核心商業目標很簡單，就是要找出哪些研發、行政或行銷的投資，才是推動企業獲利成長的關鍵槓桿。",
    3: "本專案遵循標準的 CRISP-DM 流程，從商業理解開始，歷經資料準備、建模與評估，確保分析邏輯的嚴謹性。",
    4: "在資料準備階段，我們使用 Pipeline 和 ColumnTransformer，對地區資料進行獨熱編碼，建立多重線性迴歸模型。",
    5: "為了驗證特徵的影響力，我們設計了兩條分析路徑。一條是人類的商業經驗，另一條是機器的自動特徵選擇演算法。",
    6: "在人類經驗的路徑中，我們循序漸進地設計了五組實驗，逐步增加變數，比對 RMSE 和 R 平方的變化。",
    7: "而在演算法路徑，我們同時測試了包含 Filter、Wrapper 和 Embedded 等五種特徵選擇方法，來交叉驗證結果。",
    8: "實驗的結果非常一致，兩條路徑全部收斂到同一個結論，那就是研發支出是獲利最強大且最穩定的單一預測特徵。",
    9: "我們的實驗也證實，放入過多的特徵不一定會提升模型性能，複雜度反而會增加測試集的雜訊與誤差。",
    10: "從特徵權重來看，研發支出是核心驅動器，行銷支出是輔助放大器，而行政支出與地區資料，則要保守看待。",
    11: "最後，資料量雖小，但隱含的商業洞察巨大。在建立模型時，選擇能真正說對故事、有合理商業邏輯的特徵才是最關鍵的。",
    12: "報告到此結束。謝謝大家收看這份關於新創公司獲利分析的報告。我們下次見！"
}

def generate_tts_audios(project_dir):
    """使用 edge-tts 生成所有講稿的 MP3 音檔 (帶有快取機制)"""
    print("=================== 開始生成旁白語音 ===================")
    audio_dir = os.path.join(project_dir, "assets", "narration")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_durations = {}
    
    for idx, text in NARRATIONS.items():
        filename = f"slide_{idx:02d}.mp3"
        filepath = os.path.join(audio_dir, filename)
        
        # 快取：如果音檔已存在且大小大於 0，直接讀取長度，不重複下載
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                tag = TinyTag.get(filepath)
                audio_durations[idx] = tag.duration
                print(f"  使用快取 slide_{idx:02d}.mp3 -> {tag.duration:.2f} 秒")
                continue
            except Exception:
                pass
        
        # 呼叫 edge-tts 生成台灣女聲 HsiaoChenNeural，語速增快 10%
        print(f"正在生成 slide_{idx:02d} 的語音 (HsiaoChen)...")
        cmd = [
            "edge-tts",
            "--voice", "zh-TW-HsiaoChenNeural",
            "--rate", "+10%",
            "--text", text,
            "--write-media", filepath
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tag = TinyTag.get(filepath)
            audio_durations[idx] = tag.duration
            print(f"  成功！音檔長度: {tag.duration:.2f} 秒")
        except Exception as e:
            print(f"  [錯誤] 生成 slide_{idx:02d} 語音失敗: {e}")
            audio_durations[idx] = len(text) * 0.25
            
    print("=================== 旁白語音生成完畢 ===================")
    return audio_durations

def build_hyperframes_project(output_dir, project_dir):
    """建立包含語音與分層動畫的 HyperFrames 影片專案"""
    print("=================== 開始建立 HyperFrames 專案 ===================")
    os.makedirs(project_dir, exist_ok=True)
    compositions_dir = os.path.join(project_dir, "compositions")
    os.makedirs(compositions_dir, exist_ok=True)
    
    # 複製去背好的圖層到專案內
    project_output_dir = os.path.join(project_dir, "output")
    if os.path.exists(project_output_dir):
        shutil.rmtree(project_output_dir)
    shutil.copytree(output_dir, project_output_dir)
    
    # 產生音檔並取得其長度
    audio_durations = generate_tts_audios(project_dir)
    
    slide_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("slide_")])
    slide_configs = []
    
    current_time = 0.0
    transition_duration = 0.5  # 轉場淡入淡出秒數
    hold_after_voice = 0.8     # 語音結束後留白秒數
    
    for idx, s_dir in enumerate(slide_dirs):
        slide_idx = idx + 1
        meta_file = os.path.join(output_dir, s_dir, "metadata.json")
        if not os.path.exists(meta_file):
            continue
            
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        voice_dur = audio_durations.get(slide_idx, 5.0)
        dur = voice_dur + hold_after_voice
        
        # 考慮前一頁的淡入重疊 (除了第一頁)
        start_time = current_time
        if idx > 0:
            start_time -= transition_duration
            
        slide_configs.append({
            "id": s_dir,
            "index": slide_idx,
            "start": start_time,
            "duration": dur,
            "voice_duration": voice_dur,
            "width": meta["slide_width"],
            "height": meta["slide_height"],
            "layers": meta["layers"]
        })
        
        current_time = start_time + dur
        
        # 產生子合成 HTML
        generate_slide_subcomp(compositions_dir, s_dir, meta, voice_dur)
        
    # 產生主 index.html，並載入音軌
    generate_root_index(project_dir, slide_configs, transition_duration)
    print("=================== HyperFrames 專案建立完成 ===================")

def generate_slide_subcomp(compositions_dir, slide_id, meta, voice_dur):
    """為單個投影片生成子合成"""
    w = meta["slide_width"]
    h = meta["slide_height"]
    
    layers_html = []
    gsap_tweens = []
    
    layers_html.append(
        f'    <img class="bg-layer" src="../output/{slide_id}/background.png" />'
    )
    
    for layer in meta["layers"]:
        layer_id = layer["id"]
        style = (
            f"position: absolute; "
            f"left: {layer['x']}px; "
            f"top: {layer['y']}px; "
            f"width: {layer['width']}px; "
            f"height: {layer['height']}px; "
            f"z-index: {layer['z_index']};"
        )
        layers_html.append(
            f'    <img id="{layer_id}" class="layer" src="../output/{slide_id}/{layer["file"]}" style="{style}" />'
        )
        
        anim = layer["animation"]
        delay = 0.2 + (layer["z_index"] - 1) * 0.25
        
        if anim == "zoom-in":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ scale: 0.6, opacity: 0, duration: 0.7, ease: "back.out(1.5)" }}, {delay:.2f});'
            )
        elif anim == "slide-down":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ y: -60, opacity: 0, duration: 0.7, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-up":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ y: 60, opacity: 0, duration: 0.7, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-right":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ x: -60, opacity: 0, duration: 0.7, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-left":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ x: 60, opacity: 0, duration: 0.7, ease: "power3.out" }}, {delay:.2f});'
            )
        else:  # fade-in
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ opacity: 0, duration: 0.7, ease: "power2.out" }}, {delay:.2f});'
            )

    layers_str = "\n".join(layers_html)
    tweens_str = "\n".join(gsap_tweens)
    
    html_template = f"""<template id="{slide_id}-template">
  <div data-composition-id="{slide_id}" data-width="{w}" data-height="{h}" class="slide-container">
    
{layers_str}

    <style>
      .slide-container {{
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
      }}
      .bg-layer {{
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
      }}
      .layer {{
        transform-origin: center;
      }}
    </style>
    
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      
{tweens_str}
      
      window.__timelines["{slide_id}"] = tl;
    </script>
  </div>
</template>
"""
    with open(os.path.join(compositions_dir, f"{slide_id}.html"), "w", encoding="utf-8") as f:
        f.write(html_template)

def generate_root_index(project_dir, slide_configs, transition_duration):
    """產生主 index.html，包含音軌與轉場，並修復 lint 錯誤"""
    hosts_html = []
    audio_html = []
    transitions_js = []
    
    for idx, cfg in enumerate(slide_configs):
        s_id = cfg["id"]
        # 交替使用軌道軌索引 (Track 1 和 Track 2) 以免在同一軌上重疊時觸發 linter 衝突
        track_idx = 1 if idx % 2 == 0 else 2
        
        # 1. 建立投影片容器
        hosts_html.append(
            f'    <div id="{s_id}" class="slide-host" '
            f'data-composition-id="{s_id}" '
            f'data-composition-src="compositions/{s_id}.html" '
            f'data-start="{cfg["start"]:.2f}" '
            f'data-duration="{cfg["duration"]:.2f}" '
            f'data-track-index="{track_idx}"></div>'
        )
        
        # 2. 建立旁白音軌 (對齊各頁的開始時間)
        audio_filename = f"slide_{cfg['index']:02d}.mp3"
        audio_id = f"audio_{cfg['index']:02d}"
        audio_html.append(
            f'    <audio id="{audio_id}" class="clip" '
            f'src="assets/narration/{audio_filename}" '
            f'data-start="{cfg["start"]:.2f}" '
            f'data-duration="{cfg["voice_duration"]:.2f}" '
            f'data-track-index="{idx + 5}" data-volume="1"></audio>'
        )
        
        # 3. 轉場淡入動畫 (如果不是第一頁)
        if idx > 0:
            transitions_js.append(
                f'    rootTl.from("#{s_id}", {{ opacity: 0, duration: {transition_duration}, ease: "power2.inOut" }}, {cfg["start"]:.2f});'
            )

    hosts_str = "\n".join(hosts_html)
    audio_str = "\n".join(audio_html)
    transitions_str = "\n".join(transitions_js)
    
    # 計算影片總長度
    total_duration = 0.0
    if slide_configs:
        last = slide_configs[-1]
        total_duration = last["start"] + last["duration"]
        
    html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8">
  <title>PowerPoint HyperFrames Video</title>
  <style>
    body {{
      margin: 0;
      background: #000;
      overflow: hidden;
    }}
    .main-video {{
      position: relative;
      width: 1920px;
      height: 1080px;
      background: #0f141c;
    }}
    .slide-host {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
    }}
    audio {{
      display: none;
    }}
  </style>
</head>
<body>
  <!-- 根合成加入 data-start="0" -->
  <div data-composition-id="root" data-start="0" data-width="1920" data-height="1080" class="main-video">
{hosts_str}
{audio_str}
  </div>
  
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <script>
    window.__timelines = window.__timelines || {{}};
    const rootTl = gsap.timeline({{ paused: true }});
    
    // 投影片轉場淡入動畫
{transitions_str}
    
    // 結尾時整部影片淡出到黑色
    rootTl.to(".main-video", {{ opacity: 0, duration: 1.0 }}, {total_duration - 1.0:.2f});
    
    window.__timelines["root"] = rootTl;
  </script>
</body>
</html>
"""

    with open(os.path.join(project_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # 同步產生 package.json 方便直接 npm run dev
    package_json = {
      "name": "startup-profit-video",
      "version": "1.0.0",
      "scripts": {
        "dev": "npx hyperframes preview",
        "render": "npx hyperframes render --output renders/startup_profit_presentation.mp4"
      }
    }
    with open(os.path.join(project_dir, "package.json"), "w", encoding="utf-8") as f:
        json.dump(package_json, f, indent=2)

if __name__ == "__main__":
    build_hyperframes_project("output", "startup_profit_video")
