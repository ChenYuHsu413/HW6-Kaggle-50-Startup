import os
import json
import shutil

def generate_hyperframes_project(output_dir, project_dir, slide_duration=5.0, transition_duration=0.5):
    """
    讀取切割後的圖層與 metadata.json，自動生成 HyperFrames 影片專案。
    
    :param output_dir: 切割圖層來源目錄 (例如 "output")
    :param project_dir: HyperFrames 專案目標目錄 (例如 "hyperframes_project")
    :param slide_duration: 每張投影片播放秒數 (包含轉場時間)
    :param transition_duration: 投影片之間的淡入轉場時間
    """
    os.makedirs(project_dir, exist_ok=True)
    compositions_dir = os.path.join(project_dir, "compositions")
    os.makedirs(compositions_dir, exist_ok=True)
    
    # 複製切割後的 output 圖層資料夾到專案內，以便相對路徑讀取
    project_output_dir = os.path.join(project_dir, "output")
    if os.path.exists(project_output_dir):
        shutil.rmtree(project_output_dir)
    shutil.copytree(output_dir, project_output_dir)
    
    # 搜尋所有投影片資料夾
    slide_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("slide_")])
    slide_configs = []
    
    for idx, s_dir in enumerate(slide_dirs):
        meta_file = os.path.join(output_dir, s_dir, "metadata.json")
        if not os.path.exists(meta_file):
            continue
            
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # 計算此投影片在主時間軸上的起點 (包含疊加轉場時間)
        start_time = idx * (slide_duration - transition_duration)
        
        slide_configs.append({
            "id": s_dir,
            "start": start_time,
            "duration": slide_duration,
            "width": meta["slide_width"],
            "height": meta["slide_height"],
            "layers": meta["layers"]
        })
        
        # 生成該投影片的 sub-composition HTML
        generate_slide_subcomp(compositions_dir, s_dir, meta)
        
    # 生成主 root index.html
    generate_root_index(project_dir, slide_configs, transition_duration)
    
    print(f"\n[成功] 已成功建立 HyperFrames 專案於: {project_dir}")
    print("您可以執行以下指令進行影片預覽與渲染：")
    print(f"  cd \"{project_dir}\"")
    print("  npx hyperframes preview   # 即時網頁預覽")
    print("  npx hyperframes render    # 渲染為 MP4 影片")


def generate_slide_subcomp(compositions_dir, slide_id, meta):
    """生成投影片子合成的 HTML/GSAP 程式碼"""
    w = meta["slide_width"]
    h = meta["slide_height"]
    
    layers_html = []
    gsap_tweens = []
    
    # 1. 建立背景圖
    layers_html.append(
        f'    <img class="bg-layer" src="../output/{slide_id}/background.png" />'
    )
    
    # 2. 建立各個動畫圖層
    for layer in meta["layers"]:
        layer_id = layer["id"]
        # 使用絕對定位
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
        
        # 3. 根據建議動畫產生 GSAP Tween
        anim = layer["animation"]
        delay = 0.3 + (layer["z_index"] - 1) * 0.3  # 依 z-index 順序延遲載入
        
        if anim == "zoom-in":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ scale: 0.5, opacity: 0, duration: 0.8, ease: "back.out(1.7)" }}, {delay:.2f});'
            )
        elif anim == "slide-down":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ y: -80, opacity: 0, duration: 0.8, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-up":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ y: 80, opacity: 0, duration: 0.8, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-right":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ x: -80, opacity: 0, duration: 0.8, ease: "power3.out" }}, {delay:.2f});'
            )
        elif anim == "slide-left":
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ x: 80, opacity: 0, duration: 0.8, ease: "power3.out" }}, {delay:.2f});'
            )
        else:  # fade-in
            gsap_tweens.append(
                f'      tl.from("#{layer_id}", {{ opacity: 0, duration: 0.8, ease: "power2.out" }}, {delay:.2f});'
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

    subcomp_path = os.path.join(compositions_dir, f"{slide_id}.html")
    with open(subcomp_path, "w", encoding="utf-8") as f:
        f.write(html_template)


def generate_root_index(project_dir, slide_configs, transition_duration):
    """生成主 root 頁面 (index.html)"""
    hosts_html = []
    transitions_js = []
    
    for idx, cfg in enumerate(slide_configs):
        s_id = cfg["id"]
        # 建立嵌入子合成的 div
        hosts_html.append(
            f'    <div id="{s_id}" class="slide-host" '
            f'data-composition-id="{s_id}" '
            f'data-composition-src="compositions/{s_id}.html" '
            f'data-start="{cfg["start"]}" '
            f'data-duration="{cfg["duration"]}" '
            f'data-track-index="1"></div>'
        )
        
        # 如果不是第一張，主時間軸加入淡入轉場動畫 (從上一張重疊處開始淡入)
        if idx > 0:
            transitions_js.append(
                f'    rootTl.from("#{s_id}", {{ opacity: 0, duration: {transition_duration}, ease: "power2.inOut" }}, {cfg["start"]});'
            )

    hosts_str = "\n".join(hosts_html)
    transitions_str = "\n".join(transitions_js)
    
    # 計算影片總長度
    total_duration = 0.0
    if slide_configs:
        last = slide_configs[-1]
        total_duration = last["start"] + last["duration"]
        
    html_content = f"""<!DOCTYPE html>
<html>
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
    }}
    .slide-host {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
    }}
  </style>
</head>
<body>
  <div data-composition-id="root" data-width="1920" data-height="1080" class="main-video">
{hosts_str}
  </div>
  
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <script>
    window.__timelines = window.__timelines || {{}};
    const rootTl = gsap.timeline({{ paused: true }});
    
    // 建立簡報頁面之間的轉場淡入動畫
{transitions_str}
    
    // 如果需要，可以在此處為最後一頁加上全片淡出到黑色的動畫
    // rootTl.to(".main-video", {{ opacity: 0, duration: 1.0 }}, {total_duration - 1.0});
    
    window.__timelines["root"] = rootTl;
  </script>
</body>
</html>
"""

    root_path = os.path.join(project_dir, "index.html")
    with open(root_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# 測試入口
if __name__ == "__main__":
    # 使用範例
    # generate_hyperframes_project("output", "hyperframes_video")
    pass
