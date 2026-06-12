"""
Build a HyperFrames video from Startup_Profit_Code.pdf.

Pipeline:
1. Render each PDF page to PNG with PyMuPDF at 150 DPI.
2. Detect major slide elements with dynamic Otsu thresholding and dilation.
3. Cut each element into a transparent PNG with rembg, then inpaint the plate.
4. Generate upbeat Traditional Chinese narration with edge-tts.
5. Measure MP3 durations with tinytag.
6. Generate HyperFrames sub-compositions and a root index.html with GSAP.

The generated project is deterministic: no random animation, finite timelines,
paused GSAP timelines registered in window.__timelines, and alternating slide
tracks to avoid overlapping-track lint errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
from PIL import Image
from tinytag import TinyTag

try:
    import edge_tts
except ImportError:  # pragma: no cover - handled at runtime
    edge_tts = None

remove = None
new_session = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "sources" / "Startup_Profit_Code.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "startup-profit-code-video"
FPS = 30
DPI = 150
MAX_ELEMENTS = 8
OVERLAP_SECONDS = 0.5
TAIL_PADDING_SECONDS = 0.8
VOICE = "zh-TW-HsiaoChenNeural"
RATE = "+10%"


NARRATION_SCRIPTS = [
    "大家好，這支影片會快速看完新創公司獲利預測專案，從資料、模型到特徵選擇，一步步連起來。",
    "資料來自 Kaggle 五十家新創公司。目標是用研發、行銷、行政支出與州別，預測公司獲利。",
    "商業理解的核心問題是：哪一種資源投入，最能解釋利潤？這會決定後面建模與解讀方向。",
    "資料理解會檢查欄位、缺失值、重複值與統計摘要。因為只有五十筆資料，每個結論都要保守。",
    "資料準備時，數值欄位直接保留，州別用 OneHotEncoder，讓線性迴歸能處理類別資訊。",
    "模型維持 Linear Regression，重點是可解釋性，讓我們能說明投入金額如何影響獲利。",
    "評估不只看 R squared，也看 adjusted R squared、MAE、MSE 與 RMSE，避免過度高估複雜模型。",
    "結果顯示，R&D Spend 幾乎一直是最強訊號，代表產品開發與技術能力是獲利核心。",
    "Marketing Spend 是輔助放大器，可能增加市場曝光，但仍要和 R&D 一起判斷。",
    "Administration 要保守解讀。它可能是管理能力，也可能只是營運成本，需要指標佐證。",
    "州別可以當作區域控制變數，但樣本很少，不能過度推論地區差異。",
    "最後建議：以 R&D Spend 為核心，搭配 Marketing Spend 與必要控制變數，用簡潔模型支援決策。",
]


@dataclass
class Element:
    index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    animation: str
    z_index: int
    image: str


@dataclass
class Slide:
    number: int
    width: int
    height: int
    duration: float
    narration: str
    audio: str
    background: str
    elements: list[Element]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(output: Path) -> dict[str, Path]:
    dirs = {
        "root": output,
        "assets": output / "assets",
        "raw": output / "assets" / "raw_slides",
        "slides": output / "assets" / "slides",
        "audio": output / "assets" / "narration",
        "compositions": output / "compositions",
        "renders": output / "renders",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def render_pdf_pages(pdf_path: Path, raw_dir: Path) -> list[Path]:
    doc = fitz.open(pdf_path)
    page_paths = []
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        path = raw_dir / f"slide_{page_index:02d}.png"
        pix.save(path)
        image = Image.open(path).convert("RGB")
        width, height = image.size
        even_width = width + (width % 2)
        even_height = height + (height % 2)
        if (even_width, even_height) != (width, height):
            padded = Image.new("RGB", (even_width, even_height), image.getpixel((0, 0)))
            padded.paste(image, (0, 0))
            padded.save(path)
        page_paths.append(path)
    return page_paths


def dynamic_otsu_mask(gray: np.ndarray) -> np.ndarray:
    median = float(np.median(gray))
    if median >= 128:
        threshold_mode = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        threshold_mode = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, 0, 255, threshold_mode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 50))
    return cv2.dilate(binary, kernel, iterations=1)


def suggest_animation(x: int, y: int, width: int, height: int, canvas_w: int, canvas_h: int) -> str:
    cx = x + width / 2
    cy = y + height / 2
    if cy < canvas_h * 0.28:
        return "slide-down"
    if cy > canvas_h * 0.72:
        return "slide-up"
    if cx < canvas_w * 0.35:
        return "slide-right"
    if cx > canvas_w * 0.65:
        return "slide-left"
    return "scale-fade"


def find_elements(image_bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dilated = dynamic_otsu_mask(gray)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    canvas_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 60 or h < 30:
            continue
        if area > canvas_area * 0.92:
            continue
        pad_x = min(24, x)
        pad_y = min(20, y)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(width - x, w + pad_x * 2)
        h = min(height - y, h + pad_y * 2)
        boxes.append((x, y, w, h, w * h))

    boxes.sort(key=lambda item: item[4], reverse=True)
    boxes = boxes[:MAX_ELEMENTS]
    boxes.sort(key=lambda item: (item[1], item[0]))
    return boxes


def fallback_transparent_crop(crop_bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    if median >= 128:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba)


def load_rembg():
    global remove, new_session
    if remove is not None and new_session is not None:
        return remove, new_session
    from rembg import remove as loaded_remove, new_session as loaded_new_session

    remove = loaded_remove
    new_session = loaded_new_session
    return remove, new_session


def remove_background(crop_bgr: np.ndarray, session, use_rembg: bool) -> Image.Image:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_crop = Image.fromarray(crop_rgb)
    if not use_rembg or remove is None or session is None:
        return fallback_transparent_crop(crop_bgr)
    try:
        result = remove(pil_crop, session=session)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        alpha = np.array(result.getchannel("A"))
        if alpha.max() < 8:
            return fallback_transparent_crop(crop_bgr)
        return result
    except Exception:
        return fallback_transparent_crop(crop_bgr)


def process_slide_layers(
    raw_paths: Iterable[Path], slides_dir: Path, use_rembg: bool
) -> list[dict]:
    rembg_session = None
    if use_rembg:
        try:
            _, session_factory = load_rembg()
            rembg_session = session_factory("u2net")
        except Exception:
            rembg_session = None

    slide_metadata = []
    for slide_number, raw_path in enumerate(raw_paths, start=1):
        image_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read {raw_path}")
        height, width = image_bgr.shape[:2]
        slide_dir = slides_dir / f"slide_{slide_number:02d}"
        slide_dir.mkdir(parents=True, exist_ok=True)

        boxes = find_elements(image_bgr)
        inpaint_mask = np.zeros((height, width), dtype=np.uint8)
        elements = []

        for element_index, (x, y, w, h, area) in enumerate(boxes, start=1):
            crop = image_bgr[y : y + h, x : x + w]
            transparent = remove_background(crop, rembg_session, use_rembg)
            image_name = f"element_{element_index:02d}.png"
            transparent.save(slide_dir / image_name)
            cv2.rectangle(inpaint_mask, (x, y), (x + w, y + h), 255, thickness=-1)
            animation = suggest_animation(x, y, w, h, width, height)
            elements.append(
                {
                    "index": element_index,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                    "animation": animation,
                    "z_index": element_index + 2,
                    "image": image_name,
                }
            )

        # Slightly grow the mask before inpainting so glyph edges do not remain.
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        repair_mask = cv2.dilate(inpaint_mask, repair_kernel, iterations=1)
        background = cv2.inpaint(image_bgr, repair_mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(str(slide_dir / "background.png"), background)

        metadata = {
            "slide": slide_number,
            "width": width,
            "height": height,
            "background": "background.png",
            "elements": elements,
        }
        (slide_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        slide_metadata.append(metadata)
    return slide_metadata


async def generate_one_audio(text: str, output_path: Path) -> None:
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed.")
    communicate = edge_tts.Communicate(text=text, voice=VOICE, rate=RATE)
    await communicate.save(str(output_path))


async def generate_audio_files(audio_dir: Path) -> list[Path]:
    paths = []
    for index, text in enumerate(NARRATION_SCRIPTS, start=1):
        path = audio_dir / f"slide_{index:02d}.mp3"
        await generate_one_audio(text, path)
        paths.append(path)
    return paths


def measure_audio(path: Path) -> float:
    tag = TinyTag.get(str(path))
    if not tag.duration:
        raise RuntimeError(f"Could not measure audio duration: {path}")
    return float(tag.duration)


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def gsap_from_vars(animation: str) -> str:
    mapping = {
        "slide-down": "{ y: -90, autoAlpha: 0, duration: 0.7, ease: 'power3.out' }",
        "slide-up": "{ y: 90, autoAlpha: 0, duration: 0.7, ease: 'power3.out' }",
        "slide-right": "{ x: -110, autoAlpha: 0, duration: 0.7, ease: 'power3.out' }",
        "slide-left": "{ x: 110, autoAlpha: 0, duration: 0.7, ease: 'power3.out' }",
        "scale-fade": "{ scale: 0.92, autoAlpha: 0, duration: 0.7, ease: 'power3.out' }",
    }
    return mapping.get(animation, mapping["scale-fade"])


def write_slide_composition(slide: Slide, compositions_dir: Path) -> None:
    comp_id = f"slide-{slide.number:02d}"
    rel_base = f"../assets/slides/slide_{slide.number:02d}"
    elements_html = []
    timeline_lines = []

    for element in slide.elements:
        class_name = f"layer-{element.index:02d}"
        elements_html.append(
            f'''      <img class="slide-layer {class_name}" src="{rel_base}/{element.image}" '''
            f'''data-layout-allow-overflow '''
            f'''style="left:{element.x}px; top:{element.y}px; width:{element.width}px; '''
            f'''height:{element.height}px; z-index:{element.z_index};" alt="" />'''
        )
        start = min(0.2 + (element.index - 1) * 0.25, max(slide.duration - 1.0, 0.2))
        timeline_lines.append(
            f"      tl.from('.{class_name}', {gsap_from_vars(element.animation)}, {start:.2f});"
        )

    html = f"""<template id="{comp_id}-template">
  <div id="{comp_id}" data-composition-id="{comp_id}" data-width="{slide.width}" data-height="{slide.height}">
    <div class="slide-stage">
      <img class="slide-bg" src="{rel_base}/background.png" alt="" />
{os.linesep.join(elements_html)}
    </div>
    <style>
      #{comp_id} {{
        position: relative;
        width: {slide.width}px;
        height: {slide.height}px;
        overflow: hidden;
        background: #050505;
      }}
      #{comp_id} .slide-stage {{
        position: absolute;
        inset: 0;
        overflow: hidden;
      }}
      #{comp_id} .slide-bg {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 1;
      }}
      #{comp_id} .slide-layer {{
        position: absolute;
        object-fit: contain;
        will-change: transform, opacity;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true, defaults: {{ ease: 'power2.out' }} }});
{os.linesep.join(timeline_lines)}
      window.__timelines["{comp_id}"] = tl;
    </script>
  </div>
</template>
"""
    (compositions_dir / f"slide_{slide.number:02d}.html").write_text(
        html, encoding="utf-8"
    )


def write_index(slides: list[Slide], output: Path) -> None:
    starts = []
    current = 0.0
    for slide in slides:
        starts.append(current)
        current += max(slide.duration - OVERLAP_SECONDS, 0.1)
    total_duration = starts[-1] + slides[-1].duration

    clips = []
    audio = []
    transition_lines = []
    for idx, (slide, start) in enumerate(zip(slides, starts), start=1):
        track = 1 if idx % 2 == 1 else 2
        clips.append(
            f'''      <div id="slide-{idx:02d}-host" class="slide-host" '''
            f'''data-composition-id="slide-{idx:02d}" '''
            f'''data-composition-src="compositions/slide_{idx:02d}.html" '''
            f'''data-start="{start:.3f}" data-duration="{slide.duration:.3f}" '''
            f'''data-track-index="{track}"></div>'''
        )
        audio.append(
            f'''      <audio id="audio-{idx:02d}" src="assets/narration/slide_{idx:02d}.mp3" '''
            f'''data-start="{start:.3f}" data-duration="{slide.duration - TAIL_PADDING_SECONDS:.3f}" '''
            f'''data-track-index="{20 + idx}" data-volume="1"></audio>'''
        )
        if idx > 1:
            transition_lines.append(
                f"      master.from('#slide-{idx:02d}-host', "
                f"{{ opacity: 0, duration: 0.5, ease: 'power1.out' }}, {start:.3f});"
            )

    fade_start = max(total_duration - 1.0, 0)
    html = f"""<!doctype html>
<html id="startup-profit-code-document" class="clip" data-start="0">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Startup Profit Code Video</title>
  </head>
  <body>
    <div id="startup-profit-code-main" data-composition-id="startup-profit-code-main" data-start="0" data-width="{slides[0].width}" data-height="{slides[0].height}" data-duration="{total_duration:.3f}">
{os.linesep.join(clips)}
{os.linesep.join(audio)}
      <div id="black-fade" class="clip" data-start="0" data-duration="{total_duration:.3f}" data-track-index="99"></div>
    </div>
    <style>
      html,
      body {{
        margin: 0;
        width: 100%;
        height: 100%;
        background: #000;
      }}
      #startup-profit-code-main {{
        position: relative;
        width: {slides[0].width}px;
        height: {slides[0].height}px;
        overflow: hidden;
        background: #000;
        font-family: Arial, sans-serif;
      }}
      .slide-host {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }}
      #black-fade {{
        position: absolute;
        inset: 0;
        background: #000;
        opacity: 0;
        z-index: 1000;
        pointer-events: none;
      }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <script>
      window.__timelines = window.__timelines || {{}};
      const master = gsap.timeline({{ paused: true }});
{os.linesep.join(transition_lines)}
      master.to('#black-fade', {{ opacity: 1, duration: 1, ease: 'power1.inOut' }}, {fade_start:.3f});
      window.__timelines["startup-profit-code-main"] = master;
    </script>
  </body>
</html>
"""
    (output / "index.html").write_text(html, encoding="utf-8")


def write_project_files(output: Path, total_duration: float) -> None:
    package_json = {
        "scripts": {
            "lint": "hyperframes lint",
            "inspect": "hyperframes inspect",
            "preview": "hyperframes preview",
            "render": "hyperframes render --output renders/startup-profit-code-video.mp4 --quality standard",
        },
        "dependencies": {"hyperframes": "^0.6.91"},
        "devDependencies": {},
    }
    (output / "package.json").write_text(
        json.dumps(package_json, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Startup Profit Code Video\n\n"
        "Generated HyperFrames project from `sources/Startup_Profit_Code.pdf`.\n\n"
        f"Duration: {total_duration:.2f} seconds.\n\n"
        "Commands:\n\n"
        "```bash\n"
        "npx hyperframes lint\n"
        "npx hyperframes render --output renders/startup-profit-code-video.mp4 --quality standard\n"
        "```\n",
        encoding="utf-8",
    )


def build_project(pdf_path: Path, output: Path, use_rembg: bool) -> None:
    reset_dir(output)
    dirs = ensure_dirs(output)

    raw_paths = render_pdf_pages(pdf_path, dirs["raw"])
    if len(raw_paths) != len(NARRATION_SCRIPTS):
        raise RuntimeError(
            f"Expected {len(NARRATION_SCRIPTS)} slides but rendered {len(raw_paths)} pages."
        )

    metadata = process_slide_layers(raw_paths, dirs["slides"], use_rembg)
    asyncio.run(generate_audio_files(dirs["audio"]))

    slides = []
    for item in metadata:
        n = item["slide"]
        audio_rel = f"assets/narration/slide_{n:02d}.mp3"
        audio_path = output / audio_rel
        speech_duration = measure_audio(audio_path)
        duration = speech_duration + TAIL_PADDING_SECONDS
        elements = [Element(**element) for element in item["elements"]]
        slide = Slide(
            number=n,
            width=item["width"],
            height=item["height"],
            duration=duration,
            narration=NARRATION_SCRIPTS[n - 1],
            audio=audio_rel,
            background=f"assets/slides/slide_{n:02d}/background.png",
            elements=elements,
        )
        slides.append(slide)
        write_slide_composition(slide, dirs["compositions"])

    write_index(slides, output)
    total_duration = sum(s.duration for s in slides) - OVERLAP_SECONDS * (len(slides) - 1)
    write_project_files(output, total_duration)

    manifest = {
        "source_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        "voice": VOICE,
        "rate": RATE,
        "dpi": DPI,
        "overlap_seconds": OVERLAP_SECONDS,
        "tail_padding_seconds": TAIL_PADDING_SECONDS,
        "use_rembg": use_rembg,
        "total_duration_seconds": total_duration,
        "slides": [
            {
                "slide": slide.number,
                "duration": slide.duration,
                "narration": slide.narration,
                "audio": slide.audio,
                "background": slide.background,
                "element_count": len(slide.elements),
                "elements": [element.__dict__ for element in slide.elements],
            }
            for slide in slides
        ],
    }
    (output / "assets" / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if total_duration > 120:
        raise RuntimeError(f"Generated video is too long: {total_duration:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--use-rembg",
        action="store_true",
        help="Use rembg for each detected element. Slower; default uses OpenCV alpha masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    output = args.output.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    build_project(pdf_path, output, use_rembg=args.use_rembg)
    print(f"Generated HyperFrames project: {output}")


if __name__ == "__main__":
    main()
