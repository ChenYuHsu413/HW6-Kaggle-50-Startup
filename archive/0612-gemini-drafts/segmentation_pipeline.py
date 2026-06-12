import os
import cv2
import json
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from rembg import remove

def convert_pdf_to_png(pdf_path, output_dir, dpi=150):
    """將 PDF 轉成高解析度的投影片圖檔"""
    print(f"正在從 {pdf_path} 導出 PNG 投影片...")
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    slide_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(output_dir, f"slide_{i+1:02d}.png")
        pix.save(image_path)
        slide_paths.append(image_path)
        print(f"  匯出投影片 {i+1}/{len(doc)} -> {image_path}")
    return slide_paths

def segment_slide(image_path, slide_output_dir):
    """分析單張投影片影像，擷取獨立區塊去背並儲存"""
    os.makedirs(slide_output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 動態檢測背景亮度並使用大津二值化 (Otsu's Thresholding)
    median_val = np.median(gray)
    if median_val < 127:
        # 暗色背景，前景色較亮，不需要反轉
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    else:
        # 亮色背景，前景色較暗，需要反轉使前景色為白色
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        
    _, thresh = cv2.threshold(gray, 0, 255, thresh_type)
        
    # 用較大的膨脹核 (橫向 120px, 縱向 50px)，將文字段落或圖表群組合為一個物件
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 50))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    layers_metadata = []
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    layer_idx = 1
    
    bg_img = img.copy()
    
    bounding_boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # 嚴格過濾微小物件 (如單一字母、標點符號或小雜點)，只保留主要物件
        if cw < 60 or ch < 30:
            continue
        # 過濾背景
        if cw > w * 0.92 and ch > h * 0.92:
            continue
            
        bounding_boxes.append((x, y, cw, ch))
        
    # 由上至下、由左至右排序
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 50, b[0]))
    
    # 如果過濾後的物件過多，只挑選面積前 8 大的物件，以防動畫過於雜亂
    if len(bounding_boxes) > 8:
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[2] * b[3], reverse=True)[:8]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 50, b[0]))
        
    print(f"正在對 {os.path.basename(image_path)} 進行分割，共擷取 {len(bounding_boxes)} 個主要元素...")
    for idx, (x, y, cw, ch) in enumerate(bounding_boxes):
        crop_img = img[y:y+ch, x:x+cw]
        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        
        try:
            nobg_pil = remove(crop_pil)
        except Exception as e:
            print(f"  去背失敗: {e}，改用不透明背景")
            nobg_pil = crop_pil.convert("RGBA")
            
        layer_filename = f"layer_{layer_idx:02d}.png"
        nobg_pil.save(os.path.join(slide_output_dir, layer_filename))
        
        # 在背景遮罩上畫出該物件區域
        cv2.rectangle(bg_mask, (max(0, x-3), max(0, y-3)), (min(w, x+cw+3), min(h, y+ch+3)), 255, -1)
        
        # 動畫建議
        if y < h * 0.25:
            anim = "slide-down"
        elif cw > w * 0.5:
            anim = "fade-in"
        elif x < w * 0.35:
            anim = "slide-right"
        elif x > w * 0.65:
            anim = "slide-left"
        else:
            anim = "zoom-in"
            
        layers_metadata.append({
            "id": f"layer_{layer_idx:02d}",
            "file": layer_filename,
            "x": x,
            "y": y,
            "width": cw,
            "height": ch,
            "z_index": layer_idx,
            "animation": anim
        })
        layer_idx += 1
        
    # 背景填平
    if len(bounding_boxes) > 0:
        clean_bg = cv2.inpaint(bg_img, bg_mask, 5, cv2.INPAINT_TELEA)
    else:
        clean_bg = bg_img
        
    cv2.imwrite(os.path.join(slide_output_dir, "background.png"), clean_bg)
    
    metadata = {
        "slide_width": w,
        "slide_height": h,
        "layers": layers_metadata
    }
    with open(os.path.join(slide_output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"  投影片 {os.path.basename(image_path)} 處理完成！")

def run_full_pipeline(pdf_path, temp_slides_dir, output_dir):
    print("=================== 啟動投影片自動化處理管線 ===================")
    # 清除舊的 output 和 temp_slides 內容以防混雜
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(temp_slides_dir):
        shutil.rmtree(temp_slides_dir)
        
    slide_paths = convert_pdf_to_png(pdf_path, temp_slides_dir)
    
    for idx, slide_path in enumerate(slide_paths):
        slide_name = f"slide_{idx+1:02d}"
        slide_out_dir = os.path.join(output_dir, slide_name)
        segment_slide(slide_path, slide_out_dir)
    print("=================== 投影片分層處理全部完成 ===================")

import shutil
if __name__ == "__main__":
    run_full_pipeline("sources/Startup_Profit_Code.pdf", "temp_slides", "output")
