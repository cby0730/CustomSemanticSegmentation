#!/usr/bin/env python3
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from model import get_model
import cv2
from utils import convert_video, convert_to_common_mp4  # 從 utils 載入轉檔函數

def get_color_map():
    """
    定義每個類別對應的顏色
    假設類別順序為：
      0: background → 黑色
      1: beach      → 黃色
      2: ocean      → 藍色
      3: sky        → 淺藍
    """
    color_map = {
        0: (0, 0, 0),
        1: (255, 255, 0),
        2: (0, 0, 255),
        3: (135, 206, 235)
    }
    return color_map

def colorize_mask(mask, color_map):
    """
    將單通道的分割 mask 轉換成彩色圖
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[mask == cls] = color
    return color_mask

def overlay_images(image, mask_color, alpha=0.5):
    """
    將原圖與分割彩色圖疊加
    """
    overlay = (alpha * image + (1 - alpha) * mask_color).astype(np.uint8)
    return overlay

def process_image(file_path, output_dir, model, transform, device):
    """
    處理圖片：讀取、推論、生成分割圖及疊圖後儲存
    """
    image = Image.open(file_path).convert("RGB")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    color_map = get_color_map()
    pred_color = colorize_mask(pred, color_map)
    pred_color_image = Image.fromarray(pred_color)
    if pred_color_image.size != original_size:
        pred_color_image = pred_color_image.resize(original_size, resample=Image.NEAREST)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    seg_path = os.path.join(output_dir, f"{base_name}_segmentation.png")
    pred_color_image.save(seg_path)
    print(f"Segmentation image saved to {seg_path}")

    original_np = np.array(image)
    pred_color_np = np.array(pred_color_image)
    overlay_np = overlay_images(original_np, pred_color_np, alpha=0.5)
    overlay_image = Image.fromarray(overlay_np)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    overlay_image.save(overlay_path)
    print(f"Overlay image saved to {overlay_path}")

def process_video(file_path, output_dir, model, transform, device):
    """
    處理影片：
      1. 利用 ffmpeg 將輸入影片轉換成 mp4 格式（預處理）
      2. 逐幀推論、產生分割及疊圖影片
      3. 將輸出影片再轉換成通用 mp4 格式
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_video_path = os.path.join(output_dir, f"temp_{base_name}.mp4")
    
    try:
        convert_video(file_path, temp_video_path)
    except Exception as e:
        print(f"FFmpeg conversion failed for {file_path}: {e}")
        return
    
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        print(f"無法開啟轉換後的影片：{temp_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    seg_video_path = os.path.join(output_dir, f"{base_name}_segmentation.mp4")
    overlay_video_path = os.path.join(output_dir, f"{base_name}_overlay.mp4")
    out_seg = cv2.VideoWriter(seg_video_path, fourcc, fps, (width, height))
    out_overlay = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        original_size = image_pil.size

        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        color_map = get_color_map()
        pred_color = colorize_mask(pred, color_map)
        pred_color_image = Image.fromarray(pred_color)
        if pred_color_image.size != original_size:
            pred_color_image = pred_color_image.resize(original_size, resample=Image.NEAREST)

        original_np = np.array(image_pil)
        pred_color_np = np.array(pred_color_image)
        overlay_np = overlay_images(original_np, pred_color_np, alpha=0.5)

        seg_bgr = cv2.cvtColor(np.array(pred_color_image), cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR)

        out_seg.write(seg_bgr)
        out_overlay.write(overlay_bgr)
        frame_count += 1

    cap.release()
    out_seg.release()
    out_overlay.release()
    print(f"影片 {file_path} 處理完成，共 {frame_count} 幀")
    print(f"初步分割影片儲存於 {seg_video_path}")
    print(f"初步疊圖影片儲存於 {overlay_video_path}")

    final_seg_video_path = os.path.join(output_dir, f"{base_name}_segmentation_final.mp4")
    final_overlay_video_path = os.path.join(output_dir, f"{base_name}_overlay_final.mp4")
    try:
        convert_to_common_mp4(seg_video_path, final_seg_video_path)
        convert_to_common_mp4(overlay_video_path, final_overlay_video_path)
    except Exception as e:
        print(f"Final conversion failed: {e}")
        return
    else:
        os.remove(seg_video_path)
        os.remove(overlay_video_path)
        print(f"最終分割影片儲存於 {final_seg_video_path}")
        print(f"最終疊圖影片儲存於 {final_overlay_video_path}")

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 4
    # 建立模型：呼叫 model.py 中的 get_model()，並根據命令列參數設定 encoder 及模型名稱
    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    model.to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    os.makedirs(args.output_dir, exist_ok=True)
    input_dir = args.input_dir

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in image_exts:
                print(f"Processing image: {file_path}")
                process_image(file_path, args.output_dir, model, transform, device)
            elif ext in video_exts:
                print(f"Processing video: {file_path}")
                process_video(file_path, args.output_dir, model, transform, device)
            else:
                print(f"跳過不支援的檔案: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints",
                        help="訓練好模型的 checkpoint 路徑 (.pth 檔案)")
    parser.add_argument("--input_dir", type=str, default="test",
                        help="包含待測試圖片與影片的資料夾")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="儲存分割結果圖與疊圖的資料夾")
    
    # 新增模型選擇參數
    parser.add_argument("--model_name", type=str, default="segformer",
                        help="所使用的 segmentation model，選項包括：unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer")
    parser.add_argument("--encoder_name", type=str, default="mit_b1",
                        help="Encoder backbone（預設: resnet34 ）")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="Encoder 預訓練權重（預設: imagenet）")
    
    args = parser.parse_args()
    main(args)
