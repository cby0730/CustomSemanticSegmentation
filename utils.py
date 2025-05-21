# utils.py
import subprocess
import os
import torch
import matplotlib.pyplot as plt

def convert_video(input_path, output_path):
    """
    使用 ffmpeg 將影片轉換成 mp4 格式 (用於輸入影片的預處理)
    """
    cmd = [
         "ffmpeg",
         "-y",  # 自動覆蓋輸出檔案
         "-i", input_path,
         "-c:v", "libx264",
         "-preset", "fast",
         "-crf", "23",
         "-c:a", "copy",
         output_path
    ]
    print("Running ffmpeg command: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def convert_to_common_mp4(input_path, output_path):
    """
    使用 ffmpeg 將輸出影片轉換成通用 mp4 格式，
    加入參數 -profile:v high -level 4.0 與 -pix_fmt yuv420p 以提高播放兼容性
    """
    cmd = [
         "ffmpeg",
         "-y",  # 自動覆蓋輸出檔案
         "-i", input_path,
         "-c:v", "libx264",
         "-profile:v", "high",
         "-level", "4.0",
         "-pix_fmt", "yuv420p",
         output_path
    ]
    print("Running ffmpeg for output conversion: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def calculate_miou(outputs, targets, eps=1e-6):
    """
    計算 Mean Intersection over Union (mIoU)

    Args:
         outputs (torch.Tensor): 預測輸出，shape: [B, classes, H, W]
         targets (torch.Tensor): 真實標籤，shape: [B, H, W]
         eps (float): 避免除以零的小數

    Returns:
         float: mIoU 值
    """
    preds = torch.argmax(outputs, dim=1)
    num_classes = outputs.shape[1]
    iou_sum = 0.0
    for cls in range(num_classes):
         intersection = ((preds == cls) & (targets == cls)).float().sum()
         union = ((preds == cls) | (targets == cls)).float().sum()
         iou = (intersection + eps) / (union + eps)
         iou_sum += iou
    miou = iou_sum / num_classes
    return miou.item()

def update_training_plot(train_loss_list, val_loss_list, val_miou_list, best_loss, best_miou, param_count, save_dir):
    """
    更新並儲存訓練過程的圖表 (training_progress.png)
    此圖表除了顯示損失與 mIoU 變化外，也會顯示模型的參數量資訊

    Args:
         train_loss_list (list): 每個 epoch 的訓練損失
         val_loss_list (list): 每個 epoch 的驗證損失
         val_miou_list (list): 每個 epoch 的驗證 mIoU
         best_loss (float): 目前最佳的驗證損失
         best_miou (float): 目前最佳的驗證 mIoU
         param_count (int): 模型的參數總數
         save_dir (str): 儲存圖表的資料夾路徑
    """
    plt.figure(figsize=(8,6))
    epochs = range(1, len(train_loss_list) + 1)
    if val_loss_list:
         # 繪製損失圖表
         plt.subplot(2, 1, 1)
         plt.plot(epochs, train_loss_list, label='Train Loss')
         plt.plot(epochs, val_loss_list, label='Val Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.legend()
         # 繪製 mIoU 圖表
         plt.subplot(2, 1, 2)
         plt.plot(epochs, val_miou_list, label='Val mIoU')
         plt.xlabel('Epoch')
         plt.ylabel('mIoU')
         plt.legend()
         # 在總標題中加入最佳資訊與模型參數量
         plt.suptitle(f'Best Val Loss: {best_loss:.4f} | Best Val mIoU: {best_miou:.4f}\nModel Params: {param_count:,}')
    else:
         plt.plot(epochs, train_loss_list, label='Train Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.legend()
         plt.title(f'Model Params: {param_count:,}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_progress.png"))
    plt.close()
