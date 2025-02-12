# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class SegmentationDataset(Dataset):
    """
    自訂的語意分割資料集，讀取指定資料夾中的原圖與 mask。
    假設：
      - images 資料夾中存放原圖（支援 .png, .jpg, .jpeg）
      - masks 資料夾中存放對應的 mask，檔名格式：<原圖檔名基底>_mask.png
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): 原圖資料夾路徑。
            masks_dir (str): Mask 資料夾路徑。
            transform (callable, optional): 對原圖進行轉換（例如 Resize、ToTensor 等）。
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # 讀取 images 資料夾內的所有圖片檔名（排序以便對應）
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 取得原圖檔名與完整路徑
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        # 根據原圖檔名構造 mask 檔名，例如 A.jpg → A_mask.png
        mask_filename = os.path.splitext(img_filename)[0] + '_mask.png'
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # 載入原圖（轉成 RGB）與 mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # 如果有設定 transform，作用於原圖（例如 Resize、Normalization）
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # 將 mask 轉成 numpy 陣列，再轉成 LongTensor（每個像素值代表類別索引）
        mask = np.array(mask)
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask
