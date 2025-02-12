# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SegmentationDataset
from model import get_model
from trainer import Trainer
import torchvision.transforms as transforms

def main(args):
    # 設定設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 訓練與驗證資料夾路徑
    train_images_dir = os.path.join(args.train_data_dir, "images")
    train_masks_dir = os.path.join(args.train_data_dir, "masks")
    valid_images_dir = os.path.join(args.valid_data_dir, "images")
    valid_masks_dir = os.path.join(args.valid_data_dir, "masks")
    
    # 定義轉換（例如將影像 resize 為 640x640，再轉為 Tensor）
    transform = transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor(),
    ])
    
    # 建立訓練與驗證資料集
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=transform)
    valid_dataset = SegmentationDataset(valid_images_dir, valid_masks_dir, transform=transform)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")
    
    # 讀取 _classes.csv 以取得類別數（假設該檔案在 train_data_dir 中）
    classes_csv = os.path.join(args.train_data_dir, "_classes.csv")
    if os.path.exists(classes_csv):
        with open(classes_csv, "r") as f:
            lines = f.readlines()
        if len(lines) > 1:
            # 第一行為 header，因此類別數為行數 - 1
            num_classes = len(lines) - 1
        else:
            num_classes = len(lines)
    else:
        print("Warning: _classes.csv not found. Defaulting num_classes to 2.")
        num_classes = 2
    print(f"Number of classes: {num_classes}")
    
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
    
    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 建立 Trainer 並開始訓練
    trainer = Trainer(model, train_dataset, valid_dataset, optimizer, criterion, device,
                      batch_size=args.batch_size, num_workers=args.num_workers, save_dir=args.checkpoints_dir)
    trainer.train(num_epochs=args.epochs, save_interval=args.save_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="data/train",
                        help="訓練資料夾路徑，內含 images, masks 與 _classes.csv")
    parser.add_argument("--valid_data_dir", type=str, default="data/valid",
                        help="驗證資料夾路徑，內含 images 與 masks")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="儲存 checkpoint 的資料夾")
    parser.add_argument("--epochs", type=int, default=50, help="訓練 epoch 數")
    parser.add_argument("--batch_size", type=int, default=4, help="batch 大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers 數")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="學習率")
    parser.add_argument("--save_interval", type=int, default=5, help="每隔幾個 epoch 儲存一次 checkpoint")
    
    # 新增模型選擇參數
    parser.add_argument("--model_name", type=str, default="segformer",
                        help="所使用的 segmentation model，選項包括：unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer")
    parser.add_argument("--encoder_name", type=str, default="mit_b0",
                        help="Encoder backbone（預設: resnet34）")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="Encoder 預訓練權重（預設: imagenet）")
    
    args = parser.parse_args()
    main(args)
