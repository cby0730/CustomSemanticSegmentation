# train.py
import os
import argparse
import torch
import torch.nn as nn
# 由 optimizer.py 匯入 get_optimizer
from optimizer import get_optimizer
from dataset import SegmentationDataset
from model import get_model
from trainer import Trainer
import torchvision.transforms as transforms
from loss import get_loss  # 若需要使用 loss.py 中的損失函式

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
        transforms.Resize((640, 640)),
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
    
    # 建立模型：根據命令列參數設定 encoder 及模型名稱
    model = get_model(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    model.to(device)
    
    # 選擇損失函式（此處可根據需求調整，預設若選 "crossentropy" 則使用 nn.CrossEntropyLoss()）
    if args.loss_fn.lower() == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = get_loss(args.loss_fn, mode="multiclass", classes=list(range(num_classes)), from_logits=True)
    
    # 使用 optimizer.py 建立優化器
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.learning_rate)

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
    parser.add_argument("--epochs", type=int, default=100, help="訓練 epoch 數")
    parser.add_argument("--batch_size", type=int, default=4, help="batch 大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers 數")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="學習率")
    parser.add_argument("--save_interval", type=int, default=50, help="每隔幾個 epoch 儲存一次 checkpoint")
    
    # 模型相關參數
    parser.add_argument("--model_name", type=str, default="deeplabv3+",
                        help="所使用的 segmentation model，選項包括： unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer")
    parser.add_argument("--encoder_name", type=str, default="resnext50_32x4d",
                        help="Encoder backbone（預設: resnet34 ），選項包括： resnet18, resnet34, resnet50, resnext50_32x4d, \
                                                                        vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, \
                                                                        mobilenet_v2, mobileone_s0, mobileone_s1, mobileone_s2, mobileone_s3, mobileone_s4, \
                                                                        efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, \
                                                                        timm-efficientnet-b0, timm-efficientnet-b1, timm-efficientnet-b2, timm-efficientnet-b3, timm-efficientnet-b4, timm-efficientnet-b5, \
                                                                        timm-tf_efficientnet_lite0, timm-tf_efficientnet_lite1, timm-tf_efficientnet_lite2, timm-tf_efficientnet_lite3, timm-tf_efficientnet_lite4, \
                                                                        mit_b0, mit_b1, mit_b2, \
                                                                        ")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="Encoder 預訓練權重（預設: imagenet）")
    
    # 損失函式選擇參數
    parser.add_argument("--loss_fn", type=str, default="focal",
                        help="選擇使用的損失函式，可選項包括：crossentropy, jaccard, dice, tversky, focal, lovasz, softcrossentropy; Binary Segmentation: mcc, softbce")
    
    # 新增優化器選擇參數
    parser.add_argument("--optimizer", type=str, default="radam",
                        help="選擇使用的優化器，可選項包括：adam, adamw, sgd, radam, rmsprop, adadelta, adagrad")
    
    args = parser.parse_args()
    main(args)
