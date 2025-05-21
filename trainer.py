# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from utils import calculate_miou, update_training_plot

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, optimizer, criterion, device,
                 batch_size=4, num_workers=4, save_dir='checkpoints'):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset is not None else None
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)  # 預測輸出 shape: [B, classes, H, W]
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        miou_total = 0.0
        count = 0
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * images.size(0)
                
                miou = calculate_miou(outputs, masks)
                miou_total += miou
                count += 1
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_miou = miou_total / count if count > 0 else 0.0
        return epoch_loss, epoch_miou

    def train(self, num_epochs, save_interval=5):
        train_loss_list = []
        val_loss_list = []
        val_miou_list = []
        best_loss = float('inf')
        best_miou = 0.0
        
        # 計算模型參數量 (這裡計算的是總參數數量)
        param_count = sum(p.numel() for p in self.model.parameters())
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss = self.train_epoch()
            train_loss_list.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            if self.val_loader is not None:
                val_loss, val_miou = self.validate_epoch()
                val_loss_list.append(val_loss)
                val_miou_list.append(val_miou)
                print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_miou = val_miou
                    self.save_checkpoint(epoch, best_loss, best=True)
                if epoch % save_interval == 0:
                    self.save_checkpoint(epoch, best_loss)
            else:
                if epoch % save_interval == 0:
                    self.save_checkpoint(epoch, train_loss)
            
            # 更新訓練過程圖表 (每個 epoch 都更新同一張圖片)，並傳入模型參數量資訊
            update_training_plot(train_loss_list, val_loss_list, val_miou_list, best_loss, best_miou, param_count, self.save_dir)
    
    def save_checkpoint(self, epoch, loss, best=False):
        filename = f"checkpoint_epoch_{epoch}.pth"
        if best:
            filename = "best_model.pth"
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }, save_path)
        print(f"Checkpoint saved: {save_path}")
