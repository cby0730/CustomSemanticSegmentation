# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, optimizer, criterion, device,
                 batch_size=4, num_workers=4, save_dir='checkpoints'):
        """
        Args:
            model (torch.nn.Module): 要訓練的模型。
            train_dataset (Dataset): 訓練資料集。
            val_dataset (Dataset): 驗證資料集（可為 None）。
            optimizer: 優化器。
            criterion: 損失函數（例如 CrossEntropyLoss）。
            device (torch.device): 設備 ("cuda" 或 "cpu")。
            batch_size (int): 每個 batch 的大小。
            num_workers (int): DataLoader 的 workers 數量。
            save_dir (str): 儲存 checkpoint 的資料夾。
        """
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
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def train(self, num_epochs, save_interval=5):
        best_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            if self.val_loader is not None:
                val_loss = self.validate_epoch()
                print(f"Val Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(epoch, best_loss, best=True)
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, best_loss)
    
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
