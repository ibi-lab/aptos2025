#!/usr/bin/env python

import os
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
import click
from tqdm import tqdm
import wandb

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APTOSClassifier(nn.Module):
    def __init__(self, num_classes: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        # ViTモデルの読み込み
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 入力サイズを動画フレーム用に変更 (32フレーム x 3チャンネル x 256 x 256)
        self.vit.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        
        # 分類層の変更
        self.vit.heads = nn.Linear(768, num_classes)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # バッチ内の各フレームを個別に処理
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)  # (batch*frames, channels, height, width)
        
        # ViTで特徴量抽出
        x = self.vit(x)  # (batch*frames, num_classes)
        
        # フレームごとの予測を平均化
        x = x.view(batch_size, num_frames, -1)  # (batch, frames, num_classes)
        x = torch.mean(x, dim=1)  # (batch, num_classes)
        
        return x

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: str) -> Dict[str, float]:
    """1エポックの学習を実行"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        features = batch['features'].to(device)
        labels = batch['phase_id'].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
            device: str) -> Dict[str, float]:
    """検証を実行"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch['features'].to(device)
            labels = batch['phase_id'].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return {
        'loss': running_loss / len(val_loader),
        'accuracy': 100. * correct / total
    }

@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), required=True,
              help="APTOSデータセットファイルのパス")
@click.option("--batch-size", type=int, default=8, help="バッチサイズ")
@click.option("--epochs", type=int, default=50, help="学習エポック数")
@click.option("--lr", type=float, default=1e-4, help="学習率")
@click.option("--output-dir", type=click.Path(), default="outputs",
              help="モデルと結果の出力ディレクトリ")
@click.option("--use-wandb", is_flag=True, help="Weights & Biasesを使用する")
def train(dataset_path: str, batch_size: int, epochs: int, lr: float, 
         output_dir: str, use_wandb: bool):
    """APTOSデータセットを使用してViTモデルを学習"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # データセットの読み込み
    logger.info("データセットを読み込み中...")
    data = torch.load(dataset_path)
    
    from datam import APTOSDataset
    train_dataset = APTOSDataset(data['features_dict'], data['train_df'])
    val_dataset = APTOSDataset(data['features_dict'], data['val_df'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)

    # モデルの初期化
    model = APTOSClassifier(num_classes=data['num_classes'], device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5)

    if use_wandb:
        wandb.init(project="aptos-classifier", name="vit-classifier")
        wandb.config.update({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "model": "ViT-B-16"
        })

    best_val_acc = 0.0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # 学習
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.2f}%")

        # 検証
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.2f}%")

        if use_wandb:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "train_acc": train_metrics['accuracy'],
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy']
            })

        # モデルの保存
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))

        scheduler.step(val_metrics['loss'])

    logger.info("学習完了")
    logger.info(f"最高検証精度: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()