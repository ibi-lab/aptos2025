#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APTOSデータセットを使用してViTモデルを学習するスクリプト
このスクリプトは、APTOSデータセットを使用して、手術フェーズ分類のためのVision Transformer (ViT) モデルを学習します。
主な機能:
- データセットの読み込みと前処理
- ViTモデルの定義
- 学習と検証のループ
- Weights & Biasesを使用したメトリクスの記録
- 最良モデルの保存
- コマンドライン引数の処理
"""

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

# ロガーの設定: INFO レベルでタイムスタンプ付きのログを出力
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APTOSClassifier(nn.Module):
    """手術フェーズ分類用のViTベースの分類器
    
    Args:
        num_classes (int): 分類するフェーズの数
        device (str): 使用するデバイス（'cuda' or 'cpu'）
    """
    def __init__(self, num_classes: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        # ImageNet事前学習済みのViTモデルを読み込み
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # パッチ埋め込み層を動画フレーム用にカスタマイズ
        # 入力: 32フレーム x 3チャンネル x 256 x 256
        # パッチサイズ: 16x16
        self.vit.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        
        # 最終の分類層をタスクに合わせて変更
        self.vit.heads = nn.Linear(768, num_classes)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播の実行
        
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, num_frames, channels, height, width)
        
        Returns:
            torch.Tensor: クラス予測スコア (batch_size, num_classes)
        """
        # バッチ内の各フレームを個別に処理するために次元を変換
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)  # (batch*frames, channels, height, width)
        
        # ViTで各フレームから特徴量を抽出
        x = self.vit(x)  # (batch*frames, num_classes)
        
        # フレームごとの予測を平均化して最終予測を生成
        x = x.view(batch_size, num_frames, -1)  # (batch, frames, num_classes)
        x = torch.mean(x, dim=1)  # (batch, num_classes)
        
        return x

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: str) -> Dict[str, float]:
    """1エポックの学習を実行
    
    Args:
        model: 学習するモデル
        train_loader: 訓練データのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザ
        device: 使用するデバイス
    
    Returns:
        Dict[str, float]: 訓練損失と精度を含む辞書
    """
    model.train()  # 訓練モードに設定
    running_loss = 0.0
    correct = 0
    total = 0

    # ミニバッチごとの処理
    for batch in tqdm(train_loader, desc="Training"):
        # データをデバイスに転送
        features = batch['features'].to(device)
        labels = batch['phase_id'].to(device)

        # 勾配を初期化
        optimizer.zero_grad()
        
        # 順伝播、損失計算、逆伝播
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 統計情報の更新
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # エポック全体の統計を計算
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
            device: str) -> Dict[str, float]:
    """バリデーションセットでの評価を実行
    
    Args:
        model: 評価するモデル
        val_loader: 検証データのDataLoader
        criterion: 損失関数
        device: 使用するデバイス
    
    Returns:
        Dict[str, float]: 検証損失と精度を含む辞書
    """
    model.eval()  # 評価モードに設定
    running_loss = 0.0
    correct = 0
    total = 0

    # 勾配計算を行わない
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch['features'].to(device)
            labels = batch['phase_id'].to(device)

            # 順伝播と損失計算
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 統計情報の更新
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # バリデーション全体の統計を計算
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
    """APTOSデータセットを使用してViTモデルを学習
    
    Args:
        dataset_path: データセットファイルのパス
        batch_size: バッチサイズ
        epochs: 学習エポック数
        lr: 学習率
        output_dir: 出力ディレクトリ
        use_wandb: W&Bを使用するかどうか
    """
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # データセットの読み込みと前処理
    logger.info("データセットを読み込み中...")
    data = torch.load(dataset_path)
    
    from datam import APTOSDataset
    # 訓練データとバリデーションデータのデータセットを作成
    train_dataset = APTOSDataset(data['features_dict'], data['train_df'])
    val_dataset = APTOSDataset(data['features_dict'], data['val_df'])

    # DataLoaderの設定
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)

    # モデルとトレーニング関連コンポーネントの初期化
    model = APTOSClassifier(num_classes=data['num_classes'], device=device)
    criterion = nn.CrossEntropyLoss()  # 多クラス分類用の損失関数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 検証損失が改善しない場合に学習率を調整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=5)

    # Weights & Biasesの設定（指定された場合）
    if use_wandb:
        wandb.init(project="aptos-classifier", name="vit-classifier")
        wandb.config.update({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "model": "ViT-B-16"
        })

    # 学習ループ
    best_val_acc = 0.0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # 訓練フェーズ
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.2f}%")

        # 検証フェーズ
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.2f}%")

        # Weights & Biasesにメトリクスを記録
        if use_wandb:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "train_acc": train_metrics['accuracy'],
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy']
            })

        # 最良モデルの保存
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))

        # 学習率の調整
        scheduler.step(val_metrics['loss'])

    logger.info("学習完了")
    logger.info(f"最高検証精度: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()