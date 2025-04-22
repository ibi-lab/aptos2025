#!/usr/bin/env python

import os
import sys
from typing import Dict, Any

import click
import pandas as pd
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Lambda
# from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     UniformTemporalSubsample,
#     ShortSideScale,
#     Normalize,
# )
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
# from torchvision.transforms import Compose
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

alpha = 4

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VideoProcessor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self._load_model()
        self.transform = self._create_transform()
        logger.info(f"Using device: {self.device}")

    def _load_model(self) -> torch.nn.Module:
        """モデルのロードと初期化"""
        try:
            model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
            sys.exit(1)

    def _create_transform(self) -> Compose:
        """動画前処理用のtransform作成"""
        # return Compose([
        #     UniformTemporalSubsample(32),
        #     ShortSideScale(256),
        #     Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        # ])
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    # PackPathway()
                ]
            ),
        )

    def extract_video_segment(self, video_path: str, start_sec: float, end_sec: float, 
                            output_path: str) -> bool:
        """動画の指定区間を切り出して保存"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"動画を開けません: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if start_frame <= frame_count < end_frame:
                    out.write(frame)
                elif frame_count >= end_frame:
                    break
                    
                frame_count += 1
            
            cap.release()
            out.release()
            return True

        except Exception as e:
            logger.error(f"動画切り出し中にエラー: {str(e)}")
            return False

    def extract_features(self, video_path: str) -> np.ndarray:
        """動画から特徴量を抽出"""
        try:
            logger.info(f"動画エンコード")
            video = EncodedVideo.from_path(video_path)
            logger.info(f"クリップ取得")
            video_data = video.get_clip(0, video.duration)
            # logger.info(f"テンソル取得")
            # video_tensor = video_data["video"]

            # logger.info(f"テンソル変換")
            # [T, H, W, C] -> [C, T, H, W]に変換
            # video_tensor = video_tensor.permute(3, 0, 1, 2)
            # float32に変換し、[0-255]から[0-1]にスケール
            # video_tensor = video_tensor.float() / 255.0

            logger.info(f"前処理")            
            # 前処理を適用
            video_tensor = self.transform(video_data)
            # バッチ次元を追加
            # video_tensor = video_tensor.unsqueeze(0)
            # video_tensor = video_tensor.to(self.device)

            logger.info(f'video_tensor: {video_tensor["video"].cpu().numpy().shape}')
            
            logger.info(f"inputs変換")            

            inputs = video_tensor["video"]
            inputs = [i.to(self.device)[None, ...] for i in inputs]
            
            logger.info(f"model実行")            
            with torch.no_grad():
                features = self.model(inputs)

            logger.info(f"CPUモードへ切り替え")
                
            return features.cpu().numpy()

        except Exception as e:
            logger.error(f"特徴量抽出中にエラー: {str(e)}")
            return None

class APTOSDataset(torch.utils.data.Dataset):
    """APTOSデータセット"""
    def __init__(self, features_dir: str, annotation_df: pd.DataFrame):
        self.features_dir = features_dir
        self.annotations = annotation_df
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        feature_path = os.path.join(self.features_dir, f"{row['video_id']}_{idx:04d}.npz")
        
        try:
            data = np.load(feature_path)
            return {
                'features': torch.from_numpy(data['features']).float(),
                'phase_id': torch.tensor(data['phase_id'], dtype=torch.long),
                'video_id': row['video_id'],
                'start_time': row['start'],
                'end_time': row['end']
            }
        except Exception as e:
            logger.error(f"特徴量ファイルの読み込みエラー {feature_path}: {str(e)}")
            return None

@click.group()
def cli():
    """APTOSデータセット処理ツール"""
    pass

@cli.command()
@click.option("--csv-path", required=True, type=click.Path(exists=True),
              help="アノテーションCSVファイルのパス")
@click.option("--video-dir", required=True, type=click.Path(exists=True),
              help="元動画が格納されているディレクトリ")
@click.option("--output-video-dir", default="output_videos", type=click.Path(),
              help="切り出した動画の出力先ディレクトリ (デフォルト: output_videos)")
def extract_segments(csv_path: str, video_dir: str, output_video_dir: str):
    """動画から指定された区間を切り出して保存"""
    # 出力ディレクトリの作成
    os.makedirs(output_video_dir, exist_ok=True)

    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    processor = VideoProcessor()

    # 各動画セグメントの処理
    for idx, row in df.iterrows():
        video_id = row["video_id"]
        start_sec = float(row["start"])
        end_sec = float(row["end"])

        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"動画ファイルが見つかりません: {video_path}")
            continue

        # セグメント出力パス
        segment_path = os.path.join(output_video_dir, f"{video_id}_{idx:04d}.mp4")
        logger.info(f"処理中: {video_id} ({start_sec:.2f}-{end_sec:.2f})")

        # 動画セグメントの切り出し
        if not processor.extract_video_segment(video_path, start_sec, end_sec, segment_path):
            logger.error(f"動画セグメントの切り出しに失敗: {video_path}")

@cli.command()
@click.option("--csv-path", required=True, type=click.Path(exists=True),
              help="アノテーションCSVファイルのパス")
@click.option("--video-dir", required=True, type=click.Path(exists=True),
              help="切り出した動画が格納されているディレクトリ")
@click.option("--output-feature-dir", default="output_features", type=click.Path(),
              help="特徴量ファイルの出力先ディレクトリ (デフォルト: output_features)")
@click.option("--save-dataset", default="aptos_dataset.pth", type=click.Path(),
              help="保存するデータセットファイルのパス (デフォルト: aptos_dataset.pth)")
def create_dataset(csv_path: str, video_dir: str, output_feature_dir: str, save_dataset: str):
    """切り出した動画から特徴量を抽出してデータセットを作成"""
    os.makedirs(output_feature_dir, exist_ok=True)

    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    processor = VideoProcessor()

    # 各動画セグメントから特徴量を抽出
    for idx, row in df.iterrows():
        video_id = row["video_id"]
        start_time = row["start"]
        end_time = row["end"]
        phase_id = int(row["phase_id"])
        
        # video_pathの構築を修正
        video_path = os.path.join(video_dir, f"{video_id}_{start_time}_{end_time}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"動画ファイルが見つかりません: {video_path}")
            continue

        feature_path = os.path.join(output_feature_dir, f"{video_id}_{start_time}_{end_time}.npz")
        logger.info(f"特徴量抽出中: {video_path}")

        # 特徴量の抽出
        features = processor.extract_features(video_path)

        logger.info(f"特徴量抽出終了")
        
        if features is not None:
            # 特徴量とphase_idを保存
            logger.info(f"特徴量保存中: {feature_path}")

            np.savez(feature_path, features=features, phase_id=phase_id)
            logger.info(f"特徴量を保存: {feature_path}")
        else:
            logger.error(f"特徴量の抽出に失敗: {video_path}")
            continue

    # データセットの作成
    logger.info("データセットの作成を開始")
    
    # DataFrameをトレーニングとバリデーションに分割
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # データセットの作成
    train_dataset = APTOSDataset(output_feature_dir, train_df)
    val_dataset = APTOSDataset(output_feature_dir, val_df)
    
    # データセットの保存
    torch.save({
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': len(df['phase_id'].unique())
    }, save_dataset)
    
    logger.info(f"データセットを保存しました: {save_dataset}")
    logger.info(f"トレーニングデータ数: {len(train_dataset)}")
    logger.info(f"バリデーションデータ数: {len(val_dataset)}")

if __name__ == "__main__":
    cli()