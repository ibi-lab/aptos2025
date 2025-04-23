#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APTOSデータセット処理スクリプト

このスクリプトは、手術フェーズ分類のための動画データの前処理と特徴量抽出を行います。

主な機能:
1. 動画セグメントの切り出し
   - 長い手術動画から指定された時間区間を切り出し
   - 各セグメントを個別のファイルとして保存

2. 特徴量抽出
   - 切り出された動画セグメントから特徴量を抽出
   - 事前学習済みのSlowFast networkを使用
   - 抽出された特徴量をPyTorchのデータセット形式で保存

3. データセットの作成
   - 特徴量とアノテーションを組み合わせてデータセットを作成
   - トレーニング用とバリデーション用にデータを分割
   - オプションで均等なクラス分布になるようにサンプリング

使用方法:
1. 動画セグメントの切り出し:
   python datam.py extract-segments --csv-path annotations.csv --video-dir videos

2. データセットの作成:
   python datam.py create-dataset --csv-path annotations.csv --video-dir segments

注意事項:
- GPUが利用可能な場合は自動的に使用されます
- 大容量のメモリが必要になる場合があります
"""

import os
import sys
from typing import Dict, Any

import click
import pandas as pd
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Lambda
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
    """SlowFast networkのための動画フレーム変換モジュール
    
    このモジュールは動画フレームを2つの経路（slow pathwayとfast pathway）に分割します：
    - slow pathway: 低フレームレートで空間的に詳細な特徴を抽出
    - fast pathway: 高フレームレートで時間的な特徴を抽出
    
    Attributes:
        alpha (int): fast pathwayとslow pathwayのフレームレート比
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        """フレームを2つの経路に分割
        
        Args:
            frames (torch.Tensor): 入力フレーム (T, C, H, W)
        
        Returns:
            List[torch.Tensor]: [slow_pathway, fast_pathway]
        """
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
    """動画処理とモデル推論を行うクラス
    
    このクラスは以下の機能を提供します：
    1. 事前学習済みのSlowFast networkのロードと初期化
    2. 動画の前処理（リサイズ、正規化など）
    3. 動画セグメントの切り出し
    4. 特徴量の抽出
    
    Attributes:
        device (str): 使用するデバイス（'cuda' or 'cpu'）
        model (nn.Module): ロードされたSlowFast network
        transform (Compose): 動画前処理用の変換パイプライン
    """
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
        """動画前処理用のtransform作成
        
        以下の処理を順に適用します：
        1. フレーム数の統一（32フレーム）
        2. ピクセル値の正規化（0-1に変換）
        3. チャネルごとの正規化
        4. サイズの統一（256x256にリサイズ）
        5. テンソル次元の並べ替え
        
        Returns:
            Compose: 変換パイプライン
        """
        num_frames = 32
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    Lambda(lambda x: torch.nn.functional.interpolate(
                        x, size=(256, 256), mode="bilinear", align_corners=False
                    )),
                    Lambda(lambda x: x.permute(1, 0, 2, 3)),  # [C,T,H,W] -> [T,C,H,W]に変換
                    # PackPathway()  # 必要に応じて有効化
                ]
            ),
        )

    def extract_video_segment(self, video_path: str, start_sec: float, end_sec: float, 
                            output_path: str) -> bool:
        """動画から指定時間区間を切り出して保存
        
        Args:
            video_path (str): 入力動画のパス
            start_sec (float): 切り出し開始時間（秒）
            end_sec (float): 切り出し終了時間（秒）
            output_path (str): 出力ファイルパス
        
        Returns:
            bool: 切り出しが成功した場合True
        """
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

            logger.info(f"前処理")            
            # 前処理を適用
            video_tensor = self.transform(video_data)
            # バッチ次元を追加
            # video_tensor = video_tensor.unsqueeze(0)
            # video_tensor = video_tensor.to(self.device)

            logger.info(f'video_tensor: {video_tensor["video"].cpu().numpy().shape}')
            
            # logger.info(f"inputs変換")            

            # inputs = video_tensor["video"]
            # inputs = [i.to(self.device)[None, ...] for i in inputs]
            
            # logger.info(f"model実行")            
            # with torch.no_grad():
            #     features = self.model(inputs)

            # logger.info(f"CPUモードへ切り替え")
                
            # return features.cpu().numpy()
            return video_tensor["video"].cpu().numpy()

        except Exception as e:
            logger.error(f"特徴量抽出中にエラー: {str(e)}")
            return None

class APTOSDataset(torch.utils.data.Dataset):
    """APTOSデータセットのPyTorch実装
    
    特徴量とアノテーションを組み合わせてデータセットを作成します。
    
    Attributes:
        features_dict (Dict[str, np.ndarray]): 動画IDをキーとする特徴量の辞書
        annotations (pd.DataFrame): アノテーション情報のDataFrame
    """
    def __init__(self, features_dict: Dict[str, np.ndarray], annotation_df: pd.DataFrame):
        self.features_dict = features_dict
        self.annotations = annotation_df
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """データセットからサンプルを取得
        
        Args:
            idx (int): サンプルのインデックス
        
        Returns:
            Dict: {
                'features': 特徴量テンソル,
                'phase_id': フェーズID,
                'video_id': 動画ID,
                'start_time': 開始時間,
                'end_time': 終了時間
            }
        """
        row = self.annotations.iloc[idx]
        video_id = row['video_id']
        start_time = row['start']
        end_time = row['end']
        
        # features_dictから特徴量を取得
        feature_key = f"{video_id}_{start_time}_{end_time}"
        features = self.features_dict[feature_key]
        
        return {
            'features': torch.from_numpy(features).float(),
            'phase_id': torch.tensor(row['phase_id'], dtype=torch.long),
            'video_id': video_id,
            'start_time': start_time,
            'end_time': end_time
        }

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

def sample_balanced_data(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """各フェーズから均等にサンプリング
    
    Args:
        df (pd.DataFrame): 元のデータフレーム
        n_samples (int): サンプリングする総数
    
    Returns:
        pd.DataFrame: サンプリング後のデータフレーム
    
    注意:
        - クラスごとのサンプル数が不足する場合は置換サンプリングを行います
        - 総サンプル数はクラス数で割り切れる必要があります
    """
    # phase_idごとのグループを取得
    groups = df.groupby('phase_id')
    
    # 各グループから取得するサンプル数を計算
    n_classes = len(groups)
    samples_per_class = n_samples // n_classes
    
    sampled_dfs = []
    for _, group in groups:
        # 各クラスからサンプリング（グループサイズより多く要求された場合は置換でサンプリング）
        replace = len(group) < samples_per_class
        sampled = group.sample(n=samples_per_class, replace=replace)
        sampled_dfs.append(sampled)
    
    # すべてのサンプリングされたデータを結合
    return pd.concat(sampled_dfs).reset_index(drop=True)

def load_excludes(excludes_file: str) -> set:
    """除外するビデオIDのリストを読み込む
    
    Args:
        excludes_file (str): 除外リストを含むテキストファイルのパス
        
    Returns:
        set: 除外するビデオIDのセット
    
    Note:
        テキストファイルは1行に1つのビデオIDを記載
        空行とコメント行(#で始まる)は無視
    """
    if not os.path.exists(excludes_file):
        logger.warning(f"除外リストファイルが見つかりません: {excludes_file}")
        return set()
        
    try:
        with open(excludes_file, 'r') as f:
            # 空行とコメント行を除去し、各行をトリムしてセットに変換
            excludes = {line.strip() for line in f 
                       if line.strip() and not line.startswith('#')}
        logger.info(f"除外リストを読み込みました: {len(excludes)}件")
        return excludes
    except Exception as e:
        logger.error(f"除外リストの読み込みでエラー: {str(e)}")
        return set()

def add_to_excludes(entry: str, excludes_file: str) -> None:
    """excludes_fileにエントリを追加"""
    with open(excludes_file, "a") as f:
        f.write(entry + "\n")

def remove_from_excludes(entry: str, excludes_file: str) -> None:
    """excludes_fileからエントリを削除"""
    if not os.path.exists(excludes_file):
        return
    with open(excludes_file, "r") as f:
        lines = f.readlines()
    with open(excludes_file, "w") as f:
        for line in f:
            if line.strip() != entry:
                f.write(line)

@cli.command()
@click.option("--csv-path", required=True, type=click.Path(exists=True),
              help="アノテーションCSVファイルのパス")
@click.option("--video-dir", required=True, type=click.Path(exists=True),
              help="切り出した動画が格納されているディレクトリ")
@click.option("--save-dataset", default="aptos_dataset.pth", type=click.Path(),
              help="保存するデータセットファイルのパス (デフォルト: aptos_dataset.pth)")
@click.option("--n-samples", default=None, type=int,
              help="サンプリングする総データ数（指定しない場合は全データを使用）")
@click.option("--excludes-file", default="excludes.txt", type=click.Path(),
              help="除外するビデオIDのリストを含むテキストファイル")
@click.option("--intermediate-save-dir", default=None, type=click.Path(),
              help="中間結果として各動画の特徴量を保存するディレクトリ（指定しない場合はvideo-dirを使用）")
def create_dataset(csv_path: str, video_dir: str, save_dataset: str, 
                   n_samples: int, excludes_file: str, intermediate_save_dir: str):
    """動画から特徴量を抽出してデータセットを作成
    （処理中のファイルはexcludes.txtに一時的に登録され、正常終了時に削除されます）
    処理手順:
    1. CSVファイルからアノテーション情報を読み込み
    2. 除外リストに基づいてデータをフィルタリング
    3. 必要に応じてデータをサンプリング
    4. 各動画から特徴量を抽出／中間結果があればロード（npz形式）
    5. トレーニング用とバリデーション用にデータを分割
    6. データセットをファイルに保存
    """
    import numpy as np  # ここでnumpyを利用
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    
    # 除外リストの読み込みと適用
    if excludes_file:
        excludes = load_excludes(excludes_file)
        if excludes:
            original_len = len(df)
            df = df[~df['video_id'].isin(excludes)]
            filtered_len = len(df)
            logger.info(f"除外リストを適用: {original_len - filtered_len}件のデータを除外")
    else:
        os.path.touch(excludes_file)
        logger.info(f"除外リストファイルが存在しません: {excludes_file}")
        logger.info(f"新しい除外リストファイルを作成しました")
    
    # サンプル数が指定されている場合、データをサンプリング
    if n_samples is not None:
        logger.info(f"データを{n_samples}サンプルにサンプリングします")
        df = sample_balanced_data(df, n_samples)
        logger.info(f"各phase_idのサンプル数:\n{df['phase_id'].value_counts()}")
    
    # intermediate_save_dirが指定されていなければ video_dir を利用
    if intermediate_save_dir is None:
        intermediate_save_dir = video_dir
    # ディレクトリが存在しなければ作成
    os.makedirs(intermediate_save_dir, exist_ok=True)
    
    processor = VideoProcessor()
    features_dict = {}

    # 各動画セグメントから特徴量を抽出
    for idx, row in df.iterrows():
        video_id = row["video_id"]
        start_time = row["start"]
        end_time = row["end"]
        
        # 動画識別子（キー）を作成
        feature_key = f"{video_id}_{start_time}_{end_time}"
        # 処理中のファイルをexcludes.txtに書き込み
        add_to_excludes(feature_key, excludes_file)
        
        # video_pathの構築
        video_path = os.path.join(video_dir, f"{video_id}_{start_time}_{end_time}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"動画ファイルが見つかりません: {video_path}")
            continue

        # npzファイルのパスを決定
        npz_filename = f"{video_id}_{start_time}_{end_time}.npz"
        npz_path = os.path.join(intermediate_save_dir, npz_filename)
        
        logger.info(f"特徴量抽出中 ({idx + 1}/{len(df)}): {video_path}")
        
        # 既にnpzファイルがあれば、特徴量を読み込む
        if os.path.exists(npz_path):
            try:
                logger.info(f"中間結果を読み込み: {npz_path}")
                data = np.load(npz_path)
                features = data["features"]
            except Exception as e:
                logger.error(f"npzファイルの読み込みに失敗: {str(e)}")
                features = None
        else:
            features = None
        
        # npzがなければ特徴量を抽出して保存
        if features is None:
            features = processor.extract_features(video_path)
            if features is not None:
                try:
                    np.savez_compressed(npz_path, features=features)
                    logger.info(f"中間結果を保存: {npz_path}")
                except Exception as e:
                    logger.error(f"中間結果の保存に失敗: {str(e)}")
            else:
                logger.error(f"特徴量の抽出に失敗: {video_path}")
                continue
        
        # 正常に処理が完了した動画はexcludes.txtからエントリを削除
        remove_from_excludes(feature_key, excludes_file)
        # 辞書に保存
        features_dict[feature_key] = features

    logger.info("データセットの作成を開始")
    
    # DataFrameをトレーニングとバリデーションに分割
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # データセットの作成
    train_dataset = APTOSDataset(features_dict, train_df)
    val_dataset = APTOSDataset(features_dict, val_df)
    
    # データセットの保存
    torch.save({
        'features_dict': features_dict,
        'train_df': train_df,
        'val_df': val_df,
        'num_classes': len(df['phase_id'].unique())
    }, save_dataset)
    
    logger.info(f"データセットを保存しました: {save_dataset}")
    logger.info(f"トレーニングデータ数: {len(train_dataset)}")
    logger.info(f"バリデーションデータ数: {len(val_dataset)}")

if __name__ == "__main__":
    cli()