"""
数据集类 - 语音评分
"""

import json
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np


class SpeechScoringDataset(Dataset):
    """语音评分数据集"""
    
    def __init__(self, json_path, split='train', sample_rate=16000, 
                 max_length=30, n_mels=128, n_fft=400, hop_length=160):
        """
        Args:
            json_path: 数据集 JSON 文件路径
            split: 'train' 或 'val'
            sample_rate: 采样率
            max_length: 最大音频长度（秒）
            n_mels: Mel 频率数量
            n_fft: FFT 窗口大小
            hop_length: 帧移
        """
        # 加载数据集
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        
        # 过滤出 train split 的数据
        self.data = [item for item in all_data if item['split'] == 'train']
        
        # 数据集划分标记
        self.split = split
        
        # 音频参数
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = sample_rate * max_length
        
        # Mel-Spectrogram 变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # 振幅转分贝
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        print(f"加载 {split} 数据集: {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        item = self.data[idx]
        
        # 加载音频
        audio_path = item['audio_path']
        score = item['score_overall']
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样（如果需要）
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 裁剪或填充到固定长度
            if waveform.shape[1] > self.max_samples:
                # 裁剪
                waveform = waveform[:, :self.max_samples]
            else:
                # 填充
                pad_length = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # 提取 Mel-Spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # 转换为分贝
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            # 归一化
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # 转换为 tensor
            score_tensor = torch.tensor(score, dtype=torch.float32)
            
            return mel_spec_db, score_tensor
            
        except Exception as e:
            print(f"加载音频失败 {audio_path}: {e}")
            # 返回零填充
            mel_spec = torch.zeros((1, 128, 3000))  # 默认形状
            score_tensor = torch.tensor(3.0, dtype=torch.float32)  # 默认中间分数
            return mel_spec, score_tensor


def create_dataloaders(json_path, train_ratio=0.8, batch_size=16, 
                       num_workers=4, **kwargs):
    """
    创建训练和验证数据加载器
    
    Args:
        json_path: 数据集 JSON 路径
        train_ratio: 训练集比例
        batch_size: batch size
        num_workers: 数据加载线程数
        **kwargs: 传递给 Dataset 的其他参数
    
    Returns:
        train_loader, val_loader, dataset_sizes
    """
    from torch.utils.data import DataLoader, random_split
    
    # 创建完整数据集
    full_dataset = SpeechScoringDataset(json_path, split='train', **kwargs)
    
    # 计算划分大小
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # 随机划分
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_sizes = {
        'train': train_size,
        'val': val_size
    }
    
    print(f"\n数据集划分:")
    print(f"  训练集: {train_size} 个样本")
    print(f"  验证集: {val_size} 个样本")
    print(f"  总计: {total_size} 个样本\n")
    
    return train_loader, val_loader, dataset_sizes
