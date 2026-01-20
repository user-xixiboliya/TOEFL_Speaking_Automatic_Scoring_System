"""
推理脚本 - 预测单个音频文件的评分
"""

import torch
import torchaudio
import argparse
import os

from config import Config
from model import create_model


def predict_audio(audio_path, model, config):
    """
    预测单个音频文件的评分
    
    Args:
        audio_path: 音频文件路径
        model: 训练好的模型
        config: 配置对象
    
    Returns:
        score: 预测的评分
    """
    model.eval()
    
    # 音频预处理
    # 加载音频
    waveform, sr = torchaudio.load(audio_path)
    
    # 重采样
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 裁剪或填充到固定长度
    max_samples = config.SAMPLE_RATE * config.MAX_AUDIO_LENGTH
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        pad_length = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    # 提取 Mel-Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    mel_spec = mel_transform(waveform)
    
    # 转换为分贝
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # 归一化
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # 添加 batch 维度
    mel_spec_db = mel_spec_db.unsqueeze(0)  # [1, 1, n_mels, time]
    
    # 移动到设备
    mel_spec_db = mel_spec_db.to(config.DEVICE)
    
    # 预测
    with torch.no_grad():
        output = model(mel_spec_db)
        score = output.item()
    
    return score


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预测音频评分')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', type=str, 
                       default='/root/autodl-tmp/models/checkpoints/best_model.pth',
                       help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在: {args.audio}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 加载配置
    config = Config()
    
    # 创建模型
    model = create_model(config)
    
    # 加载模型权重
    checkpoint = torch.load(args.model, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"已加载模型: {args.model}")
    print(f"音频文件: {args.audio}")
    print()
    
    # 预测
    score = predict_audio(args.audio, model, config)
    
    print("="*40)
    print(f"预测评分: {score:.2f}")
    print("="*40)


if __name__ == "__main__":
    main()
