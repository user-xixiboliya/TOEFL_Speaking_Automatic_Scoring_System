"""
测试脚本 - 验证环境、数据和模型
"""

import os
import sys
import json
import torch
import torchaudio

def check_environment():
    """检查环境"""
    print("\n" + "="*60)
    print("环境检查")
    print("="*60 + "\n")
    
    # Python 版本
    print(f"Python 版本: {sys.version.split()[0]}")
    
    # PyTorch
    print(f"PyTorch 版本: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用: ✓")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"CUDA 可用: ✗ (将使用CPU)")
    
    # torchaudio
    print(f"torchaudio 版本: {torchaudio.__version__}")
    
    print("\n✓ 环境检查完成\n")


def check_dataset(json_path):
    """检查数据集"""
    print("="*60)
    print("数据集检查")
    print("="*60 + "\n")
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 数据集文件不存在: {json_path}")
        return False
    
    print(f"✓ 数据集文件存在: {json_path}")
    
    # 加载数据集
    try:
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        print(f"✓ 成功加载数据集")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return False
    
    # 统计信息
    train_data = [d for d in dataset if d['split'] == 'train']
    
    print(f"\n数据集统计:")
    print(f"  Train 样本: {len(train_data)}")
    print(f"  总样本: {len(dataset)}")
    
    # 检查第一个样本
    if train_data:
        sample = train_data[0]
        print(f"\n第一个样本:")
        print(f"  ID: {sample['id']}")
        print(f"  音频路径: {sample['audio_path']}")
        print(f"  评分: {sample['score_overall']}")
        
        # 检查音频文件是否存在
        if os.path.exists(sample['audio_path']):
            print(f"  ✓ 音频文件存在")
            
            # 尝试加载音频
            try:
                waveform, sr = torchaudio.load(sample['audio_path'])
                duration = waveform.shape[1] / sr
                print(f"  ✓ 音频加载成功")
                print(f"    采样率: {sr} Hz")
                print(f"    时长: {duration:.2f} 秒")
                print(f"    波形形状: {waveform.shape}")
            except Exception as e:
                print(f"  ❌ 音频加载失败: {e}")
                return False
        else:
            print(f"  ❌ 音频文件不存在")
            return False
    
    print("\n✓ 数据集检查完成\n")
    return True


def test_model():
    """测试模型"""
    print("="*60)
    print("模型测试")
    print("="*60 + "\n")
    
    try:
        from model import CNN_LSTM_Regressor
        
        # 创建模型
        model = CNN_LSTM_Regressor()
        print("✓ 模型创建成功")
        
        # 测试前向传播
        batch_size = 2
        n_mels = 128
        time_steps = 3000
        
        # 随机输入
        x = torch.randn(batch_size, 1, n_mels, time_steps)
        
        # 前向传播
        output = model(x)
        
        print(f"✓ 前向传播成功")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出值: {output.squeeze().tolist()}")
        
        # 模型参数
        size = model.get_model_size()
        print(f"\n模型参数:")
        print(f"  总参数量: {size['total']:,}")
        print(f"  可训练参数: {size['trainable']:,}")
        
        print("\n✓ 模型测试完成\n")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loader():
    """测试数据加载器"""
    print("="*60)
    print("数据加载器测试")
    print("="*60 + "\n")
    
    try:
        from config import Config
        from dataset import create_dataloaders
        
        config = Config()
        
        print("创建数据加载器...")
        train_loader, val_loader, dataset_sizes = create_dataloaders(
            json_path=config.DATASET_JSON,
            train_ratio=config.TRAIN_RATIO,
            batch_size=2,  # 小 batch size 用于测试
            num_workers=0,  # 测试时不使用多线程
            sample_rate=config.SAMPLE_RATE,
            max_length=config.MAX_AUDIO_LENGTH,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        
        print("✓ 数据加载器创建成功")
        
        # 测试加载一个 batch
        print("\n测试加载一个 batch...")
        features, labels = next(iter(train_loader))
        
        print(f"✓ Batch 加载成功")
        print(f"  特征形状: {features.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签值: {labels.tolist()}")
        
        print("\n✓ 数据加载器测试完成\n")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("CNN-LSTM 语音评分系统 - 环境测试")
    print("="*60)
    
    # 1. 检查环境
    check_environment()
    
    # 2. 检查数据集
    dataset_path = "/root/autodl-tmp/sla_dataset_concat.json"
    if not check_dataset(dataset_path):
        print("数据集检查失败，请先生成数据集")
        return
    
    # 3. 测试模型
    if not test_model():
        print("模型测试失败")
        return
    
    # 4. 测试数据加载器
    if not test_dataset_loader():
        print("数据加载器测试失败")
        return
    
    # 全部通过
    print("="*60)
    print("✓ 所有测试通过！")
    print("="*60)
    print("\n可以开始训练了:")
    print("  python train.py")
    print()


if __name__ == "__main__":
    main()
