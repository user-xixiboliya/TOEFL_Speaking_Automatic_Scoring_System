"""
配置文件 - 语音评分系统
"""

import torch

class Config:
    # ==================== 路径配置 ====================
    # 数据集 JSON 文件路径
    DATASET_JSON = "/root/autodl-tmp/sla_dataset_concat.json"
    
    # 模型保存路径
    MODEL_SAVE_DIR = "/root/autodl-tmp/models/checkpoints"
    
    # 最佳模型保存路径
    BEST_MODEL_PATH = "/root/autodl-tmp/models/best_model.pth"
    
    # ==================== 数据配置 ====================
    # 训练集验证集划分比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2
    
    # 随机种子
    RANDOM_SEED = 42
    
    # ==================== 音频配置 ====================
    # 采样率
    SAMPLE_RATE = 16000
    
    # 音频最大长度（秒）
    MAX_AUDIO_LENGTH = 30  # 30秒
    
    # Mel-Spectrogram 参数
    N_FFT = 400
    HOP_LENGTH = 160
    N_MELS = 128
    
    # ==================== 模型配置 ====================
    # CNN 配置
    CNN_CHANNELS = [32, 64, 128]  # 每层的通道数
    
    # LSTM 配置
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # 全连接层配置
    FC_HIDDEN_SIZE = 128
    FC_DROPOUT = 0.3
    
    # ==================== 训练配置 ====================
    # Batch size
    BATCH_SIZE = 16
    
    # 训练轮数
    NUM_EPOCHS = 100
    
    # 学习率
    LEARNING_RATE = 1e-3
    
    # 权重衰减
    WEIGHT_DECAY = 1e-4
    
    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 15
    
    # 学习率调度器 patience
    SCHEDULER_PATIENCE = 5
    
    # 学习率衰减因子
    SCHEDULER_FACTOR = 0.5
    
    # ==================== 设备配置 ====================
    # 自动检测 GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DataLoader 工作线程数
    NUM_WORKERS = 4
    
    # ==================== 评分范围 ====================
    MIN_SCORE = 1.0
    MAX_SCORE = 6.0
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "="*60)
        print("配置信息")
        print("="*60)
        print(f"数据集: {cls.DATASET_JSON}")
        print(f"设备: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"Early Stopping Patience: {cls.EARLY_STOPPING_PATIENCE}")
        print("="*60 + "\n")
