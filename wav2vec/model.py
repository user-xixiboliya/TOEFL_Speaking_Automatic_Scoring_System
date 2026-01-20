"""
CNN-LSTM 模型 - 语音评分
"""

import torch
import torch.nn as nn


class CNN_LSTM_Regressor(nn.Module):
    """
    CNN-LSTM 回归模型
    
    架构:
        Input: Mel-Spectrogram [batch, 1, n_mels, time]
        ↓
        CNN 层 (提取局部特征)
        ↓
        LSTM 层 (时序建模)
        ↓
        全局池化
        ↓
        全连接层 (回归)
        ↓
        Output: Score [batch, 1]
    """
    
    def __init__(self, 
                 n_mels=128,
                 cnn_channels=[32, 64, 128],
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 lstm_dropout=0.3,
                 fc_hidden_size=128,
                 fc_dropout=0.3):
        """
        Args:
            n_mels: Mel 频率数量
            cnn_channels: CNN 每层的通道数
            lstm_hidden_size: LSTM 隐藏层大小
            lstm_num_layers: LSTM 层数
            lstm_dropout: LSTM dropout
            fc_hidden_size: 全连接隐藏层大小
            fc_dropout: 全连接层 dropout
        """
        super(CNN_LSTM_Regressor, self).__init__()
        
        # ==================== CNN 层 ====================
        self.cnn_layers = nn.ModuleList()
        
        in_channels = 1  # 输入是单通道 Mel-Spectrogram
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        
        # 计算 CNN 输出后的频率维度
        # 每经过一次 MaxPool2d(2, 2)，频率维度减半
        self.freq_dim_after_cnn = n_mels // (2 ** len(cnn_channels))
        
        # CNN 输出特征维度
        self.cnn_out_features = cnn_channels[-1] * self.freq_dim_after_cnn
        
        # ==================== LSTM 层 ====================
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # LSTM 输出维度（双向）
        lstm_output_size = lstm_hidden_size * 2
        
        # ==================== 全连接层 ====================
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, 1)  # 回归输出
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, 1, n_mels, time]
        
        Returns:
            score: [batch, 1]
        """
        batch_size = x.size(0)
        
        # ==================== CNN 特征提取 ====================
        # x: [batch, 1, n_mels, time]
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        # x: [batch, channels, freq, time]
        
        # 重塑为 LSTM 输入格式
        # [batch, channels, freq, time] -> [batch, time, channels*freq]
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.reshape(batch, time, channels * freq)  # [batch, time, features]
        
        # ==================== LSTM 时序建模 ====================
        # x: [batch, time, features]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, time, hidden*2]
        
        # ==================== 全局池化 ====================
        # 使用平均池化和最大池化的结合
        mean_pool = torch.mean(lstm_out, dim=1)  # [batch, hidden*2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [batch, hidden*2]
        
        # 使用平均池化（也可以尝试拼接两者）
        pooled = mean_pool
        
        # ==================== 回归输出 ====================
        score = self.fc(pooled)  # [batch, 1]
        
        return score
    
    def get_model_size(self):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params
        }


def create_model(config):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象
    
    Returns:
        model: CNN-LSTM 模型
    """
    model = CNN_LSTM_Regressor(
        n_mels=config.N_MELS,
        cnn_channels=config.CNN_CHANNELS,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_dropout=config.LSTM_DROPOUT,
        fc_hidden_size=config.FC_HIDDEN_SIZE,
        fc_dropout=config.FC_DROPOUT
    )
    
    # 移动到设备
    model = model.to(config.DEVICE)
    
    # 打印模型信息
    model_size = model.get_model_size()
    print("\n模型信息:")
    print(f"  总参数量: {model_size['total']:,}")
    print(f"  可训练参数量: {model_size['trainable']:,}")
    print(f"  设备: {config.DEVICE}\n")
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = CNN_LSTM_Regressor()
    
    # 创建随机输入
    x = torch.randn(4, 1, 128, 3000)  # [batch=4, channels=1, n_mels=128, time=3000]
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output.squeeze()}")
    
    # 模型大小
    size = model.get_model_size()
    print(f"\n模型参数量: {size['total']:,}")
