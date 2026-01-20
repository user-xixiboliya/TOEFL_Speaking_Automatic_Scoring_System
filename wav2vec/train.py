"""
训练脚本 - 语音评分模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import time

from config import Config
from dataset import create_dataloaders
from model import create_model


class Trainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置对象
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # Early stopping 计数器
        self.early_stopping_counter = 0
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # 移动到设备
            features = features.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE).unsqueeze(1)  # [batch] -> [batch, 1]
            
            # 前向传播
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item() * features.size(0)
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.train_loader.dataset)
        
        # 计算 MAE 和 MSE
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        
        return epoch_loss, mae, mse
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                # 移动到设备
                features = features.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE).unsqueeze(1)
                
                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item() * features.size(0)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # 计算平均损失
        epoch_loss = running_loss / len(self.val_loader.dataset)
        
        # 计算 MAE 和 MSE
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        mae = mean_absolute_error(all_labels, all_preds)
        mse = mean_squared_error(all_labels, all_preds)
        
        # 计算 Pearson 相关系数
        correlation, _ = pearsonr(all_labels, all_preds)
        
        return epoch_loss, mae, mse, correlation, all_preds, all_labels
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60 + "\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_mae, train_mse = self.train_epoch()
            
            # 验证
            val_loss, val_mae, val_mse, val_corr, _, _ = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # 计算 epoch 时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印信息
            print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | MSE: {train_mse:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | MSE: {val_mse:.4f} | Corr: {val_corr:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            print()
        
        print("="*60)
        print("训练完成")
        print("="*60 + "\n")
    
    def save_checkpoint(self, name='checkpoint'):
        """保存模型检查点"""
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.config.MODEL_SAVE_DIR,
            f'{name}_model.pth'
        )
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"已加载检查点: {checkpoint_path}")


def evaluate_model(model, val_loader, config):
    """
    评估模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        config: 配置
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(config.DEVICE)
            
            outputs = model(features)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    correlation, p_value = pearsonr(all_labels, all_preds)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson': correlation,
        'P-value': p_value
    }
    
    return metrics, all_preds, all_labels


def main():
    """主函数"""
    # 加载配置
    config = Config()
    config.print_config()
    
    # 创建数据加载器
    train_loader, val_loader, dataset_sizes = create_dataloaders(
        json_path=config.DATASET_JSON,
        train_ratio=config.TRAIN_RATIO,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        sample_rate=config.SAMPLE_RATE,
        max_length=config.MAX_AUDIO_LENGTH,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    
    # 创建模型
    model = create_model(config)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # 训练
    trainer.train()
    
    # 加载最佳模型
    best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型: {best_model_path}\n")
    
    # 最终评估
    print("="*60)
    print("最终测试集评估")
    print("="*60 + "\n")
    
    metrics, preds, labels = evaluate_model(model, val_loader, config)
    
    print("测试集指标:")
    print(f"  MSE (均方误差):        {metrics['MSE']:.4f}")
    print(f"  RMSE (均方根误差):     {metrics['RMSE']:.4f}")
    print(f"  MAE (平均绝对误差):    {metrics['MAE']:.4f}")
    print(f"  Pearson 相关系数:      {metrics['Pearson']:.4f}")
    print(f"  P-value:              {metrics['P-value']:.6f}")
    print()
    
    # 打印一些预测示例
    print("预测示例 (前10个):")
    print("  真实值 | 预测值 | 误差")
    print("  " + "-"*30)
    for i in range(min(10, len(labels))):
        error = abs(labels[i] - preds[i])
        print(f"  {labels[i]:.2f}   | {preds[i]:.2f}   | {error:.2f}")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
