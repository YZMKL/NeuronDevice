"""
HIL (Hardware-in-the-Loop) 训练框架
HIL Training Framework

核心理念:
- Forward: 使用模拟器件（或硬件模型）
- Backward: 使用数字梯度 + STE
- Update: 更新数字影子权重 W_float，真实器件不写入

训练流程:
1. 数字输入 → DAC
2. Crossbar MVM (V·(G⁺-G⁻))
3. ADC → 数字输出
4. + bias → 激活 → y_hat
5. Loss 计算 → Backward (STE)
6. 优化器更新 W_float
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, List, Tuple
from tqdm import tqdm
import numpy as np


class HILTrainer:
    """
    HIL训练器
    封装完整的训练和评估流程
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Args:
            model: 使用Crossbar层的神经网络模型
            optimizer: 优化器
            criterion: 损失函数
            device: 训练设备
            scheduler: 学习率调度器
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        normalize_input: bool = True
    ) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            normalize_input: 是否将输入归一化到 [0, 1]
            
        Returns:
            平均损失, 准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 输入归一化到 [0, 1]
            if normalize_input:
                data = self._normalize_input(data)
            
            self.optimizer.zero_grad()
            
            # Forward pass (使用模拟器件)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass (STE 梯度)
            loss.backward()
            
            # 更新数字影子权重
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(
        self,
        val_loader: DataLoader,
        normalize_input: bool = True
    ) -> Tuple[float, float]:
        """
        评估模型
        
        Args:
            val_loader: 验证数据加载器
            normalize_input: 是否归一化输入
            
        Returns:
            平均损失, 准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if normalize_input:
                    data = self._normalize_input(data)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        normalize_input: bool = True,
        save_best: bool = True,
        save_path: str = 'best_model.pth'
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            normalize_input: 是否归一化输入
            save_best: 是否保存最佳模型
            save_path: 模型保存路径
            
        Returns:
            训练历史
        """
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, normalize_input)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # 验证
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, normalize_input)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # 保存最佳模型
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                    print(f'Best model saved with accuracy: {val_acc:.2f}%')
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
        
        return self.history
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入归一化到 [0, 1]
        
        Args:
            x: 输入张量
            
        Returns:
            归一化后的张量
        """
        # 假设输入已经在合理范围，做简单的min-max归一化
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-10:
            return (x - x_min) / (x_max - x_min)
        else:
            return x


class HILInference:
    """
    HIL推理器
    使用训练好的模型进行推理
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        add_device_noise: bool = True
    ):
        """
        Args:
            model: 训练好的Crossbar模型
            device: 推理设备
            add_device_noise: 推理时是否添加器件噪声
        """
        self.model = model.to(device)
        self.device = device
        self.add_device_noise = add_device_noise
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 如果不需要噪声，禁用各层的噪声
        if not add_device_noise:
            self._disable_noise()
    
    def _disable_noise(self):
        """禁用所有层的噪声"""
        for module in self.model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = 0.0
    
    def predict(
        self,
        x: torch.Tensor,
        normalize_input: bool = True
    ) -> torch.Tensor:
        """
        预测
        
        Args:
            x: 输入张量
            normalize_input: 是否归一化输入
            
        Returns:
            预测结果
        """
        x = x.to(self.device)
        
        if normalize_input:
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min > 1e-10:
                x = (x - x_min) / (x_max - x_min)
        
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
        normalize_input: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用蒙特卡洛采样估计预测不确定性
        
        Args:
            x: 输入张量
            n_samples: 采样次数
            normalize_input: 是否归一化输入
            
        Returns:
            预测均值, 预测标准差
        """
        x = x.to(self.device)
        
        if normalize_input:
            x_min = x.min()
            x_max = x.max()
            if x_max - x_min > 1e-10:
                x = (x - x_min) / (x_max - x_min)
        
        # 临时启用噪声进行采样
        self.model.train()  # 启用训练模式以添加噪声
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                predictions.append(output)
        
        self.model.eval()
        
        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std

