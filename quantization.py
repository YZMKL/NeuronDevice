"""
量化模块 - DAC和ADC的实现
Quantization Module - DAC and ADC implementation
"""

import torch
import torch.nn as nn
from typing import Optional


class StraightThroughEstimator(torch.autograd.Function):
    """
    直通估计器 (STE) - 用于量化操作的梯度传递
    前向：执行量化
    反向：梯度直通
    """
    
    @staticmethod
    def forward(ctx, x, n_bits, x_min, x_max):
        # 量化级数
        n_levels = 2 ** n_bits - 1
        
        # 归一化到 [0, 1]
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        x_norm = torch.clamp(x_norm, 0, 1)
        
        # 量化
        x_quant = torch.round(x_norm * n_levels) / n_levels
        
        # 反归一化
        x_out = x_quant * (x_max - x_min) + x_min
        
        return x_out
    
    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直通
        return grad_output, None, None, None


class DAC(nn.Module):
    """
    数模转换器 (DAC)
    将数字输入转换为模拟电压
    
    流程：数字输入 x ∈ [0, 1] → 量化 → 电压 V ∈ [0, V_max]
    """
    def __init__(
        self,
        n_bits: int = 8,
        v_max: float = 1.0,
        noise_std: float = 0.0,
        momentum: float = 0.1        # 新增
    ):
        super().__init__()
        self.n_bits = n_bits
        self.v_max = v_max
        self.noise_std = noise_std
        self.momentum = momentum
        
        # 新增：记录运行时输入范围
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 动态归一化，替换原来的硬clamp
        if self.training:
            x_min = x.min().detach()
            x_max = x.max().detach()
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * x_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * x_max
        else:
            x_min = self.running_min
            x_max = self.running_max
        
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        x_norm = torch.clamp(x_norm, 0, 1)  # 防止数值越界，不是截断信息
        
        # 量化（这部分不变）
        v = StraightThroughEstimator.apply(
            x_norm * self.v_max,
            self.n_bits,
            torch.tensor(0.0, device=x.device),
            torch.tensor(self.v_max, device=x.device)
        )
        
        # 添加噪声（这部分不变）
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(v) * self.noise_std * self.v_max
            v = v + noise
            v = torch.clamp(v, 0, self.v_max)
        
        return v

class ADC(nn.Module):
    """
    模数转换器 (ADC)
    将模拟电流转换为数字信号
    
    流程：电流 I → 归一化 → 量化 → 数字输出 ∈ [0, 1]
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        noise_std: float = 0.0,
        calibrated: bool = True
    ):
        super().__init__()
        self.n_bits = n_bits
        self.noise_std = noise_std
        self.calibrated = calibrated
        
        # 动态范围参数（可学习或固定）
        self.register_buffer('i_min', torch.tensor(0.0))
        self.register_buffer('i_max', torch.tensor(1.0))
        
    def calibrate(self, i_min: float, i_max: float):
        """校准ADC范围"""
        self.i_min.fill_(i_min)
        self.i_max.fill_(i_max)
        
    def forward(self, current: torch.Tensor, i_max: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            current: 输入电流
            i_max: 可选的动态最大电流范围
            
        Returns:
            量化后的数字信号 [0, 1]
        """
        if i_max is not None:
            _i_max = i_max
        else:
            _i_max = self.i_max
            
        _i_min = self.i_min
        
        # 归一化
        normalized = (current - _i_min) / (_i_max - _i_min + 1e-10)
        normalized = torch.clamp(normalized, 0, 1)
        
        # 量化
        quantized = StraightThroughEstimator.apply(
            normalized,
            self.n_bits,
            torch.tensor(0.0, device=current.device),
            torch.tensor(1.0, device=current.device)
        )
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(quantized) * self.noise_std
            quantized = quantized + noise
            quantized = torch.clamp(quantized, 0, 1)
        
        return quantized


class DynamicADC(nn.Module):
    """
    动态范围ADC
    根据每层的输出范围自动调整量化范围
    """
    
    def __init__(
        self,
        n_bits: int = 8,
        noise_std: float = 0.0,
        momentum: float = 0.1
    ):
        super().__init__()
        self.n_bits = n_bits
        self.noise_std = noise_std
        self.momentum = momentum
        
        self.register_buffer('running_max', torch.tensor(1.0))
        self.register_buffer('running_min', torch.tensor(0.0))
        
    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: 输入电流
            
        Returns:
            量化后的数字信号
        """
        if self.training:
            current_max = current.max().detach()
            current_min = current.min().detach()
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * current_max
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * current_min

        # 训练和推理都用running统计
        i_max = self.running_max
        i_min = self.running_min
        
        # 归一化
        normalized = (current - i_min) / (i_max - i_min + 1e-10)
        normalized = torch.clamp(normalized, 0, 1)
        
        # 量化
        quantized = StraightThroughEstimator.apply(
            normalized,
            self.n_bits,
            torch.tensor(0.0, device=current.device),
            torch.tensor(1.0, device=current.device)
        )
        
        # 反归一化（保持原始尺度）
        output = quantized * (i_max - i_min) + i_min
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(output) * self.noise_std
            output = output + noise
        
        return output

