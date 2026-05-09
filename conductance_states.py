"""
电导态模块 - 从真实器件数据加载稳定可达的电导态
Conductance States Module - Load stable reachable conductance states from real device data

数据格式要求:
- 第一列: 脉冲数 (pulse_count)
- 第二列: 平均电导值 (G_mean)
- 第三列: 标准差 (G_std)

核心功能:
1. 加载器件电导态数据
2. 构造等效器件权重集合 G = {(G_k, σ_k)}
3. 权重 → 最近电导态映射
4. Forward时使用高斯采样 G ~ N(G_k, σ_k²)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass


@dataclass
class ConductanceStateData:
    """电导态数据结构"""
    pulse_count: np.ndarray   # 脉冲数
    g_mean: np.ndarray        # 平均电导值 (S)
    g_std: np.ndarray         # 电导标准差 (S)
    
    @property
    def n_states(self) -> int:
        return len(self.g_mean)
    
    @property
    def g_min(self) -> float:
        return float(self.g_mean.min())
    
    @property
    def g_max(self) -> float:
        return float(self.g_mean.max())
    
    def __repr__(self):
        return (f"ConductanceStateData(n_states={self.n_states}, "
                f"g_range=[{self.g_min:.2e}, {self.g_max:.2e}] S)")


class ConductanceStates:
    """
    器件电导态管理类
    
    从真实器件数据加载稳定可达的电导态集合:
    G = {(G_k, σ_k)}_{k=1}^K
    
    其中:
    - G_k 是量化中心（平均电导）
    - σ_k 是forward噪声来源（标准差）
    """
    
    def __init__(
        self,
        g_mean: np.ndarray,
        g_std: np.ndarray,
        pulse_count: Optional[np.ndarray] = None
    ):
        """
        Args:
            g_mean: 平均电导值数组 [K]，单位 S
            g_std: 电导标准差数组 [K]，单位 S
            pulse_count: 脉冲数数组 [K]（可选）
        """
        # 确保按电导值排序
        sort_idx = np.argsort(g_mean)
        self.g_mean = np.array(g_mean)[sort_idx].astype(np.float64)
        self.g_std = np.array(g_std)[sort_idx].astype(np.float64)
        
        if pulse_count is not None:
            self.pulse_count = np.array(pulse_count)[sort_idx]
        else:
            self.pulse_count = np.arange(len(g_mean))
        
        # 电导范围
        self.g_min = self.g_mean.min()
        self.g_max = self.g_mean.max()
        self.n_states = len(self.g_mean)
        
        # 计算归一化电导态（用于权重映射）
        # G_normalized = (G - G_min) / (G_max - G_min) ∈ [0, 1]
        self.g_normalized = (self.g_mean - self.g_min) / (self.g_max - self.g_min + 1e-20)
        
        # 归一化标准差
        self.g_std_normalized = self.g_std / (self.g_max - self.g_min + 1e-20)
    
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        pulse_col: int = 0,
        mean_col: int = 1,
        std_col: int = 2,
        skip_rows: int = 1,
        delimiter: str = ','
    ) -> 'ConductanceStates':
        """
        从CSV文件加载电导态数据
        
        Args:
            file_path: CSV文件路径
            pulse_col: 脉冲数列索引
            mean_col: 平均电导列索引
            std_col: 标准差列索引
            skip_rows: 跳过的行数（表头）
            delimiter: 分隔符
        """
        df = pd.read_csv(file_path, delimiter=delimiter, header=None, skiprows=skip_rows)
        
        pulse_count = df.iloc[:, pulse_col].values.astype(float)
        g_mean = df.iloc[:, mean_col].values.astype(float)
        g_std = df.iloc[:, std_col].values.astype(float)
        
        # 过滤无效数据
        valid_mask = ~(np.isnan(g_mean) | np.isnan(g_std))
        pulse_count = pulse_count[valid_mask]
        g_mean = g_mean[valid_mask]
        g_std = g_std[valid_mask]
        
        return cls(g_mean, g_std, pulse_count)
    
    @classmethod
    def from_excel(
        cls,
        file_path: str,
        sheet_name: Union[str, int] = 0,
        pulse_col: int = 0,
        mean_col: int = 1,
        std_col: int = 2,
        skip_rows: int = 1
    ) -> 'ConductanceStates':
        """
        从Excel文件加载电导态数据
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称或索引
            pulse_col: 脉冲数列索引
            mean_col: 平均电导列索引
            std_col: 标准差列索引
            skip_rows: 跳过的行数
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=skip_rows)
        
        pulse_count = df.iloc[:, pulse_col].values.astype(float)
        g_mean = df.iloc[:, mean_col].values.astype(float)
        g_std = df.iloc[:, std_col].values.astype(float)
        
        # 过滤无效数据
        valid_mask = ~(np.isnan(g_mean) | np.isnan(g_std))
        pulse_count = pulse_count[valid_mask]
        g_mean = g_mean[valid_mask]
        g_std = g_std[valid_mask]
        
        return cls(g_mean, g_std, pulse_count)
    
    @classmethod
    def from_arrays(
        cls,
        pulse_count: np.ndarray,
        g_mean: np.ndarray,
        g_std: np.ndarray
    ) -> 'ConductanceStates':
        """
        从NumPy数组创建
        
        Args:
            pulse_count: 脉冲数数组
            g_mean: 平均电导数组
            g_std: 标准差数组
        """
        return cls(g_mean, g_std, pulse_count)
    
    @classmethod
    def create_linear(
        cls,
        g_min: float = 1e-9,
        g_max: float = 1e-5,
        n_states: int = 64,
        relative_std: float = 0.05
    ) -> 'ConductanceStates':
        """
        创建线性分布的电导态（用于测试）
        
        Args:
            g_min: 最小电导
            g_max: 最大电导
            n_states: 电导态数量
            relative_std: 相对标准差（相对于电导值）
        """
        pulse_count = np.arange(n_states)
        g_mean = np.linspace(g_min, g_max, n_states)
        g_std = g_mean * relative_std
        return cls(g_mean, g_std, pulse_count)
    
    @classmethod
    def create_log(
        cls,
        g_min: float = 1e-9,
        g_max: float = 1e-5,
        n_states: int = 64,
        relative_std: float = 0.05
    ) -> 'ConductanceStates':
        """
        创建对数分布的电导态（更适合跨多个数量级的器件）
        
        Args:
            g_min: 最小电导
            g_max: 最大电导
            n_states: 电导态数量
            relative_std: 相对标准差
        """
        pulse_count = np.arange(n_states)
        log_min = np.log10(g_min)
        log_max = np.log10(g_max)
        g_mean = 10 ** np.linspace(log_min, log_max, n_states)
        g_std = g_mean * relative_std
        return cls(g_mean, g_std, pulse_count)
    
    def get_info(self) -> dict:
        """获取电导态信息"""
        return {
            'n_states': self.n_states,
            'g_min': self.g_min,
            'g_max': self.g_max,
            'g_range_ratio': self.g_max / self.g_min,
            'avg_relative_std': np.mean(self.g_std / self.g_mean),
            'pulse_range': (int(self.pulse_count.min()), int(self.pulse_count.max())),
        }
    
    def __repr__(self):
        info = self.get_info()
        return (f"ConductanceStates(\n"
                f"  n_states={info['n_states']},\n"
                f"  g_range=[{info['g_min']:.2e}, {info['g_max']:.2e}] S,\n"
                f"  dynamic_ratio={info['g_range_ratio']:.1f}x,\n"
                f"  avg_relative_std={info['avg_relative_std']*100:.2f}%,\n"
                f"  pulse_range={info['pulse_range']}\n"
                f")")


class ConductanceStatesTorchLUT(nn.Module):
    """
    PyTorch版本的电导态查找表
    用于权重到电导的映射和采样
    """
    
    def __init__(self, conductance_states: ConductanceStates):
        """
        Args:
            conductance_states: ConductanceStates对象
        """
        super().__init__()
        
        self.n_states = conductance_states.n_states
        self.g_min = conductance_states.g_min
        self.g_max = conductance_states.g_max
        
        # 注册buffer（不参与梯度）
        self.register_buffer('g_mean', 
            torch.tensor(conductance_states.g_mean, dtype=torch.float32))
        self.register_buffer('g_std',
            torch.tensor(conductance_states.g_std, dtype=torch.float32))
        self.register_buffer('g_normalized',
            torch.tensor(conductance_states.g_normalized, dtype=torch.float32))
        self.register_buffer('g_std_normalized',
            torch.tensor(conductance_states.g_std_normalized, dtype=torch.float32))
    
    def map_weight_to_state_index(self, normalized_weight: torch.Tensor) -> torch.Tensor:
        """
        将归一化权重 [0, 1] 映射到最近电导态的索引
        
        实现公式:
        G(Ŵ) = argmin_{G_k ∈ G} |Ŵ - (G_k - G_min)/(G_max - G_min)|
        
        Args:
            normalized_weight: 归一化权重 [0, 1]
            
        Returns:
            电导态索引
        """
        w_flat = normalized_weight.flatten()
    
        # g_normalized 已经是升序 [0,1]，直接 bucketize
        idx = torch.bucketize(w_flat, self.g_normalized).clamp(0, self.n_states - 1)
    
        # 检查左右邻居哪个更近
        idx_left = (idx - 1).clamp(0, self.n_states - 1)
        dist_right = torch.abs(w_flat - self.g_normalized[idx])
        dist_left  = torch.abs(w_flat - self.g_normalized[idx_left])
        indices = torch.where(dist_left < dist_right, idx_left, idx)
    
        return indices.view(normalized_weight.shape)
    
    def get_conductance(
        self,
        normalized_weight: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取映射后的电导值
        
        Args:
            normalized_weight: 归一化权重 [0, 1]
            add_noise: 是否添加噪声 G ~ N(G_k, σ_k²)
            
        Returns:
            conductance: 电导值
            state_indices: 电导态索引
        """
        # 获取最近电导态索引
        indices = self.map_weight_to_state_index(normalized_weight)
        indices_flat = indices.view(-1)
        
        # 获取电导均值
        g_mean = self.g_mean[indices_flat].view(normalized_weight.shape)
        
        if add_noise:
            # 获取对应的标准差
            g_std = self.g_std[indices_flat].view(normalized_weight.shape)
            # 高斯采样: G ~ N(G_k, σ_k²)
            noise = torch.randn_like(g_mean) * g_std
            conductance = g_mean + noise
            # 裁剪到有效范围
            conductance = torch.clamp(conductance, self.g_min, self.g_max)
        else:
            conductance = g_mean
        
        return conductance, indices
    
    def get_normalized_conductance(
        self,
        normalized_weight: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取归一化电导值 [0, 1]
        
        Args:
            normalized_weight: 归一化权重 [0, 1]
            add_noise: 是否添加噪声
            
        Returns:
            normalized_conductance: 归一化电导 [0, 1]
            state_indices: 电导态索引
        """
        g, indices = self.get_conductance(normalized_weight, add_noise)
        g_normalized = (g - self.g_min) / (self.g_max - self.g_min + 1e-20)
        return g_normalized, indices


class WeightToConductanceMapper(nn.Module):
    """
    权重到电导的完整映射器
    
    实现流程:
    1. Layer-wise归一化: W → Ŵ ∈ [-1, 1]
    2. 差分拆分: Ŵ → (Ŵ⁺, Ŵ⁻)
    3. 映射到最近电导态: (Ŵ⁺, Ŵ⁻) → (G⁺, G⁻)
    4. Forward时高斯采样: G ~ N(G_k, σ_k²)
    """
    
    def __init__(self, conductance_states: ConductanceStates):
        """
        Args:
            conductance_states: ConductanceStates对象，包含真实器件电导态数据
        """
        super().__init__()
        
        self.lut = ConductanceStatesTorchLUT(conductance_states)
        self.g_min = conductance_states.g_min
        self.g_max = conductance_states.g_max
        self.n_states = conductance_states.n_states
    
    def normalize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 1: Layer-wise权重归一化
        
        Ŵ = W / max|W|
        
        Args:
            weight: 原始权重
            
        Returns:
            normalized_weight: 归一化权重 [-1, 1]
            scale: 缩放因子（用于恢复输出尺度）
        """
        scale = weight.abs().max() + 1e-10
        normalized_weight = weight / scale
        return normalized_weight, scale
    
    def differential_split(
        self,
        normalized_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 2: 差分拆分
        
        Ŵ⁺ = max(Ŵ, 0)
        Ŵ⁻ = max(-Ŵ, 0)
        
        Args:
            normalized_weight: 归一化权重 [-1, 1]
            
        Returns:
            w_pos: 正权重 [0, 1]
            w_neg: 负权重 [0, 1]
        """
        w_pos = torch.clamp(normalized_weight, min=0)
        w_neg = torch.clamp(-normalized_weight, min=0)
        return w_pos, w_neg
    
    def map_to_conductance(
        self,
        w_normalized: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 3-4: 映射到最近电导态
        
        G(Ŵ) = argmin_{G_k} |Ŵ - (G_k - G_min)/(G_max - G_min)|
        
        Step 5: Forward时高斯采样
        G ~ N(G_k, σ_k²)
        
        Args:
            w_normalized: 归一化权重 [0, 1]
            add_noise: 是否添加噪声
            
        Returns:
            conductance: 电导值
            state_indices: 电导态索引
        """
        return self.lut.get_normalized_conductance(w_normalized, add_noise)
    
    def forward(
        self,
        weight: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的权重到电导映射
        
        Args:
            weight: 原始权重
            add_noise: 是否添加噪声
            
        Returns:
            g_pos: 正电导 G⁺ (归一化)
            g_neg: 负电导 G⁻ (归一化)
            scale: 缩放因子
        """
        # Step 1: 归一化
        w_norm, scale = self.normalize_weights(weight)
        
        # Step 2: 差分拆分
        w_pos, w_neg = self.differential_split(w_norm)
        
        # Step 3-5: 映射到电导态（带噪声采样）
        g_pos, _ = self.map_to_conductance(w_pos, add_noise)
        g_neg, _ = self.map_to_conductance(w_neg, add_noise)
        
        return g_pos, g_neg, scale
    
    def get_effective_weight(
        self,
        weight: torch.Tensor,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        获取有效权重 (G⁺ - G⁻) * scale
        
        这是实际用于计算 I = V · (G⁺ - G⁻) 的权重
        
        Args:
            weight: 原始权重
            add_noise: 是否添加噪声
            
        Returns:
            有效权重
        """
        g_pos, g_neg, scale = self.forward(weight, add_noise)
        return (g_pos - g_neg) * scale
    
    def compute_mapping_error(self, weight: torch.Tensor) -> dict:
        """
        计算权重映射误差
        
        Args:
            weight: 原始权重
            
        Returns:
            误差统计字典
        """
        with torch.no_grad():
            # 不加噪声的映射
            effective_weight = self.get_effective_weight(weight, add_noise=False)
            
            # 计算误差
            error = weight - effective_weight
            
            return {
                'mse': float((error ** 2).mean()),
                'rmse': float(torch.sqrt((error ** 2).mean())),
                'mae': float(error.abs().mean()),
                'max_error': float(error.abs().max()),
                'relative_error': float((error.abs() / (weight.abs() + 1e-10)).mean()),
            }

