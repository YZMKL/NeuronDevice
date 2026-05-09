"""
工具函数模块
Utility functions for device data loading and analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os


def load_device_data(
    set_file: str,
    reset_file: str,
    max_rows: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载器件SET/RESET数据
    
    Args:
        set_file: SET过程数据文件路径
        reset_file: RESET过程数据文件路径
        max_rows: 最大读取行数
        
    Returns:
        set_conductances: SET过程电导数据 [rows, devices]
        reset_conductances: RESET过程电导数据 [rows, devices]
    """
    # 读取数据
    set_df = pd.read_csv(set_file)
    reset_df = pd.read_csv(reset_file)
    
    # 提取有效数据列（排除空列）
    set_cols = [c for c in set_df.columns if not c.startswith('Unnamed')]
    reset_cols = [c for c in reset_df.columns if not c.startswith('Unnamed')]
    
    # 提取电导值
    set_values = set_df[set_cols].iloc[1:max_rows+1].values.astype(float)
    reset_values = reset_df[reset_cols].iloc[1:max_rows+1].values.astype(float)
    
    # 去除NaN
    set_values = np.nan_to_num(set_values, nan=0.0)
    reset_values = np.nan_to_num(reset_values, nan=0.0)
    
    return set_values, reset_values


def extract_conductance_states(
    conductances: np.ndarray,
    n_states: int = 64,
    method: str = 'log_uniform'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从实验电导数据中提取代表性电导态
    
    Args:
        conductances: 所有电导值
        n_states: 目标电导态数量
        method: 提取方法 ('log_uniform', 'linear', 'percentile')
        
    Returns:
        states: 电导态中心值
        stds: 各电导态的标准差
    """
    # 展平并过滤
    flat = conductances.flatten()
    flat = flat[flat > 0]  # 过滤0值
    
    g_min = flat.min()
    g_max = flat.max()
    
    if method == 'log_uniform':
        # 对数域均匀分布
        log_min = np.log10(g_min + 1e-20)
        log_max = np.log10(g_max)
        log_states = np.linspace(log_min, log_max, n_states)
        states = 10 ** log_states
        
    elif method == 'linear':
        # 线性均匀分布
        states = np.linspace(g_min, g_max, n_states)
        
    elif method == 'percentile':
        # 按百分位数分布（更好地匹配数据分布）
        percentiles = np.linspace(0, 100, n_states)
        states = np.percentile(flat, percentiles)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 计算每个态的局部标准差
    stds = np.zeros(n_states)
    for i, s in enumerate(states):
        # 找到最近的点
        distances = np.abs(flat - s)
        nearest_idx = np.argsort(distances)[:min(10, len(flat))]
        stds[i] = np.std(flat[nearest_idx]) if len(nearest_idx) > 1 else 0
    
    return states, stds


def analyze_device_characteristics(
    set_conductances: np.ndarray,
    reset_conductances: np.ndarray
) -> dict:
    """
    分析器件特性
    
    Args:
        set_conductances: SET过程电导
        reset_conductances: RESET过程电导
        
    Returns:
        特性字典
    """
    # 合并所有数据
    all_g = np.concatenate([
        set_conductances.flatten(),
        reset_conductances.flatten()
    ])
    all_g = all_g[all_g > 0]
    
    # 计算统计量
    stats = {
        'g_min': float(all_g.min()),
        'g_max': float(all_g.max()),
        'g_mean': float(all_g.mean()),
        'g_std': float(all_g.std()),
        'g_median': float(np.median(all_g)),
        'dynamic_range': float(all_g.max() / all_g.min()),
        'log_dynamic_range': float(np.log10(all_g.max() / all_g.min())),
        'n_samples': len(all_g),
    }
    
    # SET/RESET范围
    set_flat = set_conductances.flatten()
    set_flat = set_flat[set_flat > 0]
    reset_flat = reset_conductances.flatten()
    reset_flat = reset_flat[reset_flat > 0]
    
    stats['set_range'] = (float(set_flat.min()), float(set_flat.max()))
    stats['reset_range'] = (float(reset_flat.min()), float(reset_flat.max()))
    
    return stats


def calculate_weight_mapping_error(
    weights: np.ndarray,
    n_states: int,
    g_min: float,
    g_max: float
) -> dict:
    """
    计算权重映射误差
    
    Args:
        weights: 原始权重
        n_states: 电导量化级数
        g_min: 最小电导
        g_max: 最大电导
        
    Returns:
        误差统计
    """
    # 归一化权重
    w_max = np.abs(weights).max()
    w_norm = weights / (w_max + 1e-10)
    
    # 差分拆分
    w_pos = np.maximum(w_norm, 0)
    w_neg = np.maximum(-w_norm, 0)
    
    # 创建电导态
    states = np.linspace(0, 1, n_states)
    
    # 量化
    def quantize(w):
        distances = np.abs(w[..., np.newaxis] - states.reshape([1]*w.ndim + [-1]))
        indices = np.argmin(distances, axis=-1)
        return states[indices]
    
    w_pos_q = quantize(w_pos)
    w_neg_q = quantize(w_neg)
    
    # 重构权重
    w_recon = (w_pos_q - w_neg_q) * w_max
    
    # 计算误差
    error = weights - w_recon
    
    return {
        'mse': float(np.mean(error ** 2)),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
        'max_error': float(np.abs(error).max()),
        'relative_error': float(np.mean(np.abs(error) / (np.abs(weights) + 1e-10))),
    }


def print_model_crossbar_info(model) -> None:
    """
    打印模型中Crossbar层的信息
    
    Args:
        model: PyTorch模型
    """
    from .crossbar_layers import CrossbarLinear, CrossbarConv2d
    
    print('\n' + '='*60)
    print('Crossbar层信息')
    print('='*60)
    
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, CrossbarLinear):
            layer_idx += 1
            print(f'\n层 {layer_idx}: {name} (CrossbarLinear)')
            print(f'  输入维度: {module.in_features}')
            print(f'  输出维度: {module.out_features}')
            print(f'  DAC位宽: {module.dac.n_bits}')
            print(f'  ADC位宽: {module.adc.n_bits}')
            print(f'  电导态数: {module.weight_mapper.n_states}')
            print(f'  Crossbar大小: {module.in_features} × {module.out_features}')
            print(f'  差分对器件数: {module.in_features * module.out_features * 2}')
            
        elif isinstance(module, CrossbarConv2d):
            layer_idx += 1
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            crossbar_size = module.in_channels * kernel_size
            print(f'\n层 {layer_idx}: {name} (CrossbarConv2d)')
            print(f'  输入通道: {module.in_channels}')
            print(f'  输出通道: {module.out_channels}')
            print(f'  卷积核大小: {module.kernel_size}')
            print(f'  DAC位宽: {module.dac.n_bits}')
            print(f'  ADC位宽: {module.adc.n_bits}')
            print(f'  电导态数: {module.weight_mapper.n_states}')
            print(f'  展开后Crossbar大小: {crossbar_size} × {module.out_channels}')
            print(f'  差分对器件数: {crossbar_size * module.out_channels * 2}')
    
    print('\n' + '='*60)

