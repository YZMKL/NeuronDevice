"""
Benchmark 模块 - 标准神经网络对比基准
Standard Neural Network Benchmark for Comparison

用于与 Crossbar 仿真结果进行对比的标准神经网络实现。
"""

from .models import (
    StandardMLP,
    StandardCNN,
    StandardVGG,
    MNISTStandardMLP,
    MNISTStandardCNN,
    CIFAR10StandardMLP,
    CIFAR10StandardCNN,
    CIFAR10StandardVGG
)

from .train import (
    train_standard_model,
    BenchmarkLogger,
    create_benchmark_logger
)

__all__ = [
    # 模型
    'StandardMLP',
    'StandardCNN',
    'StandardVGG',
    'MNISTStandardMLP',
    'MNISTStandardCNN',
    'CIFAR10StandardMLP',
    'CIFAR10StandardCNN',
    'CIFAR10StandardVGG',
    # 训练
    'train_standard_model',
    'BenchmarkLogger',
    'create_benchmark_logger',
]

