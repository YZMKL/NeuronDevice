"""
日志模块 - 训练日志记录
Logger Module - Training Log Recording

日志命名规则:
    {网络类型}_crossbar_{neuronactivation_如果有}_{数据集}_{时间戳}.txt
    
例如:
    MLP_crossbar_mnist_2026-1-29-12-54.txt
    CNN_crossbar_cifar10_2026-1-29-13-20.txt
    MLP_crossbar_neuronactivation_mnist_2026-1-29-14-00.txt
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, List, Any
from io import StringIO


class TrainingLogger:
    """
    训练日志记录器
    
    功能:
    1. 同时输出到终端和日志文件
    2. 自动生成日志文件名
    3. 记录配置信息和训练结果
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_type: str = "MLP",
        dataset: str = "mnist",
        use_device_neuron: bool = False,
        enabled: bool = True
    ):
        """
        Args:
            log_dir: 日志保存目录
            model_type: 模型类型 (MLP, CNN, VGG 等)
            dataset: 数据集名称 (mnist, cifar10 等)
            use_device_neuron: 是否使用自定义器件激活函数
            enabled: 是否启用日志
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.model_type = model_type.upper()
        self.dataset = dataset.lower()
        self.use_device_neuron = use_device_neuron
        
        self.log_file = None
        self.log_path = None
        self.buffer = StringIO()
        
        if self.enabled:
            self._setup_log_file()
    
    def _setup_log_file(self):
        """设置日志文件"""
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        # 生成文件名
        if self.use_device_neuron:
            filename = f"{self.model_type}_crossbar_neuronactivation_{self.dataset}_{timestamp}.txt"
        else:
            filename = f"{self.model_type}_crossbar_{self.dataset}_{timestamp}.txt"
        
        self.log_path = os.path.join(self.log_dir, filename)
        
        # 打开日志文件
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # 写入日志头
        self._write_header()
    
    def _write_header(self):
        """写入日志头信息"""
        header = f"""
{'='*70}
训练日志 - Training Log
{'='*70}
创建时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
模型类型: {self.model_type}
数据集: {self.dataset.upper()}
使用器件激活函数: {'是' if self.use_device_neuron else '否'}
日志文件: {self.log_path}
{'='*70}
"""
        if self.log_file:
            self.log_file.write(header)
            self.log_file.flush()
    
    def print(self, *args, **kwargs):
        """
        同时打印到终端和日志文件
        """
        # 生成消息
        output = StringIO()
        print(*args, file=output, **kwargs)
        message = output.getvalue()
        
        # 打印到终端
        print(message, end='')
        
        # 写入日志文件
        if self.enabled and self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def log_config(self, config: Any):
        """
        记录配置信息
        
        Args:
            config: 配置对象 (DeviceNeuronConfig 或 MyConfig)
        """
        self.print("\n" + "="*70)
        self.print("配置信息 - Configuration")
        self.print("="*70)
        
        if hasattr(config, 'print_config'):
            # 捕获 print_config 的输出
            old_stdout = sys.stdout
            sys.stdout = captured = StringIO()
            config.print_config()
            sys.stdout = old_stdout
            self.print(captured.getvalue())
        else:
            # 尝试打印配置属性
            self.print(f"\n{config}")
    
    def log_crossbar_config(self, crossbar_config: Any):
        """
        记录 Crossbar 配置
        
        Args:
            crossbar_config: CrossbarConfig 对象
        """
        self.print("\n" + "-"*50)
        self.print("Crossbar 配置:")
        self.print("-"*50)
        
        info = crossbar_config.get_info()
        self.print(f"  电导态数量: {info['n_states']}")
        self.print(f"  电导范围: [{info['g_range'][0]:.2e}, {info['g_range'][1]:.2e}] S")
        self.print(f"  DAC/ADC位宽: {info['dac_bits']}/{info['adc_bits']} bits")
        self.print(f"  DAC噪声: {info['dac_noise']}")
        self.print(f"  ADC噪声: {info['adc_noise']}")
    
    def log_model_info(self, model: Any, model_desc: str = ""):
        """
        记录模型信息
        
        Args:
            model: PyTorch 模型
            model_desc: 模型描述
        """
        self.print("\n" + "-"*50)
        self.print("模型信息:")
        self.print("-"*50)
        
        if model_desc:
            self.print(f"  结构: {model_desc}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.print(f"  总参数: {total_params:,}")
        self.print(f"  可训练参数: {trainable_params:,}")
    
    def log_dataset_info(self, train_size: int, test_size: int, img_info: str = ""):
        """
        记录数据集信息
        
        Args:
            train_size: 训练集大小
            test_size: 测试集大小
            img_info: 图像信息
        """
        self.print("\n" + "-"*50)
        self.print("数据集信息:")
        self.print("-"*50)
        self.print(f"  训练集: {train_size} 样本")
        self.print(f"  测试集: {test_size} 样本")
        if img_info:
            self.print(f"  图像: {img_info}")
    
    def log_training_params(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: str = "Adam",
        device: str = "cpu"
    ):
        """
        记录训练参数
        """
        self.print("\n" + "-"*50)
        self.print("训练参数:")
        self.print("-"*50)
        self.print(f"  Epochs: {epochs}")
        self.print(f"  Batch Size: {batch_size}")
        self.print(f"  Learning Rate: {learning_rate}")
        self.print(f"  Optimizer: {optimizer}")
        self.print(f"  Device: {device}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录 epoch 开始"""
        self.print(f"\nEpoch {epoch}/{total_epochs}")
        self.print("-"*50)
    
    def log_epoch_result(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """
        记录 epoch 结果
        
        Args:
            epoch: Epoch 编号（0 表示初始验证）
            train_loss: 训练损失（Epoch 0 时为 None）
            train_acc: 训练准确率（Epoch 0 时为 None）
            val_loss: 验证损失
            val_acc: 验证准确率
            is_best: 是否是最佳模型
        """
        if epoch == 0:
            # Epoch 0: 只显示验证结果（初始状态）
            self.print(f"Initial Val Loss: {val_loss:.4f}, Initial Val Acc: {val_acc:.2f}%")
        else:
            self.print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if is_best:
                self.print(f"✓ Best model saved with accuracy: {val_acc:.2f}%")
    
    def log_mapping_stats(self, stats: List[Dict]):
        """
        记录权重映射统计
        
        Args:
            stats: 映射统计列表
        """
        self.print("\n" + "-"*50)
        self.print("权重映射统计:")
        self.print("-"*50)
        
        for i, stat in enumerate(stats):
            self.print(f"\n  Layer {i+1}:")
            self.print(f"    RMSE: {stat.get('rmse', 0):.6f}")
            self.print(f"    MAE: {stat.get('mae', 0):.6f}")
            self.print(f"    Max Error: {stat.get('max_error', 0):.6f}")
            self.print(f"    Relative Error: {stat.get('relative_error', 0)*100:.2f}%")
    
    def log_final_result(
        self,
        best_acc: float,
        final_train_acc: float,
        total_time: float = None
    ):
        """
        记录最终结果
        """
        self.print("\n" + "="*70)
        self.print("训练完成 - Training Complete")
        self.print("="*70)
        self.print(f"  最佳验证准确率: {best_acc:.2f}%")
        self.print(f"  最终训练准确率: {final_train_acc:.2f}%")
        if total_time:
            self.print(f"  总训练时间: {total_time:.1f} 秒")
        self.print("="*70)
    
    def log_custom_section(self, title: str, content: str):
        """
        记录自定义部分
        """
        self.print(f"\n{title}")
        self.print("-"*len(title))
        self.print(content)
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.print(f"\n日志已保存到: {self.log_path}")
            self.log_file.close()
            self.log_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_logger(
    log_dir: str = None,
    model_type: str = "MLP",
    dataset: str = "mnist",
    use_device_neuron: bool = False,
    enabled: bool = True
) -> TrainingLogger:
    """
    创建日志记录器的便捷函数
    
    Args:
        log_dir: 日志目录 (默认为 DeviceNeuron/logs)
        model_type: 模型类型
        dataset: 数据集
        use_device_neuron: 是否使用器件激活函数
        enabled: 是否启用
        
    Returns:
        TrainingLogger 实例
    """
    if log_dir is None:
        # 默认日志目录
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs"
        )
    
    return TrainingLogger(
        log_dir=log_dir,
        model_type=model_type,
        dataset=dataset,
        use_device_neuron=use_device_neuron,
        enabled=enabled
    )

