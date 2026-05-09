"""
使用自定义交叉阵列和激活函数训练 MNIST
Train MNIST with Custom Crossbar and Activation Function

交叉阵列数据: trainingdata/device2.csv
激活函数: I(V) = 3.386 * (exp(-V / 44283.495) - 1), V ∈ [-4, 0]

运行方式:
    cd /home/zhc/Projects/neuromorphic
    python -m DeviceNeuron.examples.train_my_model
    
    # 指定隐藏层
    python -m DeviceNeuron.examples.train_my_model --hidden 512 256 128
    
    # 指定训练轮数
    python -m DeviceNeuron.examples.train_my_model --epochs 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DeviceNeuron import (
    MY_CONFIG,
    create_my_mlp_model,
    train_my_model,
    HILTrainer
)


def get_data_loaders(batch_size=64):
    """加载 MNIST 数据集"""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def main(args):
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    
    # 加载数据
    print('\n加载 MNIST 数据集...')
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    print(f'训练集: {len(train_loader.dataset)} 样本')
    print(f'测试集: {len(test_loader.dataset)} 样本')
    
    # 使用便捷函数训练
    model, history = train_my_model(
        train_loader=train_loader,
        test_loader=test_loader,
        hidden_sizes=args.hidden,
        epochs=args.epochs,
        device=device,
        save_path='my_custom_model_best.pth'
    )
    
    print('\n' + '='*70)
    print('训练完成!')
    print('模型已保存到: my_custom_model_best.pth')
    print('='*70)


def demo():
    """演示如何使用"""
    print('''
================================================================================
使用自定义交叉阵列和激活函数
================================================================================

你的配置:
  - 交叉阵列数据: trainingdata/device2.csv
    - 50个电导态
    - 电导范围: [5.94e-7, 4.11e-6] S
    
  - 器件激活函数: I(V) = 3.386 * (exp(-V / 44283.495) - 1)
    - 电压范围: [-4, 0] V

================================================================================
使用方法
================================================================================

方法1: 一键训练
---------------
from DeviceNeuron import train_my_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 准备数据
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, 
                          transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False,
                          transform=transforms.ToTensor()), batch_size=64)

# 训练
model, history = train_my_model(train_loader, test_loader)


方法2: 分步操作
---------------
from DeviceNeuron import (
    MY_CONFIG,
    create_my_crossbar_config,
    create_my_device_activation,
    create_my_mlp_model
)

# 打印配置
MY_CONFIG.print_config()

# 创建模型
model = create_my_mlp_model(hidden_sizes=[256, 128])

# 或者手动创建
from DeviceNeuron import UnifiedMLPWithDeviceNeuron

config = create_my_crossbar_config()      # 你的交叉阵列
activation = create_my_device_activation() # 你的激活函数

model = UnifiedMLPWithDeviceNeuron(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    config=config,
    device_activation=activation
)


方法3: 修改配置
---------------
编辑文件: DeviceNeuron/my_config.py

修改 MyConfig 类中的参数:
  - DEVICE_I_S: 饱和电流
  - DEVICE_R_VT: R * V_t 参数
  - MLP_HIDDEN_SIZES: 隐藏层大小
  - EPOCHS: 训练轮数
  - 等等...

================================================================================
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='使用自定义交叉阵列和激活函数训练 MNIST'
    )
    
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128],
                        help='隐藏层大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--demo', action='store_true',
                        help='显示使用说明')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        main(args)

