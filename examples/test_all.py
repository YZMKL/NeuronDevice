"""
完整测试脚本 - 验证统一Crossbar架构
Comprehensive Test Script - Verify Unified Crossbar Architecture

运行方式:
    cd /home/zhc/Projects/neuromorphic
    python -m DeviceNeuron.examples.test_all
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_conductance_states():
    """测试电导态管理"""
    print('\n' + '='*60)
    print('测试 1: 电导态管理 (ConductanceStates)')
    print('='*60)
    
    from DeviceNeuron import ConductanceStates
    
    # 测试创建
    states = ConductanceStates.create_log(
        g_min=1e-9,
        g_max=1e-5,
        n_states=64,
        relative_std=0.05
    )
    
    print(f'电导态数量: {states.n_states}')
    print(f'电导范围: [{states.g_min:.2e}, {states.g_max:.2e}] S')
    print(f'动态范围: {states.g_max/states.g_min:.0f}x')
    print(f'平均相对标准差: {np.mean(states.g_std/states.g_mean)*100:.1f}%')
    
    print('\n✅ 电导态管理测试通过')
    return True


def test_crossbar_config():
    """测试Crossbar配置"""
    print('\n' + '='*60)
    print('测试 2: Crossbar配置 (CrossbarConfig)')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig
    
    # 创建默认配置
    config = CrossbarConfig.create_default(
        n_states=64,
        g_min=1e-9,
        g_max=1e-5,
        relative_std=0.05,
        dac_bits=8,
        adc_bits=8
    )
    
    info = config.get_info()
    print(f'电导态数量: {info["n_states"]}')
    print(f'电导范围: [{info["g_range"][0]:.2e}, {info["g_range"][1]:.2e}] S')
    print(f'DAC位宽: {info["dac_bits"]} bits')
    print(f'ADC位宽: {info["adc_bits"]} bits')
    
    print('\n✅ Crossbar配置测试通过')
    return True


def test_quantization():
    """测试DAC/ADC量化"""
    print('\n' + '='*60)
    print('测试 3: DAC/ADC量化')
    print('='*60)
    
    from DeviceNeuron import DAC, ADC, DynamicADC
    
    # 测试DAC
    dac = DAC(n_bits=8, v_max=1.0)
    x = torch.linspace(0, 1, 10)
    v = dac(x)
    print(f'DAC输入: {x[:5].tolist()}...')
    print(f'DAC输出: {v[:5].tolist()}...')
    
    # 测试ADC
    adc = DynamicADC(n_bits=8)
    adc.train()
    current = torch.rand(10)
    digital = adc(current)
    print(f'\nADC输入范围: [{current.min():.4f}, {current.max():.4f}]')
    print(f'ADC输出范围: [{digital.min():.4f}, {digital.max():.4f}]')
    
    print('\n✅ DAC/ADC量化测试通过')
    return True


def test_weight_mapping():
    """测试权重映射"""
    print('\n' + '='*60)
    print('测试 4: 权重到电导映射')
    print('='*60)
    
    from DeviceNeuron import ConductanceStates, WeightToConductanceMapper
    
    # 创建电导态
    states = ConductanceStates.create_log(n_states=64)
    mapper = WeightToConductanceMapper(states)
    
    # 测试权重
    weight = torch.randn(4, 4)
    print(f'原始权重:\n{weight}')
    
    # 映射
    g_pos, g_neg, scale = mapper(weight, add_noise=False)
    print(f'\nG+ (归一化):\n{g_pos}')
    print(f'\nG- (归一化):\n{g_neg}')
    print(f'\n缩放因子: {scale.item():.4f}')
    
    # 计算误差
    error = mapper.compute_mapping_error(weight)
    print(f'\n映射误差:')
    print(f'  RMSE: {error["rmse"]:.6f}')
    print(f'  相对误差: {error["relative_error"]*100:.2f}%')
    
    print('\n✅ 权重映射测试通过')
    return True


def test_crossbar_linear():
    """测试统一Crossbar全连接层"""
    print('\n' + '='*60)
    print('测试 5: 统一Crossbar全连接层')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig, UnifiedCrossbarLinear
    
    config = CrossbarConfig.create_default(n_states=64)
    layer = UnifiedCrossbarLinear(784, 128, config)
    
    print(f'层结构: {layer.in_features} → {layer.out_features}')
    print(f'权重形状: {layer.weight.shape}')
    
    # 前向传播
    x = torch.rand(16, 784)
    y = layer(x)
    print(f'\n输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    
    # 测试梯度
    y.sum().backward()
    print(f'\n权重梯度形状: {layer.weight.grad.shape}')
    
    # 映射误差
    stats = layer.get_mapping_stats()
    print(f'映射RMSE: {stats["rmse"]:.6f}')
    
    print('\n✅ Crossbar全连接层测试通过')
    return True


def test_crossbar_conv2d():
    """测试统一Crossbar卷积层"""
    print('\n' + '='*60)
    print('测试 6: 统一Crossbar卷积层')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig, UnifiedCrossbarConv2d
    
    config = CrossbarConfig.create_default(n_states=64)
    layer = UnifiedCrossbarConv2d(1, 32, 3, config, padding=1)
    
    print(f'卷积核形状: {layer.weight.shape}')
    
    # 前向传播
    x = torch.rand(8, 1, 28, 28)
    y = layer(x)
    print(f'\n输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    
    # 测试梯度
    y.sum().backward()
    print(f'\n权重梯度形状: {layer.weight.grad.shape}')
    
    print('\n✅ Crossbar卷积层测试通过')
    return True


def test_unified_mlp():
    """测试统一MLP模型"""
    print('\n' + '='*60)
    print('测试 7: 统一MLP模型')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig, UnifiedMLP, MNISTUnifiedMLP
    
    config = CrossbarConfig.create_default(n_states=64)
    model = MNISTUnifiedMLP(config, hidden_sizes=[128, 64])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数总数: {total_params:,}')
    
    # 前向传播
    x = torch.rand(16, 1, 28, 28)
    y = model(x)
    print(f'\n输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    
    # 测试梯度
    loss = nn.CrossEntropyLoss()(y, torch.randint(0, 10, (16,)))
    loss.backward()
    print(f'\n损失值: {loss.item():.4f}')
    
    # 映射统计
    stats = model.get_all_mapping_stats()
    print(f'\n各层映射RMSE:')
    for i, s in enumerate(stats):
        print(f'  层{i+1}: {s["rmse"]:.6f}')
    
    print('\n✅ 统一MLP模型测试通过')
    return True


def test_unified_cnn():
    """测试统一CNN模型"""
    print('\n' + '='*60)
    print('测试 8: 统一CNN模型')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig, MNISTUnifiedCNN
    
    config = CrossbarConfig.create_default(n_states=64)
    model = MNISTUnifiedCNN(config, conv_channels=[16, 32], fc_sizes=[64])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数总数: {total_params:,}')
    
    # 前向传播
    x = torch.rand(8, 1, 28, 28)
    y = model(x)
    print(f'\n输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    
    # 测试梯度
    loss = nn.CrossEntropyLoss()(y, torch.randint(0, 10, (8,)))
    loss.backward()
    print(f'\n损失值: {loss.item():.4f}')
    
    print('\n✅ 统一CNN模型测试通过')
    return True


def test_custom_neuron():
    """测试自定义器件神经元"""
    print('\n' + '='*60)
    print('测试 9: 自定义器件神经元')
    print('='*60)
    
    from DeviceNeuron import (
        DeviceTransferCurve,
        CustomDeviceNeuron,
        SigmoidDeviceNeuron
    )
    
    # 创建器件曲线
    curve = DeviceTransferCurve.create_default('sigmoid')
    print(f'电压范围: [{curve.v_min:.2f}, {curve.v_max:.2f}]')
    print(f'电流范围: [{curve.i_min:.4f}, {curve.i_max:.4f}]')
    
    # 测试神经元
    neuron = SigmoidDeviceNeuron(k_int_input=1.0, k_int_output=1.0)
    
    current = torch.rand(4, 8)
    voltage = neuron(current)
    print(f'\n输入电流形状: {current.shape}')
    print(f'输出电压形状: {voltage.shape}')
    print(f'输出范围: [{voltage.min():.4f}, {voltage.max():.4f}]')
    
    # 梯度测试
    voltage.sum().backward()
    print(f'梯度计算成功')
    
    print('\n✅ 自定义器件神经元测试通过')
    return True


def test_unified_mlp_with_device_neuron():
    """测试带器件激活的统一MLP"""
    print('\n' + '='*60)
    print('测试 10: 带器件激活的统一MLP')
    print('='*60)
    
    from DeviceNeuron import (
        CrossbarConfig,
        UnifiedMLPWithDeviceNeuron,
        DeviceTransferCurve
    )
    
    config = CrossbarConfig.create_default(n_states=64)
    curve = DeviceTransferCurve.create_default('sigmoid')
    
    model = UnifiedMLPWithDeviceNeuron(
        input_size=784,
        hidden_sizes=[64],
        output_size=10,
        config=config,
        transfer_curve=curve
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数总数: {total_params:,}')
    
    # 前向传播
    x = torch.rand(8, 784)
    y = model(x)
    print(f'\n输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    
    # 测试梯度
    loss = nn.CrossEntropyLoss()(y, torch.randint(0, 10, (8,)))
    loss.backward()
    print(f'\n损失值: {loss.item():.4f}')
    
    print('\n✅ 带器件激活的统一MLP测试通过')
    return True


def test_hil_trainer():
    """测试HIL训练器"""
    print('\n' + '='*60)
    print('测试 11: HIL训练器')
    print('='*60)
    
    from DeviceNeuron import CrossbarConfig, MNISTUnifiedMLP, HILTrainer
    from torch.utils.data import TensorDataset, DataLoader
    
    config = CrossbarConfig.create_default(n_states=32)
    model = MNISTUnifiedMLP(config, hidden_sizes=[64])
    
    # 创建假数据
    x_train = torch.rand(100, 784)
    y_train = torch.randint(0, 10, (100,))
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建训练器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    trainer = HILTrainer(model, optimizer, criterion, device='cpu')
    
    # 训练1个epoch
    print('训练1个epoch...')
    train_loss, train_acc = trainer.train_epoch(train_loader, normalize_input=False)
    print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
    
    print('\n✅ HIL训练器测试通过')
    return True


def main():
    """运行所有测试"""
    print('\n' + '#'*60)
    print('# DeviceNeuron 统一架构测试')
    print('#'*60)
    
    tests = [
        ('电导态管理', test_conductance_states),
        ('Crossbar配置', test_crossbar_config),
        ('DAC/ADC量化', test_quantization),
        ('权重映射', test_weight_mapping),
        ('Crossbar全连接层', test_crossbar_linear),
        ('Crossbar卷积层', test_crossbar_conv2d),
        ('统一MLP', test_unified_mlp),
        ('统一CNN', test_unified_cnn),
        ('自定义器件神经元', test_custom_neuron),
        ('器件激活MLP', test_unified_mlp_with_device_neuron),
        ('HIL训练器', test_hil_trainer),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f'\n❌ {name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print('\n' + '#'*60)
    print('# 测试总结')
    print('#'*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = '✅ 通过' if result else '❌ 失败'
        print(f'  {name}: {status}')
    
    print(f'\n总计: {passed}/{total} 测试通过')
    
    if passed == total:
        print('\n🎉 所有测试通过！')
    else:
        print('\n⚠️ 部分测试失败')


if __name__ == '__main__':
    main()

