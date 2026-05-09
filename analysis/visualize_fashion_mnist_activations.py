"""
Fashion-MNIST 模型激活可视化脚本
Visualize activations for Fashion-MNIST models

为三种模型（标准、交叉阵列、交叉阵列+自定义激活函数）生成热图：
- Conv1: 前4个kernel的激活前、激活后、池化后热图（12个图）
- Conv2: 前4个kernel的激活前、激活后、池化后热图（12个图）
- 输出层: 激活前、激活后的热图（2个图）
总共26个图，保存为PDF格式

cd /home/zhc/Projects/neuromorphic
python -m DeviceNeuron.anaysis.visualize_fashion_mnist_activations.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeviceNeuron import (
    FashionMNISTUnifiedCNN,
    FashionMNISTUnifiedCNNWithDeviceNeuron,
    CrossbarConfig,
    create_real_device_config,
    CONFIG
)
from DeviceNeuron.benchmark.models import FashionMNISTStandardCNN
from DeviceNeuron.my_device_activation import MyDeviceActivation


class ActivationExtractor:
    """提取模型中间层激活的辅助类"""
    
    def __init__(self, model, model_type='standard'):
        self.model = model
        self.model_type = model_type
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册hook来提取中间层输出"""
        if self.model_type == 'standard':
            self._register_standard_hooks()
        elif self.model_type == 'crossbar':
            self._register_crossbar_hooks()
        elif self.model_type == 'crossbar_device':
            self._register_crossbar_device_hooks()
    
    def _register_standard_hooks(self):
        """为标准CNN注册hook"""
        # Conv1: 激活前
        def conv1_pre_act_hook(module, input, output):
            self.activations['conv1_pre_act'] = output.detach()
        
        # Conv1: 激活后
        def conv1_post_act_hook(module, input, output):
            self.activations['conv1_post_act'] = output.detach()
        
        # Conv1: 池化后
        def conv1_pool_hook(module, input, output):
            self.activations['conv1_pool'] = output.detach()
        
        # Conv2: 激活前
        def conv2_pre_act_hook(module, input, output):
            self.activations['conv2_pre_act'] = output.detach()
        
        # Conv2: 激活后
        def conv2_post_act_hook(module, input, output):
            self.activations['conv2_post_act'] = output.detach()
        
        # Conv2: 池化后
        def conv2_pool_hook(module, input, output):
            self.activations['conv2_pool'] = output.detach()
        
        # 输出层: 激活前
        def output_pre_act_hook(module, input, output):
            self.activations['output_pre_act'] = output.detach()
        
        # 输出层: 激活后（标准模型没有输出层激活，所以就是激活前）
        self.activations['output_post_act'] = None
        
        # 注册hook
        conv_layers = list(self.model.conv.children())
        # Conv1: index 0 (Conv2d), 1 (ReLU), 2 (MaxPool2d)
        self.hooks.append(conv_layers[0].register_forward_hook(conv1_pre_act_hook))
        self.hooks.append(conv_layers[1].register_forward_hook(conv1_post_act_hook))
        self.hooks.append(conv_layers[2].register_forward_hook(conv1_pool_hook))
        # Conv2: index 3 (Conv2d), 4 (ReLU), 5 (MaxPool2d)
        self.hooks.append(conv_layers[3].register_forward_hook(conv2_pre_act_hook))
        self.hooks.append(conv_layers[4].register_forward_hook(conv2_post_act_hook))
        self.hooks.append(conv_layers[5].register_forward_hook(conv2_pool_hook))
        # 输出层: 最后一个Linear层
        fc_layers = list(self.model.fc.children())
        output_layer = [layer for layer in fc_layers if isinstance(layer, nn.Linear)][-1]
        self.hooks.append(output_layer.register_forward_hook(output_pre_act_hook))
    
    def _register_crossbar_hooks(self):
        """为交叉阵列CNN注册hook"""
        # 类似标准模型，但需要找到交叉阵列层
        conv_layers = list(self.model.conv_layers.children())
        # Conv1: UnifiedCrossbarConv2dWithActivation (index 0), MaxPool2d (index 1)
        # Conv2: UnifiedCrossbarConv2dWithActivation (index 2), MaxPool2d (index 3)
        
        def conv1_conv_hook(module, input, output):
            # UnifiedCrossbarConv2dWithActivation 内部有激活，需要提取激活前后的值
            # 这里我们只能获取最终输出（激活后）
            self.activations['conv1_post_act'] = output.detach()
        
        def conv1_pool_hook(module, input, output):
            self.activations['conv1_pool'] = output.detach()
        
        def conv2_conv_hook(module, input, output):
            self.activations['conv2_post_act'] = output.detach()
        
        def conv2_pool_hook(module, input, output):
            self.activations['conv2_pool'] = output.detach()
        
        # 对于交叉阵列模型，我们需要手动提取激活前的值
        # 这需要修改forward方法或使用更复杂的hook
        
        # 输出层
        fc_layers = list(self.model.fc_layers.children())
        output_layer = [layer for layer in fc_layers if hasattr(layer, 'forward_linear')][-1]
        
        def output_hook(module, input, output):
            self.activations['output_pre_act'] = output.detach()
        
        self.hooks.append(conv_layers[0].register_forward_hook(conv1_conv_hook))
        self.hooks.append(conv_layers[1].register_forward_hook(conv1_pool_hook))
        self.hooks.append(conv_layers[2].register_forward_hook(conv2_conv_hook))
        self.hooks.append(conv_layers[3].register_forward_hook(conv2_pool_hook))
        self.hooks.append(output_layer.register_forward_hook(output_hook))
        
        # 对于激活前的值，我们需要修改模型或使用wrapper
        # 这里我们先用激活后的值，稍后通过修改模型来获取激活前的值
    
    def _register_crossbar_device_hooks(self):
        """为交叉阵列+自定义激活函数CNN注册hook"""
        # 类似交叉阵列模型
        self._register_crossbar_hooks()
    
    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        """清空激活值"""
        self.activations = {}


def extract_activations_manual(model, x, model_type='standard'):
    """
    手动提取激活值（通过修改forward方法）
    
    对于交叉阵列模型，我们需要手动提取激活前后的值
    """
    activations = {}
    
    if model_type == 'standard':
        # 标准模型: conv[Conv2d, ReLU, MaxPool2d, Conv2d, ReLU, MaxPool2d]
        # Conv1
        conv1_out = model.conv[0](x)  # 激活前
        activations['conv1_pre_act'] = conv1_out.detach()
        conv1_act = model.conv[1](conv1_out)  # 激活后
        activations['conv1_post_act'] = conv1_act.detach()
        conv1_pool = model.conv[2](conv1_act)  # 池化后
        activations['conv1_pool'] = conv1_pool.detach()
        
        # Conv2
        conv2_out = model.conv[3](conv1_pool)  # 激活前
        activations['conv2_pre_act'] = conv2_out.detach()
        conv2_act = model.conv[4](conv2_out)  # 激活后
        activations['conv2_post_act'] = conv2_act.detach()
        conv2_pool = model.conv[5](conv2_act)  # 池化后
        activations['conv2_pool'] = conv2_pool.detach()
        
        # 全连接层
        x_flat = conv2_pool.reshape(conv2_pool.size(0), -1)
        fc1_out = model.fc[0](x_flat)
        activations['fc1_pre_act'] = fc1_out.detach()
        fc1_act = model.fc[1](fc1_out)
        activations['fc1_post_act'] = fc1_act.detach()
        output_pre = model.fc[2](fc1_act)  # 输出层（无激活）
        activations['output_pre_act'] = output_pre.detach()
        # 输出层应用 ReLU
        output_post = F.relu(output_pre)
        activations['output_post_act'] = output_post.detach()
    
    elif model_type == 'crossbar':
        # 交叉阵列模型: conv_layers[UnifiedCrossbarConv2d, ReLU, MaxPool2d, UnifiedCrossbarConv2d, ReLU, MaxPool2d]
        # Conv1
        conv1_out = model.conv_layers[0](x)  # UnifiedCrossbarConv2d输出（激活前，因为后面有ReLU）
        activations['conv1_pre_act'] = conv1_out.detach()
        conv1_act = model.conv_layers[1](conv1_out)  # ReLU激活后
        activations['conv1_post_act'] = conv1_act.detach()
        conv1_pool = model.conv_layers[2](conv1_act)  # 池化后
        activations['conv1_pool'] = conv1_pool.detach()
        
        # Conv2
        conv2_out = model.conv_layers[3](conv1_pool)  # 激活前
        activations['conv2_pre_act'] = conv2_out.detach()
        conv2_act = model.conv_layers[4](conv2_out)  # 激活后
        activations['conv2_post_act'] = conv2_act.detach()
        conv2_pool = model.conv_layers[5](conv2_act)  # 池化后
        activations['conv2_pool'] = conv2_pool.detach()
        
        # 全连接层
        x_flat = conv2_pool.reshape(conv2_pool.size(0), -1)
        fc1_out = model.fc_layers[0](x_flat)  # UnifiedCrossbarLinear
        activations['fc1_pre_act'] = fc1_out.detach()
        fc1_act = model.fc_layers[1](fc1_out)  # ReLU
        activations['fc1_post_act'] = fc1_act.detach()
        output_pre = model.fc_layers[2](fc1_act)  # 输出层
        activations['output_pre_act'] = output_pre.detach()
        # 输出层应用 ReLU
        output_post = F.relu(output_pre)
        activations['output_post_act'] = output_post.detach()
    
    elif model_type == 'crossbar_device':
        # 交叉阵列+自定义激活函数模型: conv_layers[UnifiedCrossbarConv2dWithActivation, MaxPool2d, ...]
        # UnifiedCrossbarConv2dWithActivation内部已经包含激活函数
        # 我们需要手动提取激活前后的值
        
        # 创建一个wrapper来提取激活前的值（归一化后的值，在激活函数之前）
        def extract_conv_pre_act(layer, x_input):
            """提取卷积层激活前的值（归一化后的crossbar输出，范围[-1,1]）"""
            try:
                batch_size = x_input.shape[0]
                
                # 手动执行forward的前几步来获取激活前的值
                v_in = layer.crossbar_core.apply_dac(x_input)
                v_unfold = F.unfold(v_in, kernel_size=layer.kernel_size, 
                                   stride=layer.stride, padding=layer.padding)
                
                H_out = (x_input.shape[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                W_out = (x_input.shape[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                
                weight_flat = layer.weight.view(layer.out_channels, -1)
                g_pos, g_neg, scale = layer.crossbar_core.map_weight(weight_flat, add_noise=False)
                g_diff = g_pos - g_neg
                
                v_unfold_t = v_unfold.transpose(1, 2)
                i_out = torch.matmul(v_unfold_t, g_diff.t()) * scale
                i_out = i_out.transpose(1, 2)
                
                i_digital = layer.crossbar_core.apply_adc(i_out)
                i_normalized, _ = layer._normalize_for_activation(i_digital)
                
                # Reshape并返回（这是激活前的值，归一化到[-1,1]）
                pre_act = i_normalized.view(batch_size, layer.out_channels, H_out, W_out)
                
                return pre_act
            except Exception as e:
                print(f"ERROR: extract_conv_pre_act failed: {e}")
                import traceback
                traceback.print_exc()
                # 返回零张量作为占位符
                batch_size = x_input.shape[0]
                H_out = (x_input.shape[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                W_out = (x_input.shape[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                return torch.zeros(batch_size, layer.out_channels, H_out, W_out, device=x_input.device)
        
        # 创建一个wrapper来提取激活后的值（在激活函数之后，应该是非负的）
        def extract_conv_post_act(layer, x_input):
            """提取卷积层激活后的值（经过激活函数，应该是非负的）"""
            try:
                batch_size = x_input.shape[0]
                
                # 执行完整的forward，但提取激活后的值（在缩放和偏置之前）
                v_in = layer.crossbar_core.apply_dac(x_input)
                v_unfold = F.unfold(v_in, kernel_size=layer.kernel_size, 
                                   stride=layer.stride, padding=layer.padding)
                
                H_out = (x_input.shape[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                W_out = (x_input.shape[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                
                weight_flat = layer.weight.view(layer.out_channels, -1)
                g_pos, g_neg, scale = layer.crossbar_core.map_weight(weight_flat, add_noise=False)
                g_diff = g_pos - g_neg
                
                v_unfold_t = v_unfold.transpose(1, 2)
                i_out = torch.matmul(v_unfold_t, g_diff.t()) * scale
                i_out = i_out.transpose(1, 2)
                
                i_digital = layer.crossbar_core.apply_adc(i_out)
                i_normalized, i_scale = layer._normalize_for_activation(i_digital)
                
                # 激活函数（CustomDeviceNeuron内部会处理取反等逻辑）
                shape_before = i_normalized.shape
                i_normalized_flat = i_normalized.reshape(-1)
                
                # CustomDeviceNeuron的forward流程：
                # 1. 输入积分器：将电流转换为电压（0-1范围）
                # 2. 激活函数：MyDeviceActivation（内部会取反）
                # 3. 输出积分器：将电流转换为电压（0-1范围）
                activation_module = layer.activation  # 这是CustomDeviceNeuron
                y_activated_flat = activation_module(i_normalized_flat)
                y_activated = y_activated_flat.reshape(shape_before)
                
                # 这是激活后的值（在缩放之前），应该是[0,1]范围（因为CustomDeviceNeuron输出0-1）
                post_act = y_activated.view(batch_size, layer.out_channels, H_out, W_out)
                
                # 检查：激活后的值应该是非负的
                if (post_act < 0).any():
                    print(f"WARNING: extract_conv_post_act found negative values (min={post_act.min():.4f})")
                
                return post_act
            except Exception as e:
                print(f"ERROR: extract_conv_post_act failed: {e}")
                import traceback
                traceback.print_exc()
                # 返回零张量作为占位符
                batch_size = x_input.shape[0]
                H_out = (x_input.shape[2] + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                W_out = (x_input.shape[3] + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                return torch.zeros(batch_size, layer.out_channels, H_out, W_out, device=x_input.device)
        
        # Conv1
        print(f"  Extracting Conv1 activations...")
        conv1_pre = extract_conv_pre_act(model.conv_layers[0], x)
        print(f"    conv1_pre: shape={conv1_pre.shape}, range=[{conv1_pre.min():.4f}, {conv1_pre.max():.4f}]")
        activations['conv1_pre_act'] = conv1_pre.detach()
        
        # 提取激活后的值（在缩放之前，应该是[0,1]范围）
        conv1_post_raw = extract_conv_post_act(model.conv_layers[0], x)
        print(f"    conv1_post_raw: shape={conv1_post_raw.shape}, range=[{conv1_post_raw.min():.4f}, {conv1_post_raw.max():.4f}]")
        # 使用原始激活后的值（[0,1]范围），而不是经过缩放的值
        activations['conv1_post_act'] = conv1_post_raw.detach()
        
        # 但是池化需要使用完整的forward输出（经过缩放和偏置）
        conv1_post_full = model.conv_layers[0](x)  # 完整的输出（经过缩放和偏置）
        conv1_pool = model.conv_layers[1](conv1_post_full)  # 池化后
        print(f"    conv1_pool: shape={conv1_pool.shape}, range=[{conv1_pool.min():.4f}, {conv1_pool.max():.4f}]")
        activations['conv1_pool'] = conv1_pool.detach()
        
        # Conv2
        print(f"  Extracting Conv2 activations...")
        conv2_pre = extract_conv_pre_act(model.conv_layers[2], conv1_pool)
        print(f"    conv2_pre: shape={conv2_pre.shape}, range=[{conv2_pre.min():.4f}, {conv2_pre.max():.4f}]")
        activations['conv2_pre_act'] = conv2_pre.detach()
        
        # 提取激活后的值
        conv2_post_raw = extract_conv_post_act(model.conv_layers[2], conv1_pool)
        print(f"    conv2_post_raw: shape={conv2_post_raw.shape}, range=[{conv2_post_raw.min():.4f}, {conv2_post_raw.max():.4f}]")
        activations['conv2_post_act'] = conv2_post_raw.detach()
        
        # 池化使用完整输出
        conv2_post_full = model.conv_layers[2](conv1_pool)
        conv2_pool = model.conv_layers[3](conv2_post_full)  # 池化后
        print(f"    conv2_pool: shape={conv2_pool.shape}, range=[{conv2_pool.min():.4f}, {conv2_pool.max():.4f}]")
        activations['conv2_pool'] = conv2_pool.detach()
        
        # 全连接层
        x_flat = conv2_pool.reshape(conv2_pool.size(0), -1)
        print(f"  Extracting FC layer activations...")
        
        # 提取FC层激活前的值
        def extract_fc_pre_act(layer, x_input):
            """提取全连接层激活前的值（归一化后的crossbar输出，范围[-1,1]）"""
            try:
                # 检查是否是 UnifiedCrossbarLinearWithActivation
                if hasattr(layer, '_normalize_for_activation'):
                    # 有激活函数的层
                    v_in = layer.crossbar_core.apply_dac(x_input)
                    g_pos, g_neg, scale = layer.crossbar_core.map_weight(layer.weight, add_noise=False)
                    g_diff = g_pos - g_neg
                    i_diff = torch.matmul(v_in, g_diff.t()) * scale
                    i_digital = layer.crossbar_core.apply_adc(i_diff)
                    i_normalized, _ = layer._normalize_for_activation(i_digital)
                    return i_normalized
                else:
                    # 没有激活函数的层（如输出层），直接返回ADC后的值（在加偏置之前）
                    v_in = layer.crossbar_core.apply_dac(x_input)
                    g_pos, g_neg, scale = layer.crossbar_core.map_weight(layer.weight, add_noise=False)
                    g_diff = g_pos - g_neg
                    i_diff = torch.matmul(v_in, g_diff.t()) * scale
                    i_digital = layer.crossbar_core.apply_adc(i_diff)
                    # 对于没有激活函数的层，ADC后的值就是"激活前"的值
                    return i_digital
            except Exception as e:
                print(f"ERROR: extract_fc_pre_act failed: {e}")
                import traceback
                traceback.print_exc()
                return torch.zeros(x_input.shape[0], layer.out_features, device=x_input.device)
        
        # 提取FC层激活后的值
        def extract_fc_post_act(layer, x_input):
            """提取全连接层激活后的值（经过激活函数，应该是非负的）"""
            try:
                # 检查是否是 UnifiedCrossbarLinearWithActivation
                if not hasattr(layer, '_normalize_for_activation'):
                    # 没有激活函数的层，返回None（由调用者处理）
                    return None
                
                v_in = layer.crossbar_core.apply_dac(x_input)
                g_pos, g_neg, scale = layer.crossbar_core.map_weight(layer.weight, add_noise=False)
                g_diff = g_pos - g_neg
                i_diff = torch.matmul(v_in, g_diff.t()) * scale
                i_digital = layer.crossbar_core.apply_adc(i_diff)
                i_normalized, i_scale = layer._normalize_for_activation(i_digital)
                
                # 激活函数（CustomDeviceNeuron内部会处理取反等逻辑）
                i_normalized_flat = i_normalized.reshape(-1)
                activation_module = layer.activation  # 这是CustomDeviceNeuron
                y_activated_flat = activation_module(i_normalized_flat)
                
                # 确保返回正确的形状 [batch_size, out_features]
                if y_activated_flat.dim() == 0:
                    # 如果是标量，需要reshape
                    batch_size = x_input.shape[0]
                    post_act = y_activated_flat.unsqueeze(0).expand(batch_size, layer.out_features)
                elif y_activated_flat.dim() == 1:
                    # 如果是1D，需要reshape为2D
                    batch_size = x_input.shape[0]
                    post_act = y_activated_flat.view(batch_size, -1)
                else:
                    post_act = y_activated_flat
                
                # 检查：激活后的值应该是非负的
                if (post_act < 0).any():
                    print(f"WARNING: extract_fc_post_act found negative values (min={post_act.min():.4f})")
                
                return post_act
            except Exception as e:
                print(f"ERROR: extract_fc_post_act failed: {e}")
                import traceback
                traceback.print_exc()
                return torch.zeros(x_input.shape[0], layer.out_features, device=x_input.device)
        
        fc1_pre = extract_fc_pre_act(model.fc_layers[0], x_flat)
        print(f"    fc1_pre: shape={fc1_pre.shape}, range=[{fc1_pre.min():.4f}, {fc1_pre.max():.4f}]")
        activations['fc1_pre_act'] = fc1_pre.detach()
        
        # 提取激活后的值（在缩放之前，应该是[0,1]范围）
        fc1_post_raw = extract_fc_post_act(model.fc_layers[0], x_flat)
        if fc1_post_raw is not None:
            print(f"    fc1_post_raw: shape={fc1_post_raw.shape}, range=[{fc1_post_raw.min():.4f}, {fc1_post_raw.max():.4f}]")
            activations['fc1_post_act'] = fc1_post_raw.detach()
        else:
            # 如果没有激活函数，使用完整的forward输出
            fc1_post_full = model.fc_layers[0](x_flat)
            activations['fc1_post_act'] = fc1_post_full.detach()
        
        # 输出层（无激活函数，直接提取）
        fc1_post_full = model.fc_layers[0](x_flat)  # 完整的FC1输出（经过缩放和偏置）
        output_pre = extract_fc_pre_act(model.fc_layers[1], fc1_post_full)  # 输出层激活前
        print(f"    output_pre: shape={output_pre.shape}, range=[{output_pre.min():.4f}, {output_pre.max():.4f}]")
        activations['output_pre_act'] = output_pre.detach()
        
        # 输出层应用自定义激活函数（使用与FC1相同的激活函数）
        # 需要手动模拟 UnifiedCrossbarLinearWithActivation 的激活过程
        # 1. 归一化到 [-1, 1]（使用与 UnifiedCrossbarLinearWithActivation 相同的逻辑）
        i_digital = output_pre  # 这是ADC后的值
        # 按样本归一化到 [-1, 1]
        i_min = i_digital.min(dim=1, keepdim=True)[0]
        i_max = i_digital.max(dim=1, keepdim=True)[0]
        i_range = i_max - i_min
        i_range = torch.clamp(i_range, min=1e-10)
        i_normalized = (i_digital - i_min) / i_range * 2.0 - 1.0  # 归一化到 [-1, 1]
        
        # 2. 应用自定义激活函数（使用模型的激活函数模板）
        from DeviceNeuron.custom_neuron import CustomDeviceNeuron
        # 获取激活函数模板
        activation_template = model._activation_template
        # 创建 CustomDeviceNeuron（使用与模型相同的参数）
        device_neuron = CustomDeviceNeuron(
            activation=activation_template,
            k_int_input=model.k_int_input,
            k_int_output=model.k_int_output,
            device_noise=0.0  # 可视化时不添加噪声
        )
        # 应用激活函数
        i_normalized_flat = i_normalized.reshape(-1)
        output_post_flat = device_neuron(i_normalized_flat)
        output_post = output_post_flat.reshape(output_pre.shape)
        
        print(f"    output_post: shape={output_post.shape}, range=[{output_post.min():.4f}, {output_post.max():.4f}]")
        activations['output_post_act'] = output_post.detach()
    
    return activations


def plot_heatmap(data, title, save_path, vmin=-1.0, vmax=1.0):
    """
    绘制热图并保存为PDF（无坐标轴、无色带、无标题、正方形）
    
    Args:
        data: 2D numpy array
        title: 图标题（不使用，保留用于兼容性）
        save_path: 保存路径
        vmin, vmax: 颜色范围（统一使用[-1,1]）
    """
    # 检查数据有效性
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    data_min = data.min()
    data_max = data.max()
    
    # 检查数据是否全为非负值（激活后的数据）
    is_non_negative = data_min >= -1e-10
    
    # 统一使用[-1, 1]色带，但根据数据特性选择归一化方式
    if abs(data_max - data_min) > 1e-10:
        if is_non_negative:
            # 对于非负数据（激活后），归一化到[0, 1]，只使用色带的正半部分
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            # 对于有负数的数据（激活前），归一化到[-1, 1]
            data_normalized = (data - data_min) / (data_max - data_min) * 2.0 - 1.0
    else:
        # 如果数据范围太小（几乎为常数）
        if is_non_negative:
            data_normalized = np.zeros_like(data) if abs(data_min) < 1e-10 else np.ones_like(data) * 1.0
        else:
            data_normalized = np.ones_like(data) * (-1.0 if data_min < 0 else 1.0)
    
    # 计算正方形尺寸（基于数据维度）
    h, w = data.shape
    size = max(h, w) * 0.1  # 每个像素约0.1英寸
    fig, ax = plt.subplots(figsize=(size, size))
    
    # 使用统一的颜色范围[-1, 1]
    im = ax.imshow(data_normalized, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # 去掉所有坐标轴和标签
    ax.axis('off')
    
    # 添加黑色边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    
    plt.tight_layout(pad=0)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_fc1_heatmap(data, title, save_path, vmin=-1.0, vmax=1.0):
    """
    绘制FC1层热图（1D数据，横条排列，又瘦又高，无坐标轴、无色带、无标题、无边框）
    
    Args:
        data: 1D numpy array (128维)
        title: 图标题（不使用，保留用于兼容性）
        save_path: 保存路径
        vmin, vmax: 颜色范围（统一使用[-1,1]）
    """
    data_min = data.min()
    data_max = data.max()
    
    # 检查数据是否全为非负值（激活后的数据）
    is_non_negative = data_min >= -1e-10
    
    # 统一使用[-1, 1]色带，但根据数据特性选择归一化方式
    if abs(data_max - data_min) > 1e-10:
        if is_non_negative:
            # 对于非负数据（激活后），归一化到[0, 1]，只使用色带的正半部分
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            # 对于有负数的数据（激活前），归一化到[-1, 1]
            data_normalized = (data - data_min) / (data_max - data_min) * 2.0 - 1.0
    else:
        # 常数数据
        if is_non_negative:
            data_normalized = np.zeros_like(data)
        else:
            data_normalized = np.ones_like(data) * -1.0
    
    # 创建2D热图（128行，1列）- 横条排列，又瘦又高
    data_2d = data_normalized.reshape(-1, 1)  # [128, 1]
    h, w = data_2d.shape  # h=128, w=1
    
    # 计算图像尺寸（又瘦又高）
    # 每个横条的高度（像素）
    bar_height = 2  # 每个横条2像素高
    total_height = h * bar_height
    total_width = 40  # 宽度固定为20像素（很窄）
    
    # 创建数据数组
    final_data = np.zeros((total_height, total_width))
    
    # 填充数据：每个横条
    for i in range(h):
        row_start = i * bar_height
        # 将单个值扩展到整个横条
        final_data[row_start:row_start+bar_height, :] = data_2d[i, 0]
    
    # 计算图像尺寸（英寸）- 又瘦又高
    height_inch = 4.0  # 高度4英寸
    width_inch = 0.3  # 宽度0.1英寸（很窄）
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    # 使用统一的颜色范围[-1, 1]
    im = ax.imshow(final_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # 去掉所有坐标轴和标签
    ax.axis('off')
    
    # 不添加边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout(pad=0)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_output_heatmap(data, title, save_path, vmin=-1.0, vmax=1.0):
    """
    绘制输出层热图（1D数据，竖着排列，每个是小正方形，无坐标轴、无色带、无标题）
    
    Args:
        data: 1D numpy array (10个类别)
        title: 图标题（不使用，保留用于兼容性）
        save_path: 保存路径
        vmin, vmax: 颜色范围（统一使用[-1,1]）
    """
    data_min = data.min()
    data_max = data.max()
    
    # 检查数据是否全为非负值（激活后的数据）
    is_non_negative = data_min >= -1e-10
    
    # 统一使用[-1, 1]色带，但根据数据特性选择归一化方式
    if abs(data_max - data_min) > 1e-10:
        if is_non_negative:
            # 对于非负数据（激活后），归一化到[0, 1]，只使用色带的正半部分
            data_normalized = (data - data_min) / (data_max - data_min)
        else:
            # 对于有负数的数据（激活前），归一化到[-1, 1]
            data_normalized = (data - data_min) / (data_max - data_min) * 2.0 - 1.0
    else:
        # 常数数据
        if is_non_negative:
            data_normalized = np.zeros_like(data)
        else:
            data_normalized = np.ones_like(data) * -1.0
    
    # 创建2D热图（10行，1列）- 竖着排列，并在每个小正方形周围添加黑色边框
    data_2d = data_normalized.reshape(-1, 1)
    h, w = data_2d.shape  # h=10, w=1
    
    # 每个小正方形的像素大小
    square_size = 20  # 每个小正方形20x20像素
    
    # 创建数据数组：10个小正方形连续排列，不包含边框区域
    final_h = 10 * square_size
    final_w = square_size
    
    # 创建数据数组
    final_data = np.zeros((final_h, final_w))
    
    # 填充数据：每个小正方形
    for i in range(h):
        row_start = i * square_size
        # 将单个值扩展到 square_size x square_size 的正方形
        final_data[row_start:row_start+square_size, :] = data_2d[i, 0]
    
    # 计算正方形尺寸（英寸）
    size = 0.3  # 每个小正方形约0.3英寸
    total_height = size * 10  # 10个小正方形
    total_width = size  # 1列数据
    fig, ax = plt.subplots(figsize=(total_width, total_height))
    
    # 使用统一的颜色范围[-1, 1]
    im = ax.imshow(final_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # 去掉所有坐标轴和标签
    ax.axis('off')
    
    # 使用 Rectangle 绘制每个小正方形的黑色边框
    # 相邻小正方形共享边框，所以它们之间只有一条黑线
    from matplotlib.patches import Rectangle
    
    for i in range(h):
        # 计算当前小正方形的位置（数据坐标）
        # 注意：imshow 的坐标系统，y 轴是反向的（从上到下），所以 row_start 从 0 开始
        row_start = i * square_size
        # 绘制黑色边框（使用 Rectangle，edgecolor='black', facecolor='none'）
        rect = Rectangle(
            (-0.5, row_start - 0.5),  # 左下角位置（数据坐标）
            square_size,  # 宽度
            square_size,  # 高度
            linewidth=1,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # 添加整体黑色边框（外边框），确保覆盖整个图像包括最下面
    # 设置坐标轴范围，确保外边框正确显示
    ax.set_xlim(-0.5, final_w - 0.5)
    ax.set_ylim(final_h - 0.5, -0.5)  # y 轴反向
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    
    plt.tight_layout(pad=0)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_colorbar(save_path, vmin=-1.0, vmax=1.0):
    """
    单独绘制色带PDF
    
    Args:
        save_path: 保存路径
        vmin, vmax: 颜色范围（统一使用[-1,1]）
    """
    fig, ax = plt.subplots(figsize=(0.5, 6))
    
    # 创建颜色条数据
    gradient = np.linspace(vmax, vmin, 256).reshape(256, 1)
    
    im = ax.imshow(gradient, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, 255, 11))  # 11个刻度点
    ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(vmax, vmin, 11)])
    ax.set_ylabel('Value', rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_model(model, model_type, model_name, image, output_dir):
    """
    为单个模型生成所有热图
    
    Args:
        model: 模型
        model_type: 模型类型 ('standard', 'crossbar', 'crossbar_device')
        model_name: 模型名称（用于文件夹命名）
        image: 输入图像 [1, 1, 28, 28]
        output_dir: 输出目录
    """
    model.eval()
    
    # 创建输出目录
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取激活值
    with torch.no_grad():
        activations = extract_activations_manual(model, image, model_type)
    
    # 调试信息：检查激活值的范围
    if model_type == 'crossbar_device':
        print(f"\n[{model_name}] 激活值范围检查:")
        for key, val in activations.items():
            if val is not None:
                if val.dim() > 0:
                    val_np = val[0].cpu().numpy()
                else:
                    val_np = val.cpu().numpy()
                if val_np.size > 0:
                    print(f"  {key}: shape={val_np.shape}, min={val_np.min():.4f}, max={val_np.max():.4f}, mean={val_np.mean():.4f}, std={val_np.std():.4f}")
                    if np.isnan(val_np).any():
                        print(f"    ⚠️  包含 NaN 值!")
                    if np.allclose(val_np, 0):
                        print(f"    ⚠️  全为0!")
                else:
                    print(f"  {key}: empty array")
    
    # 生成Conv1热图（前4个kernel）
    conv1_pre = activations['conv1_pre_act'][0].cpu().numpy()  # [32, 28, 28]
    conv1_post = activations['conv1_post_act'][0].cpu().numpy()
    conv1_pool = activations['conv1_pool'][0].cpu().numpy()  # [32, 14, 14]
    
    # 检查数据有效性
    if model_type == 'crossbar_device':
        if np.isnan(conv1_pre).any() or np.isnan(conv1_post).any() or np.isnan(conv1_pool).any():
            print(f"WARNING [{model_name}]: Conv1 contains NaN values")
        if np.allclose(conv1_pre, 0) and np.allclose(conv1_post, 0):
            print(f"WARNING [{model_name}]: Conv1 activations are all zeros, extraction may have issues")
    
    for i in range(4):
        # 激活前
        plot_heatmap(
            conv1_pre[i],
            f'Conv1 Kernel {i+1} - Before Activation',
            model_dir / f'conv1_kernel{i+1}_pre_act.pdf'
        )
        
        # 激活后
        plot_heatmap(
            conv1_post[i],
            f'Conv1 Kernel {i+1} - After Activation',
            model_dir / f'conv1_kernel{i+1}_post_act.pdf'
        )
        
        # 池化后
        plot_heatmap(
            conv1_pool[i],
            f'Conv1 Kernel {i+1} - After Pooling',
            model_dir / f'conv1_kernel{i+1}_pool.pdf'
        )
    
    # 生成Conv2热图（前4个kernel）
    conv2_pre = activations['conv2_pre_act'][0].cpu().numpy()  # [64, 14, 14]
    conv2_post = activations['conv2_post_act'][0].cpu().numpy()
    conv2_pool = activations['conv2_pool'][0].cpu().numpy()  # [64, 7, 7]
    
    for i in range(4):
        # 激活前
        plot_heatmap(
            conv2_pre[i],
            f'Conv2 Kernel {i+1} - Before Activation',
            model_dir / f'conv2_kernel{i+1}_pre_act.pdf'
        )
        
        # 激活后
        plot_heatmap(
            conv2_post[i],
            f'Conv2 Kernel {i+1} - After Activation',
            model_dir / f'conv2_kernel{i+1}_post_act.pdf'
        )
        
        # 池化后
        plot_heatmap(
            conv2_pool[i],
            f'Conv2 Kernel {i+1} - After Pooling',
            model_dir / f'conv2_kernel{i+1}_pool.pdf'
        )
    
    # 生成FC1层热图
    fc1_pre = activations['fc1_pre_act'][0].cpu().numpy()  # [128]
    fc1_post = activations['fc1_post_act'][0].cpu().numpy()  # [128]
    
    plot_fc1_heatmap(
        fc1_pre,
        'FC1 Layer - Before Activation',
        model_dir / 'fc1_pre_act.pdf'
    )
    
    plot_fc1_heatmap(
        fc1_post,
        'FC1 Layer - After Activation',
        model_dir / 'fc1_post_act.pdf'
    )
    
    # 生成输出层热图
    output_pre = activations['output_pre_act'][0].cpu().numpy()  # [10]
    if activations['output_post_act'] is not None:
        output_post = activations['output_post_act'][0].cpu().numpy()
    else:
        output_post = output_pre  # 如果输出层无激活，则使用激活前的值
    
    plot_output_heatmap(
        output_pre,
        'Output Layer - Before Activation',
        model_dir / 'output_pre_act.pdf'
    )
    
    plot_output_heatmap(
        output_post,
        'Output Layer - After Activation',
        model_dir / 'output_post_act.pdf'
    )
    
    print(f"✓ {model_name}: Generated 28 heatmaps to {model_dir}")


def main():
    # 设置路径
    script_dir = Path(__file__).parent
    model_dir = script_dir
    output_dir = script_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    # 选择第一张图片
    image, label = dataset[0]
    image = image.unsqueeze(0)  # [1, 1, 28, 28]
    
    print(f"使用图片: 类别 {label}")
    print("="*70)
    
    # 1. 加载标准模型
    print("\n1. 加载标准模型...")
    standard_model = FashionMNISTStandardCNN()
    standard_model.load_state_dict(torch.load(model_dir / 'benchmark_fashion-mnist_cnn_best.pth'))
    standard_model.eval()
    print("✓ 标准模型加载完成")
    
    # 2. 加载交叉阵列模型
    print("\n2. 加载交叉阵列模型...")
    config = create_real_device_config().get_crossbar_config()
    crossbar_model = FashionMNISTUnifiedCNN(config=config)
    crossbar_model.load_state_dict(torch.load(model_dir / 'unified_fashion-mnist_cnn_best.pth'))
    crossbar_model.eval()
    print("✓ 交叉阵列模型加载完成")
    
    # 3. 加载交叉阵列+自定义激活函数模型
    print("\n3. 加载交叉阵列+自定义激活函数模型...")
    device_activation = MyDeviceActivation()
    crossbar_device_model = FashionMNISTUnifiedCNNWithDeviceNeuron(
        config=config,
        device_activation=device_activation,
        conv_channels=CONFIG.MNIST_CNN_CONV_CHANNELS,
        fc_sizes=CONFIG.MNIST_CNN_FC_SIZES,
        k_int_input=CONFIG.INTEGRATOR_K_INPUT,
        k_int_output=CONFIG.INTEGRATOR_K_OUTPUT,
        device_noise=CONFIG.DEVICE_NOISE_STD
    )
    crossbar_device_model.load_state_dict(
        torch.load(model_dir / 'unified_fashion-mnist_cnn_device_neuron_best.pth')
    )
    crossbar_device_model.eval()
    print("✓ 交叉阵列+自定义激活函数模型加载完成")
    
    # 生成热图
    print("\n" + "="*70)
    print("开始生成热图...")
    print("="*70)
    
    visualize_model(
        standard_model,
        'standard',
        'standard_cnn',
        image,
        output_dir
    )
    
    visualize_model(
        crossbar_model,
        'crossbar',
        'crossbar_cnn',
        image,
        output_dir
    )
    
    visualize_model(
        crossbar_device_model,
        'crossbar_device',
        'crossbar_device_neuron_cnn',
        image,
        output_dir
    )
    
    # 生成单独的色带PDF
    print("\n生成色带...")
    colorbar_path = output_dir / 'colorbar.pdf'
    plot_colorbar(colorbar_path)
    print(f"✓ 色带已保存到: {colorbar_path}")
    
    print("\n" + "="*70)
    print("All heatmaps generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

