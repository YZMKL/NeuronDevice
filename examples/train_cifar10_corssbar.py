"""
CIFAR-10 上 VGG 风格 Crossbar 训练（对应 train_unified 的 --dataset cifar10 且
--model vgg 或 --model vgg_device_neuron）

运行（建议使用 conda 环境 neu_comp）:
    conda activate neu_comp
    cd /home/zhc/Projects/neuromorphic
    python -m DeviceNeuron.examples.train_cifar10_corssbar
    python -m DeviceNeuron.examples.train_cifar10_corssbar --model vgg_device_neuron
    python -m DeviceNeuron.examples.train_cifar10_corssbar --config real
    python -m DeviceNeuron.examples.train_cifar10_corssbar --model vgg_device_neuron --config real
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DeviceNeuron import (
    CIFAR10VGGStyleCNN,
    CIFAR10UnifiedVGGWithDeviceNeuron,
    CONFIG,
    create_real_device_config,
    create_high_precision_config,
    create_fast_test_config,
)
from DeviceNeuron.logger import create_logger


def get_cifar10_loaders(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_device_neuron = args.model == 'vgg_device_neuron'

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )

    logger = create_logger(
        log_dir=log_dir,
        model_type=args.model,
        dataset='cifar10',
        use_device_neuron=use_device_neuron,
        enabled=not args.no_log
    )

    start_time = time.time()
    logger.print(f'Using device: {device}')

    try:
        logger.print('\n' + '='*60)
        logger.print('Step 0: 加载配置 (DeviceNeuron/config.py)')
        logger.print('='*60)

        if args.config == 'real':
            cfg = create_real_device_config()
            logger.print('使用真实器件数据配置')
        elif args.config == 'high':
            cfg = create_high_precision_config()
            logger.print('使用高精度配置')
        elif args.config == 'fast':
            cfg = create_fast_test_config()
            logger.print('使用快速测试配置')
        else:
            cfg = CONFIG
            logger.print('使用默认配置')

        logger.log_config(cfg)

        logger.print('\n' + '='*60)
        logger.print('Step 1: 创建Crossbar配置（所有层共用）')
        logger.print('='*60)

        config = cfg.get_crossbar_config()
        logger.log_crossbar_config(config)

        logger.print('\n' + '='*60)
        logger.print('Step 2: 加载 CIFAR-10 数据集')
        logger.print('='*60)

        train_loader, test_loader = get_cifar10_loaders(batch_size=cfg.BATCH_SIZE)

        logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            test_size=len(test_loader.dataset),
            img_info='32x32x3'
        )

        logger.print('\n' + '='*60)
        logger.print(f'Step 3: 创建模型 ({args.model} on CIFAR10)')
        logger.print('='*60)

        if args.model == 'vgg':
            model = CIFAR10VGGStyleCNN(
                config=config,
                use_batchnorm=True,
                dropout=0.5
            )
            model_desc = (
                'VGG: Conv[64,64]→Pool→Conv[128,128]→Pool→Conv[256,256]→Pool→FC[512]→10 (BN+Dropout)'
            )
        elif args.model == 'vgg_device_neuron':
            device_activation = cfg.get_device_activation()
            model = CIFAR10UnifiedVGGWithDeviceNeuron(
                config=config,
                device_activation=device_activation,
                use_batchnorm=True,
                dropout=0.5,
                k_int_input=cfg.INTEGRATOR_K_INPUT,
                k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                device_noise=cfg.DEVICE_NOISE_STD
            )
            model_desc = (
                f'DeviceNeuron VGG: VGG-style (BN+Dropout) ({type(device_activation).__name__})'
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model = model.to(device)
        logger.log_model_info(model, model_desc)

        logger.print('\n' + '='*60)
        logger.print('Step 4: 开始HIL训练')
        logger.print('='*60)

        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

        logger.log_training_params(
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            learning_rate=cfg.LEARNING_RATE,
            optimizer="Adam",
            device=str(device)
        )

        save_path = f'unified_cifar10_{args.model}_best.pth'
        best_acc = 0.0
        final_train_acc = 0.0

        logger.log_epoch_start(0, cfg.EPOCHS)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            if cfg.SAVE_BEST_MODEL:
                torch.save(model.state_dict(), save_path)

        logger.log_epoch_result(0, 0.0, 0.0, val_loss, val_acc, is_best)

        for epoch in range(1, cfg.EPOCHS + 1):
            logger.log_epoch_start(epoch, cfg.EPOCHS)

            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc='Training', leave=True)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.2f}',
                    'acc': f'{100.*train_correct/train_total:.1f}'
                })

            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            final_train_acc = train_acc

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                if cfg.SAVE_BEST_MODEL:
                    torch.save(model.state_dict(), save_path)

            logger.log_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, is_best)

            if scheduler:
                scheduler.step()

        logger.print('\n' + '='*60)
        logger.print('Step 5: 最终评估')
        logger.print('='*60)

        if hasattr(model, 'get_all_mapping_stats'):
            stats = model.get_all_mapping_stats()
            logger.log_mapping_stats(stats)

        total_time = time.time() - start_time
        logger.log_final_result(best_acc, final_train_acc, total_time)

    finally:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CIFAR-10 上训练 VGG 风格 Crossbar（标准 ReLU VGG 或器件神经元 VGG）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
模型选项:
  vgg                  标准 VGG（ReLU），与 train_unified --model vgg --dataset cifar10 一致
  vgg_device_neuron    VGG + my_device_activation 中的器件激活，与 train_unified
                       --model vgg_device_neuron --dataset cifar10 一致

配置选项:
  default  默认 (DeviceNeuron/config.py 中的 CONFIG)
  real     真实器件数据
  high     高精度
  fast     快速测试

示例:
  python -m DeviceNeuron.examples.train_cifar10_corssbar
  python -m DeviceNeuron.examples.train_cifar10_corssbar --model vgg_device_neuron --config real
  python -m DeviceNeuron.examples.train_cifar10_corssbar --config fast
'''
    )
    parser.add_argument('--model', type=str, default='vgg',
                        choices=['vgg', 'vgg_device_neuron'],
                        help='vgg: 标准 VGG；vgg_device_neuron: VGG + 器件激活')
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'real', 'high', 'fast'],
                        help='配置类型')
    parser.add_argument('--no-log', action='store_true',
                        help='禁用日志保存')
    ns = parser.parse_args()
    main(ns)
