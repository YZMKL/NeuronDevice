# DeviceNeuron/analysis/features/cifar10_deviceneuron_confusion.py
# python -m DeviceNeuron.analysis.features.cifar10_deviceneuron_confusion

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from DeviceNeuron import CIFAR10UnifiedVGGWithDeviceNeuron
from DeviceNeuron.config import create_real_device_config
from DeviceNeuron.my_device_activation import MyDeviceActivation


def plot_confusion_matrix(model_path, save_dir="DeviceNeuron/analysis/features/figures"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 数据集 ──
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    classes = test_dataset.classes

    # ── 加载模型 ──
    cfg = create_real_device_config()
    config = cfg.get_crossbar_config()
    device_activation = MyDeviceActivation()

    model = CIFAR10UnifiedVGGWithDeviceNeuron(
        config=config,
        device_activation=device_activation,
        use_batchnorm=True,
        dropout=0.5,
        k_int_input=cfg.INTEGRATOR_K_INPUT,
        k_int_output=cfg.INTEGRATOR_K_OUTPUT,
        device_noise=cfg.DEVICE_NOISE_STD
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 推理 ──
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                confusion[t][p] += 1

    acc = confusion.diagonal().sum() / confusion.sum() * 100
    print(f"整体准确率: {acc:.2f}%")

    # ── 画混淆矩阵 ──
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 归一化到百分比（按行）
    confusion_norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(confusion_norm, interpolation='nearest', cmap='Blues')
    cbar = plt.colorbar(im, ax=ax, label='Recall (%)')
    cbar.ax.tick_params(labelsize=14)  # 加这行，改数字大小
    cbar.set_label('Recall (%)', fontsize=18)  # 改label大小

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=14)
    ax.set_yticklabels(classes, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('True', fontsize=18)
    ax.set_title(f'CIFAR-10 —— vgg_device_neuron\n(Accuracy: {acc:.2f}%)', fontsize=18, pad=15)

    # 在每个格子里写数字
    thresh = confusion_norm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{confusion_norm[i,j]:.1f}',
                    ha='center', va='center', fontsize=14,
                    color='white' if confusion_norm[i, j] > thresh else 'black')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix_deviceneuron.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


if __name__ == "__main__":
    MODEL_FILE = "unified_cifar10_vgg_device_neuron_best.pth"

    plot_confusion_matrix(
        model_path=MODEL_FILE,
        save_dir="DeviceNeuron/analysis/features/figures"
    )