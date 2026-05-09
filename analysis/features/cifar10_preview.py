# DeviceNeuron/analysis/features/cifar10_preview.py
# python -m DeviceNeuron.analysis.features.cifar10_preview

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np

def preview_cifar10(save_dir="DeviceNeuron/analysis/features/demo", n_per_class=20, cols=5):
    tf_test = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)
    classes = test_dataset.classes

    os.makedirs(save_dir, exist_ok=True)
    rows = n_per_class // cols

    for class_idx, class_name in enumerate(classes):
        matching = [i for i, l in enumerate(test_dataset.targets) if l == class_idx][:n_per_class]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
        fig.suptitle(f"CIFAR-10: {class_name}", fontsize=16, y=1.01)

        for j, img_idx in enumerate(matching):
            img_tensor, _ = test_dataset[img_idx]
            img = img_tensor.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            ax = axes[j // cols][j % cols]
            ax.imshow(img)
            ax.set_title(f"#{j}  (idx:{img_idx})", fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{class_idx:02d}_{class_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path}")

    print(f"\n全部完成，保存在: {save_dir}/")


if __name__ == "__main__":
    preview_cifar10(
        save_dir="DeviceNeuron/analysis/features/demo",
        n_per_class=20,
        cols=5
    )