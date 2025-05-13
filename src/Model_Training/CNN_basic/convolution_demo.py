"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-04-15 00:28:15
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-04-15 00:29:55
FilePath: /CNN-tutorial/src/convolution_demo.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np

mlp.use("Agg")


def apply_convolution(image_path, kernel, name):
    # 打开图片并转换为灰度图
    image = Image.open(image_path).convert("L")

    # 转换为张量
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)  # 增加批次维度 (1, 1, H, W)

    # 卷积操作
    kernel_tensor = (
        torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, 3, 3)
    convolved_image = F.conv2d(image_tensor, kernel_tensor, padding=1)  # 保持尺寸

    # 转换回 NumPy 数组以便显示
    convolved_image_np = convolved_image.squeeze().detach().numpy()

    # 显示原图和卷积后的图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Convolved Image")
    plt.imshow(convolved_image_np, cmap="gray")
    plt.savefig(f"img/Convolved_img {name}.png")
    plt.close()


# 示例调用
if __name__ == "__main__":
    # 定义 3x3 卷积核（例如边缘检测）
    kernel_1 = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    kernel_2 = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    kernel_3 = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    kernel_4 = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_5 = np.random.randn(3, 3)
    kernel_5 = kernel_5 / kernel_5.sum()
    kernel_6 = np.identity(3) / 3
    kernel_7 = [[-2, -3, -2], [-3, 21, -3], [-2, -3, -2]]

    # 替换为您本地的图片路径
    image_path = "img/demo_cat.jpg"  # 请确保路径正确
    apply_convolution(image_path, kernel_1, "edge_detection")
    apply_convolution(image_path, kernel_2, "sharpen")
    apply_convolution(image_path, kernel_3, "normalize")
    apply_convolution(image_path, kernel_4, "edge_detect2")
    apply_convolution(image_path, kernel_5, "just for fun")
    apply_convolution(image_path, kernel_6, "kernel6")
    apply_convolution(image_path, kernel_7, "kernel_7")

    # the usage of surbo kernel
    kernel_surbo_1 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_surbo_2 = [[1, 2, 1], [0, 0, 0], [-1, -2, -2]]
    apply_convolution(image_path, kernel_surbo_1, "surbo_1")
    apply_convolution(image_path, kernel_surbo_2, "surbo_2")
