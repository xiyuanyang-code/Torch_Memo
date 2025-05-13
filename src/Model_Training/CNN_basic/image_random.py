"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-04-15 00:06:09
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-04-15 00:08:24
FilePath: /CNN-tutorial/src/image_random.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mlp
mlp.use("Agg")

# 1. 对一张照片施加线性变换（旋转）
def apply_linear_transform(image_path, angle):
    # 打开图片
    image = Image.open(image_path)

    # 旋转图片
    transform_rotate = transforms.RandomRotation([angle, angle])
    rotated_image = transform_rotate(image)  # 直接返回 PIL.Image.Image 对象

    # 显示原图和旋转后的图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title(f"Rotated Image ({angle}°)")
    plt.imshow(rotated_image)  # 直接显示旋转后的 PIL.Image.Image 对象
    plt.savefig("img/Rotated_img.png")
    plt.close()


# 2. 生成一个随机矩阵并转化成雪花图
def generate_snowflake_image(size):
    # 生成随机矩阵
    random_matrix = torch.rand(size, size)

    # 转换为 numpy 数组
    random_array = random_matrix.numpy()

    plt.imshow(random_array, cmap="Blues")
    plt.title("Snowflake Image")
    plt.colorbar()
    plt.savefig("img/random_img.jpg")
    plt.close()


# 示例调用
if __name__ == "__main__":
    # 替换为您本地的图片路径
    image_path = "img/Signal_with_Continuous_Effect_alpha=0.01.png"  # 请确保路径正确
    apply_linear_transform(image_path, angle=45)

    generate_snowflake_image(size=256)
