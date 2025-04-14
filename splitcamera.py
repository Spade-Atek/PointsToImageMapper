# encoding: utf-8
# @File  : splitcamera.py
# @Author: XIE Yutai
# @Date  : 2025/04/01/20:37

import cv2
import os


def process_images(input_folder, left_folder, right_folder):
    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

    # 确保输出文件夹存在
    if not os.path.exists(left_folder):
        os.makedirs(left_folder)
    if not os.path.exists(right_folder):
        os.makedirs(right_folder)

    for image_file in image_files:
        # 读取图片
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        # 假设每张图片包含左影像和右影像，左右影像的分割点是图片宽度的一半
        height, width, _ = image.shape
        left_image = image[:, :width // 2]  # 左半部分
        right_image = image[:, width // 2:]  # 右半部分

        # 构造输出文件名
        left_filename = image_file.replace('.', '_left.')
        right_filename = image_file.replace('.', '_right.')

        # 保存处理后的图片
        cv2.imwrite(os.path.join(left_folder, left_filename), left_image)
        cv2.imwrite(os.path.join(right_folder, right_filename), right_image)

        print(f"Processed {image_file}: saved as {left_filename} and {right_filename}")


# 示例：调用函数
# input_folder = 'image/new2'  # 输入图片文件夹路径
# left_folder = 'image/outimage/left_image'    # 左影像输出文件夹路径
# right_folder = 'image/outimage/right_image'  # 右影像输出文件夹路径
# input_folder = 'image/png'  # 输入图片文件夹路径
# left_folder = 'image/pngimage/left_image'  # 左影像输出文件夹路径
# right_folder = 'image/pngimage/right_image'  # 右影像输出文件夹路径
input_folder = "C:\\Users\\XIE Yutai\\Desktop\\bd\\bmp"
left_folder = 'C:\\Users\\XIE Yutai\\Desktop\\bd\\outbmp\\left_image'  # 左影像输出文件夹路径
right_folder = 'C:\\Users\\XIE Yutai\\Desktop\\bd\\outbmp\\right_image'  # 右影像输出文件夹路径

process_images(input_folder, left_folder, right_folder)
