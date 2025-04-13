# encoding: utf-8
# @File  : cali.py
# @Author: XIE Yutai
# @Date  : 2025/02/27/16:48

import cv2
import numpy as np
import glob
import os

# 1. 准备标定板的参数
# 标定板的棋盘格角点数（内角点数）
chessboard_size = (8, 6)  # 棋盘格内角点数，例如 9x6 的棋盘格

# 2. 准备存储角点坐标的数组
# 世界坐标系中的点（例如棋盘格的物理尺寸，单位可以是毫米或米）
square_size = 25.0  # 棋盘格每个方格的边长（单位：毫米）
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# 存储所有图像的角点坐标
objpoints = []  # 世界坐标系中的点
imgpoints = []  # 图像坐标系中的点

# 3. 读取标定图像
# 假设标定图像存储在 "./calibration_images" 文件夹中
images_path = "./resource/cali_pic2/*.jpg"
images = glob.glob(images_path)

if len(images) == 0:
    print("未找到标定图像，请检查路径！")
    exit()

# 4. 检测角点并存储
for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到足够的角点，则存储
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Chessboard Corners", img)
        cv2.waitKey(100)
        save_path = os.path.join("./resource/cali_output", os.path.basename(image_path))
        cv2.imwrite(save_path, img)
        print(f"Saved image with corners to: {save_path}")

cv2.destroyAllWindows()

# 5. 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 6. 打印标定结果
print("相机内参数矩阵 (Camera Matrix):")
print(camera_matrix)
print("\n畸变系数 (Distortion Coefficients):")
print(dist_coeffs)
print("\n旋转向量 (Rotation Vectors):")
print(rvecs)
print("\n平移向量 (Translation Vectors):")
print(tvecs)

fx = camera_matrix[0, 0]  # 焦距 fx
fy = camera_matrix[1, 1]  # 焦距 fy
cx = camera_matrix[0, 2]  # 主点 cx
cy = camera_matrix[1, 2]  # 主点 cy

print("\n焦距 fx (像素):", fx)
print("焦距 fy (像素):", fy)
print("主点 cx (像素):", cx)
print("主点 cy (像素):", cy)

# 8. 评估标定结果
# 计算重投影误差
total_error = 0
for i in range(len(objpoints)):
    img_points2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], img_points2, cv2.NORM_L2) / len(img_points2)
    total_error += error

print("\n平均重投影误差 (Mean Reprojection Error):", total_error / len(objpoints))