# encoding: utf-8
# @File  : undistort_image.py
# @Author: XIE Yutai
# @Date  : 2024/11/27/10:50

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from project2img import project_to_img


# 创建内参矩阵K的函数
def create_intrinsic_matrix(fl_x, fl_y, cx, cy):
    fl_x = fl_x
    fl_y = fl_y
    cx = cx
    cy = cy
    return np.array([[fl_x, 0, cx], # fx, sparse, cx
                     [0, fl_y, cy], # sparse, fy, cy
                     [0, 0, 1]], dtype=np.float32)    # sparse, sparse, 1 (齐次坐标)

def get_distortion(distortion):
    return np.array([distortion['k1'], distortion['k2'], distortion['k3'], distortion['k4']], dtype=np.float32)

def read_json(json_file_path):
    with open(json_file_path, 'r') as file:
        calibration_data = json.load(file)
        # 访问数据
        calibration_time = calibration_data['calibration_time']
        cameras = calibration_data['cameras']
        '''
                # 遍历相机数据
        for index, camera in enumerate(cameras):
            print(f"相机 {index + 1}:")
            print(f"  名称: {camera['name']}")
            print(f"  宽度: {camera['width']}")
            print(f"  高度: {camera['height']}")

            # 内参
            intrinsic = camera['intrinsic']
            print(f"  焦距 fx: {intrinsic['fl_x']}, fy: {intrinsic['fl_y']}")
            print(f"  主点 cx: {intrinsic['cx']}, cy: {intrinsic['cy']}")

            # 畸变参数
            # 畸变参数
            distortion = camera['distortion']['params']
            print(f"  畸变参数 k1: {distortion['k1']}, k2: {distortion['k2']}, k3: {distortion['k3']}, k4: {distortion['k4']}")

        '''
        # 创建左相机和右相机的内参矩阵K
        K_left = create_intrinsic_matrix(**cameras[0]['intrinsic'])
        K_right = create_intrinsic_matrix(**cameras[1]['intrinsic'])
        print(f"左相机内参 {K_left}")
        print(f"右相机内参 {K_right}")

        # 鱼眼镜头的畸变系数
        left_distortion = get_distortion(cameras[0]["distortion"]["params"])
        right_distortion = get_distortion(cameras[1]["distortion"]["params"])
        print(f"左相机畸变参数 {left_distortion}")
        print(f"右相机畸变参数 {right_distortion}")
        left_shape = [cameras[0]["width"],cameras[0]["height"]]
        right_shape = [cameras[1]["width"],cameras[1]["height"]]
        return K_left, K_right, left_distortion, right_distortion,left_shape,right_shape


def main():
    json_file_path = './info/calibration.json'
    K_left, K_right, left_distortion, right_distortion,left_shape,right_shape = read_json(json_file_path)

    # 读取鱼眼图像
    # img_path = './maize/out_image/left/6.png'
    img_path = 'info/4k/output_bymyself/images/102.png'
    image = cv2.imread(img_path)
    #image = cv2.resize(image, left_shape)
    print(image.shape[:2][::-1])
    width, height = image.shape[:2][::-1]
    print("width",width)
    print("height",height)

    # 760 * 1008  2016*1520 4032*3040
    fov = 140
    size = 512
    assert width > 0 and height > 0, f"{width}x{height}"
    print(f"width: {width}, height: {height}")
    fullscale_width, fullscale_height = 3040,4032 #3040
    scale_factor = width / fullscale_width
    # K_left = K_left*scale_factor
    print(f"fullscale image size: {fullscale_width}x{fullscale_height}, scale factor: {scale_factor}")
    new_height, new_width = size, size
    focal_new = new_width / 2 / np.tan(fov/2/180*np.pi)
    cv2_intrinsics = np.array([
        focal_new, 0, new_height/2,
        0, focal_new, new_width/2,
        0, 0, 1
    ], dtype=np.float32).reshape(3,3)
    #mapx_left, mapy_left = cv2.fisheye.initUndistortRectifyMap(K_left, left_distortion, np.eye(3), cv2_intrinsics, (new_width, new_height), cv2.CV_32FC1)
    mapx_left, mapy_left = cv2.fisheye.initUndistortRectifyMap(K_left, left_distortion, np.eye(3), K_left, (width, height), cv2.CV_32FC1)
    #mapx_right, mapy_right = cv2.fisheye.initUndistortRectifyMap(K_left, left_distortion, np.eye(3), K_left, (width, height), cv2.CV_32FC1)


    new_image = cv2.remap(image, mapx_left, mapy_left, cv2.INTER_LINEAR)
    cv2.imshow('Undistorted Image', new_image)
    cv2.imwrite('./maize/out_image/left/undistorted_image.png', new_image)



if __name__ == '__main__':
    main()

