import numpy as np
import json

def read_image():
    pass

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
            distortion = camera['distortion']['params']
            print(f"  畸变参数 k1: {distortion['k1']}, k2: {distortion['k2']}, k3: {distortion['k3']}, k4: {distortion['k4']}")

            # 从激光雷达到相机的变换
            transform = camera['transform_from_lidar']
            rotation = transform['rotation']
            position = transform['position']
            print(f"  旋转矩阵: {rotation}")
            print(f"  平移向量: {position}")
        '''
        # 创建左相机和右相机的内参矩阵K
        K_left = create_intrinsic_matrix(**cameras[0]['intrinsic'])
        K_right = create_intrinsic_matrix(**cameras[1]['intrinsic'])
        print(f"左相机内参 {K_left}")
        print(f"左相机内参 {K_right}")

        # 创建左相机和右相机的外参矩阵R和T
        R_left, T_left = create_extrinsic_matrix(**cameras[0]['transform_from_lidar'])
        R_right, T_right = create_extrinsic_matrix(**cameras[1]['transform_from_lidar'])
        return K_left, K_right, R_left, T_left, R_right, T_right

# 创建内参矩阵K的函数
def create_intrinsic_matrix(fl_x, fl_y, cx, cy):
    return np.array([[fl_x, 0, cx],
                     [0, fl_y, cy],
                     [0, 0, 1]])

# 创建旋转矩阵R和 平移向量T的函数
def create_extrinsic_matrix(rotation, position):
    R = np.array(rotation)
    T = np.array(position).reshape(-1, 1)  # 将平移向量转换为3x1矩阵
    return R, T

# k：相机内参矩阵intrinsic（fl_x、fl_y为焦距信息，cx、cy为主点信息）
def pixel_to_camera(pixel_coords, depth, k_inv):
    pixel_homogeneous = np.array([pixel_coords[0], pixel_coords[1], 1])
    # 将像素坐标转换为相机坐标系中的齐次坐标 (X_c, Y_c, Z_c)
    camera_coords_homogeneous = np.dot(k_inv, pixel_homogeneous) * depth
    return camera_coords_homogeneous

def camera_to_world(camera_coords, R, T):
    # 世界坐标系中的坐标 = R * 相机坐标系坐标 + T
    world_coords = np.dot(R, camera_coords) + T
    return world_coords

def check_pixel(pixel_coords, width, height):
    pass

def pixel_to_world(pixel_coords, depth, K, R, T):
    """
    从像素坐标系转换到世界坐标系
    :param pixel_coords: 像素坐标 (x_p, y_p)
    :param depth: 深度值 z_p
    :param K: 相机内参矩阵 (3x3)
    :param R: 旋转矩阵 (3x3)
    :param T: 平移向量 (3x1)
    :return: 世界坐标系中的坐标 (X_w, Y_w, Z_w)
    """
    # 计算相机内参矩阵的逆矩阵
    K_inv = np.linalg.inv(K)
    # 从像素坐标转换到相机坐标系
    camera_coords = pixel_to_camera(pixel_coords, depth, K_inv)
    # 从相机坐标系转换到世界坐标系
    world_coords = camera_to_world(camera_coords, R, T)
    return world_coords

def main():
    json_file_path = './info/calibration.json'
    K_left, K_right, R_left, T_left, R_right, T_right = read_json(json_file_path)

    # exp 假设读取左眼相机的坐标系变换，深度depth为5(当前欠一个深度信息）,像素坐标设定(应该由读取的图片提供）
    pixel_coords = [400, 200]  # 像素坐标 (x_p, y_p)
    depth = 5.0  # 深度 z_p（该点距离相机的距离）
    world_coords = pixel_to_world(pixel_coords, depth, K_left, R_left, T_left)
    print("世界坐标系中的坐标 (X_w, Y_w, Z_w):", world_coords)



if __name__ == '__main__':
    main()
