import numpy as np
import SimpleITK as sitk

def extract_gradient_based_keypoints(image, threshold=300):
    # 将SimpleITK图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(image)

    # 计算三维图像的梯度，可以使用Sobel算子等方法
    # 这里简单使用中心差分计算x、y和z方向的梯度
    gradient_x = np.gradient(image_array, axis=2)
    gradient_y = np.gradient(image_array, axis=1)
    gradient_z = np.gradient(image_array, axis=0)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)

    # 根据设定的阈值选择梯度幅值较大的像素点作为特征点
    keypoints = np.argwhere(gradient_magnitude > threshold)

    return keypoints

# 读取三维图像数据
image_path = "./data/scan/01_Fixed.nii"
image = sitk.ReadImage(image_path)
#x, y, z
#z, y, x
# 提取特征点
keypoints = extract_gradient_based_keypoints(image)

# 输出特征点的坐标
print("特征点坐标：", keypoints)

import matplotlib.pyplot as plt
#  将特征点的坐标转换为x、y和z坐标数组
x_coords = keypoints[:, 2]  # z坐标对应数组的第三列
y_coords = keypoints[:, 1]  # y坐标对应数组的第二列
z_coords = keypoints[:, 0]  # x坐标对应数组的第一列

# 将SimpleITK图像转换为NumPy数组，用于绘制
image_array = sitk.GetArrayFromImage(image)
# 获取图像的shape
image_shape = image_array.shape

# 翻转y坐标以调整坐标原点
y_coords = image_shape[1] - y_coords - 1
# 可视化三维图像的其中一层，这里选择z坐标最大值处的切片
slice_index = np.max(z_coords)

# 绘制图像
plt.figure()
plt.imshow(image_array[slice_index], cmap='gray')
plt.show()
# 绘制特征点，使用红色的点标记特征点
plt.scatter(x_coords[z_coords == slice_index], y_coords[z_coords == slice_index], c='red', marker='o', s=10)

# 添加标题和坐标轴标签
plt.title("3D Image with Extracted Keypoints")
plt.xlabel("X")
plt.ylabel("Y")

# 显示绘制的图像
plt.show()