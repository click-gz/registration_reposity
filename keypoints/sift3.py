import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
def extract_sift_keypoints(image_slice):
    # 提取关键点和计算描述子
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_slice, None)

    return keypoints, descriptors

# 读取三维图像数据
image_path = "./data/scan/01_Fixed.nii"
image_path1 = "./data/scan/01_Moving.nii"

image = sitk.ReadImage(image_path)
image1 = sitk.ReadImage(image_path1)


# 获取图像的shape
image_shape = image.GetSize()
image_shape1 = image1.GetSize()


# 选择z轴的切片索引
z_slice_index = image_shape[2] // 2
z_slice_index1 = image_shape1[2] // 2


# 将三维图像投影到z轴的切片上
image_slice = sitk.GetArrayFromImage(image)[:, :, z_slice_index]
image_slice = np.uint8(image_slice)
image_slice1 = sitk.GetArrayFromImage(image1)[:, :, z_slice_index1]
image_slice1 = np.uint8(image_slice1)

# 提取SIFT特征点和描述子
keypoints, descriptors = extract_sift_keypoints(image_slice)
keypoints1, descriptors1 = extract_sift_keypoints(image_slice1)


# 可视化图像和特征点
plt.figure()
plt.imshow(image_slice, cmap='gray')

# 绘制特征点，使用红色的点标记特征点
x_coords = [kp.pt[0] for kp in keypoints]
y_coords = [kp.pt[1] for kp in keypoints]
plt.scatter(x_coords, y_coords, c='red', marker='o', s=10)

# 添加标题和坐标轴标签
plt.title("Projected Slice with Extracted Keypoints")
plt.xlabel("X")
plt.ylabel("Y")

plt.figure()
plt.imshow(image_slice1, cmap='gray')
# 绘制特征点，使用红色的点标记特征点
x_coords1 = [kp.pt[0] for kp in keypoints1]
y_coords1 = [kp.pt[1] for kp in keypoints1]
plt.scatter(x_coords1, y_coords1, c='red', marker='o', s=10)
# 添加标题和坐标轴标签
plt.title("Moving Projected Slice with Extracted Keypoints")
plt.xlabel("X")
plt.ylabel("Y")

# 显示绘制的图像
plt.show()

# 使用FLANN匹配器进行特征点匹配
flann = cv2.FlannBasedMatcher_create()
matches = flann.knnMatch(descriptors, descriptors1, k=2)

# 进行筛选，只保留良好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 获取配准前后对应特征点的坐标
source_points = [keypoints[m.queryIdx].pt for m in good_matches]
target_points = [keypoints1[m.trainIdx].pt for m in good_matches]

# 输出配准前后对应特征点的坐标
print("配准前特征点坐标：", source_points)
print("配准后特征点坐标：", target_points)
# 输出关键点数量
print("Number of keypoints:", len(source_points), len(target_points))