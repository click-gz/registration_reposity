import cv2
import numpy as np

def detect_sift_keypoints_3d(volume, s):
    # 获取三维图像的尺寸
    depth, height, width = volume.shape

    # 创建SIFT特征检测器对象
    sift = cv2.SIFT_create()

    # 初始化存储关键点的空列表
    keypoints_3d = []

    # 在每个切片上应用SIFT特征检测器
    for z in range(0, 1):
        # 获取当前切片
        slice_image = volume[z]
        cv2.imwrite(s+".jpg", slice_image)
        # 转换为8位灰度图像
        slice_image_gray = np.uint8(slice_image)

        # 检测关键点和计算特征描述子
        keypoints, descriptors = sift.detectAndCompute(slice_image_gray, None)

        # 绘制关键点（可选，用于可视化）
        keypoints_image = cv2.drawKeypoints(slice_image_gray, keypoints, None)
        # cv2.imshow("Keypoints Image", keypoints_image)
        # input()
        cv2.imwrite(s+str(z)+".jpg", keypoints_image)
        # print(keypoints, descriptors)
        # 将关键点的z坐标设置为当前切片的索引
        # for kp in keypoints:
        #     kp.pt = (kp.pt[0], kp.pt[1])
        #     print(kp.pt, z)

        # 将当前切片的关键点添加到总体关键点列表中
        keypoints_3d.extend(keypoints)

    return keypoints_3d, descriptors

# 假设您已经加载了三维CT图像数据，并将其存储在名为volume的NumPy数组中
# volume的形状应该是 (depth, height, width)
import SimpleITK as sitk
image_array = sitk.GetArrayFromImage(sitk.ReadImage("./data/scan/01_Fixed.nii"))
image_array1 = sitk.GetArrayFromImage(sitk.ReadImage("./data/scan/01_Moving.nii"))
print("shape: ", image_array.shape, image_array1.shape)
# 调用函数提取三维CT图像的关键点
source_keypoints, source_descriptors = detect_sift_keypoints_3d(image_array, "f")
target_keypoints, target_descriptors = detect_sift_keypoints_3d(image_array1, "m")
# 使用FLANN匹配器进行特征点匹配
flann = cv2.FlannBasedMatcher_create()
matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)

# 进行筛选，只保留良好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 获取配准前后对应特征点的坐标
source_points = [source_keypoints[m.queryIdx].pt for m in good_matches]
target_points = [target_keypoints[m.trainIdx].pt for m in good_matches]

# 输出配准前后对应特征点的坐标
print("配准前特征点坐标：", source_points)
print("配准后特征点坐标：", target_points)
# 输出关键点数量
print("Number of keypoints:", len(source_points), len(target_points))
