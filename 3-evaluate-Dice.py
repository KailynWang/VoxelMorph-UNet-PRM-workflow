import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 读取配准后的图像和参考图像
reference_image = sitk.ReadImage("/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Reference_P099_160x192x224.nii")
registered_image = sitk.ReadImage("/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Registration_P099_BSpline_160x192x224.nii")

# 将SimpleITK图像转换为NumPy数组
registered_image_array = sitk.GetArrayFromImage(registered_image)
reference_image_array = sitk.GetArrayFromImage(reference_image)

print(f"Registered image min: {np.min(registered_image_array)}")
print(f"Registered image max: {np.max(registered_image_array)}")
print(f"Registered image median: {np.median(registered_image_array)}")

print(f"Reference image min: {np.min(reference_image_array)}")
print(f"Reference image max: {np.max(reference_image_array)}")
print(f"Reference image median: {np.median(reference_image_array)}")

# 二值化处理（如果感兴趣的区域是非背景的像素）
threshold_value_registered = 0.015612541697919369  # 调整这个值
threshold_value_reference = 0.013236876081144204     # 调整这个值
# 将图像二值化
binary_registered_image = registered_image_array > threshold_value_registered
binary_reference_image = reference_image_array > threshold_value_reference

# 可视化二值化图像的中间切片
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(binary_registered_image[binary_registered_image.shape[0] // 2], cmap='gray')
plt.title("Binary Registered Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_reference_image[binary_reference_image.shape[0] // 2], cmap='gray')
plt.title("Binary Reference Image")

plt.show()

def dice_coefficient(image1, image2):
    """
    计算两个二值化图像的Dice系数
    image1 和 image2 都是 NumPy 数组，包含0和1
    """
    # 计算交集（重叠区域）
    intersection = np.sum(image1 * image2)

    # 计算分母，避免除以零
    sum_image1 = np.sum(image1)
    sum_image2 = np.sum(image2)

    # 如果分母为0，返回0表示没有重叠
    if sum_image1 == 0 or sum_image2 == 0:
        return 0.0

    # 计算Dice系数
    return 2.0 * intersection / (sum_image1 + sum_image2)


# 计算Dice系数
dice = dice_coefficient(binary_registered_image, binary_reference_image)
print(f"Dice Coefficient: {dice}")