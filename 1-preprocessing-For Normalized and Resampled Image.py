import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# ------------------------ 原始图像信息 ------------------------
original_image = sitk.ReadImage('/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_OUT.nii')
original_size = original_image.GetSize()
original_array = sitk.GetArrayFromImage(original_image)
print("[原始图像]")
print(f"尺寸 (Z,Y,X): {original_size}")
print(f"灰度值范围: [{np.min(original_array):.2f}, {np.max(original_array):.2f}]\n")


# ------------------------ 归一化处理 --------------------------
image_array = sitk.GetArrayFromImage(original_image)
image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
normalized_image = sitk.GetImageFromArray(image_array)
normalized_image.CopyInformation(original_image)


# ------------------------ 重采样处理 --------------------------
# 设置目标尺寸和计算新spacing
target_size = [160, 192, 224]  # X, Y, Z
original_size = normalized_image.GetSize()
original_spacing = normalized_image.GetSpacing()
new_spacing = [
    original_spacing[0] * original_size[0] / target_size[0],
    original_spacing[1] * original_size[1] / target_size[1],
    original_spacing[2] * original_size[2] / target_size[2]
]

# 配置重采样器（使用线性插值）
resampler = sitk.ResampleImageFilter()
resampler.SetSize(target_size)
resampler.SetOutputSpacing(new_spacing)
resampler.SetOutputOrigin(normalized_image.GetOrigin())
resampler.SetOutputDirection(normalized_image.GetDirection())
resampler.SetInterpolator(sitk.sitkLinear)

# 执行重采样
resampled_image = resampler.Execute(normalized_image)

# 重新标准化到[0,1]
resampled_array = sitk.GetArrayFromImage(resampled_image)
resampled_array = (resampled_array - np.min(resampled_array)) / (np.max(resampled_array) - np.min(resampled_array))
resampled_image = sitk.GetImageFromArray(resampled_array)
resampled_size = resampled_image.GetSize()
resampled_image.CopyInformation(resampled_image)  # 保留spacing/origin/direction


# ------------------------ 可视化与验证 -------------------------
# 可视化
plt.imshow(resampled_array[resampled_array.shape[0] // 2], cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title("Resampled Image (Middle Slice)")
plt.show()

# 验证输出范围
stats = sitk.StatisticsImageFilter()
stats.Execute(resampled_image)
print(f"尺寸 (Z,Y,X): {resampled_size}")
print(f"Resampled Image Min: {stats.GetMinimum():.4f}, Max: {stats.GetMaximum():.4f}")  # 输出应为[0.0, 1.0]
# 保存结果
sitk.WriteImage(resampled_image, '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_OUT_160x192x224.nii')