import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像并标准化到[0,1]
original_image = sitk.ReadImage('/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_VoxelMorph_160x192x224.nii')
image_array = sitk.GetArrayFromImage(original_image)
image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
normalized_image = sitk.GetImageFromArray(image_array)
normalized_image.CopyInformation(original_image)

# 设置目标尺寸和计算新spacing
target_size = [512, 512, 224]  # X, Y, Z
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
resampled_image.CopyInformation(resampled_image)  # 保留spacing/origin/direction

# 可视化
plt.imshow(resampled_array[resampled_array.shape[0] // 2], cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.title("Resampled Image (Middle Slice)")
plt.show()

# 验证输出范围
stats = sitk.StatisticsImageFilter()
stats.Execute(resampled_image)
print(f"Resampled Image Min: {stats.GetMinimum():.4f}, Max: {stats.GetMaximum():.4f}")  # 输出应为[0.0, 1.0]

# 保存结果
sitk.WriteImage(resampled_image, '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_512x512x224.nii')