import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from PIL import Image

# 读取文件
nii_file = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_IN.nii'  ##改##

# 读取图像数据
image = sitk.ReadImage(nii_file)

# 将图像数据转换为 numpy 数组
image_array = sitk.GetArrayFromImage(image)

# 创建保存切片的文件夹
output_dir = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_In'                                 ##改##
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 处理 4D 或其他维度的情况
if len(image_array.shape) == 3:
    num_slices, height, width = image_array.shape  # 3D 图像
elif len(image_array.shape) == 4:
    num_slices, height, width, channels = image_array.shape  # 4D 图像（多通道）
    print(f"Number of channels: {channels}")
    # 可以选择处理特定通道，例如使用 image_array[..., 0] 来选择第一个通道
    image_array = image_array[..., 0]  # 选择第一个通道
else:
    raise ValueError("Unexpected image shape")

print(f"Image shape after handling: {image_array.shape}")

# 指定循环的起始位置和结束位置
start_index = 0  # 从第*张切片开始
end_index = 263   # 处理到第*张切片（包含）                                                                                 ##改##

# 遍历所有切片并保存为 PNG
for i in range(start_index, end_index):
    slice_data = image_array[i]  # 获取第 i 张切片

    # 保存为 PNG 文件
    slice_save_path = os.path.join(output_dir, f' P004_In_{i - start_index + 1}.png')                                   ##改##
    plt.imsave(slice_save_path, slice_data, cmap='gray')  # 保存为灰度图像

    print(f"Saved slice {i - start_index + 1} as {slice_save_path}")

print("All slices have been saved successfully.")