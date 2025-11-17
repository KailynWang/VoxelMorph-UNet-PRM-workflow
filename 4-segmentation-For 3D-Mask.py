import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

# 设置PNG文件路径和输出的NIfTI文件路径
png_dir = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Mask'  # PNG文件所在目录
output_nii_file = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_Mask.nii'  # 输出的NIfTI文件路径

# 获取PNG文件列表并按顺序排序
png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')], key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 初始化一个空的三维数组
num_slices = len(png_files)
height, width = Image.open(os.path.join(png_dir, png_files[0])).size
mask_array = np.zeros((num_slices, height, width), dtype=np.uint8)

# 逐个读取PNG文件并填充到三维数组中
for i, png_file in enumerate(png_files):
    slice_image = np.array(Image.open(os.path.join(png_dir, png_file)).convert('L'))  # 读取为灰度图像
    mask_array[i] = slice_image  # 填充到三维数组中

# 创建SimpleITK图像对象
mask_image = sitk.GetImageFromArray(mask_array)

# 设置图像的空间信息（可选，根据需要设置）
mask_image.SetSpacing((1.0, 1.0, 1.0))  # 设置空间分辨率
mask_image.SetOrigin((0.0, 0.0, 0.0))   # 设置原点

# 保存为NIfTI文件
sitk.WriteImage(mask_image, output_nii_file)

print(f"Successfully saved NIfTI file: {output_nii_file}")