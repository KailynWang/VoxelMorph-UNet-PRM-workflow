import nibabel as nib
import numpy as np

# 读取 NIfTI 文件
nii_path = "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_VoxelMorph_160x192x224.nii"
img = nib.load(nii_path)

# 获取数据
data = img.get_fdata()

# 计算最小值和最大值
min_val = np.min(data)
max_val = np.max(data)

print(f"图像的灰度值范围: {min_val} ~ {max_val}")

# 归一化到 [0,1]
data_fixed = (data - np.min(data)) / (np.max(data) - np.min(data))

# 检查归一化后的范围
print(f"修复后图像灰度范围: {np.min(data_fixed)} ~ {np.max(data_fixed)}")

# 创建新的 NIfTI 对象并保存
fixed_img = nib.Nifti1Image(data_fixed, affine=img.affine, header=img.header)
nib.save(fixed_img, "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_VoxelMorph_160x192x224.nii")

#print("修复后的 NIfTI 图像已保存为: Registration_P018_512x512x224_fixed.nii")