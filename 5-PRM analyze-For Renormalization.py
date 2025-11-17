import numpy as np
import nibabel as nib

# 读取归一化后的 NIfTI 图像
normalized_nii_path = "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_512x512x224.nii"
img = nib.load(normalized_nii_path)
normalized_data = img.get_fdata()

# 原始图像的灰度范围
min_val = -1023.0
max_val = 10991.0

# 反归一化操作（如果归一化到 [0,1]）
original_data = normalized_data * (max_val - min_val) + min_val


# 创建新的 NIfTI 图像并保存
original_img = nib.Nifti1Image(original_data, img.affine, img.header)
nib.save(original_img, "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_512x512x224.nii")

print("反归一化完成，保存为 renormalization.nii")