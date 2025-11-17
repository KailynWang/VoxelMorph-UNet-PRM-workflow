import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 读取配准后的图像和参考图像
registered_image = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Registration_P099_160x192x224.nii')  # 替换为你的配准后图像路径
reference_image = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Reference_P099_160x192x224.nii')    # 替换为你的参考图像路径

# 获取图像数据
registered_data = registered_image.get_fdata()
reference_data = reference_image.get_fdata()

# 计算差异图（绝对值差异）
diff_data = np.abs(registered_data - reference_data)

# 创建差异图的NIfTI图像
diff_image = nib.Nifti1Image(diff_data, registered_image.affine)

# 保存差异图
nib.save(diff_image, '/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Difference_image_P099.nii')  # 保存为NIfTI文件

# 显示差异图
# 选择一个切片进行显示（例如中间切片）
slice_index = diff_data.shape[2] // 2  # 选择中间切片
plt.imshow(diff_data[:, :, slice_index], cmap='gray')  # 使用灰度图显示
plt.colorbar(label='Difference Intensity')  # 添加颜色条
plt.title('Difference Map')  # 添加标题
plt.axis('off')  # 关闭坐标轴
plt.show()