import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# 加载NIfTI文件
fixed_img = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Reference_P099_160x192x224.nii')
registered_img = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Registration_P099_BSpline_160x192x224.nii')

# 获取图像数据
registered_data = registered_img.get_fdata()
fixed_data = fixed_img.get_fdata()

# 使用 squeeze 移除大小为1的维度
registered_data = registered_data.squeeze()
fixed_data = fixed_data.squeeze()

# 确保数据形状一致
assert registered_data.shape == fixed_data.shape, "图像大小不匹配"

# 计算差分图（绝对值）
difference_data = np.abs(registered_data - fixed_data)

# 获取中间切片的索引
depth_idx = registered_data.shape[2] // 2  # 假设我们想查看深度维度的中间切片

# 计算MSE
mse_value = mean_squared_error(fixed_data[:, :, depth_idx].flatten(), registered_data[:, :, depth_idx].flatten())

# 创建图像显示
plt.figure(figsize=(12, 4))

# 显示固定图像的中间切片
plt.subplot(1, 3, 1)
plt.imshow(fixed_data[:, :, depth_idx], cmap='gray')
plt.title('Fixed Image (P018_IN_Bspline)')

# 显示配准图像的中间切片
plt.subplot(1, 3, 2)
plt.imshow(registered_data[:, :, depth_idx], cmap='gray')
plt.title('Registered Image (Expiration to Inspiration)')

# 显示差分图的中间切片
plt.subplot(1, 3, 3)
plt.imshow(difference_data[:, :, depth_idx], cmap='hot')
plt.title('Difference Image')

plt.tight_layout()
plt.show()

# 输出MSE
print(f"MSE between Fixed and Registered Images: {mse_value}")