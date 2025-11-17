import nibabel as nib
import tensorflow as tf
from voxelmorph.tf.networks import VxmDense
import numpy as np
import matplotlib.pyplot as plt

# 载入吸气相图像(参考图像)
inspiration_image = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_IN_160x192x224.nii')
inspiration_data = inspiration_image.get_fdata()

# 载入呼气相图像（浮动图像）
expiration_image = nib.load('/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_OUT_160x192x224.nii')
expiration_data = expiration_image.get_fdata()

# 加载训练好的模型
model = tf.keras.models.load_model('/Users/wjl/Desktop/Graduation Project/code/VoxelMorph 1.0/model.test/model.h5', custom_objects={'VxmDense': VxmDense})

# 扩展维度以符合模型的输入要求
inspiration_data = np.expand_dims(inspiration_data, axis=0)  # (1, height, width, depth)
expiration_data = np.expand_dims(expiration_data, axis=0)  # (1, height, width, depth)

# 归一化处理（假设像素值在0-255之间）
inspiration_data = inspiration_data / 255.0
expiration_data = expiration_data / 255.0

# 使用训练好的模型进行配准
output = model.predict([expiration_data, inspiration_data])

# 输出结果（output包括配准后的图像和变换场）
registered_image = output[0]  # 变换后的呼气相图像

# 创建新的nifti图像对象并保存
registered_nifti = nib.Nifti1Image(registered_image[0], inspiration_image.affine)
nib.save(registered_nifti, '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_VoxelMorph_160x192x224.nii')

# 获取三个切面的索引（中间切片）
axial_idx = 120  # 横断面
coronal_idx = 110  # 冠状面
sagittal_idx = 120  # 矢状面

# 可视化三个切面
plt.figure(figsize=(15, 10))

# 横断面（Axial）
plt.subplot(2, 3, 1)
plt.imshow(inspiration_data[0, axial_idx, :, :], cmap='gray')
plt.title('Reference (Axial)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(registered_image[0, axial_idx, :, :], cmap='gray')
plt.title('Registered (Axial)')
plt.axis('off')

# 冠状面（Coronal）
plt.subplot(2, 3, 2)
plt.imshow(inspiration_data[0, :, coronal_idx, :], cmap='gray')
plt.title('Reference (Coronal)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(registered_image[0, :, coronal_idx, :], cmap='gray')
plt.title('Registered (Coronal)')
plt.axis('off')

# 矢状面（Sagittal）
plt.subplot(2, 3, 3)
plt.imshow(inspiration_data[0, :, :, sagittal_idx], cmap='gray')
plt.title('Reference (Sagittal)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(registered_image[0, :, :, sagittal_idx], cmap='gray')
plt.title('Registered (Sagittal)')
plt.axis('off')

plt.tight_layout()
plt.show()