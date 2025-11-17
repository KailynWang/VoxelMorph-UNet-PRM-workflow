
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 加载三维掩膜图像
mask_path = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_Mask_512x512x224.nii'  # 替换为你的掩膜文件路径
mask_image = sitk.ReadImage(mask_path)
mask_array = sitk.GetArrayFromImage(mask_image)

# 加载吸气相和呼气相三维图像
insp_image_path = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Reference_P004_512x512x224.nii'  # 吸气相图像
exp_image_path = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_512x512x224.nii'  # 呼气相图像
insp_image = sitk.ReadImage(insp_image_path)
exp_image = sitk.ReadImage(exp_image_path)

# 将图像数据转换为 numpy 数组
insp_array = sitk.GetArrayFromImage(insp_image)
exp_array = sitk.GetArrayFromImage(exp_image)

# PRM分类规则
prm_map = np.zeros_like(insp_array, dtype=np.uint8)

# 应用掩膜，得到掩膜区域的图像数据
insp_masked = np.where(mask_array == 1, insp_array, np.nan)
exp_masked = np.where(mask_array == 1, exp_array, np.nan)

# 进行PRM分类
prm_map[(insp_masked > -950) & (exp_masked > -856)] = 1  # 正常肺组织（绿色）
prm_map[(insp_masked > -950) & (exp_masked <= -856)] = 2  # 功能性小气道病变（黄色）
prm_map[insp_masked <= -950] = 3  # 肺气肿（红色）

# 定义颜色映射
colors = {
    0: [0, 0, 0],       # 背景（黑色）
    1: [0, 255, 0],     # 正常肺组织（绿色）
    2: [255, 255, 0],   # 功能性小气道病变（黄色）
    3: [255, 0, 0]      # 肺气肿（红色）
}

# 生成彩色PRM图
prm_color_map = np.zeros((*prm_map.shape, 3), dtype=np.uint8)
for value, color in colors.items():
    prm_color_map[prm_map == value] = color

# 可视化PRM分类结果的某一切片（例如第 97 张切片）
slice_index = 97  # 选择需要查看的切片索引
plt.imshow(prm_color_map[slice_index], cmap='gray')
plt.title(f"Parametric Response Map (PRM) Slice {slice_index}")
plt.axis('off')
plt.show()

# 保存PRM分类结果为 NIfTI 文件
prm_image = sitk.GetImageFromArray(prm_map)
sitk.WriteImage(prm_image, '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_PRM_result.nii')
print("PRM result saved as PRM_result.nii")

# 创建保存切片的文件夹
#output_dir = '/Users/wjl/Desktop/Graduation Project/code/workflow/5-PRM analyze/P003_PRM_result'
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

# 遍历并保存每个切片
num_slices = prm_map.shape[0]
#for i in range(num_slices):
    # 获取当前切片的彩色PRM图
   # current_slice = prm_color_map[i]

    # 将当前切片转换为PIL图像
  #  slice_image = Image.fromarray(current_slice)

    # 保存为PNG文件
  #  slice_save_path = os.path.join(output_dir, f'PRM_P003_{i + 1}.png')
  #  slice_image.save(slice_save_path)

  #  print(f"Saved slice {i + 1} as {slice_save_path}")

#print("All PRM slices have been saved successfully.")


# 计算每种分类的像素数量
total_pixels = np.sum(mask_array == 1)  # 总像素数（掩膜区域）
normal_pixels = np.sum(prm_map == 1)    # 正常肺组织像素数
fsad_pixels = np.sum(prm_map == 2)      # 功能性小气道病变像素数
emphysema_pixels = np.sum(prm_map == 3) # 肺气肿像素数

# 计算每种分类所占的比例
normal_ratio = normal_pixels / total_pixels * 100
fsad_ratio = fsad_pixels / total_pixels * 100
emphysema_ratio = emphysema_pixels / total_pixels * 100

# 输出结果
print(f"Normal lung tissue (green): {normal_ratio:.2f}%")
print(f"Functional small airway disease (yellow): {fsad_ratio:.2f}%")
print(f"Emphysema (red): {emphysema_ratio:.2f}%")

import matplotlib.pyplot as plt

# 获取图像的尺寸
depth, height, width, _ = prm_color_map.shape

# 指定切面的索引（替换为您的值）
slice_index_sagittal = 200  # 矢状面（Sagittal）的指定位置
slice_index_coronal = 150   # 冠状面（Coronal）的指定位置
slice_index_axial = 100     # 横断面（Axial）的指定位置

# 可视化三个切面
plt.figure(figsize=(15, 5))

# 横断面（Axial）
plt.subplot(1, 3, 1)
plt.imshow(prm_color_map[slice_index_axial, :, :])  # 去掉 cmap='gray'
plt.title(f'Axial Slice {slice_index_axial}')
plt.axis('off')

# 冠状面（Coronal）
plt.subplot(1, 3, 2)
plt.imshow(prm_color_map[:, slice_index_coronal, :])  # 去掉 cmap='gray'
plt.title(f'Coronal Slice {slice_index_coronal}')
plt.axis('off')

# 矢状面（Sagittal）
plt.subplot(1, 3, 3)
plt.imshow(prm_color_map[:, :, slice_index_sagittal])  # 去掉 cmap='gray'
plt.title(f'Sagittal Slice {slice_index_sagittal}')
plt.axis('off')

plt.tight_layout()
plt.show()