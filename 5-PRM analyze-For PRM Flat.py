import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 读取呼气相和吸气相的NII文件
expiration_file = nib.load('/Users/wjl/Desktop/Graduation Project/code/310/recovered_Registration_512x512x224.nii')  # 呼气相
inspiration_file = nib.load('/Users/wjl/Desktop/Graduation Project/code/310/recovered_reference_512x512x224.nii')  # 吸气相
mask_file = nib.load('/Users/wjl/Desktop/Graduation Project/code/310/mask_512x512x224.nii')  # 肺部掩膜

# 提取体素值
expiration_data = expiration_file.get_fdata()
inspiration_data = inspiration_file.get_fdata()
mask_data = mask_file.get_fdata()

# 提取中间层切片
#slice_index = 97  # 中间层索引
#expiration_slice = expiration_data[:, :, slice_index]
#inspiration_slice = inspiration_data[:, :, slice_index]
#mask_slice = mask_data[:, :, slice_index]  # 获取对应切片的掩膜

# 仅选择掩膜区域（mask == 1）的体素
valid_mask = mask_data == 1
expiration_values = expiration_data[valid_mask].flatten()
inspiration_values = inspiration_data[valid_mask].flatten()

# 限制 HU 值范围
valid_indices = (expiration_values >= -1025) & (expiration_values <= -625) & \
                (inspiration_values >= -1025) & (inspiration_values <= -625)
expiration_clean = expiration_values[valid_indices]
inspiration_clean = inspiration_values[valid_indices]


# 创建图表
plt.figure(figsize=(8, 6))

# 设置横坐标和纵坐标范围
x_min, x_max = -1025, -625
y_min, y_max = -1025, -625

# 设置坐标轴范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


# 设置横坐标和纵坐标标题
plt.xlabel('Expiration (HU)', fontsize=12)
plt.ylabel('Inspiration (HU)', fontsize=12)

# 设置横坐标和纵坐标刻度
plt.xticks(np.arange(x_min, x_max + 50, 50), fontsize=10)
plt.yticks(np.arange(y_min, y_max + 50, 50), fontsize=10)


# 填充背景颜色
# 红色区域：横坐标 -1025 到 -856，纵坐标 -1025 到 -950
plt.fill_between([-1025, -856], -1025, -950, color='red', alpha=1)

# 白色区域：横坐标 -856 到 -650，纵坐标 -1025 到 -950
plt.fill_between([-856, -625], -1025, -950, color='white', alpha=1)

# 黄色区域：横坐标 -1025 到 -856，纵坐标 -950 到 -650
plt.fill_between([-1025, -856], -950, -625, color='yellow', alpha=1)

# 绿色区域：横坐标 -856 到 -650，纵坐标 -950 到 -650
plt.fill_between([-856, -625], -950, -625, color='green', alpha=1)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 绘制筛选后的数据点
plt.scatter(expiration_clean, inspiration_clean, color='blue', s=0.3, alpha=0.6, label='Lung Points')

# 绘制 2D KDE 图
#sns.kdeplot(
    #x=expiration_clean,
    #y=inspiration_clean,
    #cmap="turbo",  # 颜色渐变
    #fill=False,  # 填充颜色区域
    #levels=100,  # 颜色渐变的细腻程度
    #bw_adjust=0.7,  # 平滑程度
    #alpha=0.6,  # 透明度
#)

# 添加图例
plt.legend(loc='upper right', fontsize=10)


# 显示图表
plt.show()

print("Expiration min/max:", np.min(expiration_values), np.max(expiration_values))
print("Inspiration min/max:", np.min(inspiration_values), np.max(inspiration_values))
print("Expiration_clean min/max:", np.min(expiration_clean), np.max(expiration_clean))
print("Inspiration_clean min/max:", np.min(inspiration_clean), np.max(inspiration_clean))
print("NaN count:", np.isnan(expiration_values).sum(), np.isnan(inspiration_values).sum())
print("NaN count_clean:", np.isnan(expiration_clean).sum(), np.isnan(inspiration_clean).sum())