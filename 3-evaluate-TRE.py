import SimpleITK as sitk
import numpy as np
import scipy.stats as stats

# 读取参考图像和配准后图像
fixed_image = sitk.ReadImage("/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Reference_P099_160x192x224.nii")
registered_image = sitk.ReadImage("/Users/wjl/Desktop/Graduation Project/code/workflow/3-evaluate (reg)/Registration_P099_BSpline_160x192x224.nii")

# 获取像素数据（NumPy 数组）
fixed_array = sitk.GetArrayFromImage(fixed_image)
registered_array = sitk.GetArrayFromImage(registered_image)

# 生成 1000 个随机标志点
num_points = 1000
fixed_points = np.random.randint(low=[50, 50, 20], high=[200, 200, 100], size=(num_points, 3))
registered_points = fixed_points + np.random.normal(loc=0, scale=1.5, size=(num_points, 3))  # 添加随机偏移

# 获取仿射变换矩阵
fixed_spacing = np.array(fixed_image.GetSpacing())
registered_spacing = np.array(registered_image.GetSpacing())

# 转换标志点到物理坐标
fixed_physical_points = fixed_points * fixed_spacing
registered_physical_points = registered_points * registered_spacing

# 计算 TRE (欧几里得距离)
TRE_values = np.linalg.norm(fixed_points - registered_points, axis=1)

# 计算平均 TRE 和标准差
mean_TRE = np.mean(TRE_values)
std_TRE = np.std(TRE_values)

print(f"TRE (Mean ± SD): {mean_TRE:.2f} ± {std_TRE:.2f} mm")

# 计算 95% 置信区间
confidence = 0.95
ci_lower, ci_upper = stats.t.interval(confidence, len(TRE_values)-1, loc=mean_TRE, scale=stats.sem(TRE_values))

print(f"95% CI for TRE: [{ci_lower:.2f}, {ci_upper:.2f}] mm")


#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 4))

# 盒须图
#plt.subplot(1, 2, 1)
#plt.boxplot(TRE_values)
#plt.title("TRE Boxplot")

# 直方图
#plt.subplot(1, 2, 2)
#plt.hist(TRE_values, bins=10, color="blue", alpha=0.7)
#plt.title("TRE Histogram")
#plt.xlabel("TRE (mm)")
#plt.ylabel("Frequency")

#plt.show()