import cv2
import numpy as np

# 读取前后两张 PNG 图像
prev_img = cv2.imread("/Users/wjl/Desktop/Graduation Project/code/313/4-segmentation/Mask/ In_P099_127.png", cv2.IMREAD_GRAYSCALE)  # 替换为前一张的文件名
next_img = cv2.imread("/Users/wjl/Desktop/Graduation Project/code/313/4-segmentation/Mask/ In_P099_132.png", cv2.IMREAD_GRAYSCALE)      # 替换为后一张的文件名

# 确保两张图像尺寸一致
if prev_img.shape != next_img.shape:
    raise ValueError("前后两张图片尺寸不一致，无法计算平均值！")

# 计算平均值（确保数据类型为 uint8）
avg_img = ((prev_img.astype(np.float32) + next_img.astype(np.float32)) / 2).astype(np.uint8)
# 二值化处理（设定阈值为 128，可根据需求调整）
_, binary_img = cv2.threshold(avg_img, 128, 255, cv2.THRESH_BINARY)


# 保存替换后的图像
cv2.imwrite("/Users/wjl/Desktop/Graduation Project/code/313/4-segmentation/Mask/ In_P099_128_new.png", binary_img)

print("已成功生成替换图像，保存为 replaced.png")