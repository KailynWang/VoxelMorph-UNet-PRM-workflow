import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 加载训练好的模型
model = load_model('/Users/wjl/Desktop/Graduation Project/code/U-Net 1.0/model.test/unet_best_model.h5')

# 输入文件夹和输出文件夹
input_dir = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/IN-Slice'
output_dir = '/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Mask'
os.makedirs(output_dir, exist_ok=True)

# 起止范围（文件名编号）
start_idx = 0  # 起始编号
end_idx = 263    # 结束编号


# 遍历文件夹中的图像
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # 提取文件名中的编号（假设文件名为 "Registration_P018_132.png"）
        try:
            idx = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            continue  # 跳过文件名格式不匹配的文件

        # 判断是否在起止范围内
        if start_idx <= idx <= end_idx:
            # 加载并处理图像
            img_path = os.path.join(input_dir, filename)
            img = image.load_img(img_path, target_size=(512, 512), color_mode="grayscale")
            img_array = image.img_to_array(img)  # 转换为NumPy数组
            img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
            img_array = img_array / 255.0  # 归一化

            # 使用训练好的模型进行预测
            predicted_mask = model.predict(img_array)

            # 将预测结果转换为二值图像（阈值=0.5）
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

            # 保存预测结果
            output_img = Image.fromarray(predicted_mask[0, :, :, 0] * 255)  # 转换为0-255范围的图像
            output_path = os.path.join(output_dir, filename)
            output_img.save(output_path)

            print(f"处理并保存: {filename}")