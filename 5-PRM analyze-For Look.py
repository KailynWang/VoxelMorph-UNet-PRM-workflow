import SimpleITK as sitk

# 定义 NIfTI 文件路径
nii_files = {
    "File1": "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_512x512x224.nii",
    "File2": "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Registration_P004_VoxelMorph_160x192x224.nii",
    "File3": "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/Reference_P004_512x512x224.nii",
    "File4": "/Users/wjl/Desktop/Graduation Project/code/workflow/P004/P004_IN_160x192x224.nii"
}

# 遍历文件并打印信息
for name, path in nii_files.items():
    # 读取 NIfTI 文件
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)

    # 获取尺寸和灰度值范围
    shape = array.shape  # 形状 (Z, Y, X)
    min_val = array.min()  # 最小灰度值
    max_val = array.max()  # 最大灰度值

    print(f"{name} - 尺寸: {shape}, 灰度值范围: {min_val} ~ {max_val}")