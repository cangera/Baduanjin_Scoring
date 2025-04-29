import time


from fastdtw import fastdtw


import pandas as pd

from dtw import dtw
from scipy.spatial.distance import euclidean

def FastDtw():
        # 读取两个 Excel 文件
        file_path_a = r"D:\KK\bdjdatastes\template_keypoints.xlsx"  # 替换为你的第一个文件路径
        file_path_b = r"D:\KK\bdjdatastes\duanjin8_12person\1.xlsx"  # 替换为你的第二个文件路径

        data_a = pd.read_excel(file_path_a)
        data_b = pd.read_excel(file_path_b)

        # 将数据转换为 NumPy 数组
        array_a = data_a.values  # shape: (n_samples, n_features)
        array_b = data_b.values  # shape: (m_samples, n_features)
        # 记录开始时间
        start_time = time.time()
        distance, path = fastdtw(array_a, array_b, dist=euclidean)
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("fastdtw整体的欧几里得距离:", distance)
        print(f"fastDTW算法耗时: {elapsed_time:.4f} 秒")
        return distance



def Dtw():
    # 读取两个 Excel 文件
    file_path_a = r"D:\KK\bdjdatastes\template_keypoints.xlsx"  # 替换为你的第一个文件路径
    file_path_b = r"D:\KK\bdjdatastes\duanjin8_12person\1.xlsx"  # 替换为你的第二个文件路径

    data_a = pd.read_excel(file_path_a)
    data_b = pd.read_excel(file_path_b)

    # 转换为 NumPy 数组
    array_a = data_a.values
    array_b = data_b.values

    # 记录开始时间
    start_time = time.time()

    # 使用标准 DTW 计算距离
    distance, _, _, _ = dtw(array_a, array_b, dist=euclidean)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("DTW整体的欧几里得距离:", distance)
    print(f"DTW算法耗时: {elapsed_time:.4f} 秒")
    return distance

dis1=Dtw()
dis2=FastDtw()
print(f"误差{100*(dis2-dis1)/dis1:.2f} %")