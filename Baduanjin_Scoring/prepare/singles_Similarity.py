import os

import numpy as np
from matplotlib import pyplot as plt


def Cosine_Value(a, b):
    dot_product = np.dot(a, b)  # Vector dot product
    norm_a = np.linalg.norm(a)  # Calculate the L2 norm of array_a
    norm_b = np.linalg.norm(b)  # Calculate the L2 norm of array_b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine_value = dot_product / (norm_a * norm_b)
    return cosine_value
def singles_sim(a, b):
    # Check whether the array length is 34

    if len(a) != 34 or len(b) != 34:
        raise ValueError("The length of the input array should be 34")

    #x corresponds to the subscript ×2, and y is ×2+1
    a_neck_x = (a[10] + a[12])/2
    a_neck_y = (a[11] + a[13])/2
    b_neck_x = (b[10] + b[12]) / 2
    b_neck_y = (b[11] + b[13]) / 2

    a_hip_x = (a[22] + a[24])/2
    a_hip_y = (a[23] + a[25]) / 2
    b_hip_x = (b[22] + b[24]) / 2
    b_hip_y = (b[23] + b[25]) / 2

    a_v1 = np.array([a_neck_x - a[0], a_neck_y - a[1]])
    b_v1 = np.array([b_neck_x - b[0], b_neck_y - b[1]])
    a_v2 = np.array([a_neck_x - a_hip_x, a_neck_y - a_hip_y])
    b_v2 = np.array([b_neck_x - b_hip_x, b_neck_y - b_hip_y])
    a_v3 = np.array([a_neck_x - a[10], a_neck_y - a[11]])
    b_v3 = np.array([b_neck_x - b[10], b_neck_y - b[11]])
    a_v4 = np.array([a[10] - a[14], a[11] - a[15]])
    b_v4 = np.array([b[10] - b[14], b[11] - b[15]])
    a_v5 = np.array([a[14] - a[18], a[15] - a[19]])
    b_v5 = np.array([b[14] - b[18], b[15] - b[19]])
    a_v6 = np.array([a_neck_x - a[12], a_neck_y - a[13]])
    b_v6 = np.array([b_neck_x - b[12], b_neck_y - b[13]])
    a_v7 = np.array([a[12] - a[16], a[13] - a[17]])
    b_v7 = np.array([b[12] - b[16], b[13] - b[17]])
    a_v8 = np.array([a[16] - a[20], a[17] - a[21]])
    b_v8 = np.array([b[16] - b[20], b[17] - b[21]])
    a_v9 = np.array([a[10] - a[22], a[11] - a[23]])
    b_v9 = np.array([b[10] - b[22], b[11] - b[23]])
    a_v10 = np.array([a[12] - a[24], a[13] - a[25]])
    b_v10 = np.array([b[12] - b[24], b[13] - b[25]])
    a_v11 = np.array([a_hip_x - a[22], a_hip_y - a[23]])
    b_v11 = np.array([b_hip_x - b[22], b_hip_y - b[23]])
    a_v12 = np.array([a_hip_x - a[24], a_hip_y - a[25]])
    b_v12 = np.array([b_hip_x - b[24], b_hip_y - b[25]])
    a_v13 = np.array([a[22] - a[26], a[23] - a[27]])
    b_v13 = np.array([b[22] - b[26], b[23] - b[27]])
    a_v14 = np.array([a[26] - a[30], a[27] - a[31]])
    b_v14 = np.array([b[26] - b[30], b[27] - b[31]])
    a_v15 = np.array([a[24] - a[28], a[25] - a[29]])
    b_v15 = np.array([b[24] - b[28], b[25] - b[29]])
    a_v16 = np.array([a[28] - a[32], a[29] - a[33]])
    b_v16 = np.array([b[28] - b[32], b[29] - b[33]])


    # Calculate the cosine similarity of the vector

    s_v1 = Cosine_Value(a_v1, b_v1)
    s_v2 = Cosine_Value(a_v2, b_v2)
    s_v3 = Cosine_Value(a_v3, b_v3)
    s_v4 = Cosine_Value(a_v4, b_v4)
    s_v5 = Cosine_Value(a_v5, b_v5)
    s_v6 = Cosine_Value(a_v6, b_v6)
    s_v7 = Cosine_Value(a_v7, b_v7)
    s_v8 = Cosine_Value(a_v8, b_v8)
    s_v9 = Cosine_Value(a_v9, b_v9)
    s_v10 = Cosine_Value(a_v10, b_v10)
    s_v11 = Cosine_Value(a_v11, b_v11)
    s_v12 = Cosine_Value(a_v12, b_v12)
    s_v13 = Cosine_Value(a_v13, b_v13)
    s_v14 = Cosine_Value(a_v14, b_v14)
    s_v15 = Cosine_Value(a_v15, b_v15)
    s_v16 = Cosine_Value(a_v16, b_v16)

    #The coefficient is obtained through experiments by Coefficient_experiment.py
    Avg_s = s_v1 * 0.1135 + s_v2 * 0.0519 + s_v3 * 0.0256 + s_v4 * 0.0804 + s_v5 * 0.098 + s_v6 * 0.0256 + s_v7 * 0.0759 + s_v8 * 0.0911 + s_v9 * 0.0607 + s_v10 * 0.0594 + s_v11 * 0.0156 + s_v12 * 0.0156 + s_v13 * 0.0624 + s_v14 * 0.0802 + s_v15 * 0.0630 + s_v16 * 0.0726

    return Avg_s

def draw_pic(list, count = 1):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘制图形
    plt.title("单帧相似度对比结果")
    plt.xlabel("肢体向量段")
    plt.ylabel("分数")
    list = list[:-1]
    y = list/count
    # 示例数据
    x = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16']

    # 绘制柱状图
    plt.bar(x, y)

    # 给定目录
    output_directory = r'D:\KK\bdjdatastes'  # 替换为你希望保存图形的文件夹路径
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # 如果目录不存在则创建

    # 保存图像到指定目录
    output_file = os.path.join(output_directory, 'similarity_comparison.png')  # 设置文件名和路径
    plt.savefig(output_file)

    print(f"图像已保存到 {output_file}")

    # 显示图形
    plt.show()

    return 0