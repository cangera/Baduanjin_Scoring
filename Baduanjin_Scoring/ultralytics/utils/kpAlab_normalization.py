import math

import torch


h = 0

def kp_normalization(data):
    # 将 tensor 数据从 GPU 移动到 CPU
    data_cpu = data.cpu()

    # 提取 x 和 y
    x = data_cpu[0, 0].item()  # 第一个数
    y = data_cpu[0, 1].item()  # 第二个数


    # 处理从第7个数开始的关键点
    keypoints = data_cpu[0, 6:].reshape(-1, 3)  # 每3个数为一组
    modified_keypoints = []
    global h
    if h == 0:
        vx1 = (keypoints[5][0] + keypoints[6][0]) / 2
        vy1 = (keypoints[5][1] + keypoints[6][1]) / 2
        vx2 = (keypoints[11][0] + keypoints[12][0]) / 2
        vy2 = (keypoints[11][1] + keypoints[12][1]) / 2
        h = math.sqrt(math.pow((vx1 - vx2), 2)+math.pow((vy1 - vy2), 2))


    for kp in keypoints:
        modified_kp = [(kp[0] - x) / h, (kp[1] - y) / h]
        modified_keypoints.append(modified_kp)

    # 将结果写入 txt 文件
    with open(r'D:\KK\bdj_datasets\kp_normalization.txt_1', 'a') as f:
        for kp in modified_keypoints:
            f.write(f"{kp[0]:.6f} {kp[1]:.6f} ")  # 保持在同一行，空格分隔
        f.write('\n')  # 写入换行符

    print("修改后的关键点已写入 kp_normalization.txt 文件。")
    print(h)
def label_normalization(data):
    # 将 tensor 数据从 GPU 移动到 CPU
    data_cpu = data.cpu()
    label_value = data_cpu[0, 5].item()
    with open(r'D:\KK\bdj_datasets\labels.txt_1', 'a') as f:
        f.write(f"{label_value} ")  # 追加空格以分隔数值

    print("标签已保存。")