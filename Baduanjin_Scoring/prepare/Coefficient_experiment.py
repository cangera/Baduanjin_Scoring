import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def singles_sim(a):
    if len(a) != 34:
        raise ValueError("The length of the input array should be 34")

    a_neck_x = (a[10] + a[12])/2
    a_neck_y = (a[11] + a[13])/2
    a_hip_x = (a[22] + a[24])/2
    a_hip_y = (a[23] + a[25]) / 2

    v_list = [
        np.array([a_neck_x - a[0], a_neck_y - a[1]]),
        np.array([a_neck_x - a_hip_x, a_neck_y - a_hip_y]),
        np.array([a_neck_x - a[10], a_neck_y - a[11]]),
        np.array([a[10] - a[14], a[11] - a[15]]),
        np.array([a[14] - a[18], a[15] - a[19]]),
        np.array([a_neck_x - a[12], a_neck_y - a[13]]),
        np.array([a[12] - a[16], a[13] - a[17]]),
        np.array([a[16] - a[20], a[17] - a[21]]),
        np.array([a[10] - a[22], a[11] - a[23]]),
        np.array([a[12] - a[24], a[13] - a[25]]),
        np.array([a_hip_x - a[22], a_hip_y - a[23]]),
        np.array([a_hip_x - a[24], a_hip_y - a[25]]),
        np.array([a[22] - a[26], a[23] - a[27]]),
        np.array([a[26] - a[30], a[27] - a[31]]),
        np.array([a[24] - a[28], a[25] - a[29]]),
        np.array([a[28] - a[32], a[29] - a[33]])
    ]

    return v_list

def compute_motion_distribution(array_a):
    n_frames = array_a.shape[0]
    n_segments = 16
    segment_motion_sums = np.zeros(n_segments)
    prev_vectors = None

    for i in range(n_frames):
        frame = array_a[i]
        cur_vectors = singles_sim(frame)

        if prev_vectors is not None:
            for j in range(n_segments):
                diff = cur_vectors[j] - prev_vectors[j]
                motion = np.linalg.norm(diff)
                segment_motion_sums[j] += motion

        prev_vectors = cur_vectors

    total_motion = np.sum(segment_motion_sums)
    motion_distribution = segment_motion_sums / total_motion

    # Output the result of the amount of exercise
    print("\nThe amount of motion and proportion of each vector segment:")
    for i in range(n_segments):
        print(f"a_v{i+1}:Total exercise volume= {segment_motion_sums[i]:.4f}, Proportion = {motion_distribution[i]*100:.2f}%")

    print(f"\nTotal exercise volume: {total_motion:.4f}")

    return segment_motion_sums, total_motion, motion_distribution

def plot_motion_distribution(motion_distribution):
    labels = [f'a_v{i+1}' for i in range(len(motion_distribution))]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, motion_distribution * 100, color='skyblue')
    plt.ylabel("Percentage of total displacement (%)")
    plt.title("The displacement proportion of each vector segment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path_a = r"D:\KK\bdjdatastes\template_keypoints.xlsx"
    data_a = pd.read_excel(file_path_a)
    array_a = data_a.values

    segment_motion_sums, total_motion, motion_distribution = compute_motion_distribution(array_a)

    # Visualization
    plot_motion_distribution(motion_distribution)
