import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from prepare.singles_Similarity import singles_sim


def dtw_o_distance(array_b):

        file_path_a = r'D:\KK\bdjdatastes\template_keypoints.xlsx'  # Replace it with the path of the template file
        data_a = pd.read_excel(file_path_a)
        array_a = data_a.values  # shape: (n_samples, n_features)

        # Calculate the Fastdtw between the two sample sets
        distance, path = fastdtw(array_a, array_b, dist=euclidean)

        #Calculate the keyframe similarity through   singles_sim()
        for i in range(0, len(path), 50):
                count = 1
                sum = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                a_idx, b_idx = path[i]  # Extract index
                s = np.array(singles_sim(array_a[a_idx], array_b[b_idx]))
                sum = sum + s


        d_max = 159022.30069531524  #The considered maximum distance
        d_min = 0
        Overall_similarity_score = 100 * (1 - ((distance - d_min) / (d_max - d_min)))
        cosin_score = 100 * (sum[16] / count)
        # draw_pic(sum, count)
        fin_score = 0.7 * Overall_similarity_score + 0.3 * cosin_score
        # singles_sim(array_a[50],array_b[50])
        print("The overall Euclidean distance:", distance)
        print("Overall score:", Overall_similarity_score)
        print("The score of the keyframe:", cosin_score)
        print("Fusion score：", fin_score)
        return fin_score, sum, count

if __name__ == '__main__':
        file_path_b = r'D:\KK\bdjdatastes\add_studies\60.xlsx'
        data_b = pd.read_excel(file_path_b)
        array_b = data_b.values  # shape: (m_samples, n_features)
        dtw_o_distance(array_b)
