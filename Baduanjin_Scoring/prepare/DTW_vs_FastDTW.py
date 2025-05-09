import time


from fastdtw import fastdtw


import pandas as pd

from dtw import dtw
from scipy.spatial.distance import euclidean

def FastDtw():
        # Read two Excel files
        file_path_a = r"D:\KK\bdjdatastes\template_keypoints.xlsx"  # Replace it with your first file path
        file_path_b = r"D:\KK\bdjdatastes\duanjin8_12person\1.xlsx"  # Replace it with your second file path

        data_a = pd.read_excel(file_path_a)
        data_b = pd.read_excel(file_path_b)

        # Convert the data into a NumPy array
        array_a = data_a.values  # shape: (n_samples, n_features)
        array_b = data_b.values  # shape: (m_samples, n_features)
        # Record the start time
        start_time = time.time()
        distance, path = fastdtw(array_a, array_b, dist=euclidean)
        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("fastdtwThe overall Euclidean distance：", distance)
        print(f"fastDTWAlgorithm time consumption {elapsed_time:.4f} s")
        return distance



def Dtw():
    
    file_path_a = r"D:\KK\bdjdatastes\template_keypoints.xlsx"  
    file_path_b = r"D:\KK\bdjdatastes\duanjin8_12person\1.xlsx"  

    data_a = pd.read_excel(file_path_a)
    data_b = pd.read_excel(file_path_b)

 
    array_a = data_a.values
    array_b = data_b.values

   
    start_time = time.time()

    
    distance, _, _, _ = dtw(array_a, array_b, dist=euclidean)

   
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("DTWThe overall Euclidean distance:", distance)
    print(f"The time consumption of the DTW algorithm：{elapsed_time:.4f} 秒")
    return distance

dis1=Dtw()
dis2=FastDtw()
print(f"Error rate{100*(dis2-dis1)/dis1:.2f} %")
