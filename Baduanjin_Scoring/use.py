import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from prepare.DTW_O_use import dtw_o_distance

# Load a model
model = YOLO("yolov8l-pose.pt")  # pretrained YOLO11n model

# Run a video that needs to be detected
results = model(r"D:\KK\bdjdatastes\5.mp4", stream=True)  # return a generator of Results objects
h=0
keypoints_array = []
count = 0  #It is used to record the number of frames in which no human was detected
# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    if boxes.xyxy.size(0) > 0:  # Determine whether there is data
        x_min = boxes.xyxy[0][0]
        # Subsequent processing code
    else:
        # If there is no data, the current loop can be skipped or other processing can be done
        count +=1
        continue  # Skip the current loop
    y_min = boxes.xyxy[0][1]
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    # Normalize keypoints
    if h == 0:
        x1 = ((keypoints.xy[0][5]+keypoints.xy[0][6])/2)[0]
        y1 = ((keypoints.xy[0][5]+keypoints.xy[0][6])/2)[1]
        x2 = ((keypoints.xy[0][11]+keypoints.xy[0][12])/2)[0]
        y2 = ((keypoints.xy[0][11]+keypoints.xy[0][12])/2)[1]
        h = math.sqrt(math.pow((x1 - x2), 2)+math.pow((y1 - y2), 2))
    frame_keypoints = np.zeros(34)
    for i, k in enumerate(keypoints.xy[0]):  # Enumerate over the 17 keypoints
        if k[0] != 0:  # Avoid processing invalid keypoints
            # Normalize by subtracting the minimum box value and dividing by height
            normalized_x = (k[0] - x_min) / h
            normalized_y = (k[1] - y_min) / h
            frame_keypoints[2 * i] = normalized_x  # Store normalized x
            frame_keypoints[2 * i + 1] = normalized_y  # Store normalized y

        # Add the frame's keypoints array to the keypoints list
    keypoints_array.append(frame_keypoints)
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen

    # result.save(filename="out_bdj_3.jpg")  # save to disk

keypoints_array = np.array(keypoints_array)
# print(keypoints_array)


#Save the data run by yolo
# # List of key point names
# keypoint_names = [
#     "nose-x", "nose-y", "l-eye-x", "l-eye-y", "r-eye-x", "r-eye-y",
#     "l-ear-x", "l-ear-y", "r-ear-x", "r-ear-y", "l-shoulder-x", "l-shoulder-y",
#     "r-shoulder-x", "r-shoulder-y", "l-elbow-x", "l-elbow-y", "r-elbow-x",
#     "r-elbow-y", "l-wrist-x", "l-wrist-y", "r-wrist-x", "r-wrist-y",
#     "l-hip-x", "l-hip-y", "r-hip-x", "r-hip-y", "l-knee-x", "l-knee-y",
#     "r-knee-x", "r-knee-y", "l-ankle-x", "l-ankle-y", "r-ankle-x", "r-ankle-y"
# ]
# df = pd.DataFrame(keypoints_array, columns=keypoint_names)
#
# # Save as an Excel file
# df.to_excel(r"D:\KK\bdjdatastes\5.xlsx", index=False, engine='openpyxl')
#
# print("The key point data has been saved to the 'xlsx' file. There are {} frames where no people were detected".format(count))

dtw_o_distance(keypoints_array)  #Calculation score
