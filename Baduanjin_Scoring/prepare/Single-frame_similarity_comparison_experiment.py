import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import math
from ultralytics import YOLO
model = YOLO(r"D:\KK\ultralytics-main_first\yolov8l-pose.pt")  # pretrained YOLO11n model
img1_path = r"D:\KK\bdjdatastes\add_studies\Pose_structure_modeling\imgs\1.jpg"
img2_path = r"D:\KK\bdjdatastes\add_studies\Pose_structure_modeling\imgs\9526.jpg"

# Read the picture as a grayscale image
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check whether the picture has been loaded successfully
if img1 is None or img2 is None:
    raise ValueError("One of the pictures has an incorrect path or cannot be read. Please check if the path is correct!")

# Uniform size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

def normalization(result):
    keypoints = result.keypoints  # Keypoints object for pose outputs
    x1 = ((keypoints.xy[0][5] + keypoints.xy[0][6]) / 2)[0]
    y1 = ((keypoints.xy[0][5] + keypoints.xy[0][6]) / 2)[1]
    x2 = ((keypoints.xy[0][11] + keypoints.xy[0][12]) / 2)[0]
    y2 = ((keypoints.xy[0][11] + keypoints.xy[0][12]) / 2)[1]
    h = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    frame_keypoints = np.zeros(34)
    boxes = result.boxes
    x_min = boxes.xyxy[0][0]
    y_min = boxes.xyxy[0][1]
    for i, k in enumerate(keypoints.xy[0]):  # Enumerate over the 17 keypoints
        if k[0] != 0:  # Avoid processing invalid keypoints
            # Normalize by subtracting the minimum box value and dividing by height
            normalized_x = (k[0] - x_min) / h
            normalized_y = (k[1] - y_min) / h
            frame_keypoints[2 * i] = normalized_x  # Store normalized x
            frame_keypoints[2 * i + 1] = normalized_y  # Store normalized y
    return frame_keypoints
# 1. MSE
def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

# 2. PSNR
def psnr(imageA, imageB):
    mse_val = mse(imageA, imageB)
    if mse_val == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

# 3. SSIM
def calc_ssim(imageA, imageB):
    s, _ = ssim(imageA, imageB, full=True)
    return s

# 4. Cosine Similarity（Flatten the image into a vector）
def cosine_sim(imageA, imageB):
    vecA = imageA.flatten().reshape(1, -1)
    vecB = imageB.flatten().reshape(1, -1)
    return cosine_similarity(vecA, vecB)[0][0]

def cosine(img1, img2):
    # Flatten into a vector and normalize (to prevent the influence of pixel intensity)
    vec1 = img1.flatten().reshape(1, -1).astype(np.float32)
    vec2 = img2.flatten().reshape(1, -1).astype(np.float32)

    #Optional normalization
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    # Calculate the cosine similarity
    cos_sim = cosine_similarity(vec1, vec2)[0][0]
    return cos_sim
def Cosine_Value(a, b):
    dot_product = np.dot(a, b)  
    norm_a = np.linalg.norm(a)  
    norm_b = np.linalg.norm(b)  
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine_value = dot_product / (norm_a * norm_b)
    return cosine_value
def singles_sim(a, b):
    

    if len(a) != 34 or len(b) != 34:
        raise ValueError("The length of the input array should be 34")

    # x corresponds to the subscript ×2, and y is ×2+1
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
    
    # Calculate the cosine similarity

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


    Avg_s = s_v1*0.1135 + s_v2*0.0519 + s_v3*0.0256 + s_v4*0.0804 + s_v5*0.098 + s_v6*0.0256 + s_v7*0.0759 + s_v8*0.0911 + s_v9*0.0607 + s_v10*0.0594 +s_v11*0.0156 + s_v12*0.0156 + s_v13*0.0624 + s_v14*0.0802 + s_v15*0.0630 +s_v16*0.0726
    return Avg_s

re1 = model(img1_path)
re2 = model(img2_path)
re1_norm = normalization(re1[0])
re2_norm = normalization(re2[0])
# compare_custom = singles_sim(re1_norm, re2_norm)



print("🔍 Image similarity comparison index：")
print(f"✅ MSE: {mse(img1, img2):.4f}")
print(f"✅ PSNR: {psnr(img1, img2):.2f} dB")
print(f"✅ SSIM: {calc_ssim(img1, img2):.4f}")
print(f"✅ Cosine: {cosine(img1, img2):.4f}")
print(f"✅ MY Cosine Similarity: {singles_sim(re1_norm, re2_norm):.4f}")
# print(f"✅ Custom Model Score: {compare_custom(img1, img2):.4f}")
