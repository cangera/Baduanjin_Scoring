from ultralytics import YOLO

# model = YOLO(r"E:\AIGC\ComfyUI-aki-v1.4\runs\pose\train88\weights\best.pt")
# m = model.val(data=r"D:\KK\ultralytics-main\datasets\Baduanjin_data\Baduanjin_data.yaml")


model = YOLO(r"D:\KK\ultralytics-main\ultralytics\cfg\models\v8\yolov8_pose_MSGAConv.yaml")