from ultralytics import YOLO


#This file trains the model
#Explanation: the ultralytics/utils/metrics. Py has modified the Ciou Piou, so here are replaced with Piou. If you need the original Ciou need to ultralytics/utils/metrics. The change in py
#The running results are placed in runs/ Poses. The corresponding results train1-6 correspond respectively to YOLOv8-Pose, YOLOv8-pose-GhostConv, YOLOv8-pose-MSGRConv, YOLOv8-pose-Piou, and YOLOv8-pose-GhostConv-Piou YOLOv8-Pose-MSGRConv-PIoU



#This code is for training yolov8-pose-Piou
# model = YOLO(r"D:\KK\Baduanjin_Scoring\ultralytics\cfg\models\v8\yolov8-pose.yaml")
# model.train(data=r"D:\KK\Baduanjin_Scoring\datasets\Baduanjin_data\pose_baduanjin.yaml", epochs=120, batch=32)
# metrics = model.val()

# #This code is for training YOLOv8-Pose-GhostConv-Piou
# model = YOLO(r"D:\KK\Baduanjin_Scoring\ultralytics\cfg\models\v8\yolov8-ghost.yaml")
# model.train(data=r"D:\KK\Baduanjin_Scoring\datasets\Baduanjin_data\pose_baduanjin.yaml", epochs=120, batch=32)
# metrics = model.val()
#
# #This code is for training YOLOv8-Pose-MSGRConv-Piou
model = YOLO(r"D:\KK\Baduanjin_Scoring\ultralytics\cfg\models\v8\yolov8_pose_MSGRConv.yaml")
model.train(data=r"D:\KK\Baduanjin_Scoring\datasets\Baduanjin_data\pose_baduanjin.yaml", epochs=120, batch=32)
metrics = model.val()