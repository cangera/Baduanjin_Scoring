#This pt page is an initial version and has not been optimized yet. The operation speed might be relatively slow. If you want to test it, you can go to use.py

import torch
from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
import math
import numpy as np
from matplotlib import pyplot as plt


from prepare.DTW_O_use import dtw_o_distance


os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO 

class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        self.h = 0
        self.keypoints_array = []
        self.list = []
        self.count = 0
        # Set interface
        self.setupUI()

        self.camBtn.clicked.connect(self.startCamera)
        self.videoBtn.clicked.connect(self.startVideoFile)
        self.stopBtn.clicked.connect(self.stop)
        self.similarity.clicked.connect(self.sim)
        self.score.clicked.connect(self.jisuan)
        self.biaoji = []

       
        self.timer_camera = QtCore.QTimer()
       
        self.timer_camera.timeout.connect(self.show_camera)

        # Automatically select the device (GPU or CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        self.model = YOLO(r'D:\KK\ultralytics-main_first\yolov8l-pose.pt').to(device)  
        # self.model = YOLO(r"E:\AIGC\ComfyUI-aki-v1.4\runs\pose\train88\weights\best.pt").to(device)
        # The video frame image queue to be processed currently holds only one frame of image
        self.frameToAnalyze = []

        # Start the independent thread for processing video frames
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()


        # Define a timer to control the frame rate of the displayed video file
        self.timer_videoFile = QtCore.QTimer()
        
        self.timer_videoFile.timeout.connect(self.show_videoFile)

        # The frame number of the video to be played currently
        self.vframeIdx = 0

        # cv2.VideoCapture 
        self.cap = None

        self.stopFlag = False

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('Baduanjin scoring System')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # The main layout in the central Widget
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # The upper part of the interface: the graphic display section
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setFixedSize(520,400)
        self.label_treated.setFixedSize(520,400)
        # self.label_ori_video.setMinimumSize(520,400)
        # self.label_treated.setMinimumSize(520,400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # The lower half of the interface: output boxes and buttons
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout =  QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()

        bottomLayout.addWidget(self.textLog)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️Video file')
        self.camBtn   = QtWidgets.QPushButton('📹Camera')
        self.stopBtn  = QtWidgets.QPushButton('🛑stop')
        self.score = QtWidgets.QPushButton('Calculate the score')
        self.similarity = QtWidgets.QPushButton('Keyframe similarity display')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.score)
        btnLayout.addWidget(self.similarity)
        bottomLayout.addLayout(btnLayout)


    def startCamera(self):

       
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Camera No. 1 cannot be turned on")
            return

        if self.timer_camera.isActive() == False:  # If the timer is not started
            self.timer_camera.start(30)
            self.stopFlag = False


    def show_camera(self):

        ret, frame = self.cap.read()  # Read from the video stream
        if not ret:
            return


        # Reset the size of the read frame
        frame = cv2.resize(frame, (520, 400))

        self.setFrameToOriLabel(frame)

    def setFrameToOriLabel(self,frame):

        # The video color is converted back to RGB, OpenCV images as BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                 QtGui.QImage.Format_RGB888)  
        
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage)) 

        
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):

        while True:
            if not self.frameToAnalyze:
                # time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)

            results = self.model(frame)[0]

            """Process data=============================================================="""
            if results:
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    x_min = boxes.xyxy[0][0]
                    y_min = boxes.xyxy[0][1]
                    masks = result.masks  # Masks object for segmentation masks outputs
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    if self.h == 0:
                        x1 = ((keypoints.xy[0][5] + keypoints.xy[0][6]) / 2)[0]
                        y1 = ((keypoints.xy[0][5] + keypoints.xy[0][6]) / 2)[1]
                        x2 = ((keypoints.xy[0][11] + keypoints.xy[0][12]) / 2)[0]
                        y2 = ((keypoints.xy[0][11] + keypoints.xy[0][12]) / 2)[1]
                        h = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))

                    frame_keypoints = np.zeros(34)

                    # Normalize keypoints
                    for i, k in enumerate(keypoints.xy[0]):  # Enumerate over the 17 keypoints
                        if k[0] != 0:  # Avoid processing invalid keypoints
                            # Normalize by subtracting the minimum box value and dividing by height
                            normalized_x = (k[0] - x_min) / h
                            normalized_y = (k[1] - y_min) / h
                            frame_keypoints[2 * i] = normalized_x  # Store normalized x
                            frame_keypoints[2 * i + 1] = normalized_y  # Store normalized y

                        # Add the frame's keypoints array to the keypoints list
                    self.keypoints_array.append(frame_keypoints)
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                # result.show()  # display to screen

                # result.save(filename="out_bdj_3.jpg")  # save to disk



            """Process data========================================================================"""

            img = results.plot(line_width=1)    

            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                    QtGui.QImage.Format_RGB888)  

            if self.stopFlag == False:
                self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))  

            time.sleep(0.5)



    def stop(self, ):
        self.stopFlag = True     
        self.timer_camera.stop() 
        self.timer_videoFile.stop()  

        if self.cap:
            self.cap.release() 

        
        self.label_ori_video.clear()        
        self.label_treated.clear()

        self.h = 0
        self.keypoints_array = []
        
        


    def startVideoFile(self):

        # Close the one that was originally turned on first
        self.stop()
        self.textLog.clear()
        videoPath, _  = QtWidgets.QFileDialog.getOpenFileName(
            self,             
            "Select the video file",       
            ".",               
            "Picture type (*.mp4 *.avi)" 
        )

        print('videoPath is', videoPath)
        if not videoPath:
            return


        self.cap = cv2.VideoCapture(videoPath)
        if not self.cap.isOpened():
            print("Failed to open the file")
            return


        self.timer_videoFile.start(30)
        self.stopFlag = False

        print("ok")


    def show_videoFile(self):
        # Select the position of the video frame，
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.vframeIdx)  
        self.vframeIdx += 1
        ret, frame = self.cap.read()  

        # The reading failed. It should be that the video playback has ended
        if not ret:
            self.keypoints_array = np.array(self.keypoints_array)
            # print(keypoints_array)
            if self.keypoints_array.size != 0:
                self.biaoji = self.keypoints_array
                self.textLog.setText(str('finall'))
            self.keypoints_array = []


            # self.stop()
            return
       
        frame = cv2.resize(frame, (520, 300))
        self.setFrameToOriLabel(frame)

    def sim(self, ):
        count = self.count
        
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False 

        # Draw graphics
        plt.title("Single-frame similarity comparison result")
        plt.xlabel("Limb vector segment")
        plt.ylabel("score")
        list = self.list[:-1]
        y = list / count
        self.count = 0
        self.list = []
      
        x = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16']

        # Draw the bar chart
        plt.bar(x, y)

        
        output_directory = r'D:\KK\bdjdatastes' 
        if not os.path.exists(output_directory):
            os.makedirs(output_directory) 

        
        output_file = os.path.join(output_directory, 'similarity_comparison.png')  
        plt.savefig(output_file)

        

       
        plt.show()

        return 0
    def jisuan(self,):
        if self.biaoji.size !=0:
            fin_score, list, count = dtw_o_distance(self.biaoji)
            print(fin_score)
            self.textLog.setText(str(fin_score))
            self.count = count
            self.list = list
            self.biaoji = []
        return 0
app = QtWidgets.QApplication()
window = MWindow()
window.show()
app.exec()
