#这个pt页面是一个初代版本，还未优化，运算速度可能会比较慢，要测试的话可以去use.py中

import torch
from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
import math
import numpy as np
from matplotlib import pyplot as plt


from prepare.DTW_O_use import dtw_o_distance

# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO 

class MWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        self.h = 0
        self.keypoints_array = []
        self.list = []
        self.count = 0
        # 设置界面
        self.setupUI()

        self.camBtn.clicked.connect(self.startCamera)
        self.videoBtn.clicked.connect(self.startVideoFile)
        self.stopBtn.clicked.connect(self.stop)
        self.similarity.clicked.connect(self.sim)
        self.score.clicked.connect(self.jisuan)
        self.biaoji = []

        # 定义定时器，用于控制显示摄像头视频的帧率
        self.timer_camera = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        self.timer_camera.timeout.connect(self.show_camera)

        # 自动选择设备（GPU 或 CPU）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU
        self.model = YOLO(r'D:\KK\ultralytics-main_first\yolov8l-pose.pt').to(device)  # 加载模型到相应的设备
        # self.model = YOLO(r"E:\AIGC\ComfyUI-aki-v1.4\runs\pose\train88\weights\best.pt").to(device)
        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc,daemon=True).start()


        # 定义定时器，用于控制显示视频文件的帧率
        self.timer_videoFile = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        self.timer_videoFile.timeout.connect(self.show_videoFile)

        # 当前要播放的视频帧号
        self.vframeIdx = 0

        # cv2.VideoCapture 实例
        self.cap = None

        self.stopFlag = False

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('八段锦评分系统')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
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

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout =  QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()

        bottomLayout.addWidget(self.textLog)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn   = QtWidgets.QPushButton('📹摄像头')
        self.stopBtn  = QtWidgets.QPushButton('🛑停止')
        self.score = QtWidgets.QPushButton('计算得分')
        self.similarity = QtWidgets.QPushButton('关键帧相似度显示')
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.score)
        btnLayout.addWidget(self.similarity)
        bottomLayout.addLayout(btnLayout)


    def startCamera(self):

        # 参考 https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多， 
        # 在 Linux/Mac上 指定 V4L, FFMPEG 或者 GSTREAMER
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("1号摄像头不能打开")
            return

        if self.timer_camera.isActive() == False:  # 若定时器未启动
            self.timer_camera.start(30)
            self.stopFlag = False


    def show_camera(self):

        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return


        # 把读到的帧的大小重新设置 
        frame = cv2.resize(frame, (520, 400))

        self.setFrameToOriLabel(frame)

    def setFrameToOriLabel(self,frame):

        # 视频色彩转换回RGB，OpenCV images as BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 变成QImage形式
        # 往显示视频的Label里 显示QImage
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage)) 

        # 如果当前没有处理任务
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):

        while True:
            if not self.frameToAnalyze:
                # time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)

            results = self.model(frame)[0]

            """处理数据=============================================================="""
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



            """处理数据========================================================================"""

            img = results.plot(line_width=1)    

            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 变成QImage形式

            if self.stopFlag == False:
                self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))  # 往显示Label里 显示QImage

            time.sleep(0.5)



    def stop(self, ):
        self.stopFlag = True      # 让 frameAnalyzeThreadFunc 不要再设置 label_treated
        self.timer_camera.stop()  # 关闭定时器
        self.timer_videoFile.stop()  # 关闭定时器

        if self.cap:
            self.cap.release()  # 释放视频流

        # 清空视频显示区域 
        self.label_ori_video.clear()        
        self.label_treated.clear()

        self.h = 0
        self.keypoints_array = []
        # # 延时500ms清除，有的定时器处理任务可能会在当前时间点后处理完最后一帧
        # QtCore.QTimer.singleShot(500, clearLabels)


    def startVideoFile(self):

        # 先关闭原来打开的
        self.stop()
        self.textLog.clear()
        videoPath, _  = QtWidgets.QFileDialog.getOpenFileName(
            self,             # 父窗口对象
            "选择视频文件",        # 标题
            ".",               # 起始目录
            "图片类型 (*.mp4 *.avi)" # 选择类型过滤项，过滤内容在括号中
        )

        print('videoPath is', videoPath)
        if not videoPath:
            return


        self.cap = cv2.VideoCapture(videoPath)
        if not self.cap.isOpened():
            print("打开文件失败")
            return


        self.timer_videoFile.start(30)
        self.stopFlag = False

        print("ok")


    def show_videoFile(self):
        # 选取视频帧位置，
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.vframeIdx)  
        self.vframeIdx += 1
        ret, frame = self.cap.read()  # 从视频流中读取

        # 读取失败，应该是视频播放结束了
        if not ret:
            self.keypoints_array = np.array(self.keypoints_array)
            # print(keypoints_array)
            if self.keypoints_array.size != 0:
                self.biaoji = self.keypoints_array
                self.textLog.setText(str('finall'))
            self.keypoints_array = []


            # self.stop()
            return
        # 把读到的帧的大小重新设置
        frame = cv2.resize(frame, (520, 300))
        self.setFrameToOriLabel(frame)

    def sim(self, ):
        count = self.count
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 绘制图形
        plt.title("单帧相似度对比结果")
        plt.xlabel("肢体向量段")
        plt.ylabel("分数")
        list = self.list[:-1]
        y = list / count
        self.count = 0
        self.list = []
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