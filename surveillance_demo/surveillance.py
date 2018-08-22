#! /usr/bin/python

"""Surveillance Demo: Tracking Pedestrians in Camera Feed

The application opens a video (could be a camera or a video file)
and tracks pedestrians in the video.
"""

import cv2
import numpy as np
import os.path as path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm",
                    help="m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())


def center(points):
    print(points)
    """计算矩阵的中心坐标"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


font = cv2.FONT_HERSHEY_SIMPLEX


class Pedestrian():
    """Pedestrian class

  each pedestrian is composed of a ROI, an ID and a Kalman filter
  so we create a Pedestrian class to hold the object state
  """

    def __init__(self, id, frame, track_window):
        # 初始化Roi范围
        self.id = int(id)
        x, y, w, h = track_window
        self.track_window = track_window
        # HSV:色调（H），饱和度（S），明度（V）
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        # 通过均值漂移算法找出Roi区域
        roi_hist = cv2.calcHist([self.roi], [0], None, [180], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 初始化卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print("Pedestrian %d destroyed" % self.id)

    def update(self, frame):
        # 计算当前帧的HSV直方图的反向投影
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # 使用CamShift计算出新的Roi区域
        # ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        # self.center = center(pts)
        # cv2.polylines(frame, [pts], True, 255, 1)

        # 使用均值漂移计算出新的Roi区域
        ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        self.center = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # 以Roi的中心值坐标作为参数训练卡尔曼滤波器，并预测出下一帧位置
        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
        # 通过输出两个字体，使字体带有阴影效果
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
                    font, 0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)
        # actual info
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
                    font, 0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA)


def main():
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "traffic.flv"))
    camera = cv2.VideoCapture(path.join(path.dirname(__file__), "768x576.avi"))
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "..", "movie.mpg"))
    # camera = cv2.VideoCapture(0)
    history = 20
    # 创建KNN背景分割器
    bs = cv2.createBackgroundSubtractorKNN()

    # MOG subtractor
    # bs = cv2.bgsegm.createBackgroundSubtractorMOG(history = history)
    # bs.setHistory(history)

    # GMG
    # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = history)

    cv2.namedWindow("surveillance")
    pedestrians = {}
    firstFrame = True
    frames = 0
    while True:
        print(" -------------------- FRAME %d --------------------" % frames)
        grabbed, frane = camera.read()
        if (grabbed is False):
            print("failed to grab frame.")
            break

        ret, frame = camera.read()
        fgmask = bs.apply(frame)

        # 将视频初始的一部分数据用于训练背景分割器
        if frames < history:
            frames += 1
            continue

        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        # 图片腐蚀（对图片反卷积，平滑图片，减少噪音干扰）
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # 图片膨胀（对图片卷积，提前特征）
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        # 找出图片中的轮廓
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        # 遍历每一个找到的轮廓
        for c in contours:
            # 只显示面积大于500的轮廓
            if cv2.contourArea(c) > 500:
                # 在原图中画出轮廓
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # 只在第一帧（不包含用于背景分割训练的帧）中依据轮廓的个数生成行人对象
                if firstFrame is True:
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                counter += 1
        # 在每一帧中更新CMAShift和卡尔曼滤波器的值
        for i, p in pedestrians.items():
            p.update(frame)

        # firstFrame = False
        frames += 1

        cv2.imshow("surveillance", frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()


if __name__ == "__main__":
    main()
