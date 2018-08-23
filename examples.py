import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#========================================== 生成灰度和色彩图片==========================================
def example_1():
    numpyArray = np.random.randint(0, 255, 120000)

    # creat gray image
    grayImage = numpyArray.reshape(300, 400)
    cv2.imwrite('GrayImage.png', grayImage)

    # creat color image
    bgrImage = numpyArray.reshape(100, 400, 3)
    cv2.imwrite('ColorImage.png', bgrImage)

#========================================== 显示图片==========================================
def example_2():
    img = cv2.imread('ColorImage.png')
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


#========================================== 捕获摄像头内容 ==========================================
def example_3():
    cameraCapture = cv2.VideoCapture(0)
    fps = 30 #  这是个假设值
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('OutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    success, frame = cameraCapture.read()
    numFrames = 10 * fps - 1
    while success and numFrames > 0:
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
        numFrames -= 1
    cameraCapture.release()


#========================================== 追踪视频左上角的物体 ==========================================
def example_4():
    # 设置初始化的窗口位置
    r,h,c,w = 0,300,0,300 # 设置初试窗口位置和大小
    track_window = (c,r,w,h)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ret, frame= cap.read()

    # 设置追踪的区域
    roi = frame[r:r+h, c:c+w]
    # roi区域的hsv图像
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 取值hsv值在(0,60,32)到(180,255,255)之间的部分
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    # 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    # 归一化
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    # 设置终止条件，迭代10次或者至少移动1次
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret, frame = cap.read()
        if ret == True:
            # 计算每一帧的hsv图像
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 计算反向投影
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
            cv2.imshow('img2',img2)


        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#========================================== 基于ORB特征匹配 ==========================================
def example_5():
    img1 = cv2.imread('./feature_detection/icon.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./feature_detection/logo.jpg', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # 使用暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


# ========================================== 基于FLANN特征匹配 ==========================================


if __name__ == "__main__":
    example_5()