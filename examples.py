import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


# ========================================== 生成灰度和色彩图片==========================================
def example_1():
    numpyArray = np.random.randint(0, 255, 120000)

    # creat gray image
    grayImage = numpyArray.reshape(300, 400)
    cv2.imwrite('GrayImage.png', grayImage)

    # creat color image
    bgrImage = numpyArray.reshape(100, 400, 3)
    cv2.imwrite('ColorImage.png', bgrImage)


# ========================================== 显示图片==========================================
def example_2():
    img = cv2.imread('ColorImage.png')
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# ========================================== 捕获摄像头内容 ==========================================
def example_3():
    cameraCapture = cv2.VideoCapture(0)
    fps = 30  # 这是个假设值
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('OutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    success, frame = cameraCapture.read()
    numFrames = 10 * fps - 1
    while success and numFrames > 0:
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
        numFrames -= 1
    cameraCapture.release()


# ========================================== 追踪视频左上角的物体 ==========================================
def example_4():
    # 设置初始化的窗口位置
    r, h, c, w = 0, 300, 0, 300  # 设置初试窗口位置和大小
    track_window = (c, r, w, h)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ret, frame = cap.read()

    # 设置追踪的区域
    roi = frame[r:r + h, c:c + w]
    # roi区域的hsv图像
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 取值hsv值在(0,60,32)到(180,255,255)之间的部分
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    # 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    # 归一化
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 设置终止条件，迭代10次或者至少移动1次
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while (1):
        ret, frame = cap.read()
        if ret == True:
            # 计算每一帧的hsv图像
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 计算反向投影
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# ========================================== 基于ORB特征匹配 ==========================================
def example_5():
    img1 = cv2.imread('feature_detection/icon.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('feature_detection/logo.jpg', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 使用暴力匹配
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


# ========================================== 基于FLANN特征匹配 ==========================================
def example_6():
    queryImage = cv2.imread('feature_detection/icon.jpg', cv2.IMREAD_GRAYSCALE)
    trainingImage = cv2.imread('feature_detection/test_3.jpg', cv2.IMREAD_GRAYSCALE)

    # 创造SIFT并计算出描述符
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainingImage, None)

    # FLANN匹配器
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(des1, des2, k=2)

    # 准备一个空的容器装载匹配结果
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    drawParams = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                      flags=0)

    resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
    plt.imshow(resultImage)
    plt.show()


# ========================================== 基于FLANN单应性匹配(图像畸变、尺寸变化后依然能匹配) ==========================================
def example_7():
    # 设定需要多少个好的匹配点才认为匹配成功（一般大于10）
    MIN_MATCH_COUNT = 20

    img1 = cv2.imread('feature_detection/icon.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('feature_detection/test_4.jpg', cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算出描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 应用LOWE的算法过滤错误值，存储匹配值
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # 大于阈值才认为匹配到图像
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 通过单应性匹配图像
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        # 显示结果
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()


if __name__ == "__main__":
    example_7()
