import cv2
import numpy
import os

TARGET_IMAGE = './face images/feng.jpg'


def static_detect(image):
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.namedWindow('Face Detect')
    cv2.imshow('Face Detect', img)
    cv2.imwrite('./vikings.jpg', img)
    cv2.waitKey(0)


def dynamic_detect():
    # 加载模型
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    # 初始化摄像头
    camera = cv2.VideoCapture(0)
    # 获取视频帧
    while True:
        status, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 脸部侦测，如果找到回返回矩形框尺寸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 获得矩形框尺寸
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 只在脸部区域内寻找眼睛
            face_img = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_img, 1.03, 5, 0, (40, 40))
            # 获得眼睛查找结果
            for (ex, ey, ew, eh) in eyes:
                # 基础坐标为 （x+ex， y+ey）
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        # 显示结果
        # 注意：cv2.rectangle 是直接在输入图像上进行改动，并且保留会输入图像中！所以frame已经被添加了矩形框
        cv2.imshow('camera', frame)
        # 按q退出
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    dynamic_detect()
