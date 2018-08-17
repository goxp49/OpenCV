import cv2
import numpy
import os

#========================================== 生成灰度和色彩图片==========================================
def example_1():
    numpyArray = numpy.random.randint(0, 255, 120000)

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

if __name__ == "__main__":
    example_3()