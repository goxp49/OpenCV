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