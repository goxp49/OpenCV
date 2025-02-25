import cv2
import numpy as np
import os

def mousemove(event, x, y, s, p):
    global frame, current_measurement, measurements, last_measurement, last_prediction, current_prediction
    last_measurement = current_measurement
    last_prediction = current_prediction
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 255, 255))
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))


frame = np.zeros((800, 800, 3), np.uint8)
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.array((2, 1), np.float32)

cv2.namedWindow('kalman')
cv2.setMouseCallback('kalman', mousemove)

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    cv2.imshow('kalman', frame)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()