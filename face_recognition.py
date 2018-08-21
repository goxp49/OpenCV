import cv2
import numpy as np
import os


# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))
# generate生成路径
GENERATE_PATH = os.path.join(Project_PATH, 'face_recognition', 'gen')
# 输出路径
OUTPUT_PATH = os.path.join(Project_PATH, 'face_recognition', 'output')

def generate(id, resize=False):
    # 加载模型
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    # 初始化摄像头
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        status, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 获得矩形框尺寸
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 依据参数决定是否将图片改为 200 * 200，LBPH对图片尺寸无要求
            if resize:
                f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            # cv2.imwrite('./FaceRecognition/gen/%s.pgm' % str(count), f)
            cv2.imencode('.pgm', f)[1].tofile(os.path.join(GENERATE_PATH, 'user_%s_%s.pgm' % (id, str(count))))   # 正确方法
            print('已保存图片')
            count += 1
        # 显示结果
        cv2.imshow('camera', frame)
        # 按q退出
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
        if count >= 50:
            break

    camera.release()
    cv2.destroyAllWindows()


def read_images(image_paths):
    faces_sample, lable = [], []
    # os.listdir()用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for image_path in os.listdir(image_paths):
        image = cv2.imread(os.path.join(image_paths, image_path), cv2.IMREAD_GRAYSCALE)
        print(image_path.split(".")[0].split('_')[-1])
        image_id = int(image_path.split(".")[0].split('_')[1])
        faces_sample.append(np.asarray(image, dtype=np.uint8))
        lable.append(image_id)
    return [faces_sample, lable]


def face_rec():
    names = ['wang', 'll']
    [faces_sample, lables] = read_images(GENERATE_PATH)
    lables = np.asarray(lables, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(faces_sample), np.asarray(lables))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    while True:
        status, frame = camera.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            # 已经训练结果对脸部进行判断
            try:
                params = model.predict(roi)
                print('标签：%s， 相似率：%.2f' % (params[0], params[1]))
                cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow('camera', frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cv2.destoryAllWindows()


if __name__ == "__main__":
    # generate(0)
    face_rec()