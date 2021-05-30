import numpy as np
import cv2
from tensorflow.keras.models import load_model

# path
SAVE_PATH = "resnet_model"
PROTOTXT_PATH = "face_detector/deploy.prototxt"
CAFFE_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

# load face net
faceNet = cv2.dnn.readNet(PROTOTXT_PATH, CAFFE_PATH)


model = load_model(SAVE_PATH, compile=True)


labels = ["with_mask", "without_mask"]
# 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap = cv2.VideoCapture(0)
while True:
    # 从摄像头读取图片
    sucess, img = cap.read()
    # 转为灰度图片
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    """
    Face Detector
    """
    # img_resize = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    blobShape = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), [104, 117, 123], True, False)
    faceNet.setInput(blobShape)
    detections = faceNet.forward()

    height = img.shape[0]
    width = img.shape[1]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # face confidence
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            

            try:
                img_face = img[y1:y2, x1:x2]
                # img_face = img[(y1 - 30):(y2 + 30), (x1 - 30):(x2 + 30)]
                # cv2.rectangle(img, (x1 - 30, y1 - 30), (x2 + 30, y2 + 30),
                #           (0, 255, 255), thickness=1)

                img_resize = cv2.resize(img_face, dsize=(224, 224),
                                        interpolation=cv2.INTER_CUBIC)
                img_resize = cv2.normalize(img_resize, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # cv2.imshow("img",img)
                
                """
                Mask Detector
                """
                img_1 = img_resize[np.newaxis, :]
                prediction = model.predict(img_1)
                predicted_index = np.argmax(prediction, axis=1)
                predicted_proba = np.max(prediction)
                # print(predicted_index)

                text = labels[predicted_index[0].item()] + " " + \
                    str(round(predicted_proba, 2))

                if labels[predicted_index[0].item()] == "with_mask":
                    cv2.putText(img, text, (x1-15, y1-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif labels[predicted_index[0].item()] == "without_mask":
                    cv2.putText(img, text, (x1-15, y1-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except:
                break

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(height / 150)), 8)
                          
    cv2.imshow("img", img)

    k = cv2.waitKey(1)
    if k == 27:
        # 通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k == ord("s"):
        # 通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg", img)
        cv2.destroyAllWindows()
        break

# 关闭摄像头
cap.release()
