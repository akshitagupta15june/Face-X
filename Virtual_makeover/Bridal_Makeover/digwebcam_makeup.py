# pylint: skip-file
import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy

jewel_img = cv2.imread("jewelery.png")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame,(432, 576))
    face_landmarks_list = face_recognition.face_landmarks(frame)
    for face_landmarks in face_landmarks_list:

        shape_chin = face_landmarks['chin']
        x = shape_chin[3][0]
        y = shape_chin[5][1]

        img_width = abs ( shape_chin[3][0] - shape_chin[14][0])
        print(f'shape_chin[3][0]:{shape_chin[3][0]}')
        print(f'shape_chin[14][0]:{shape_chin[14][0]}')
        img_height = int( 1.02 * img_width)
        print(f'imgWidth:{img_width}')
        print(f'imgheight:{img_height}')
        jewel_img = cv2.resize(jewel_img, (img_width,img_height), interpolation=cv2.INTER_AREA)
        jewel_gray = cv2.cvtColor(jewel_img, cv2.COLOR_BGR2GRAY)
        thresh, jewel_mask = cv2.threshold(jewel_gray, 230, 255, cv2.THRESH_BINARY)
        jewel_img[jewel_mask == 255] = 0
        jewel_area = frame[y:y+img_height, x:x+img_width]
        masked_jewel_area = cv2.bitwise_and(jewel_area, jewel_area, mask=jewel_mask)
        final_jewel = cv2.add(masked_jewel_area, jewel_img)

        frame[y:y+img_height, x:x+img_width] = final_jewel

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        draw.polygon(face_landmarks['left_eyebrow'], fill=(23, 26, 31, 100))
        draw.polygon(face_landmarks['right_eyebrow'], fill=(23, 26, 31, 100))
        draw.polygon(face_landmarks['top_lip'], fill=(158, 63, 136, 100))
        draw.polygon(face_landmarks['bottom_lip'], fill=(158, 63, 136, 100))
        draw.polygon(face_landmarks['left_eye'], fill=(23, 26, 31, 100))
        draw.polygon(face_landmarks['right_eye'], fill=(23, 26, 31, 100))

        x_centre_eyebrow = face_landmarks['nose_bridge'][0][0]
        y_centre_eyebrow = face_landmarks['left_eyebrow'][4][1]
        r = int( 1/4 * abs(face_landmarks['left_eyebrow'][4][0] - face_landmarks['right_eyebrow'][0][0]) )
        print("r:",r)
        print("x_centre_eyebrow:",x_centre_eyebrow)
        print("y_centre_eyebrow",y_centre_eyebrow)

        draw.ellipse((x_centre_eyebrow-r, y_centre_eyebrow-r, x_centre_eyebrow+r, y_centre_eyebrow+r), fill =(128, 0, 128, 100))

        pil_img.show()
        opencv_img = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame',opencv_img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()