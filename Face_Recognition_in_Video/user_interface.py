import cv2
import numpy as np
import face_recognition
import os
import datetime
import glob
import pathlib

def image_list():
    path="Images"
    images=[]
    class_name=[]
    my_list=os.listdir(path)
    # print(my_list)
    for c1 in my_list:
        cur_img=cv2.cv2.imread(f"{path}/{c1}")
        images.append(cur_img)
        class_name.append(os.path.splitext(c1)[0])
    return images, class_name
# images, class_name=image_list()
# print(class_name)

def findEncodings(images):
    encode_list=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        encode=face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def register_face(name):
    with open("found_faces.csv",'r+') as file:
        my_data_list=file.readlines()
        name_list=[]
        for line in my_data_list:
            entry=line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now=datetime.datetime.now()
            dt_string=now.strftime("%H:%M:%S")
            file.writelines(f"\n{name},{dt_string}")


def face_recog(video_capture,encode_list_known,class_name):
    found=0
    while True:
        list_of_files = glob.glob(f"{pathlib.Path().resolve()}/Images/*.jpg")  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file_name=os.path.basename(latest_file)
        success, img=video_capture.read()
        if not success:
            return found
        img_small=cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small=cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)

        faces_cur_frame=face_recognition.face_locations(img_small)
        encodes_cur_frame = face_recognition.face_encodings(img_small,faces_cur_frame)

        for encode_face,face_loc in zip(encodes_cur_frame,faces_cur_frame):
            matches=face_recognition.compare_faces(encode_list_known,encode_face)
            face_dis=face_recognition.face_distance(encode_list_known,encode_face)
            print(face_dis)
            match_index=np.argmin(face_dis)

            if matches[match_index]:
                name=class_name[match_index].upper()
                print(name)
                y1,x2,y2,x1=face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                register_face(name)
                if f"{name}.JPG"==latest_file_name.upper():
                    found=1
                    return found


        cv2.imshow("Video",img)
        cv2.waitKey(1)

