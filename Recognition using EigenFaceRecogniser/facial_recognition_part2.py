import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
data_path='/home/akshita/Desktop/Face_reco/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,Labels=[],[]

for i, files in enumerate(onlyfiles):
    image_path=data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels=np.asarray(Labels,dtype=np.int32)

model=cv2.face.EigenFaceRecognizer_create()

model.train(np.asarray(Training_data),np.asarray(Labels))
print("Model Training Complete")
