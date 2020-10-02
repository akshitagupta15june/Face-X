import cv2
import numpy as np
from os import listdir, getenv
from os.path import isfile, join
import dotenv

dotenv.load_dotenv()
data_path=getenv("DATA_PATH")

onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,Labels=[],[]

for i, files in enumerate(onlyfiles):
    image_path=data_path + onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels=np.asarray(Labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_data),np.asarray(Labels))
print("Model Training Complete")
