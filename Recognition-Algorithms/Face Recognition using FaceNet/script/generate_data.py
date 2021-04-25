import os
import numpy as np
from PIL import Image,ImageFont
from detect_faces import detect_faces
from fx import prewhiten,l2_normalize
from keras.models import load_model
from sklearn.linear_model import LinearRegression

faces_path='../data/faces/'
model_path='../model/facenet_keras.h5'

choice=int(input("""Choose one:
        1. Have faces already.
        2. Want to detect faces from pictures in './data/images/'\n"""))
if choice==2:
    detect_faces()
    print("\n\n\nFaces saved in './data/faces/'\n\n\n")
    print("\n\n\nNOTE: PLEASE DELETE UNNECESSARY FACES BEFORE PROCEEDING AND RENAME FACES TO THEIR NAMES.\n\n")
    input("Press ENTER to continue.(IF YOU'VE DELETED UNNECESSARY FACES AND RENAMED FACES)\n")
elif choice!=1:
    print('Wrong Choice')
    input()
    quit()

os.chdir(faces_path)

names=os.listdir()
if len(names)==0:
    print("No Face Found")
    input()
    quit()

names.sort()
names=np.array(names)
faces=[]
for i in names:
    img=Image.open(i).resize((160,160))
    img=np.array(img)
    faces.append(img)
    
faces=np.array(faces)
faces=prewhiten(faces)


model=load_model(model_path)

embs=model.predict(faces)
embs=l2_normalize(embs)

for i in range(len(names)):
    names[i]=names[i][:-4]

font_path='../font/Calibri Regular.ttf'

slope=[]
intercept=[]
for i in names:
    x=[]
    y=[]
    for j in range(1,100):
        font=ImageFont.truetype(font_path,j)
        x.append(j)
        y.append(font.getsize(i)[0])
    lin=LinearRegression().fit(np.array(y).reshape(-1,1),np.array(x))
    slope.append(lin.coef_)
    intercept.append(lin.intercept_)
slope=np.array(slope)
intercept=np.array(intercept)


np.savez_compressed('../arrays/vars.npz',a=slope,b=intercept)
np.savez_compressed('../arrays/embeddings.npz',a=embs,b=names)
