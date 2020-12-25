#Pre-Proccessed Images Data Collection
import cv2
import numpy as np
import os
photo = cv2.imread("path of image") #photo should be of 640x480 pixels and axis must match.
name = input("Enter your name : ")

frames = []
outputs = []

frames.append(photo.flatten())
outputs.append([name])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)
