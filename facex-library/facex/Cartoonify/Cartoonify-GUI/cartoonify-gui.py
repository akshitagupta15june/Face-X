import tkinter as tk
import numpy as np
from tkinter import *
from tkinter import messagebox
import cv2

import easygui
import sys
import os

def upload():
    imagepath = easygui.fileopenbox()
    cartoon(imagepath)

def cartoon(imagepath):

    image = cv2.imread(imagepath)

    if image is None:
        print('Choose another file')
        sys.exit()
    height, width, channels = image.shape
    print(width, height, channels)
    #image_resize
    if height >=900 and width >=1200:
        resized_image = cv2.resize(image, (800, int(700*0.8)))
    else:
        resized_image = cv2.resize(image, (width, int(width*0.8)))
    #sharpen image

    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])

    sharpen_image = cv2.filter2D(resized_image, -1, filter)



    gray_image = cv2.cvtColor(sharpen_image, cv2.COLOR_BGR2GRAY)


    blurred = cv2.medianBlur(gray_image, 9)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 11)

    original_image = cv2.bilateralFilter(resized_image, 13, 150, 150)

    cartoon = cv2.bitwise_and(original_image, original_image, mask=thresh)
    if cartoon.shape[0] >=900 and cartoon.shape[1] >=1200:
        cartoon_resize = cv2.resize(cartoon, (800, int(700*0.8)))
    else:
        cartoon_resize = cv2.resize(cartoon, (cartoon.shape[1], int(cartoon.shape[0]*0.8)))
    #cartoon_resize = cv2.resize(cartoon, (width, int(width*0.8)))

    cv2.imshow("Cartoonified", cartoon_resize)
    cv2.imshow("Main Image", image)
    cv2.imshow("Sharped Image", sharpen_image)
    save1 = Button(GUI, text="Save cartoon image", command=lambda: save_image(cartoon_resize, imagepath ), padx=30, pady=5)
    save1.configure(background='black', foreground='white', font=('calibri', 10, 'bold'))
    save1.pack(side=TOP, pady=50)


def save_image(cartoon_resize, imagepath):
    name= "Cartoonified"
    file = os.path.dirname(os.path.realpath(imagepath))
    last_name = os.path.splitext(imagepath)[1]
    path = os.path.join(file, name + last_name  )
    cv2.imwrite(path, cartoon_resize)
    full_name = "Image " + name + "saved at" + path

    tk.messagebox.showinfo(message=full_name)


#create GUI platform
GUI = tk.Tk()
GUI.geometry('800*650')
GUI.title("Cartoonify your image!!")
GUI.configure(background='gray')
label=Label(GUI, background='black', font=('calibri',20,'bold'))

upload=Button(GUI, text="Cartoonify an Image",command=upload, padx=10,pady=5)
upload.configure(background='black', foreground='white',font=('calibri',10,'bold'))
upload.pack(side=TOP,pady=50)

GUI.mainloop()


