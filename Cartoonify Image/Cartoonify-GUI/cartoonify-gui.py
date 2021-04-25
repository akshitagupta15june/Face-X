#Importing the necessary libraries
import tkinter as tk
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import cv2
import easygui
import sys
import os

#Function Defined for Uploading function:
def upload():
    imagepath = easygui.fileopenbox()
    cartoon(imagepath)

#Function to convert image to cartoon
def cartoon(imagepath):
    #Image variable takes image using imagepath
    image = cv2.imread(imagepath)

    if image is None:
        print('Choose another file')
        sys.exit()
    height, width, channels = image.shape
    print(width, height, channels)

    #Image_resize
    if height >=900 and width >=1200:
        resized_image = cv2.resize(image, (800, int(700*0.8)))
    else:
        resized_image = cv2.resize(image, (width, int(width*0.8)))
    #sharpen image

    #Putting a filter using numpy array
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #Sharpening Image using Open CV filter2D function
    sharpen_image = cv2.filter2D(resized_image, -1, filter)
    #Converting to Fray Image Scale
    gray_image = cv2.cvtColor(sharpen_image, cv2.COLOR_BGR2GRAY)
    #Blurring the Image
    blurred = cv2.medianBlur(gray_image, 9)
    # For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 11)
    #Original Image 
    original_image = cv2.bilateralFilter(resized_image, 13, 150, 150)

    cartoon = cv2.bitwise_and(original_image, original_image, mask=thresh)
    if cartoon.shape[0] >=900 and cartoon.shape[1] >=1200:
        cartoon_resize = cv2.resize(cartoon, (800, int(700*0.8)))
    else:
        cartoon_resize = cv2.resize(cartoon, (cartoon.shape[1], int(cartoon.shape[0]*0.8)))
    #cartoon_resize = cv2.resize(cartoon, (width, int(width*0.8)))

    #Displaying the Main,Cartoonified and Sharpened Image
    cv2.imshow("Cartoonified", cartoon_resize)
    cv2.imshow("Main Image", image)
    cv2.imshow("Sharped Image", sharpen_image)
    save1 = Button(GUI, text="Save cartoon image", command=lambda: save_image(cartoon_resize, imagepath ), padx=30, pady=5)
    save1.configure(background='black', foreground='white', font=('calibri', 12, 'bold'))
    save1.pack(side=TOP, pady=50)

#Saving Image 
def save_image(cartoon_resize, imagepath):
    name= "CartooniFied"
    file = os.path.dirname(os.path.realpath(imagepath))
    last_name = os.path.splitext(imagepath)[1]
    path = os.path.join(file, name + last_name  )
    cv2.imwrite(path, cartoon_resize)
    full_name = "Image " + name + "saved at" + path

    tk.messagebox.showinfo(message=full_name)


#create GUI Interface:

#Defining the basic structure of the application
GUI = tk.Tk()
GUI.geometry('650x500')
GUI.title("Cartoonify Image")
GUI.configure(background='skyblue')
#Loading the Background Image for the Application
load=Image.open("D:\\GitRepo\\Face-X\\Cartoonify Image\\Cartoonify-GUI\\background.png")
render=ImageTk.PhotoImage(load)
img=Label(GUI,image=render)
img.place(x=0,y=0)

#Defining Buttons
label=Label(GUI, background='black', font=('calibri',20,'bold'))
upload=Button(GUI, text="Cartoonify Image",command=upload, padx=30,pady=5)
upload.configure(background='black', foreground='white',font=('calibri',12,'bold'))
upload.pack(side=TOP,pady=50)

GUI.mainloop()


