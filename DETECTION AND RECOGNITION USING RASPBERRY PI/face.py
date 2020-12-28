import RPi.GPIO as GPIO

import time
from datetime import datetime

print('1')
print(str(datetime.now()))

import numpy as np
import cv2
import os

#path for saving images taken in the raspberry Pi
path = '/home/pi/Desktop/face_recognition'

#Importing Libaries for OLED display
import Adafruit_GPIO.SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

#Setting initial pin values for the oled display
RST = 24
DC = 1
SPI_PORT = 0
SPI_DEVICE = 0

#Function for clearing the Contents of OLED display
def create_screen():
    disp = Adafruit_SSD1306.SSD1306_128_64(rst = RST)

    disp.begin()
    disp.clear()
    disp.display()

    width = disp.width
    height = disp.height
    image = Image.new('1',(width,height))

    draw = ImageDraw.Draw(image)
    draw.rectangle((0,0,width,height), outline=0, fill=0)
    return draw, image, disp


#Setting up position of content on OLED Display
padding = 2
shape_width = 20
top = padding

# Move left to right keeping track of the current x position for drawing shapes.
pad = padding

font = ImageFont.truetype('custom.ttf', 12)

#code for facila recognition begins here
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("2")
print(str(datetime.now()))
# Load the trained mode
recognizer.read('trainer/trainer.yml')

print("3")
print(str(datetime.now()))
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

print("4")
print(str(datetime.now()))
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
# font = cv2.FONT_HERSHEY_SIMPLEX
draw, image, disp = create_screen()
draw.text((pad,top), 'Press Button', font = font, fill = 255)
disp.image(image)
disp.display()


#Function for face detection and face recognition
def facial_recognition():

    draw, image, disp = create_screen()

    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
    im = []
    for i in range(3):
        ret,im = cap.read() # return a single frame in variable `frame`
        cv2.imwrite('opencv'+str(i)+'.png', im)
        if(i == 2):
            cv2.imwrite(os.path.join(path , str(datetime.now())+'.png'), im)

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 2,5)

    # For each face in faces
    #draw.text((x,top), 'not detected', font = font, fill = 255)
    detect = False

    for(x,y,w,h) in faces:
        # Create rectangle around the face
        #cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist
        if(Id[0] == 1 and Id[1] <= 70):
            name = "Gagandeep Nagpal"
        #If not exist, then it is Unknown
        elif(Id[0] == 2 and Id[1] <= 70):
            name = "Gagandeep Singh"

        elif(Id[0] == 3 and Id[1] <= 70):

            name = "Jappreet Singh"

        elif(Id[0] == 4 and Id[1] <= 70):
            name = "Nishant Rao"

        else:
            print(Id[0])
            name = "All Good!"

        # Put text describe who is in the picture
        #cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        #cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
        print(Id[0])
        print(Id[1])
        print(name)
        print("5")
        print(str(datetime.now()))

        draw.text((pad,top), name, font = font, fill = 255)
        disp.image(image)
        disp.display()
        detect = True

    # if face is not detected in the image
    if(detect==False):
        draw.text((pad,top), 'not detected', font = font, fill = 255)
        print('Not detected')
        disp.image(image)
        disp.display()

    # Display the video frame with the bounded rectangle
    #cv2.imshow('im',im)

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN)
GPIO.add_event_detect(4, GPIO.RISING)

while True:
    if GPIO.event_detected(4):
        facial_recognition()
