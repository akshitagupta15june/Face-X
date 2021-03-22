#importing our libraries that we will be using for emotion detection
from operator import index
from textwrap import fill
from tkinter import Label, Tk
import numpy as np
import cv2
import keras
from tkinter import *
import pandas as pd
import webbrowser

win = Tk() #main application window
win.geometry('600x500')
win.title("Emotional Recommender")
label = Label(win,text="Welcome To Emotional Recommender",font=50,relief = RAISED,bg = 'red').pack(fill=X,padx=15,pady=30)
user_label = Label(win,text="Here's How this application works: \n 1) Click on Capture button to Open up your camera. \n 2) The Model will detect your emotions, \n you can exit the camera window by clicking 'q' on your keyboard. \n 3) The Result will be displayed in the window",font=50,relief = RAISED,bg = 'red').pack(fill=X,padx=15,pady=30)
win.iconbitmap(r'Face-Emotions-Recognition\Emotion-recognition-with-GUI\\images\\Icons8-Ios7-Logos-Google-Drive-Copyrighted.ico') #giving the window an icon
new = 1

cap = cv2.VideoCapture(0) #used for capturing the  video using the webcam
model_path = 'Face-Emotions-Recognition\Emotion-recognition-with-GUI\model_optimal.h5' #path of our model
model = keras.models.load_model(model_path) #loading our model that we will use to make predictions of emotions


emotion_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'} #dictionary containing different values
def videocapture():
    while True: #continuous loop to keep the window running
        isTrue,frame = cap.read() #reading our  frames from the capture
        facecascade = cv2.CascadeClassifier('Face-Emotions-Recognition\Emotion-recognition-with-GUI\haarcascade_frontalface_default.xml') #using the cascade classifier for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting whatever we are reading into gray
        faces = facecascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5) #this helps in finding features and location in our images - we are passing in our grayscale input, we are scaling down the image which is done with scaleFactor 
        # min neighbors helps in determining the quality of detection 

        for (x, y, w, h) in faces: #drawing rectangles on the faces detected and also adding text
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) #used to draw our rectangle, we are specifying the start and end point and also color and width of our rectangle
            roi_gray = gray[y:y + h, x:x + w] #ROI - Region of interest, in this we are trying to select the rows starting from y to y+h and then columns from x to x+h - this works like NumPy slicing
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) #resizing the inputs in order to get  them in the same shape as the images on which our images were trained
            prediction = model.predict(cropped_img) #making predictions on the face detected
            maxindex = int(np.argmax(prediction)) #getting the maximum index out of all the predicted indices
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) #printing the emotion label corresponding to the output index from our emotio dictionary
        
        
        cv2.imshow('Video', cv2.resize(frame,(700,500),interpolation = cv2.INTER_CUBIC)) #creating our video window in which we will be detecting emotions
        if cv2.waitKey(1) & 0xFF == ord('q'): #we will have to press q if we wish to exit our window
            break
        
    var = emotion_dict[maxindex]
    label1 = Label(win,text="Emotion Detected " + " ==> "+ var,font=50,relief=RAISED,bg = 'red').pack(fill=X,padx=15,pady=30)

    cap.release() #this will release the hardware and software resources that are being used
    cv2.destroyAllWindows() #destroys the window that we created for emotion detection


Button(win,text="Capture",command = videocapture,relief=RAISED,width=15,font=10,bg = 'black',fg = 'green').pack(pady=20)
Button(win,text="Exit Application",command = win.destroy,relief=RAISED,width=15,font=10).pack(pady=5)
win.mainloop()