import cv2
import numpy as np
import mediapipe as mp
import pyautogui as pagui
import time as time
from tensorflow import keras

def multiple_clicker(key1, key2):
    pagui.keyDown(key1)
    pagui.press(key2)
    pagui.keyUp(key1)

def clicker(key):
    pagui.keyDown(key)
    time.sleep(1)
    pagui.keyUp(key)

def ytcontrol():

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    model=keras.models.load_model("Youtube Controller\ytcontroller\mp_hand_gesture")

    f = open('Youtube Controller\ytcontroller\gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        x, y, c = frame.shape

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        
        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

  
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("Output is :", frame) 

        if cv2.waitKey(1) == ord('q'):
            break
        print(className)
        checkGesture(className)

    cap.release()

    cv2.destroyAllWindows()

def checkGesture(hd): 

    if hd=='rock':
        #fullscreen
        clicker('f')
    elif hd=='thumbs up':
        #volume up
        clicker('up')
    elif hd=='thumbs down':
        #volume down
        clicker('down')
    elif hd=='smile' or hd=='call me':
        #forward
        clicker('right')
    elif hd=='peace':
        #reverse
        clicker('left')
    elif hd=='okay':
        #pause/play
        clicker('space')
    return

ytcontrol()