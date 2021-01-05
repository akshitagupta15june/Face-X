from __future__ import absolute_import, print_function

import argparse
import math
import os
import sys
import threading
import time
from os import listdir
from os.path import isfile, join
from sys import platform as _platform
from threading import Thread
import cv2
from PIL import Image, ImageTk
import dlib
from imutils import face_utils, rotate_bound

if sys.version_info.major >= 3:
    from tkinter import SUNKEN, RAISED, Tk, PhotoImage, Button, Label
else:
    from Tkinter import SUNKEN, RAISED, Tk, PhotoImage, Button, Label


_streaming = False


def put_sprite(num):
    global SPRITES, BTNS
    SPRITES[num] = 1 - SPRITES[num]  
    if SPRITES[num]:
        BTNS[num].config(relief=SUNKEN)
    else:
        BTNS[num].config(relief=RAISED)


def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imgH: 
        sprite = sprite[0 : imgH - y_offset, :, :]

    if x_offset + w >= imgW:  
        sprite = sprite[:, 0 : imgW - x_offset, :]

    if x_offset < 0:  
        sprite = sprite[:, abs(x_offset) : :, :]
        w = sprite.shape[1]
        x_offset = 0

    
    for c in range(3):
       
        frame[y_offset : y_offset + h, x_offset : x_offset + w, c] = sprite[:, :, c] * (
            sprite[:, :, 3] / 255.0
        ) + frame[y_offset : y_offset + h, x_offset : x_offset + w, c] * (
            1.0 - sprite[:, :, 3] / 255.0
        )
    return frame


def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0 * head_width / w_sprite
    sprite = cv2.resize(
        sprite, (0, 0), fx=factor, fy=factor
    )  
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig = (
        head_ypos - h_sprite if ontop else head_ypos
    ) 
    if(
        y_orig < 0
    ):  
        sprite = sprite[abs(y_orig) : :, :, :]  
        y_orig = 0 
    return (sprite, y_orig)



def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
    sprite = cv2.imread(path2sprite, -1)
    
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image, sprite, x, y_final)



def calculate_inclination(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl


def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:, 0])
    y = min(list_coordinates[:, 1])
    w = max(list_coordinates[:, 0]) - x
    h = max(list_coordinates[:, 1]) - y
    return (x, y, w, h)


def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x, y, w, h) = calculate_boundbox(points[17:22])  # left eyebrow
    elif face_part == 2:
        (x, y, w, h) = calculate_boundbox(points[22:27])  # right eyebrow
    elif face_part == 3:
        (x, y, w, h) = calculate_boundbox(points[36:42])  # left eye
    elif face_part == 4:
        (x, y, w, h) = calculate_boundbox(points[42:48])  # right eye
    elif face_part == 5:
        (x, y, w, h) = calculate_boundbox(points[29:36])  # nose
    elif face_part == 6:
        (x, y, w, h) = calculate_boundbox(points[48:68])  # mouth
    return (x, y, w, h)


def cvloop(run_event, read_camera=0, virtual_camera=0):
    global panelA
    global SPRITES

    dir_ = "./sprites/flyes/"
    flies = [
        f for f in listdir(dir_) if isfile(join(dir_, f))
    ]  # image of flies to make the "animation"
    i = 0
    video_capture = cv2.VideoCapture(read_camera)  # read from webcam
    (x, y, w, h) = (0, 0, 10, 10)  # whatever initial values

    # Filters path
    detector = dlib.get_frontal_face_detector()

    # Facial landmarks
    # print("[INFO] loading facial landmark predictor...")
    model = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(
        model
    )  
    stream_camera = None
    img_counter = 0
    while run_event.is_set(): 
        ret, image = video_capture.read()

        if not ret:
            print("Error reading camera, exiting")
            break

        if _streaming:
            if stream_camera is None:
                if virtual_camera:
                    h, w = image.shape[:2]
                    stream_camera = pyfakewebcam.FakeWebcam(
                        "/dev/video{}".format(virtual_camera), w, h
                    )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        for face in faces:  # if there are faces
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            # *** Facial Landmarks detection
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            incl = calculate_inclination(
                shape[17], shape[26]
            )  # inclination based on eyebrows

            # condition to see if mouth is open
            is_mouth_open = (
                shape[66][1] - shape[62][1]
            ) >= 10  # y coordiantes of landmark points of lips

            # hat condition
            if SPRITES[0]:
                apply_sprite(image, "./sprites/hat.png", w, x, y, incl)

            # mustache condition
            if SPRITES[1]:
                (x1, y1, w1, h1) = get_face_boundbox(shape, 6)
                apply_sprite(image, "./sprites/mustache.png", w1, x1, y1, incl)

            # glasses condition
            if SPRITES[3]:
                (x3, y3, _, h3) = get_face_boundbox(shape, 1)
                apply_sprite(
                    image, "./sprites/glasses.png", w, x, y3, incl, ontop=False
                )

            # flies condition
            if SPRITES[2]:
                # to make the "animation" we read each time a different image of that folder
                # the images are placed in the correct order to give the animation impresion
                apply_sprite(image, dir_ + flies[i], w, x, y, incl)
                i += 1
                i = (
                    0 if i >= len(flies) else i
                )  # when done with all images of that folder, begin again

            # doggy condition
            (x0, y0, w0, h0) = get_face_boundbox(shape, 6)  # bound box of mouth
            if SPRITES[4]:
                (x3, y3, w3, h3) = get_face_boundbox(shape, 5)  # nose
                apply_sprite(
                    image, "./sprites/doggy_nose.png", w3, x3, y3, incl, ontop=False
                )

                apply_sprite(image, "./sprites/doggy_ears.png", w, x, y, incl)

                if is_mouth_open:
                    apply_sprite(
                        image,
                        "./sprites/doggy_tongue.png",
                        w0,
                        x0,
                        y0,
                        incl,
                        ontop=False,
                    )
            if SPRITES[5]:
                img_name = "image{}.png".format(img_counter)
                cv2.imwrite(img_name, image)
                # print("Success")
                
                break
            img_counter += 1
        # OpenCV represents image as BGR; PIL but RGB, we need to change the chanel order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if _streaming:
            if virtual_camera:
                stream_camera.schedule_frame(image)

        # conerts to PIL format
        image = Image.fromarray(image)
        # Converts to a TK format to visualize it in the GUI
        image = ImageTk.PhotoImage(image)
        # Actualize the image in the panel to show it
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()


# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--read_camera", type=int, default=0, help="Id to read camera from")
parser.add_argument(
    "--virtual_camera",
    type=int,
    default=0,
    help="If different from 0, creates a virtual camera with results on that id (linux only)",
)
args = parser.parse_args()

# Initialize GUI object
root = Tk()
root.title("Snap chat filters")
this_dir = os.path.dirname(os.path.realpath(__file__))
# Adds a custom logo
imgicon = PhotoImage(file=os.path.join(this_dir, "imgs", "icon.gif"))
root.tk.call("wm", "iconphoto", root._w, imgicon)

##Create 5 buttons and assign their corresponding function to active sprites
btn1 = Button(root, text="Hat", command=lambda: put_sprite(0))
btn1.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn2 = Button(root, text="Mustache", command=lambda: put_sprite(1))
btn2.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn3 = Button(root, text="Flies", command=lambda: put_sprite(2))
btn3.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn4 = Button(root, text="Glasses", command=lambda: put_sprite(3))
btn4.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn5 = Button(root, text="Doggy", command=lambda: put_sprite(4))
btn5.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn6 = Button(root, text="Capture", command=lambda: put_sprite(5))
btn6.pack(side="top", fill="both", expand="no", padx="5", pady="5")


# Create the panel where webcam image will be shown
panelA = Label(root)
panelA.pack(padx=10, pady=10)

# Variable to control which sprite you want to visualize
SPRITES = [
    0,
    0,
    0,
    0,
    0,
    0,
]  # hat, mustache, flies, glasses, doggy -> 1 is visible, 0 is not visible
BTNS = [btn1, btn2, btn3, btn4, btn5,btn6]


# Creates a thread where the magic ocurs
run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event, args.read_camera, args.virtual_camera))
action.setDaemon(True)
action.start()


# Function to close all properly, aka threads and GUI
def terminate():
    global root, run_event, action
    print("Closing ...")
    run_event.clear()
    time.sleep(1)
    # action.join() #strangely in Linux this thread does not terminate properly, so .join never finishes
    root.destroy()
    print("Hope You Enjoyed")


# When the GUI is closed it actives the terminate function
root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop()  # creates loop of GUI
