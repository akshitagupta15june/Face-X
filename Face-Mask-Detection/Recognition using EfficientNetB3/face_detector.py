#importing required libraries
import os
import numpy as np
import cv2
from keras.models import load_model



#input shape on which we have trained our model

input_shape = (120,120,3)
labels_dict = {0: 'WithMask', 1: 'WithoutMask'}
color_dict = {0 : (0,255,0), 1:(0,0,255)} #if 1 - RED color, 0 - GREEN color
model = load_model('best_model.hdf5')

# !pip install mtcnn #toinstall the model mtcnn ## https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
from mtcnn.mtcnn import MTCNN #importing the model
detector = MTCNN() # instatiating the model

## RESEARCH PAPER LINK: https://arxiv.org/abs/1604.02878

size = 4
webcam = cv2.VideoCapture(0)  # Use camera 0 - default webcam

#
while True: #we are reading frame by frame
    (rval, im) = webcam.read()
    # im = cv2.flip(im, 1, 1)  # Flip to act as a mirror
#
#     # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    rgb_image = cv2.cvtColor(mini, cv2.COLOR_BGR2RGB) # MTCNN needs the file in RGB format, but cv2 reads in BGR format. Hence we are converting.
    faces = detector.detect_faces(mini) # detecting faces---> we will have (x,y,w,h) coordinates


#
#     # Draw rectangles around each face
    for f in faces:
        x, y, w, h = [v * size for v in f['box']]


#         # cropping the face portion from the entire image
        face_img = im[y:y + h, x:x + w]
        # print(face_img)
        resized = cv2.resize(face_img, (input_shape[0],input_shape[1])) # resizing the image to our reuired input size on which we have trained our model

        reshaped = np.reshape(resized, (1, input_shape[0],input_shape[1], 3)) # we have used ImageDatagenerator and we have trained our model in batches
                                                                        # hence input shape to our model is (batch_size,height,width,color_depth)
                                                                        # we are converting the image into this format. i.e. (height,width,color_depth) ---> (batch_size,height,width,color_depth)

        result = model.predict(reshaped) #predicting
#         # print(result)
#
        label = np.argmax(result, axis=1)[0] #getting the index for the maximum value
#
        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2) # Bounding box (Big rectangle around the face)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1) # small rectangle above BBox where we will put our text
                                                                        #Thickness of -1 px will fill the rectangle shape by the specified color.
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
#
#     # Show the image
    cv2.imshow('LIVE FACE DETECTION', im)
    key = cv2.waitKey(10)
#     # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# # Stop video
webcam.release()
#
# # Close all started windows
cv2.destroyAllWindows()

## SPECIAL THANKS TO
## https://github.com/mk-gurucharan/Face-Mask-Detection/blob/master/FaceMask-Detection.ipynb
