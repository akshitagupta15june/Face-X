import cv2
import math
import argparse


def highlightFace(net, frame, conf_threshold=0.7):
    '''
    This function detects faces on the image using the 'net' passed (if any) and returns the detection output
    as well as the cordinates of the faces detected
    '''


    frameOpencvDnn=frame.copy()
    #--------saving the image dimensions as height and width-------#
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    #-----------blob-> Preprocessing the image to required input of the model---------#
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob) #setting the image blob as input

    detections = net.forward()
    print(detections)
    print(detections.shape)

    faceBoxes=[]
    # for i in range(detections.shape[2]):
    #     confidence=detections[0,0,i,2]
    #     if confidence>conf_threshold:
    #         x1=int(detections[0,0,i,3]*frameWidth)
    #         y1=int(detections[0,0,i,4]*frameHeight)
    #         x2=int(detections[0,0,i,5]*frameWidth)
    #         y2=int(detections[0,0,i,6]*frameHeight)
    #         faceBoxes.append([x1,y1,x2,y2])
    #         cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes



#-------Creating and Parsing through the argument passed on the terminal-------------#
parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

#-----------Model File Paths----------------#
faceProto="Models/opencv_face_detector.pbtxt"
faceModel="Models/opencv_face_detector_uint8.pb"
ageProto="Models/age_deploy.prototxt"
ageModel="Models/age_net.caffemodel"
genderProto="Models/gender_deploy.prototxt"
genderModel="Models/gender_net.caffemodel"


#-----------Model Variables---------------#
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#-------------Creating the DNN------------#
faceNet= cv2.dnn.readNet(faceModel,faceProto)
ageNet= cv2.dnn.readNet(ageModel,ageProto)
genderNet= cv2.dnn.readNet(genderModel,genderProto)

#---------Instantiate the Video Capture Object-----------#
video=cv2.VideoCapture(args.image if args.image else 0) #check whether image was passed or not otherwise use the webcam
padding=20

while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    #----------------Face Detection-----------------#
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
