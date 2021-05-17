import cv2
from time import sleep


cascPath = "haarcascade_frontalface_default.xml"  # for face detection
faceCascade = cv2.CascadeClassifier(cascPath)

frame = cv2.imread("test.jpg")

mst = cv2.imread('moustache.png')
hat = cv2.imread('cowboy_hat.png')



def put_moustache(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h

    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1



    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fc[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fc

def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    hat_width = face_width+1
    hat_height = int(0.35*face_height)+1
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.45*face_height)][x+j][k] = hat[i][j][k]
    return fc

    
    

    
ch = 3
    
    

# if not video_capture.isOpened():
#     print('Unable to load camera.')
#     sleep(5)
#     pass

# Capture frame-by-frame
# ret, frame = video_capture.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40,40)
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.putText(frame,"Person Detected",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    
    if ch==2:
        frame = put_moustache(mst,frame,x,y,w,h)
    elif ch==1:
        frame = put_hat(hat,frame,x,y,w,h)
    else:
        frame = put_moustache(mst,frame,x,y,w,h)
        frame = put_hat(hat,frame,x,y,w,h)
    
      
cv2.imwrite("output.jpg",frame)  
        



# Display the resulting frame
# cv2.imshow('Video', frame)


# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

    

# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()