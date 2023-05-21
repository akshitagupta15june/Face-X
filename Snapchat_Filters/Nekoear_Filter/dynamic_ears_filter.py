import cv2
import dlib 
import numpy as np

camera_video = cv2.VideoCapture(0)
ear=cv2.imread("./Assets/earimg.png")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./Model/shape_predictor_68_face_landmarks.dat")

last_left_eye_height = 0
last_right_eye_height = 0
left_eye_height_ls = []
right_eye_height_ls = []
i = 0
height_scale = 1.3
while True:
    
    _, frame = camera_video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Clear the eye height detection memory for every 3 frames
    if i%3 == 0:
        left_eye_height_ls=[]
        right_eye_height_ls=[]
        
    # Applying ear filter on all faces detected by the camera
    for face in faces:

        # Identifying facial landmarks 
        landmarks = predictor(gray, face)
        top = (landmarks.part(19).x, landmarks.part(19).y)
        low = (landmarks.part(33).x, landmarks.part(33).y)
        left = (landmarks.part(0).x, landmarks.part(19).y)
        right = (landmarks.part(16).x, landmarks.part(24).y)
        left_eyetop = (landmarks.part(38).x, landmarks.part(38).y)
        left_eyelow = (landmarks.part(41).x, landmarks.part(41).y)
        right_eyetop = (landmarks.part(43).x, landmarks.part(43).y)
        right_eyelow = (landmarks.part(46).x, landmarks.part(46).y)
        left_eye_height_ls.append(left_eyelow[1] - left_eyetop[1])
        right_eye_height_ls.append(right_eyelow[1] - right_eyetop[1])
        
        # Determine whether to scale up or scale down the size of ear based on blink-eye detection (for every 3 frames)
        if i%3 == 0:
            if np.mean(left_eye_height_ls)>=last_left_eye_height or np.mean(right_eye_height_ls)>=last_right_eye_height:
                height_scale = 1.3
            else:
                height_scale = 1.1
            last_left_eye_height = np.mean(left_eye_height_ls)
            last_right_eye_height = np.mean(right_eye_height_ls)
        
        # Resizing the ear
        ear_width = int(1.3*abs(right[0] - left[0]))
        ear_height = int(height_scale*abs(top[1] - low[1])) 
        resized_ear = cv2.resize(ear, (ear_width, ear_height)) 

        # Identifying ear position on face
        top_left = (int(left[0] - abs(top[0]-left[0])/2), int(top[1] - ear_height * 6 / 7))

        # Determine whether to replace the ear image on original frame based on the position
        if (top_left[0] + ear_width) < camera_video.get(3) and (top_left[1] + ear_height) < camera_video.get(4) and top_left[0] > 0 and top_left[1] > 0 :
            face_area = frame[top_left[1] : top_left[1] + ear_height, top_left[0] : top_left[0] + ear_width]

            # Graying and thresholding the ear
            gray_ear = cv2.cvtColor(resized_ear, cv2.COLOR_BGR2GRAY)
            _, thresh_ear = cv2.threshold(gray_ear, 25, 255, cv2.THRESH_BINARY_INV)

            # Adding the ear on the face 
            face_area_no_face = cv2.bitwise_and(face_area, face_area, mask = thresh_ear)
            final_mask = cv2.add(face_area_no_face, resized_ear)
            frame[top_left[1] : top_left[1] + ear_height, top_left[0] : top_left[0] + ear_width] = final_mask

    cv2.imshow("Frame", frame)

    i += 1
    # Breaking the loop if 'ESC' is pressed
    key = cv2.waitKey(1) & 0xFF    
    if(key == 27):
        break

# Releasing the VideoCapture Object and closing the windows.                  
camera_video.release()
cv2.destroyAllWindows()