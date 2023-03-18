import cv2
import numpy as np
import dlib
import joblib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model = joblib.load("temperature_prediction_model.joblib")

MIN_TEMP = 34.0
MAX_TEMP = 42.0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray, 0)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
        temperature = model.predict(landmarks.reshape(1, -1))[0]
        temperature = MIN_TEMP + (MAX_TEMP - MIN_TEMP) * (temperature + 1) / 2.0
        
        cv2.putText(frame, f"Temperature: {temperature:.2f} C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

        for x, y in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    cv2.imshow("Temperature Prediction", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()