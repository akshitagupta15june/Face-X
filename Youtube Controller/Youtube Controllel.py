import cv2
import pyautogui

cap = cv2.VideoCapture(0)
hand_cascade = cv2.CascadeClassifier('hand.xml')

while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (10, 10), (60, 60), (0, 255, 0), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (hx, hy, hw, hh) in hands:
        if hx < 10:
            pyautogui.press("right")
        elif hx > 60:
            pyautogui.press("space")
    
    cv2.imshow("Video Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()