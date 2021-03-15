import cv2
frame_cap = cv2.VideoCapture(0) #Capturing each Frames from the Camera
while(True):
    ret, frame = frame_cap.read() #Reading the Captured Frames
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Applying gray filter
    blur_img = cv2.medianBlur(gray_img, 5) #Applying Median Blur
    edges = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250) #Applying Bilateral Filter
    cartoon_img = cv2.bitwise_and(color, color, mask=edges) # Bit wise And operation on color and edges images
    cv2.imshow("Cartoon Image", cartoon_img) #Displaying the cartoonified Image
    if cv2.waitKey(1) & 0xFF == ord(' '): #Press space bar to exit
        break
frame_cap.release()
cv2.destroyAllWindows()