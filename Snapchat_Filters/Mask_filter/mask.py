import cv2

eye_detector=cv2.CascadeClassifier('/home/ebey/OpenCV/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('/home/ebey/OpenCV/data/haarcascades/haarcascade_frontalface_alt.xml')

#feed = cv2.imread('face.jpg')
mask = cv2.imread('mask.png',cv2.IMREAD_UNCHANGED)

cap=cv2.VideoCapture(0)

while(cap.isOpened()):
	_,feed=cap.read()
	gray = cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)
	feed = cv2.cvtColor(feed,cv2.COLOR_BGR2BGRA)
	faces=face_detector.detectMultiScale(gray,1.1,2)
	for (x,y,w,h) in faces:
		#cv2.rectangle(feed,(x,y),(x+w,y+h),(0,0,255),2)
		eyes = eye_detector.detectMultiScale(gray[y:y+h,x:x+w],1.1,2)
		if len(eyes)<2:
			break
		X1 = eyes[0,0] -50
		Y1 = eyes[0,1] -110
		X2 = eyes[1,0] + eyes[1,2] + 50
		Y2 = eyes[1,1] + eyes[1,3] + 20
		#cv2.rectangle(feed,(x+X1,y+Y1),(x+X2,y+Y2),(255,0,0),2)
		temp = cv2.resize(mask,(X2-X1+1,Y2-Y1+1))
		for i in range(y+Y1,y+Y2+1):
			for j in range(x+X1,x+X2+1):
				if temp[i-y-Y1,j-x-X1,3] != 0:
					feed[i,j] = temp[i-y-Y1,j-x-X1]
		#feed[y+Y1:y+Y2+1,x+X1:x+X2+1] = mask
	cv2.imshow("FEED",feed)
	k=cv2.waitKey(10)
	if k == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
