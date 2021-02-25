import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
	ret, frame = camera.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:
		face = frame[y:y + h, x:x + w]
		frame = cv2.blur(frame, ksize = (10, 10))
		frame[y:y + h, x:x + w] = face

	cv2.imshow("frame", frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()