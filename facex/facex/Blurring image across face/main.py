import cv2

# camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("face.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
		face = img[y:y + h, x:x + w]
		frame = cv2.blur(img, ksize = (10, 10))
		frame[y:y + h, x:x + w] = face

cv2.imshow('face',img)
cv2.waitKey(0)
