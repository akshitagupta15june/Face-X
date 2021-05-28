import cv2
import numpy as np
import dlib

predictor_path = "./shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture("input.mp4") #loading the video, for webcam, replace "input.mp4" with 0

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(predictor_path)


while True:
	ret, frame = cam.read()
	if ret:
		#detect face
		dets = detector(frame[...,[2, 1, 0]], 1) #frame[...] -> convert to RGB from BGR
		#returns a list of detected faces with the coordinates of bounding boxes.
		if len(dets) < 1:
			continue
		d = dets[0] #take the first face
		shape = predictor(frame[...,[2, 1, 0]], d) #68 landmarks
		points = np.empty((12, 2), dtype="int")
		for i, ix in enumerate(range(49, 61)):
			points[i][0], points[i][1] = shape.part(ix).x, shape.part(ix).y
			#cv2.circle(frame, (points[i][0], points[i][1]), 2, (255,255,255), -1) #visualize point

		x, y, w, h = cv2.boundingRect(points) #get the bounding box given the points
		center = (x+w//2, y+h//2)
		#preprocessing
		diag = (w**2 + h**2)**0.5
		ratio = int(diag*1/4)
		x, y, w, h = x - ratio, y - ratio, w + ratio, h + ratio
		cropped_lips = frame[y:y+h, x:x+w] #cropping lips
		mask = np.zeros(cropped_lips.shape[:-1])
		cv2.fillConvexPoly(mask, points - np.array([x, y]), 255)
		cropped_lips = np.expand_dims(mask, 2).astype("bool")  * cropped_lips #mask out pixel of lips
		resized_lips = cv2.resize(cropped_lips, None, fx=2, fy=2) #resizing lips

		Real_thing = np.uint8(255 * (resized_lips > [0,0,0])) #creating a mask for seamless clone

		frame = cv2.seamlessClone(np.uint8(resized_lips), frame, Real_thing, center, cv2.NORMAL_CLONE) #seamless cloning-> desi bhasha me copy krdo
		#cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)#visualize rectangle
		cv2.imshow("Camera Window", frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break

cam.release()
cv2.destroyAllWindows()
