from facex import FaceX
import cv2

img = cv2.imread("3.jpg")
facex = FaceX()

faces = facex.face_detection(img)

for face in faces:
    cartoon_face = facex.cartoonify(face)
    img[face[1]:face[3], face[0]:face[2]] = cartoon_face

cv2.imshow("Cartoonified Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()