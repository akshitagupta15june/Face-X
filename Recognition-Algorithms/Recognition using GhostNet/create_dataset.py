import cv2
import os
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    # Crop all faces found
    cropped_face = 0
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y : y + h + 50, x : x + w + 50]
    return cropped_face


def check_make_dir(name, parent="train"):
    # Function checks if required directory exists
    # If not, make directory
    if not os.path.exists("Datasets/{}/{}".format(parent, name)):
        path = os.path.join(os.getcwd(), "Datasets", parent, name)
        os.makedirs(path)


def write_show_img(name, count, img):
    # Function puts image in train or val directories
    # And displays image with image count
    if count <= 400:
        file_name_path = "Datasets/train/{}/".format(name) + str(count) + ".jpg"
    else:
        file_name_path = "Datasets/val/{}/".format(name) + str(count) + ".jpg"
    cv2.imwrite(file_name_path, img)
    cv2.putText(
        img,
        str(count),
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Face Cropper", img)


# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    name = input("Enter candidate name:")
    check_make_dir(name)
    check_make_dir(name, "val")
    count = 0
    if name != "Blank":
        # Create dataset for unique faces
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (400, 400))
                write_show_img(name, count, face)
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or count == 450:
                break
    else:
        # For ground truth
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is None:
                count += 1
                bg = cv2.resize(frame, (400, 400))
                write_show_img(name, count, bg)
            else:
                print("Face found")
                pass
            if cv2.waitKey(1) == 13 or count == 450:
                break
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
