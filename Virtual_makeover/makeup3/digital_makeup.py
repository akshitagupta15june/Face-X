from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("hello4.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(59, 68, 75))
    d.polygon(face_landmarks['right_eyebrow'], fill=(59, 68, 75))
    d.line(face_landmarks['left_eyebrow'], fill=(240, 248, 255, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(240, 248, 255, 150), width=5)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(0, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(240, 248, 255), width=4)
    d.line(face_landmarks['bottom_lip'], fill=(240, 248, 255), width=4)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(0,0,0,56))
    d.polygon(face_landmarks['right_eye'], fill=(0,0,0,56))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(240, 248, 255), width=4)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(240, 248, 255), width=4)

    d.polygon(face_landmarks['chin'], fill=(0,0,0,0))
    d.line(face_landmarks['chin'], fill=(0, 0, 0,0), width=6)

    d.polygon(face_landmarks['nose_tip'], fill=(59, 68, 75))
    d.line(face_landmarks['nose_tip'], fill=(59, 68, 75), width=5)

    pil_image.show()