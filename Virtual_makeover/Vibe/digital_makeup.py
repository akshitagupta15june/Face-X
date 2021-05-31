from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("style-5.png")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Make the eyebrows into a nightmare
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 0, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 0, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    # Gloss the lips
    d.polygon(face_landmarks['top_lip'], fill=(0, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(0, 0, 0, 6), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(0, 0, 0, 6), width=8)

    # Sparkle the eyes
    d.polygon(face_landmarks['left_eye'], fill=(0,0,0))
    d.polygon(face_landmarks['right_eye'], fill=(0,0,0))

    # Apply some eyeliner
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 0), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 0), width=6)

    d.polygon(face_landmarks['chin'], fill=(0,0,0,0))
    d.line(face_landmarks['chin'], fill=(0, 0, 0,0), width=6)

    d.polygon(face_landmarks['nose_tip'], fill=(0,0,0))
    d.line(face_landmarks['nose_tip'], fill=(0, 0, 0), width=5)

    pil_image.show()