from deepface import DeepFace
#Face Detection
detector_backends=['opencv', 'retinaface','mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface']
faces=DeepFace.extract_faces("img1.jpg",detector_backend=detector_backends[2],align=True) #Image Path, #Specify any detector backend according to your choice and use case
#align == true aligns the face to straight if it is inverted or rotated or have any distortions
print(faces) # prints the coordinates of the found faces with vectors

#Face Recognition
# 1.Verify Method
models=[ 'VGG-Face', 'Facenet', 'Facenet512','OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'GhostFaceNet']
result=DeepFace.verify("img1.jpg","img2.jpg",model_name=models[2]) #you can use detector backend here as well
print(result)

# 2.Find Method
results=DeepFace.find("img1.jpg",db_path="db",) #similar to verify but finds matches over a folder and same parameters can be specified such as backend_detectors, align, threshold
print(results)

#Face Attribute Analysis
attributes=DeepFace.analyze("img2") #Reurns the age, gender, race and emotion of the person in the image.
print(attributes)

#<-----------For MORE INFORMATION VISIT DEEPFACE LIBRARY ITSELF------------------->
