# Face-Recognition-with-VGGFace2

Face Recognition has two main modes:

1) Face Identification -
In this mode, the input face is mapped against the database of many faces and give output as the probability likelihood of that input face belonging to the faces in the database. Dataset used here was MS-Celeb-1M dataset.

2) Face Verification - 
In this mode, the input face image is compared with the known face image (i.e., image of an authorized user) and checked whether they match. If Matched, the input face image is of an authorized user. If not matched, then the input face image is of an Unauthorized user.

MTCNN model is used for face detection. The VGGFace and VGGFace2 model are models developed for face recognition, and developed by researchers at the Visual Geometry Group (VGG) at the University of Oxford.
