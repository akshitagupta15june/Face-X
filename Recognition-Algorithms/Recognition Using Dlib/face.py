#Import face_recognition package
import face_recognition

#Loading assets from dataset and getting the face encoding array
image1 = face_recognition.load_image_file("images/Bill_Gates.jpg")
image2 = face_recognition.load_image_file("images/Mark_Zuckerberg.jpg")
img1_encoding = face_recognition.face_encodings(image1)[0]
img2_encoding = face_recognition.face_encodings(image2)[0]

#Storing details of person along with face encodings in a database/data structure
dic = {"Bill Gates" : img1_encoding, "Mark Zuckerberg": img2_encoding}

#Taking new sample image and finding it's face encoding array
unknown_image = face_recognition.load_image_file("Unknown.jpg")
unknown_img_encoding = face_recognition.face_encodings(unknown_image)[0]

#Comparing face encoding of unknown image with all the assets in dataset
for i in dic:
    result = face_recognition.compare_faces([dic[i]], unknown_img_encoding)
    #If face encoding matches then result is true and we can fetch name of Person
    if(result[0]==True):
        print("Hey! this is",i)
        break
else:
    print("Oops...Don't Know who is this.")

'''
Output:
Hey! This is Bill Gates
'''