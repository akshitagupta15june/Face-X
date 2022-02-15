import numpy as np
from PIL import Image
import os

# Load Images
def load(image_list):
    image_mat_array =[]
    for image in image_list:
        img = Image.open(image).convert("L")
        arr = np.array(img)
        image_mat_array.append(arr)

    return image_mat_array   


# Save a numpy array to an image
def save_image(image_obj,filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    height, width = image_obj.shape
    grayLevelImage = Image.new("L",(width,height))
    for i in range(0,width):
        for j in range(0,height):
            grayLevelImage.putpixel((i,j),int(image_obj[j][i]))
    grayLevelImage.save(filename+".png")
    

#Load Training assets
training_images_list = ["subject01.normal", "subject02.normal", "subject03.normal", 
                   "subject07.normal", "subject10.normal", "subject11.normal", 
                   "subject14.normal", "subject15.normal"]
training_images = ["Face dataset/"+ x+".jpg" for x in training_images_list]
training_images_matrices = load(training_images)



# Stacking row in each matrix to form a column vaector 
def form_column_vectors(training_images_matrices):
    training_column_vectors = [x.flatten() for x in training_images_matrices]
    return training_column_vectors


# Forming M Face vectors
training_faces = form_column_vectors(training_images_matrices)
# create a Matrix
training_faces  = np.vstack(training_faces)


training_faces = np.transpose(training_faces)


# Calculating the mean face, taking mean across column
mean_face = training_faces.mean(1)
size=training_images_matrices[0].shape   
mean_face_image = mean_face.reshape(size)
save_image(mean_face_image,"Mean Face/mean_face")


# Function to subtract mean face from each face
def subtract_mean_from_each_face(mean_face, faces):
    size = faces.shape[1]
    new_faces = np.zeros(faces.shape)
    for i in range(0,size):
        new_faces[:,i] = np.subtract(faces[:,i], mean_face)
    return new_faces


adjusted_training_faces = subtract_mean_from_each_face(mean_face, training_faces)


## -------------Eigenfaces: Training------------------------

# calculating eigenfaces of the covariance matrix C= A.A^T
# going by alternate method of Singular Value decomposition
# Find eigenvalues of L = Transpose(A).A

A = adjusted_training_faces
A_t = np.transpose(A)
L  = np.dot(A_t,A)
L.shape
L_eig_vals, L_eig_vecs = np.linalg.eig(L)
V=L_eig_vecs
eigen_face_space = np.dot(A,V)


print("Eigen values:")
print(L_eig_vals)


# Projecting each training face to Eigen_face_space
eigen_face_space_t= np.transpose(eigen_face_space)
projected_training_faces = np.dot(eigen_face_space_t,adjusted_training_faces)

#Load test assets
from os import listdir
from os.path import isfile, join
testImagesList = ['subject01.centerlight.jpg','subject01.happy.jpg',
                  'subject01.normal.jpg','subject02.normal.jpg','subject03.normal.jpg',
                  'subject07.centerlight.jpg','subject07.happy.jpg','subject07.normal.jpg',
                  'subject10.normal.jpg','subject11.centerlight.jpg','subject11.happy.jpg',
                  'subject11.normal.jpg','subject12.normal.jpg','subject14.happy.jpg',
                  'subject14.normal.jpg','subject14.sad.jpg','subject15.normal.jpg','apple1_gray.jpg']



test_images_list = testImagesList
test_images = ["Face dataset/"+ x for x in test_images_list]

test_images_matrices = load(test_images)
# Forming M Face vectors
test_faces = form_column_vectors(test_images_matrices)
# create a Matrix
test_faces  = np.vstack(test_faces)
test_faces = test_faces.T


# Saving Eigen Faces Images
for i in range(0,eigen_face_space.shape[1]):
    x=eigen_face_space[:,i]
    image_obj =x.reshape(test_images_matrices[0].shape)
    save_image(image_obj,"Eigen Faces/eigenface{}".format(i))


#Recognition of test assets in training assets
# 1. Subtract mean face from the test assets
adjusted_test_faces = subtract_mean_from_each_face(mean_face, test_faces)


# Subtract mean face from each test face
for i in range(0,adjusted_test_faces.shape[1]):
    x=adjusted_test_faces[:,i]
    image_obj = x.reshape(test_images_matrices[0].shape)
    save_image(image_obj,"Mean Subtracted Test Faces/"+testImagesList[i])


# Transpose(U).test_faces for calculating projection of test faces onto face space
eigen_face_space_t = np.transpose(eigen_face_space)
projected_test_faces = np.dot(eigen_face_space_t,adjusted_test_faces)


print("\nPCA Coefficents for Test Images:\n")
for i in range(0,projected_test_faces.shape[1]):
    print("\nPCA Coeffivents for {} :".format(test_images_list[i]))
    print(projected_test_faces[:,i])
    print("\n")
    

print("\nPCA Coefficents for Training Images:\n")
for i in range(0,projected_training_faces.shape[1]):
    print("\nPCA Coeffivents for {} :".format(training_images_list[i]))
    print(projected_training_faces[:,i])
    print("\n")


# Reconstruct input face image from eigenfaces
reconstructed_test_faces = np.dot(eigen_face_space,projected_test_faces)


# Save reconstructed test assets
for i in range(0,reconstructed_test_faces.shape[1]):
    x=reconstructed_test_faces[:,i]
    image_obj = x.reshape(test_images_matrices[0].shape)
    save_image(image_obj,"reconstructed_test_faces/{}".format(test_images_list[i]))


from numpy import  linalg as LA
# Compute distance between input face and training assets in the face space
temp_projected_test_faces = projected_test_faces.T
temp_projected_training_faces = projected_training_faces.T
min_distance_list=[]
distance_mat = np.zeros(projected_test_faces.shape)
for i in range(0,temp_projected_test_faces.shape[0]):
    distances =[LA.norm(np.subtract(temp_projected_test_faces[i], x)) for x in temp_projected_training_faces]
    distance_mat[:,i] = distances
    min_dist = min(distances)
    min_index = distances.index(min_dist)
    min_distance_list.append([min_dist,min_index])
  

# Computing Distance Between the Test Image and it's reconstructed image in the Face Space 
face_space_dist_list=[]
for i in range(0,reconstructed_test_faces.shape[1]):
    distance = LA.norm(reconstructed_test_faces[:,i] - adjusted_test_faces[:,i])
    face_space_dist_list.append(distance)
                                     
# Threshold  the distance to classify the test image as an Identical or Unknown Face
min_distance_threshold = 8.6e+7
# Print Distance of each test face from each training face in face face
# Print closest training face to the projected test face
# Classify the  Face. as Identical or Unknown
for i in range(0,distance_mat.shape[1]):
    print("\nTest Face: {}".format(test_images_list[i]))
    print("Training Face\t\t\t Distance")
    print("_______________________________________________")
    for j in range(0,distance_mat.shape[0]):
        print(training_images_list[j],"\t\t",distance_mat[j][i])
#     print("Test Face {}- {} has minimum distance with: Face {}: {}, distance:  {}".format(i+1,testImagesList[i],
#                                                                                           min_distance_list[i][1]+1,
#                                                                                           training_images[min_distance_list[i][1]],
#                                                                                           min_distance_list[i][0]))
    if(min_distance_list[i][0] < min_distance_threshold):
        print("Identity of Face: {}".format(training_images_list[min_distance_list[i][1]]))
    else:
        print("Unknown Face")


# Print the distance Between the Test Image and it's reconstructed image in the Face Space 
# Classify the test image as  a Face or Non Face Image based on the threshold set
print("Test Face\t\tDistance From Face Space\t\t\t Face classification")
print("____________________________________________________________________________________")
# Threshold to distinguish between a Face and non face image
threshold = 7e+12
for i in range(0,len(face_space_dist_list)):
        if(face_space_dist_list[i]>threshold):
            face_class ="Non Face "
        else:
            face_class ="Face"
            
        print(test_images_list[i],"\t\t\t",face_space_dist_list[i],"\t\t\t", face_class)
        

