
# GUI for face recognition using LBPH Algo
**Face Recognition:** with the facial images already extracted, cropped, resized, and usually converted to grayscale, the face recognition algorithm is responsible for finding characteristics that best describe the image.
## Local Binary Patterns Histograms
The **Local Binary Pattern Histogram(LBPH)** algorithm is a simple solution to the face recognition problem, which can recognize both front face and side face.
- LBPH is one of the easiest face recognition algorithms.
- It can represent local features in the images.
- It is possible to get great results (mainly in a controlled environment).
- It is robust against monotonic grayscale transformations.
- It is provided by the OpenCV library (Open Source Computer Vision Library).
## Steps of LBPH algorithm
 ### 1. Parameters:
- Radius
- Neighbours
- Grid X
- Grid Y
### 2. Training the Algorithm:
we need to use a dataset with the facial images of the people we want to recognize. We need to also set an ID (it may be a number or the name of the person) for each image, so the algorithm will use this information to recognize an input image and give you an output. Images of the same person must have the same ID.
### 3. Applying the LBP operation
The first computational step of the LBPH is to create an intermediate image that describes the original image in a better way, by highlighting the facial characteristics. To do so, the algorithm uses a concept of a sliding window, based on the parameters radius and neighbors.
### 4. Extracting the Histograms
Now, using the image generated in the last step, we can use the Grid X and Grid Y parameters to divide the image into multiple grids, as can be seen in the following image:
![image](https://user-images.githubusercontent.com/78999467/111055080-69cffb00-849a-11eb-9695-d142d42bd77a.png)
### 5. Performing the face recognition
In this step, the algorithm is already trained. Each histogram created is used to represent each image from the training dataset. So, given an input image, we perform the steps again for this new image and creates a histogram that represents the image.
