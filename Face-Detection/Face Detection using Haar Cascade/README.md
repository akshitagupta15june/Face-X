<h1>Face Detection using Harr Cascade</h1>
<p>Harr Cascade is a powerful detection algorithm to detect faces,eyes,lips etc.Harr cascade was proposed by Viola and Jones in the research paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. Haar features are used to identify edges,lines in the faces(ie. change in intensities of pixels.)</p>

![image](https://user-images.githubusercontent.com/51399803/115983060-a6743380-a5bc-11eb-9436-dfaef0853552.png)

<p>The haar feature continuously traverses from the top left of the image to the bottom right to search for the particular feature.
there’s a set of features which would capture certain facial structures like eyebrows or the bridge between both the eyes, or the lips etc. But originally the feature set was not limited to this. The feature set had an approx. of 180,000 of them.A boosting technique known as AdaBoost is used for feature selection as some features are irrelevant. Using AdaBoost the feature set is reduced to 6000 features.</p>
<p> Various open source trained models are available to use haar cascade, the OpenCV implementation is widely used</p>
<h3>How to use?</p>

### Step 1: Make sure to install OpenCV
```
    pip3 install opencv-python
```
### Step 2: The haarcascade_frontalface_default.xml file contains pre trained classifiers for face detection and should be downloaded and save in the working directory.</p>
```
    # Load the cascade
     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
### Step 3: Read Input Image 
```
    # Read the input image
     img = cv2.imread('test.jpg')
    # Convert into grayscale
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### Step 4: Detect faces
```
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
```

### Step 5: Draw rectangle around the faces
```
    for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```
### Step 6: Display the output

```
    cv2.imshow('img', img)
    cv2.waitKey()
```
