# Deep Dense Face Detection

A convolutional network for face detection. This work is based on "Multi-view Face Detection Using Deep Convolutional Neural Networks" by Farfade et al., although some implementation details differ.

![Faces Screenshot](./images/faces.png)

Following scripts are provided:

- scripts/download_data.py
- scripts/train_model.py
- scripts/visualization.py
- scripts/accuracy.py

### scripts/download_data.py

Downloads [Celeb Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and performs simple preprocessing. Please note that Celeb Dataset comes with its own license that you need to abide by to use it. With a little bit of effort you could adapt this code to work with a different dataset.

### scripts/train_model.py

Trains the network.

### scripts/visualization.py

Provides a few handy functions for visualizing data batching results, predictions for image crops, face detections and face heatmaps.

### scripts/accuracy.py

Provides functions to test accuracy of the trained network. OpenCV `CascadeClassifier` can also be tested for comparison.





