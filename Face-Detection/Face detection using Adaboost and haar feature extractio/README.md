# Face-detection-using-probabilistic-modelling

The goal of this project was to develop and train a face classification model using probabilistic modelling methods for the system. I used the FDDB dataset for this project. This has elliptical annotations to single out the face region in the image. I developed 5 models which are simple Gaussian distributed model, GMM (Gaussian Mixture Model), t-Distribution model, Factor Analyzer and mixture of t-Distributions. Except the simple Gaussian models, the parameters of all the other models have been calculated iteratively using a very popular algorithm called *Expectation Maximization* (EM).

* Dataset preparation --> Extracted the raw images from the FDDB dataset along with the elliptical annotations given in the txt files along with the dataset. I extracted the face patch (with some background) from the image and reduced the patch size to 20X20. The non face patch was extracted from the centre at the bottom where I was sure there won't be an entire face. From the center bottom, I extracted a 100X100 patch and reduced it to 20X20. 

* Image pre-processing --> Before feeding the images into respective models, there was some more preprocessing required. The images had the dimensions 20X20X3 = 1200 pixels. I had to make an array of size number_of_training_imagesX1200 with the each as a flattened vector. Since the array dimension is too big, the calulation of the PDF in most distributions can result in overflow. Hence I used a method called *principal component analysis* (PCA) which has in-built methods in Python. I chose 23 features out of the 1200 long feature vector of the flattened image. 

* Running the models --> The main driver code for all the codes is written in the driver.py file. The model to be trained can be specifed as a command line arguement. To get the possible arugements, on on the terminal type *driver.py help*.

# Face-detection-using-Adaboost-and-haar-feature-extraction

This project was to build a face detection module by extracting Haar-like features from the image boost these features using the adaptive boosting algorithm (Adaboost). Adaboost is a congregation of weak classifiers (each having accuracy better than a random guess, greater than 0.5) to give one strong classifier. The strong classifier finally choses the class of the image by taking into account the weights and polarity (correct/incorrect classification).

* Haar-like feature extraction --> For this I've used the code on the sklearn ofiicial website where they explain how these features can be extracted and drawn on an image.
