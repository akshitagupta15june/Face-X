# Overview

## *LBPH -> Local Binary Patterns Histogram*

It is based on local binary operator. It is widely used in facial recognition due to its computational simplicity and discriminative power. 

**It is very efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number.**
The steps involved to achieve this are:

* creating dataset
* face acquisition
* feature extraction
* classification

The LBPH algorithm is a part of opencv.


# Dependencies

    pip install numpy
    pip install opencv-python
    pip install skimage
# Images

<p><img src="https://miro.medium.com/max/667/1*J16_DKuSrnAH3WDdqwKeNA.png"><br>
Divide face images into R( for example R = 3 x 3 = 9 Regions) local regions to extract LBP histograms.</p>


<p><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Lbp_neighbors.svg/330px-Lbp_neighbors.svg.png" ><br>
Three neighborhood examples used to define a texture and calculate a local binary pattern (LBP).</p>

<p>
    <img src="https://camo.githubusercontent.com/74ab8b1f5eceb3b6e417f9393005319fb0f17c458c94fa8c0f7e550dad1ee22e/687474703a2f2f692e696d6775722e636f6d2f3342476b3133302e706e67"><br>
    After applying the LBP operation we extract the histograms of each image based on the number of grids (X and Y) passed by parameter. After extracting the histogram of each region, we concatenate all histograms and create a new one which will be used to represent the image.
</p>
    
