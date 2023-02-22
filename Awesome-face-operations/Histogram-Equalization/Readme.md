# Histogram-Equalization
Histogram equalization is a widely used technique in image processing to improve the contrast and enhance the details of an image. OpenCV, an open-source computer vision library, provides a simple and efficient method to perform histogram equalization on digital images. The process involves converting the image to grayscale and computing its histogram, which represents the frequency of pixel values in the image. The cumulative distribution function is then calculated, and the pixel values in the image are remapped according to this function to obtain a new image with improved contrast. OpenCV provides several functions to implement histogram equalization, including "equalizeHist()" and "CLAHE" (Contrast Limited Adaptive Histogram Equalization). Histogram equalization using OpenCV is a powerful tool for image enhancement and has various applications in fields such as medical imaging, remote sensing, and computer vision.
# Packages Used 
OpenCV
# Steps Involved
1. Importing Required Libraries
2. Loading Image Data
3. Coverting RGB image to gray Scale Images
4. Apply Histogram Equalization using OpenCV
5. Convert Back the grayscale image to RGB image
6. Display both the input and output images
# How to RUN
You can run the jupyter notebook file (Histogram-Equalization.ipynb) remotely using Google colab by uploading the file into remote directories and can be run locally using Anaconda.
# Sample Input and Output
![Screenshot from 2023-02-21 16-16-49](https://user-images.githubusercontent.com/86817867/220663890-8172ea20-a7ed-4f51-a815-1cdb198301c7.png)
