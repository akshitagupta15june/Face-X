[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)](https://www.python.org/downloads/release/python-360/) 
# Cartoonify_reality

Even the basics of image processing if done properly can be handy which otherwise would require a machine learning model.This project is one of such inspiration which **cartoonizes** images and videos using only core **opencv filters** and functions.It also uses K-means clustering algorithm to compress the image. This clustering gives it the basic cartoonish tinge it requires.

**Algorithm**- K_Means Clustering

**Filters**-Bialateral Filter, Contours, erode, Canny(edge detection)


### Prerequisites

What things you need to install the software and how to install them

```
scipy 
numpy 
cv2
```

## Getting Started
Download a python interpeter preferable a version beyond 3.0. Install the prerequisute libraries given above. Run vid.py file to cartonify your webcamp feed. Uncomment the last 2 lines of cartoonize.py and run to cartoonify an image.

```
$vid.py     
                    
$cartoonize.py
```
## Original Image
![The input Image to cartoonize.py]()

## Cartoon Output
![The Output Image of cartoonize.py]()

## Built With
* [python](https://www.python.org/) - The software used

## Documentation
The entire documentation and explanation of code as well as concepts can be found in this article: https://iot4beginners.com/cartoonize-reality-with-opencv-and-raspberry-pi/