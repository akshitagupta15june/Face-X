# Cartoonify Image 
Currently many of us wants to have our photo to be cartoonify, and we try to use the professional cartoonizer application available in market and most of them are not freeware. In order to have basic cartoonifying effects, we just need the bilateral filter, some edge detection mechanism and some filters. 

The bilateral filter is use to reduce the color palette, which is the most important task for cartoonifying the image and have a look like cartoon.And then comes the edge detection to produce the bold silhouettes.

<p align="center">
  <img src="logo\original.jpeg" width="250" title="hover text">
  <img src="logo\cartoonified.png" width="250" alt="accessibility text">
</p>

## Dependencies:

The Dependencies used are:

- Opencv :It provides the tool for applying computer vison techniques on the image.
- Numpy :Images are stored and processed as numbers, These are taken as arrays.

## How to create a Cartoonify Image?
- Cartoonify Images can be created using the opencv library.
- OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. It is mainly aimed at real-time computer vision and image processing. It is used to perform different operations on images which transform them using different techniques. Majorly supports all lannguages like Python, C++,Android, Java, etc.
- In Opencv there are various functions like bilateral filters, median blur, adaptive thresholding which help in cartoonify the image.

## Algorithm
- Firstly importing the cv2 and numpy library.
- Now applying the bilateral filter to reduce the color palette of the image.
- Covert the actual image to grayscale.
- Apply the median blur to reduce the image noise in the grayscale image.
- Create an edge mask from the grayscale image using adaptive thresholding.
- Finally combine the color image produced from step 1 with edge mask produced from step 4.


## Want to contribute in Cartoonify Images?
 You can refer to CONTRIBUTING.md (`https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md`)
#### Or follow the below steps - 
- Fork this repository `https://github.com/akshitagupta15june/Face-X`. 
- Clone the forked repository
``` 
git clone https://github.com/<your-username>/<repo-name> 
```
- Create a Virtual Environment(that can fulfill the required dependencies)
```
- python -m venv env
- source env/bin/activate (Linux)
- env\Scripts\activate (Windows)
```
- Install dependencies
- Go to project directory
``` 
cd Cartoonify Image
```
- Make a new branch
```
git branch < YOUR_USERNAME >
```
- Switch to Development Branch
```
git checkout < YOURUSERNAME >
```
- Make a folder and add your code file and a readme file with screenshots.
- Add your files or changes to staging area
```
git add .
```
- Commit message
```
git commit -m "Enter message"
```
- Push your code
``` 
git push
```
- Make Pull request with the Master branch of `akshitagupta15june/Face-X` repo.
- Wait for reviewers to review your PR
