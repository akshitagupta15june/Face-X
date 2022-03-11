# Differentiate between Cartoon and Human Faces

One way to discriminate between cartoon and natural scene images is to compare a given image to its "smoothed" self. The motivation behind this is that a "smoothed" cartoon image statistically will not change much, where as a natural scene image will. In other words, take an image, cartoonify (i.e. smooth) it and subtract the result from the original.

This difference (i.e. taking its mean value) will give the level of change caused by the smoothing. The index should be high for non-smooth original (natural scene) images and low for smooth original (cartoony) images.

Smoothing/Cartoonifying is done with bilateral filtering.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/image.webp?raw=true" width="400" title="Original Image">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/new.jpeg?raw=true" width="400" alt="accessibility text" title="Smoothed Image">
</p>


As for subtracting the cartoonyfied image from the original, it is done with the Hue channel of the HSV images. This means, first the images are converted from BGR to HSV and then subtracted.

<p align="center">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/img_hsv.jpeg" width="400" title="HSV of Original Image">
  <img src="https://github.com/shireenchand/Face-X/blob/cartoon/Cartoonify-Image/Differentiate%20between%20Human%20and%20Cartoon%20Faces/Media/blurred_hsv.jpeg" width="400" alt="accessibility text" title="HSV of Smoothed Image">
</p>

## Dependencies

- OpenCV - This is used to perform different operations like blurring on the given image.
- Numpy - This is used for calculating the mean of subtraction.

## Setup

- Fork the repository - Creates a copy of this project in your github.

- Clone the repository to your local machine using 
```
git clone https://github.com/akshitagupta15june/Face-X.git
```
- Use a virtual environment to keep the all dependencies in a separate enviroment for example - conda, virtualenv, pipenv, etc.

- Navigate to the Differentiate between Human and Cartoon Faces inside Cartoonify Image Folder using
```
cd Cartoonify-Image/Differentiate\ between\ Human\ and\ Cartoon\ Faces
```
  
- Install the dependencies either by using the below pip commands or by using the requirements.txt file given.

- By using pip commands
```
pip install numpy
```
```
pip install opencv-python
```

- By using requirements.txt
```
pip install -r requirements.txt
```

- Run the cartoon.py script using
```
python3 cartoon.py
```

## Want to Contribute?

- Follow the steps for Setup

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
git add.
```

- Commit Message
```
git commit -m "Enter message"
```

- Push your code
```
git push
```

- Make Pull request with the Master branch of akshitagupta15june/Face-X repo.

- Wait for reviewers to review your PR
