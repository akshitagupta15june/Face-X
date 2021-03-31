## About
A unified library for **FaceX** to run all the FaceX algorithms using only one line of code. 

## Example
### Running cartoonify using FaceX library
    import facex 
    import cv2
    
    img = facex.cartoonify('your-img.jpg', method='opencv')
    cv2.imshow(img)
    cv2.waitkey()

Similarly we can run,

    facex.face_detect('your-img.jpg', method='opencv') #Face detection
    facex.face_mask('your-img.jpg', method='opencv')   #Face mask detection
    
    And many more....

## How to use

1) Clone this repo.

2) cd `facex` from the command line.

3) open your favourite text editor and place it inside facex folder. 

4) Run the commands of [example](#Example) section.

## Current supported algorithms

### OpenCV

1) face_detection
	method : `facex.face_detect(img_path='your-img.jpg', methods='opencv')`

2) cartoonify
method : `facex.cartoonify(img_path='your-img.jpg', methods='opencv')`

3) blur background
method : `facex.blur_bg(img_path='your-img.jpg', methods='opencv')`

4) Ghost image
method : `facex.ghost_img(img_path='your-img.jpg', methods='opencv')`

5) mosaic
method : `facex.mosaic(img_path='your-img.jpg', x=219, y=61, w=460-219, h=412-61)`
Where, (x,y,w,h) are co-ordinates to apply mosaic effect on the image.

6) Sketch
method : `facex.sketch(img_path='your-img.jpg', methods='opencv')`

### Deep Learning

Deep learning algorithms shall be added soon! (Stay put)

## Pending Tasks

1) Release facex library V1.0
2) Refine the environment for easy access to the algorithms.
3) Make a **facex pip package**. 
4) Make a clear documentation.
5) Make clear documentation for the contributors to link the algorithms with the package. 
6) Add more algorithms.

## Contributions are welcome
Feel free to suggest any changes or fixes for the benefit of the package [here](https://github.com/akshitagupta15june/Face-X/discussions/323).


