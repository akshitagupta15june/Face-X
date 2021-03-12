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

3) open your favourite text editor and run the above commands.

## Current supported algorithms

1) face_detection(using opencv)
	method : `facex.face_detect(img_path='your-img.jpg', methods='opencv')`

2) cartoonify(using opencv)
method : `facex.cartoonify(img_path='your-img.jpg', methods='opencv')`

3) blur background(using opencv)
method : `facex.blur_bg(img_path='your-img.jpg', methods='opencv')`

More algorithms shall be added soon as this is still in an experimental stage. 

## Pending Tasks

1) Release facex library V1.0
2) Refine the environment for easy access to the algorithms.
3) Make a **facex pip package**. 
4) Make a clear documentation.
5) Make clear documentation for the contributors to link the algorithms with the package. 
6) Add more algorithms.

## Contributions are welcome
Feel free to suggest any changes or fixes for the benefit of the package [here]().


