## 🎊What's new 🎊

Added, 

- Face mask detection
- sketch effect [A-kriti](https://github.com/akshitagupta15june/Face-X/tree/master/Awesome-face-operations/Ghost%20Image)
- mosiac effect [Sudip Ghosh](https://github.com/AdityaNikhil/Face-X/blob/master/Awesome-face-operations/Mosaic-Effect/Mosaic.py)
- ghost image [iaditichine](https://github.com/akshitagupta15june/Face-X/blob/master/Awesome-face-operations/Pencil%20Sketch/pencil_sketch_code.py)


## About

A unified library for **FaceX** to run all the FaceX algorithms using only one line of code. 

## Example
#### Running cartoonify using FaceX library
    from facex import FaceX 
    import cv2
    
    img = FaceX.cartoonify('your-img.jpg', method='opencv')
    cv2.imshow(img)
    cv2.waitkey()

Similarly we can run,

    FaceX.face_detect('your-img.jpg', method='opencv') #Face detection
    FaceX.face_mask('your-img.jpg', method='opencv')   #Face mask detection
    
    And many more....

## How to use

You can simply run the `demo.py` file to visualize some examples. Also check the below steps to run your own code,

1) Clone this repo.

2) cd `facex-library` from the command line.

3) open your favourite text editor and place it inside `facex-library` folder. 

4) Run the commands of [example](#Example) section.

## Current supported algorithms

### OpenCV

1) **face_detection**
	**method** : `facex.face_detect(img_path='your-img.jpg', methods='opencv')`

2) **cartoonify**
**method** : `facex.cartoonify(img_path='your-img.jpg', methods='opencv')`

3) **blur background**
**method** : `facex.blur_bg(img_path='your-img.jpg', methods='opencv')`

4) **Ghost image**
**method** : `facex.ghost_img(img_path='your-img.jpg', methods='opencv')`

5) **mosaic**
**method** : `facex.mosaic(img_path='your-img.jpg', x=219, y=61, w=460-219, h=412-61)`
Where, (x,y,w,h) are co-ordinates to apply mosaic effect on the image.

6) **Sketch**
**method** : `facex.sketch(img_path='your-img.jpg', methods='opencv')`

### Deep Learning

1) **Face Mask Detection**

**method** : 

```
facex.face_mask(image='your-img.jpg') (for image)
facex.face_mask(image='your-img.jpg') (for video)
```

More deep learning algorithms shall be added soon! (Stay put)

## Pending Tasks

1) Release facex library V1.0
2) Refine the environment for easy access to the algorithms.
3) Make a **facex pip package**. 
4) Make a clear documentation.
5) Make clear documentation for the contributors to link the algorithms with the package. 
6) Add more algorithms.

## Contributions are welcome
Feel free to suggest any changes or fixes for the benefit of the package [here](https://github.com/akshitagupta15june/Face-X/discussions/323).


