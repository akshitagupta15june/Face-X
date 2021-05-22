# image_to_numpy

Load an image file into a numpy array - while automatically rotating the image based on Exif orientation. Prevents upside-down and sideways images!

![image](https://user-images.githubusercontent.com/67019423/119190463-04504a00-ba9b-11eb-80cb-483a0e4eea8e.png)

| no_exif  | exif |
| ------------- | ------------- |
| ![no_elif](https://user-images.githubusercontent.com/67019423/119191159-e9caa080-ba9b-11eb-8c0d-51deabb53c9b.PNG) | ![elif](https://user-images.githubusercontent.com/67019423/119191231-f949e980-ba9b-11eb-893b-65c812c03d0b.PNG) |

### Installation

```
pip install image_to_numpy
```

### Usage

If you have matplotlib installed, here's a quick way to show your image on the screen:

```
import matplotlib.pyplot as plt
import image_to_numpy

img = image_to_numpy.load_image_file("my_file.jpg")

plt.imshow(img)
plt.show()
```
