Background subtraction is a major preprocessing steps in many vision based applications. For example, consider the cases like visitor counter where a static camera takes the number of visitors entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In all these cases, first you need to extract the person or vehicles alone. Technically, you need to extract the moving foreground from static background.

### Algorithm used: BackgroundSubtractorMOG2

One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

Here, you have an option of selecting whether shadow to be detected or not. If `detectShadows = True` (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.

### Input:
![resframe](https://user-images.githubusercontent.com/60208804/113537714-106d6e80-95f7-11eb-8590-7d7b12e7760b.jpg)

### Output:
![resmog](https://user-images.githubusercontent.com/60208804/113537728-195e4000-95f7-11eb-8f3d-edcaf79ddc36.jpg)
