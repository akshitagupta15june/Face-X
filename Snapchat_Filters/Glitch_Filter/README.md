# <b> Cool Glitch Snapchat Filter using OpenCV and numpy</b>

## Setting up and running the script

- Install the requirements of the script as follows
```
    $ pip install -r requirements.txt
```

- Run the script as follows
```
    $ python glitch_filter.py
```

# Real-Time Glitch Filter using OpenCV and numpy

The glitch effect filter works by introducing random pixel noise to each frame of the video in real-time. The noise is generated within a specified intensity range and is added to the original frame, resulting in pixel intensity variations. This variation creates a glitch-like appearance, making the video look distorted and glitchy.

# Description 

Sure, let's explain the code and the working of the glitch effect filter.

1. Importing Libraries:
   The code starts by importing the necessary libraries: `cv2` (OpenCV) for computer vision operations and `numpy` for numerical operations.

2. Video Capture:
   The code initializes the video capture using the `cv2.VideoCapture(0)` function, which captures video from the default camera (usually the webcam).

3. Glitch Effect Loop:
   The core of the glitch effect is in the while loop. It continuously reads frames from the video capture.

4. Random Pixel Noise:
   Inside the loop, random pixel noise is generated using the `np.random.randint()` function. The noise is generated between -30 to 30, creating random intensity changes for each pixel in the frame.

5. Adding Glitch:
   The generated noise is added to the original frame using NumPy array operations. This process creates the glitch effect in the frame.

6. Combining Frames:
   The glitched frame and the original frame are combined using the `cv2.addWeighted()` function. The `alpha` value controls the intensity of the glitch effect. By adjusting this value, you can control how much of the glitch is applied to the original frame.

7. Displaying the Frame:
   The final glitched frame is displayed in a window with the title "Glitch Effect" using `cv2.imshow()`.

8. Exiting the Loop:
   The loop continues until the user presses the "ESC" key. The `cv2.waitKey(1)` function is used to wait for a key press, and the loop breaks if the pressed key is the "ESC" key (27 in ASCII code).

9. Release Video Capture and Close Windows:
   Once the loop is exited, the video capture is released using `camera_video.release()`, and all windows are closed using `cv2.destroyAllWindows()`.

# Working of the Glitch Effect Filter

The glitch effect filter works by introducing random pixel noise to each frame of the video in real-time. The noise is generated within a specified intensity range and is added to the original frame, resulting in pixel intensity variations. This variation creates a glitch-like appearance, making the video look distorted and glitchy.

The filter does not involve pixel shifting or any other complex operations. Instead, it relies on simple pixel noise addition to achieve the desired effect. The intensity of the glitch effect can be controlled by adjusting the `alpha` value in the `cv2.addWeighted()` function.

Keep in mind that this implementation provides a basic glitch effect. For more complex and visually appealing glitch effects, additional techniques like color channel splitting, image warping, and other visual distortions can be incorporated. The intensity and appearance of the glitch effect can be further customized by modifying the noise generation range, the combination of frames, and other parameters.

Note: Ensure that your system has a webcam connected and properly configured.

## Output screenshot

[Image](https://freeimage.host/i/glitch.HQbbehg)

## Author
[Subhasish Panda](https://github.com/Codingpanda252)
