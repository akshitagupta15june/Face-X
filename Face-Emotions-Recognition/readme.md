# Facial-Emotion-Recognition

Facial-Emotion-Recognition is a project that focuses on detecting human emotions from facial expressions. It utilizes artificial intelligence (AI) to recognize emotions by analyzing facial features and expressions. The goal is to develop a system that can accurately interpret and respond to human emotions, similar to how the human brain does.

## Table of Contents
- About
- Eye Blinker Counter
- Facial Expression Recognition using custom CNN
- Smile Percentage Detection
- Face Emotions Recognition using Deep Learning

## About

Facial-Emotion-Recognition is a comprehensive project that includes various components for emotion detection and recognition from facial expressions. The key components of this project are described below.

## Eye Blinker Counter
![image](https://user-images.githubusercontent.com/78999467/110669344-57cd2e80-81f4-11eb-9637-c8f5c3a267bf.png)
The Eye Blinker Counter component is responsible for detecting blinks in the eyes, which can be indicative of certain emotions. It utilizes facial landmark detection to locate the eyes in a video stream. The eye aspect ratio is calculated based on the eye coordinates, allowing the system to determine if a person is blinking. This information can be used to infer their emotional state.

- The first step is facial landmark detection to localize the eyes.
- Each eye is represented by coordinates, and the eye aspect ratio is calculated.
- A person is considered to be blinking if the eye aspect ratio rapidly approaches zero.

For more details, refer to the [Eye Blinker Counter documentation](link-to-documentation).

## Facial Expression Recognition using custom CNN
![image](https://user-images.githubusercontent.com/78999467/110669184-2f453480-81f4-11eb-9ac2-611dd5754f92.png)
![image](https://user-images.githubusercontent.com/78999467/110669183-2f453480-81f4-11eb-9a3a-a971bb7a9e95.png)

Facial Expression Recognition using custom CNN is a model that utilizes Convolutional Neural Networks (CNNs) to analyze facial images and recognize different facial expressions. The model consists of two levels: background removal and expressional vector extraction.

- The first level involves background removal to extract emotions from an image.
- The second level uses a CNN to extract the primary expressional vector (EV) by tracking relevant facial points.
- The EV represents changes in expression and is used for emotion recognition.

For more details, refer to the [Facial Expression Recognition documentation](https://ieeexplore.ieee.org/document/4427488/).

## Smile Percentage Detection
![image](https://user-images.githubusercontent.com/78999467/110666784-bc3abe80-81f1-11eb-95c6-698f8dd2116d.png)
![image](https://user-images.githubusercontent.com/78999467/110666785-bc3abe80-81f1-11eb-81a2-e8d1b86c7ecf.png)

The Smile Percentage Detection component focuses on detecting smiles in real-time. It utilizes optical flow and facial feature tracking to determine if a person is smiling.

- The first human face is detected in the initial image frame, and standard facial features are located.
- Optical flow is used to track the positions of the left and right mouth corners in subsequent frames.
- If the distance between the tracked mouth corners exceeds a threshold, a smile is detected.

For more details, refer to the [Smile Percentage Detection documentation](https://pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/).

## Face Emotions Recognition using Deep Learning
![image](https://user-images.githubusercontent.com/78999467/110668788-c8278000-81f3-11eb-81ec-e12d728b1ead.png)

Face Emotions Recognition using Deep Learning aims to classify human facial emotions using a Deep Convolutional Neural Network (DCNN) model. The system can process real-time facial images captured through a front camera and predict emotions.

- The DCNN model consists of convolution layers, pooling layers, and the sigmoid activation function.
- It is trained on a labeled facial image dataset to classify emotions into different classes.
- The model's performance is evaluated using a public database, achieving an accuracy of approximately 65%.

For more details, refer to the [Face Emotions Recognition documentation](https://ieeexplore.ieee.org/document/9752189).
## Technical Specifications

| Component                       | Specifications                                      |
|---------------------------------|-----------------------------------------------------|
| Eye Blinker Counter             | - Facial landmark detection for eye localization    |
|                                 | - Calculation of eye aspect ratio for blink detection|
| Facial Expression Recognition   | - Two-level CNN framework                            |
| Using custom CNN                | - Background removal for emotion extraction          |
|                                 | - Tracking of facial points for expressional vector  |
| Smile Percentage Detection      | - Optical flow for mouth corner tracking             |
|                                 | - Threshold-based smile detection                    |
| Face Emotions Recognition       | - Deep Convolutional Neural Network (DCNN) model     |
| Using Deep Learning             | - Convolution layers, pooling layers, sigmoid activation |
|                                 | - Trained on labeled facial image dataset            |
|                                 | - Achieves an accuracy of approximately 65%          |

## Additional Resources

- [Eye Blinker Counter documentation](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
- [Facial Expression Recognition documentation](https://ieeexplore.ieee.org/document/4427488/)
- [Smile Percentage Detection documentation](https://pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/)
- [Face Emotions Recognition documentation](https://ieeexplore.ieee.org/document/9752189)

Please refer to the respective documentation for more detailed information about each component.
