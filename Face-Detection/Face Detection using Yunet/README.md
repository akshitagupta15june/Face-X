# YuNet Face Detection Model

YuNet is a highly efficient and accurate face detection model developed by the Intel OpenVINO team. It is designed for real-time face detection and can detect multiple faces in an image, along with their facial landmarks.

## Key Features

- **High Accuracy**: YuNet is known for its high detection accuracy and reliability in various environments and conditions.
- **Real-time Performance**: Optimized for real-time performance, making it suitable for applications requiring fast processing.
- **Facial Landmarks**: In addition to detecting faces, YuNet also provides precise facial landmarks, such as eyes, nose, and mouth corners.

# Face Detection using YuNet

This repository contains a simple face detection script using the YuNet model for detecting faces in images. The script is written in Python and utilizes OpenCV for image processing and face detection.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.6+
- OpenCV (including the `opencv-contrib-python` package)
- NumPy

## Installation


1. Install the required Python packages.
   ```sh
   pip install opencv-python opencv-contrib-python numpy
   ```

2. Download the pre-trained YuNet model.
   If the model is not available locally, the script will automatically download it.

## Usage

Run the face detection script with the following command:
```sh
python face_detection.py -p /path/to/your/image.jpg
```

Replace `/path/to/your/image.jpg` with the actual path to the image you want to process.

## Script Overview

### face_detection.py

This is the main script for face detection. It includes the following key functionalities:

- **Model Loading:** Downloads the YuNet model if not already available.
- **Image Reading:** Reads the input image specified by the user.
- **Face Detection:** Detects faces and facial landmarks in the image using the YuNet model.
- **Visualization:** Draws bounding boxes and landmarks on detected faces and displays the results.

### Key Functions

- `visualize_face_detections(image_path, detections)`: Visualizes the face detections on the image.

