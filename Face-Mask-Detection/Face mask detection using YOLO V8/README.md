# yolov8 FaceMask 
* The dataset used for training the yolov8 model is from universe.roboflow.com<br/>

## Output result from testing dataset
![output_img](https://github.com/samtholathrobin/Face-X/blob/master/Face-Mask-Detection/Face%20mask%20detection%20using%20YOLO%20V8/Model/model_output.jpeg))
## Installation
1) Download and install yolov8
```
We will be using the ultralytics package in python to use YOLO V8
pip install ultralytics
```
2) Run inference 
For running inference you required trained weights which is obtained from my repo cloned as yolov5_FaceMask<br/>

```
$ python detect.py --weights last_mask_yolov5s_results.pt --conf 0.4 --source 0  # webcam
                                                                              file.jpg  # image 
                                                                              file.mp4  # video
                                                                              path/  # directory
                                                                              path/*.jpg  # glob
                                                                              rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                                                                              http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```
## Increasing accuracy and Future Scope
The dataset contains 149 Images which is very less for yolo architecture. So during training I performed augmentation and increased to 298 Images.<br/>
To get more accuracy the training dataset needs to increase.<br/>

## Output Results from open source images
![test5_out](https://user-images.githubusercontent.com/41751718/88254674-a8c6fb80-ccd3-11ea-8c09-54a7e39274f3.jpg)
![test4_out](https://user-images.githubusercontent.com/41751718/88254682-acf31900-ccd3-11ea-83b6-73659db53aa0.png)
![test3_out](https://user-images.githubusercontent.com/41751718/88254685-af557300-ccd3-11ea-9d2e-413c06820e5e.jpg)
