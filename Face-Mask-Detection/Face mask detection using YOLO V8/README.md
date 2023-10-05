# yolov8 FaceMask 
* The dataset used for training the yolov8 model is from universe.roboflow.com<br/>

## Output result from testing dataset
![output_img](Model\model_output.jpeg)

## Installation
1) Download and install yolov8
```
We will be using the ultralytics package in python to use YOLO V8
pip install ultralytics
```
2) Run inference 
For running inference you required trained weights which is obtained from my repo cloned as yolov5_FaceMask<br/>
```
$ yolo detect predict model='INSERT PATH TO /Model/maskdetektormodel.pt source='INSERT PATH TO IMAGE OR VIDEO' 
```
You can also run the inference using the cells of the notebook in Models Directory.

## Increasing accuracy and Future Scope
The dataset contains around 1400 images which gives it a level of confidence shown with the given number of epochs. The model should perform better with more data to train on and with more epochs to train.</br> 
Continue to maintain track of the mAP for preventing overfitting or enable patience while training.<br/>

## Output Results from open source images
![test5_out](Model\masktestout.jpeg)
</br>
![test4_out](Model\masktest2out.jpeg)
