
## System Overview

It detects human faces with mask or no mask  even in crowd in real time with live count status and notifies user (officer) if danger.

**System Modules:**
  
1. **Deep Learning Model :** I trained a YOLOv2,v3 and v4 on my own dataset and for YOLOv4 achieved **93.95% mAP on Test Set** whereas YOLOv3 achieved **90% mAP on Test Set** even though my test set contained realistic blur images, small + medium + large faces which represent the real world images of average quality.  
  
2. **Alert System:** It monitors the mask, no-mask counts and has  status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.



## Quick-Start
**Step 1:**
```
git clone 
```
Then, Download weights. https://bit.ly/yolov4_mask_weights 

**Step 2: Install requirements.**
```
pip install opencv-python
pip install imutils
```
**Step 3: Run yolov4 on webcam**
```
python mask-detector-image.py -y yolov4-mask-detector -u 1
```


Output :

