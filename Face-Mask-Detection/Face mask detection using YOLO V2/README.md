
## System Overview

It detects human faces with mask or no mask  even in crowd in real time with live count status and notifies user (officer) if danger.

**System Modules:**

  
**Alert System:** It monitors the mask, no-mask counts and has  status :
	1. **Safe :** When _all_ people are with mask.
	2. **Warning :** When _atleast 1_ person is without mask.



## Quick-Start
**Step 1:**
```
git clone Rep
```


**Step 2: Install requirements.**
```
pip install opencv-python
pip install imutils
```
**Step 3: Run yolov4 on webcam**
```
python mask-detector-image.py 
```


Output :
![1_output](https://user-images.githubusercontent.com/65017645/120091093-5fcaa980-c125-11eb-8150-92a1454f9992.jpg)


