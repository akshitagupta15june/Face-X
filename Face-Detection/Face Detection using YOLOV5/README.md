# yolov5
* The dataset used for training the yolov5 is from roboflow.ai<br/>

## Installation
1) Download and install yolov5
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
git clone https://github.com/pritul2/yolov5_FaceMask
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

## Output Results from open source images
![merkel_putin_trump](https://user-images.githubusercontent.com/65017645/119926902-ebadcb80-bf95-11eb-8c5d-94f65b3c9ea7.jpg)
