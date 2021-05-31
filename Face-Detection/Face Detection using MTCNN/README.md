MTCNN is based on “Joint face detection and 
alignment using multi-task cascaded convolutional networks”, it presents a method for facial 
detection and alignment in pictures. It consists of 3-part
CNN which can recognize landmarks on faces such as nose, 
forehead, eyes etc. There are 3 stages to mtcnn. In the first 
one the image is resized to create a pyramid of images so 
detection for every size can be done then it is passed through 
a neural network known as P Net which gives coordinates of 
face and bounding box as output. In the second stage faces 
which are only partly visible are dealt with a R net is used to 
give bounding boxes as output. The result from P net and R 
net is mostly similar. In the third and final stage an O net is 
applied which gives three outputs- coordinates, landmarks on 
face and confidence level of bounding boxes. After every 
stage a non max suppression method is used to remove 
bounding boxes with low confidence.

![outside_000001_yoloface](https://user-images.githubusercontent.com/65017645/120110101-dc8f6f00-c189-11eb-8331-1c32d4a5d442.jpg)

