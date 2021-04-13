# FACE MASK DETECTION FROM LIVE WEBCAM

This is a project to detect face mask using MTCNN + OPENCV + EFFICIENTNETB3 from LIVE WEBCAM.

### PROJECT WORKFLOW:

![FlowChart](https://github.com/NEERAJAP2001/Face-X/blob/master/Face-Mask-Detection/Recognition%20using%20EfficientNetB3/face%20mask%20detection-%20mtcnn.png)


## About :

Data is available at: [Link](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)
### NOTE: I have created a folder 'Test' and put a mixure of with mask and without mask images. In original dataset these images are placed under separate folders.

### Data augmentation:
1. Data augmentation encompasses a wide range of techniques used to generate “new” training samples from the original ones by applying random jitters and perturbations (but at the same time ensuring that the class labels of the data are not changed).
2.The basic idea behind the augmentation is to train the model on all kind of possible transformations of an image
3. Here we are using flow_from_directory. This is because we have limited ram and we need to get images in batches

### Callbacks
### Building Model
### Model Training

## Output :

![](https://github.com/NEERAJAP2001/Face-X/blob/master/Face-Mask-Detection/Recognition%20using%20EfficientNetB3/Mask.png)
