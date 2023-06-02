# Face obstruction Recognition

This is an approach to detecting various obstructions in faces built using CNN as well as pretrained VGG16 for classification.

The dataset has 6 classes:
- glasses
- hand
- mask
- none
- other
- sunglasses

There are two approaches shown, a multi-class single-label as well as a multi-label approach. 

## Single-label
The single-label approach shows two model architectures, one using simple CNN and the other using additionally a pretrained VGG16 model. The dataset is divided into 6 folders corresponding to the classes of the images.
## Multi-label
The multi-label approach uses the pretrained VGG16 model on a smaller version of the same dataset. This dataset however to be compatible with multi-label classification is different. All the images are placed in one folder. Additionally there is a CSV file that contains all the image names and according classes. ![pobrane](https://github.com/Ultr0x/Face-X/assets/50329232/15a24f0d-e4f1-4d90-b19b-a21fc49a3bbf)


## Dataset
The data for multiclass single-label model is available here: https://www.kaggle.com/datasets/janwidziski/face-obstructions

The data for multiclass multi-label model is available here: https://www.kaggle.com/datasets/janwidziski/face-obstructions-multilabel
