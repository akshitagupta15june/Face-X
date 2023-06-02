# Face Obstruction Recognition

This repository provides a solution for detecting various obstructions in faces using two models: a Convolutional Neural Network (CNN) and a pretrained VGG16 model. The obstructions are classified into six categories:

- Glasses
- Hand
- Mask
- None
- Other
- Sunglasses

Two classification approaches are demonstrated here: a multi-class single-label model and a multi-label model.

## Single-label Model

The single-label model is designed to recognize only one category of obstruction in each image. It showcases two architectures: one built using a simple CNN and the other with a pretrained VGG16 model. The dataset for this model is divided into six folders, with each folder corresponding to an obstruction class.

![Single-label Model](https://github.com/Ultr0x/Face-X/assets/50329232/d6ad2540-4d01-4f9d-91a5-2895ee27819b)

## Multi-label Model

The multi-label model is capable of identifying multiple obstruction categories in the same image, utilizing a pretrained VGG16 model. This model works with a differently organized, smaller version of the same dataset. All images are placed in one folder, while a CSV file provides the necessary information about each image and its corresponding classes.

| id | glasses | hand | mask | none | other | sunglasses |
| --- | --- | --- | --- | --- | --- | --- |
| i00001.jpg | 1 | 0 | 0 | 0 | 0 | 0 |
| i00002.png | 1 | 0 | 0 | 0 | 0 | 0 |
| i00003.jpg | 1 | 0 | 0 | 0 | 0 | 0 |
| i00004.png | 1 | 0 | 0 | 0 | 0 | 0 |
| i00005.png | 1 | 1 | 0 | 0 | 0 | 0 |

![Multi-label Model](https://github.com/Ultr0x/Face-X/assets/50329232/846884bf-9179-47af-b4ba-f52198e37311)

## Dataset

The dataset for the multi-class single-label model contains 19163 images and is accessible [here](https://www.kaggle.com/datasets/janwidziski/face-obstructions).

The dataset for the multi-class multi-label model, which includes 11870 images, can be found [here](https://www.kaggle.com/datasets/janwidziski/face-obstructions-multilabel).

The discrepancy in the number of images between the two datasets is a result of a deliberate attempt to mitigate class size disparities for the multi-class model. However, it is important to note that the complexity and intensity of the multi-label problem require a larger, more balanced dataset. The relatively smaller size of the multi-label dataset may have contributed to the lower performance of the multi-label model. 
This indicates potential areas for improvement, suggesting the need for further data collection and curation efforts to optimize the model's performance.

## Bias Mitigation

Efforts have been made to mitigate different biases that may affect the dataset and subsequently the model's performance. These measures include:

- Ensuring representation of various skin tones and face features.
- Incorporating diverse styles of masks.
- Taking care not to classify religious wear as a mask, which could potentially be harmful or disrespectful.
- Addressing dataset imbalances, like underrepresentation of people with darker skin tones wearing plain glasses.
- Avoiding overrepresentation of certain angles or cropped faces.
- Accommodating diversity in image sizes and facial appearances, including variations in size, angle, and position.

## Defining the Classes

The definition of obstruction classes was a critical process that involved careful consideration to ensure the classes are meaningful, distinguishable, and comprehensive.

- **Glasses:** This class includes images where the subjects are wearing glasses. This does not distinguish between prescription glasses, reading glasses, or any other specific types of glasses.

- **Hand:** Images where the subject's hand or someone else's hand is obstructing the face are included in this category. This class is irrespective of the reason for the hand's presence, posing, accidentally, or intentionally covering the face.

- **Mask:** This class primarily includes images where the subjects are wearing masks typically associated with health protection, such as medical masks or masks that attach behind the ears. Masks used for fashion purposes are also included if they bear similarity to these health-related masks in form and function. However, religious or cultural masks are intentionally excluded from this class to avoid potential misclassification or disrespect.

- **None:** The 'none' class includes images where there is no visible obstruction to the face. These images serve as a 'control group' against which the model can learn to distinguish between different instances of obstruction.

- **Other:** The 'other' class acts as an all-for category for obstructions that don't fit into the other classes. This includes but is not limited to objects, clothing items, or unconventional items that might obstruct the face.

- **Sunglasses:** This class is specifically for images where subjects are wearing sunglasses. Despite being a type of glasses, sunglasses were given a separate class due to their distinct visual characteristics and the different contexts in which they are worn.

The process of classifying these images was meticulous and aimed at ensuring a wide coverage of possible obstructions while also maintaining a respect for the diversity and individuality of the subjects involved.


