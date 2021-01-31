
# Facial-Expression-Recognition-using-custom-CNN
Recognizing facial expression with CNN
![](https://images.ctfassets.net/cnu0m8re1exe/70iMKfC0fJNNd4SN7HmgD1/cbdfd2e0595d4451fa7ff64703562d04/shutterstock_1191853330.jpg?w=650&h=433&fit=fill)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
 ## Project Environment : 

          1. Python
          2. Google Collab
          3. API Docker
 ## Data Description :

   * **Link of the Dataset**  : https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge
   * **Usage**   : 
                     
                     1. Train Data :        (28709,3)
                     
                     2. Public Test Data :   (3589,3)
                     
                     3. Private Test Data :  (3589,3)
   * **Columns** :  
                   
                   emotion
                   pixels
                   emotion
   * **Type**  : 
   
                     Image Data
                     2D Images
                     Data stored in tabular format into a comma seperated file.   (fer2013.csv)
   
   * **Iamge Shape** :
                   
                   On the dataset : ( 48,48,1 )   (Unilayered Images)
                   

                   
   * **Expressions** :
                 The expressions are encoded into numerical values. They represent :
                  
                  1:    ANGER 
                  2:    DISGUST
                  3:    FEAR 
                  4:    HAPPINESS 
                  5:    NEUTRAL
                  6:    SADNESS
                  7:    SURPRISE
 ## Model : 
        Sequential model having 
           1. Conv2D
           2. MaxPool2D
           3. Dropout
           4. Dense
           5. Flatten
 ## Model evaluation Metric :
   **Accuracy** :
          
          Train Data --> 0.6605
          Validation(Private Test) --> 0.5804
          Test(Private Test) --> 0.5887
          
   **Sparse categorical Crossentropy** :
          
          Train Data --> 0.8757
          Validation(Private Test) --> 1.1807
## Some screenshots of Classification
---
![](https://github.com/sagnik1511/Facial-Expression-Recognition-using-custom-CNN/blob/main/a.jpg)

![](https://github.com/sagnik1511/Facial-Expression-Recognition-using-custom-CNN/blob/main/b.jpg)
## Do ***STAR***  if you find it useful :)
