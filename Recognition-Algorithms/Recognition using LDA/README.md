# Face Recognition Using Linear Discriminant Analysis👨‍💻👨‍💻
Linear Discriminant Analysis (LDA) has been successfully applied to face recognition which is based on a linear projection from the image space to a low dimensional space by maximizing the between-class scatter and minimizing within-class scatter.LDA method overcomes the limitation of the Principle Component Analysis method by applying the linear discriminant criterion. Analysis (LDA) which is also called fisher face is an appearance-based technique used for dimensionality reduction and recorded a great performance in face recognition.

## Applications:

#### Face Recognition:
 In the field of Computer Vision, face recognition is a very popular application in which each face is represented by a very large number of pixel values. Linear discriminant analysis (LDA) is used here to reduce the number of features to a more manageable number before the process of classification. Each of the new dimensions generated is a linear combination of pixel values, which form a template. The linear combinations obtained using Fisher’s linear discriminant are called Fisher's faces.
#### Medical: 
In this field, Linear discriminant analysis (LDA) is used to classify the patient's disease state as mild, moderate, or severe based upon the patient various parameters and the medical treatment he is going through. This helps the doctors to intensify or reduce the pace of their treatment.
#### Customer Identification: 
Suppose we want to identify the type of customers who are most likely to buy a particular product in a shopping mall. By doing a simple question and answers survey, we can gather all the features of the customers. Here, the Linear discriminant analysis will help us to identify and select the features which can describe the characteristics of the group of customers that are most likely to buy that particular product in the shopping mall.

**The following is a demonstration of Linear Discriminant Analysis. The following has been developed in python 3.8.**

**Dataset courtesy**
- http://vis-www.cs.umass.edu/lfw/
## Proposed method
<img width="283" alt="Screenshot 2021-03-20 at 2 55 20 AM" src="https://user-images.githubusercontent.com/78999467/111843334-155ccd80-8929-11eb-8552-d935aad99e2c.png">

## Dependencies📝:
- ```pip install sklearn```
- ```pip install matplotlib```

## QuickStart✨:
- Clone this repository
` git clone https://github.com/akshitagupta15june/Face-X.git`
- Change Directory
` cd Recognition-Algorithms/Recognition Using LDA`
- Run the program with the set dataset:
``` py main.py```
- Make a folder and add your code file and a readme file with screenshots.
- Commit message
` git commit -m "Enter message"`
- Push your code
` git push`
- Make a Pull request
- Wait for reviewers to review your PR

## Result📉:
The dataset details along with the classification report and confusion matrix are printed.
The time taken by each step is also included.
![Report](https://user-images.githubusercontent.com/78999467/111842279-4a682080-8927-11eb-9b02-0d86000ae03d.png)



## Screenshot📸:
Faces with the names predicted

![Faces](https://user-images.githubusercontent.com/78999467/111842230-315f6f80-8927-11eb-8d09-8c85762d551c.png)


The fisher faces 

![FisherFaces](https://user-images.githubusercontent.com/78999467/111842163-1e4c9f80-8927-11eb-90c9-6a9e2792fa3a.png)
