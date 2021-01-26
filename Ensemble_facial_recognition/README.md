# Facial Recongtion using Ensemble Learning.

## First we have to install the deepface library.
```
pip install deepface
```
## Secondly we should install the dependencies.
```
pip install tensorflow==2.4.1
pip install keras==2.4.3
```

## For comparing two photos we can use this code.
```
obj=DeepFace.verify("Dataset\steve1.jpg","Dataset\steve2.jfif" , model_name="Ensemble") 
```
Here we insert two photos and use the ensemble model.

## For checking whether a given face is in a database.
```
df= DeepFace.find(img_path="Database\mark1.jpg",db_path="Database",model_name="Ensemble")
```
where img_path is the image you want to find resemblance, db_path is the database folder and model_name is the ensemble model.


