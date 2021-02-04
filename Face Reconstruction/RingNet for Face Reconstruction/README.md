# RingNet

![alt text](gif/celeba_reconstruction.gif)

This is an official repository of the paper Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision.
```
Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision
Soubhik Sanyal, Timo Bolkart, Haiwen Feng, Michael J. Black
CVPR 2019
```

## Download models

* Download pretrained RingNet weights from the [project website](https://ringnet.is.tue.mpg.de), downloads page. Copy this inside the **model** folder
* Download FLAME 2019 model from [here](http://flame.is.tue.mpg.de/). Copy it inside the **flame_model** folder. This step is optional and only required if you want to use the output Flame parameters to play with the 3D mesh, i.e., to neutralize the pose and
expression and only using the shape as a template for other methods like [VOCA (Voice Operated Character Animation)](https://github.com/TimoBolkart/voca).
* Download the [FLAME_texture_data](http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip) and unpack this into the **flame_model** folder.

## Demo

RingNet requires a loose crop of the face in the image. We provide two sample images in the **input_images** folder which are taken from [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

#### Output predicted mesh rendering

Run the following command from the terminal to check the predictions of RingNet
```
python -m demo --img_path ./input_images/000001.jpg --out_folder ./RingNet_output
```
Provide the image path and it will output the predictions in **./RingNet_output/images/**.

#### Output predicted mesh

If you want the output mesh then run the following command
```
python -m demo --img_path ./input_images/000001.jpg --out_folder ./RingNet_output --save_obj_file=True
```
It will save a *.obj file of the predicted mesh in **./RingNet_output/mesh/**.

#### Output textured mesh

If you want the output the predicted mesh with the image projected onto the mesh as texture then run the following command
```
python -m demo --img_path ./input_images/000001.jpg --out_folder ./RingNet_output --save_texture=True
```
It will save a *.obj, *.mtl, and *.png file of the predicted mesh in **./RingNet_output/texture/**.

#### Output FLAME and camera parameters

If you want the predicted FLAME and camera parameters then run the following command
```
python -m demo --img_path ./input_images/000001.jpg --out_folder ./RingNet_output --save_obj_file=True --save_flame_parameters=True
```
It will save a *.npy file of the predicted flame and camera parameters and in **./RingNet_output/params/**.

#### Generate VOCA templates

If you want to play with the 3D mesh, i.e. neutralize pose and expression of the 3D mesh to use it as a template in [VOCA (Voice Operated Character Animation)](https://github.com/TimoBolkart/voca), run the following command
```
python -m demo --img_path ./input_images/000013.jpg --out_folder ./RingNet_output --save_obj_file=True --save_flame_parameters=True --neutralize_expression=True
```

## Referencing RingNet

Please cite the following paper if you use the code directly or indirectly in your research/projects.
```
@inproceedings{RingNet:CVPR:2019,
title = {Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision},
author = {Sanyal, Soubhik and Bolkart, Timo and Feng, Haiwen and Black, Michael},
booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
month = jun,
year = {2019},
month_numeric = {6}
}
```
