# IMAGE SUPER RESOLUTION 

* Image Super Resolution refers to the task of enhancing the resolution of an image from low-resolution (LR) to high (HR).

# EDSR MODEL

* The EDSR architecture is based on the SRResNet architecture, consisting of multiple residual blocks. The residual block in EDSR is shown above. 
* The major difference from SRResNet is that the Batch Normalization layers are removed; removal of BN results in an improvement in accuracy. 
* The BN layers also consume memory, and removing them leads to up to a 40% memory reduction, making the network training more efficient.

# LAPSRN
*  LAPSRN consists of multiple stages. The network consists of two branches: the Feature Extraction Branch and the Image Reconstruction Branch. 
*  Each iterative stage consists of a Feature Embedding Block and Feature Upsampling Block
*  The input image is passed through a feature embedding layer to extract features in the low resolution space, which is then upsampled using transpose convolution. 
*  The output learned is a residual image which is added to the interpolated input to get the high resolution image. 
*  The output of the Feature Upsampling Block is also passed to the next stage, which is used for refining the high resolution output of this stage and scaling it to the next level. 
*  Since lower-resolution outputs are used in refining further stages, there is shared learning which helps the network to perform better.

# DOWNLOAD THE ABOVE PRETRAINED MODELS USING THE LINKS BELOW :

* EDSR MODEL - https://github.com/Saafke/EDSR_Tensorflow/tree/master/models
* LAPSRN MODEL - https://github.com/fannymonori/TF-LapSRN/tree/master/export

# INPUT IMAGE SAMPLE 
![image](https://user-images.githubusercontent.com/69035013/201017951-6f5e849b-189e-4c68-908a-9a9ce9b6a7dc.png)


# OUTPUT IMAGE
![image](https://user-images.githubusercontent.com/69035013/201018029-e14d377f-eaf4-4e70-a461-d87651cd0c87.png)
