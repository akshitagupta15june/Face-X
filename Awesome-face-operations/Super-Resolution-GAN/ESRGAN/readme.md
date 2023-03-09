# IMAGE SUPER RESOLUTION USING GAN


* ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) is an advanced image enhancement model that uses deep learning to upscale low-resolution images into high-resolution images with realistic details. It was developed by researchers from South Korea's Electronics and Telecommunications Research Institute (ETRI) in 2018.

* The ESRGAN model uses a Generative Adversarial Network (GAN) to generate high-resolution images that look similar to the original high-resolution images. The GAN is trained on a dataset of high-resolution images to learn how to generate high-quality images. Then, the model is trained to upscale low-resolution images while preserving their features and details.

* ESRGAN uses a new loss function called perceptual loss, which helps the model to generate images with realistic textures and features. This loss function is based on the VGG network, which is a popular deep neural network used for image recognition. By minimizing the perceptual loss, the ESRGAN model can produce images that are not only visually pleasing but also preserve the details and structure of the original image.

* The ESRGAN model has been shown to achieve state-of-the-art results in terms of image quality and realism, outperforming previous super-resolution models such as SRGAN and SRResNet. It has a wide range of applications, including image restoration, photo enhancement, and even video upscaling.

# STEPS INVOLVED
1. Importing Required Libraries
2. Loading Image Data
3. Downloading and Importing the Model
4. Preprocessing of Data
5. Feeding in an input image to the model to get 4x upscaled image.
6. Display both the input and output images




# DOWNLOAD THE ABOVE PRETRAINED MODEL USING THE LINKS BELOW :

* ESRGAN MODEL - https://tfhub.dev/captain-pool/esrgan-tf2/1

# HOW TO RUN

* You can run the jupyter notebook file (Histogram-Equalization.ipynb) remotely using Google colab by uploading the file into remote directories and can be run locally using Anaconda.


# INPUT IMAGE SAMPLEs
![download1lr](https://user-images.githubusercontent.com/86817867/223980738-439411cd-e0c3-4250-84a8-40981862adf1.png)
![download2lr](https://user-images.githubusercontent.com/86817867/223980843-b6433f09-8bb4-4963-ab35-43c8b497ace7.png)
![download3lr](https://user-images.githubusercontent.com/86817867/223980992-6f8f739d-dc7a-4683-b90c-0cdf8c171925.png)




# OUTPUT IMAGE SAMPLES
![download1hr](https://user-images.githubusercontent.com/86817867/223981522-63eae6bb-9f9c-4d83-ac55-3bcddafdfa24.png)
![download2hr](https://user-images.githubusercontent.com/86817867/223981555-dc6192ba-63c6-428d-88f3-5e2698c622cb.png)
![download3hr](https://user-images.githubusercontent.com/86817867/223981584-614ea7e7-8fdf-46c2-8e8c-30af56abbdce.png)



