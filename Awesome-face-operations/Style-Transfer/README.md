# Style-Transfer
In this project, I have created a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), by Gatys in PyTorch.

In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers.

## Separating Style and Content
Style transfer relies on separating the content and style of an image. Given one content image and one style image, the aim is to create a new, target image which should contain the desired content and style components:

* objects and their arrangement are similar to that of the **content image**
* style, colors, and textures are similar to that of the **style image**

In this notebook, I have used a pre-trained VGG19 Net to extract content or style features from a passed in image. I've then formalize the idea of content and style losses and use those to iteratively update the target image until I get a result that I want.

## Example

<img src="https://github.com/KKhushhalR2405/Style-Transfer/blob/master/exp1/blonde.jpg" width="50px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/KKhushhalR2405/Style-Transfer/blob/master/exp1/delaunay.jpg" width="65px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/KKhushhalR2405/Style-Transfer/blob/master/exp1/final_image.png" width="50px">

content&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;style&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;output

