## What is __Style Transfer__? 🎨

&nbsp;
- __Style Transfer__ is creating artistic imagery by separating and recombining image content and style.

  OR

- __Style Transfer__ is the task of changing the style of an image in one domain to the style of an image in another domain. 

&nbsp;

![](https://miro.medium.com/max/1200/1*XI3beonBnOwp-y5BwNOqCw.gif) 

&nbsp;

## __About this Project__ :

   In this project, I have created a style transfer method that is outlined in the paper, Image Style Transfer Using Convolutional Neural Networks, by Gatys in PyTorch.

   In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers.

 &nbsp;

  ## __Separating Style and Content__ :

Style transfer relies on separating the content and style of an image. Given one content image and one style image, the aim is to create a new, target image which should contain the desired content and style components:


- objects and their arrangement are similar to that of the content image
- style, colors, and textures are similar to that of the style image


In this notebook, I have used a pre-trained VGG19 Net to extract content or style features from a passed in image. I've then formalize the idea of content and style losses and use those to iteratively update the target image until I get a result that I want.

&nbsp;

![](https://media-exp1.licdn.com/dms/image/C4E12AQEfjA-SVxYLVQ/article-cover_image-shrink_600_2000/0/1531630356496?e=1623283200&v=beta&t=XoNFeg5J3yAJo6vj_nmIylqulnV1lek-s289ztZuya4)

### Some more examples 😎:

![](https://miro.medium.com/max/2166/1*8bbp3loQjkLXaIm_QBfD8w.jpeg)







## __Importance__ ❓

- Anyone can enjoy the pleasure that comes along with creating and sharing an artistic masterpiece.

-  Opens up endless doors in design, content generation, and the development of creativity tools.

- Artists can easily lend their creative aesthetic to others, allowing new and innovative representations of artistic styles to live alongside original masterpieces.

&nbsp;

![cool](https://media.giphy.com/media/l4FGk4HOEwB54N9qU/giphy.gif)

