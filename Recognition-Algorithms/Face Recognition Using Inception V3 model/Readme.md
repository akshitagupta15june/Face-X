<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/Unt1.png" height="80%" width="70%" /></a><br /><br /></p>

# Face Recognition Using Inception V3 model
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/facial.png" align="right"/>

### 1.Introduction 
Image classification (recognition) is one of the foremost capabilities of deep neural networks. Inception-v3 is one of the most popular convolutional neural network models for recognizing objects in images. Deep learning-powered image recognition is used by doctors to identify cancerous tissue in medical images, self-driving cars to spot road hazards, and Facebook to help users with photo tagging.
    Inception V3 by Google is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception V3 was trained using a dataset of 1,000 classes (See the list of classes here) from the original ImageNet dataset which was trained with over 1 million training images, the Tensorflow version has 1,001 classes which is due to an additional "background' class not used in the original ImageNet. Inception V3 was trained for the ImageNet Large Visual Recognition Challenge where it was a first runner up
A high-level diagram of the model is shown below:
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/inceptionv3.png" height="400px" align="center" width="100%"/>
Convolutional networks are at the core of most stateof-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter
count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over  the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error and 17.3% top-1 error.

### 2.General Design Principles
- 1.Higher dimensional representations are easier to process locally within a network. Increasing the activations per tile in a convolutional network allows for
more disentangled features. The resulting networks
will train faster
- 2.Avoid representational bottlenecks, especially early in the network. Feed-forward networks can be represented by an acyclic graph from the input layer(s) to
the classifier or regressor. This defines a clear direction for the information flow
- 3.Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. For example, before performing a more spread out (e.g. 3 × 3) convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects
- 4.Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. Increasing both the width and the depth of the network can contribute to higher quality networks. However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel. The computational budget should therefore be distributed in a balanced way between the depth and width of the network.

####  `1.Image Preprocessing`: 
Image pre-processing is the name for operations on images at the lowest level of abstraction whose aim is an
improvement of the image data that suppress undesired distortions or enhances some image features important for
further processing. It does not increase image information content. Its methods use the considerable redundancy in
images. Neighbouring pixels corresponding to one object in real images have the same or similar brightness value and
if a distorted pixel can be picked out from the image, it can be restored as an average value of neighbouring pixels.
#### `2.Training Process` : 
While training the model we use approximately 2000 image dataset, around 400 images per mammal, every image is
used multiple times through training process. Computing the layers behind the layer just before the final output layer
which performs the grouping for each image takes a substantial time. As the lower layers of the network are not being
changed their outputs can be stored and used again.
#### `3.Verification and testing process` : 
By testing, we mean evaluating the system in several conditions and observing its behavior, as stated above we are not
just providing single image as input to the inception model instead multiple images multiple times watching for defects.
By verification, we mean producing a compelling argument that the system will not misbehave under a very broad
range of circumstances so the accuracy of model will not be varied.


### 3.Factorizing Convolutions with Large Filter Size
Much of the original gains of the GoogLeNet network [20] arise from a very generous use of dimension reduction. This can be viewed as a special case of factorizing
convolutions in a computationally efficient manner. Consider for example the case of a 1 × 1 convolutional layer followed by a 3 × 3 convolutional layer. In a vision network, it is expected that the outputs of near-by activations are highly correlated. Therefore, we can expect that their activations can be reduced before aggregation and that this should result in similarly expressive local representations

Mini-network replacing the 5 × 5 convolutions: 
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/remotesensing.png"/>

This means that with suitable factorization, we can end up with more disentangled parameters and therefore with faster training. Also, we can use the computational
and memory savings to increase the filter-bank sizes of our network while maintaining our ability to train each model replica on a single comput

### 4.Spatial Factorization into Asymmetric Convolutions

The above results suggest that convolutions with filters larger 3 × 3 a might not be generally useful as they can always be reduced into a sequence of 3 × 3 convolutional 
Mini-network replacing the 3 × 3 convolutions. The lower layer of this network consists of a 3 × 1 convolution with 3 output units

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/10-png.png" align="right"/>
layers. Still we can ask the question whether one should
factorize them into smaller, for example 2×2 convolutions.
However, it turns out that one can do even better than 2 × 2
by using asymmetric convolutions, e.g. n × 1. For example
using a 3 × 1 convolution followed by a 1 × 3 convolution
is equivalent to sliding a two layer network with the same
receptive field as in a 3 × 3 convolution (see figure 3). Still
the two-layer solution is 33% cheaper for the same number
of output filters, if the number of input and output filters is
equal. By comparison, factorizing a 3 × 3 convolution into
a two 2 × 2 convolution represents only a 11% saving of
computation.

n × 1 convolution and the computational
cost saving increases dramatically as n grows (see figure 6).
In practice, we have found that employing this factorization
does not work well on early layers, but it gives very good results on medium grid-sizes (On m×m feature maps, where
m ranges between 12 and 20). On that level, very good results can be achieved by using 1 × 7 convolutions followed
by 7 × 1 convolutions.


- https://core.ac.uk/download/pdf/74351939.pdf
- https://williamkoehrsen.medium.com/facial-recognition-using-googles-convolutional-neural-network-5aa752b4240e
- https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44903.pdf
- https://ijarcce.com/wp-content/uploads/2018/06/IJARCCE-29.pdf
- https://www.cs.colostate.edu/~dwhite54/InceptionNetworkOverview.pdf
- file:///C:/Users/VISHAL%20DHANURE/Downloads/applsci-10-01245-v2.pdf
- https://www.ijrte.org/wp-content/uploads/papers/v8i1S3/A10080681S319.pdf




















<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/community.png" height="400px" align="left"/>
<p style="clear:both;">
<h1><a name="contributing"></a><a name="community"></a> <a href="https://github.com/akshitagupta15june/Face-X">Community</a> and <a href="https://github.com/akshitagupta15june/Face-X/blob/master/CONTRIBUTING.md">Contributing</a></h1>
<p>Please do! Contributions, updates, <a href="https://github.com/akshitagupta15june/Face-X/issues"></a> and <a href=" ">pull requests</a> are welcome. This project is community-built and welcomes collaboration. Contributors are expected to adhere to the <a href="https://gssoc.girlscript.tech/">GOSSC Code of Conduct</a>.
</p>
<p>
Jump into our <a href="https://discord.com/invite/Jmc97prqjb">Discord</a>! Our projects are community-built and welcome collaboration. 👍Be sure to see the <a href="https://github.com/akshitagupta15june/Face-X/blob/master/Readme.md">Face-X Community Welcome Guide</a> for a tour of resources available to you.
</p>
<p>
<i>Not sure where to start?</i> Grab an open issue with the <a href="https://github.com/akshitagupta15june/Face-X/issues">help-wanted label</a>
</p>
**`Open Source First`**
<p>We build projects to provide learning environments, deployment and operational best practices, performance benchmarks, create documentation, share networking opportunities, and more. Our shared commitment to the open source spirit pushes Face-x projects forward.</p>
