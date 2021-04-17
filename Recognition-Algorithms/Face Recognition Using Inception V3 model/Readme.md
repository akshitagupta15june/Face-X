<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/Unt1.png" height="80%" width="70%" /></a><br /><br /></p>

# Face Recognition Using Inception V3 model
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/facial.png" align="right"/>

### 1.Introduction 
Image classification `recognition` is one of the foremost capabilities of deep neural networks. Inception-v3 is one of the most popular convolutional neural network models for recognizing objects in images. Deep learning-powered image recognition is used by doctors to identify cancerous tissue in medical images, self-driving cars to spot road hazards, and Facebook to help users with photo tagging.
    `Inception V3 by Google` is the 3rd version in a series of Deep Learning Convolutional Architectures. Inception V3 was trained using a dataset of 1,000 classes (See the list of classes here) from the original ImageNet dataset which was trained with over 1 million training images, the Tensorflow version has 1,001 classes which is due to an additional "background' class not used in the original ImageNet. Inception V3 was trained for the `ImageNet` Large Visual Recognition Challenge where it was a first runner up
A high-level diagram of the model is shown below:
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/inceptionv3.png" height="400px" align="center" width="100%"/>
Convolutional networks are at the core of most stateof-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter
count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we are exploring ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the `ILSVRC 2012` classification challenge validation set demonstrate substantial gains over  the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report `3.5% top-5 error`and `17.3% top-1 error`.

### 2.About The `Inception` Versions
There are 4 versions. The first GoogLeNet must be the `Inception-v1`, but there are numerous typos in Inception-v3 which lead to wrong descriptions about Inception versions. These maybe due to the intense ILSVRC competition at that moment. Consequently, there are many reviews in the internet mixing up between `v2` and `v3`. Some of the reviews even think that v2 and v3 are the same with only some minor different settings.
Nevertheless, in` Inception-v4` , Google has a much more clear description about the version issue:
“The Inception deep convolutional architecture was introduced as`GoogLeNet` in (Szegedy et al. 2015a), here named Inception-v1. Later the Inception architecture was refined in various ways, first by the introduction of batch normalization (Ioffe and Szegedy 2015) (Inception-v2). Later by additional factorization ideas in the third iteration (Szegedy et al. 2015b) which will be referred to as Inception-v3 in this report.”
Thus, the BN-Inception / Inception-v2  is talking about batch normalization while Inception-v3 [1] is talking about factorization ideas.

### `Architectural Changes in Inception V3`:

Inception V3 is similar to and contains all the features of Inception V2 with following changes/additions:

- `Use of RMSprop optimizer`.
- `Batch Normalization in the fully connected layer of Auxiliary classifier`.
- `Use of 7×7 factorized Convolution`
- `Label Smoothing Regularization`: It is a method to regularize the classifier by estimating the effect of label-dropout during training. It prevents the classifier to predict too confidently a class. The addition of label smoothing gives 0.2% improvement from the error rate.

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


### 3.`Factorizing Convolutions with Large Filter Size`
Much of the original gains of the GoogLeNet network  arise from a very generous use of dimension reduction. This can be viewed as a special case of factorizing
convolutions in a computationally efficient manner. Consider for example the case of a` 1 × 1` convolutional layer followed by a` 3 × 3 convolutional layer`. In a vision network, it is expected that the outputs of near-by activations are highly correlated. Therefore, we can expect that their activations can be reduced before aggregation and that this should result in similarly expressive local representations

`Mini-network replacing the 5 × 5 convolutions:` 
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/remotesensing.png"/>

This means that with suitable factorization, we can end up with more disentangled parameters and therefore with faster training. Also, we can use the computational
and memory savings to increase the filter-bank sizes of our network while maintaining our ability to train each model replica on a single comput

### 4.Spatial Factorization into `Asymmetric Convolutions`

The above results suggest that convolutions with filters larger 3 × 3 a might not be generally useful as they can always be reduced into a sequence of 3 × 3 convolutional 
Mini-network replacing the `3 × 3 convolution`s. The lower layer of this network consists of a 3 × 1 convolution with 3 output units

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/10-png.png" align="right"/>

layers. Still we can ask the question whether one should factorize them into smaller, for example 2×2 convolutions. However, it turns out that one can do even better than 2 × 2 by using asymmetric convolutions, e.g. n × 1. For example using a 3 × 1 convolution followed by a 1 × 3 convolution is equivalent to sliding a two layer network with the same receptive field as in a 3 × 3 convolution . Still the two-layer solution is 33% cheaper for the same number of output filters, if the number of input and output filters is equal. By comparison, factorizing a 3 × 3 convolution into a two 2 × 2 convolution represents only a 11% saving of
computation.

n × 1 convolution and the computational cost saving increases dramatically as n grows . In practice, we have found that employing this factorization
does not work well on early layers, but it gives very good results on medium grid-sizes `(On m×m feature maps, where m ranges between 12 and 20)`. On that level, very good results can be achieved by using` 1 × 7 convolutions` followed
by 7 × 1 convolutions.

### 5.`Auxiliary classifier`: 

introduced the notion of auxiliary classifiers to improve the convergence of very deep networks. The original motivation was to push useful gradients to the lower layers to make them immediately useful and improve the convergence during training by combating the vanishing gradient problem in very deep networks. Also Lee et al argues that auxiliary classifiers promote more stable learning and better convergence. Interestingly, we found that auxiliary classifiers did not result in improved convergence
early in the training: the training progression of network with and without side head looks virtually identical before both models reach high accuracy. Near the end of training, the network with the auxiliary branches starts to overtake the accuracy of the network without any auxiliary branch and reaches a slightly higher plateau 

used two side-heads at different stages in the network. The removal of the lower auxiliary branch did not have any adverse effect on the final quality of the network.
Together with the earlier observation in the previous paragraph, this means that original the hypothesis of  that these branches help evolving the low-level features is most
likely misplaced. Instead, we argue that the auxiliary classifiers act as regularizer. This is supported by the fact that the main classifier of the network performs better if the side branch is batch-normalized or has a dropout layer. This also gives a weak supporting evidence for the conjecture that batch normalization acts as a regularizer.


an `auxiliary classifier` is a small CNN inserted between layers during training, and the loss incurred is added to the main network loss. In GoogLeNet auxiliary classifiers were used for a deeper network, whereas in` Inception v3 an auxiliary classifier` acts as a regularizer.
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/pasted.png" width="500px" height="350px" align="right" />


### 6.Grid size reduction:

Traditionally, convolutional networks used some pooling operation to decrease the grid size of the feature maps. In order to avoid a representational bottleneck, before applying maximum or average pooling the activation dimension of the network filters is expanded. For example, starting a d×d grid with k filters, if we would like to arrive at a `d/2 × d/2` grid with 2k filters, we first need to compute a stride-1 con grid size reduction is usually done by pooling operations. However, to combat the bottlenecks of computational cost, a more efficient technique is proposed:

volution with 2k filters and then apply an additional pooling step. This means that the overall computational cost is dominated by the expensive convolution on the larger grid using `2d^2k^2` operations. One possibility would be to switch to pooling with convolution and therefore resulting in `2(d/2)^2*k^2 `reducing the computational cost by a quarter. However, this creates a representational bottlenecks as the overall dimensionality of the representation drops to` ( d/2 )^2k` resulting in less expressive network
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/pas.png" height="350px" width="500px" align="right"/>

### 7.Performance on Lower Resolution Input

A typical use-case of vision networks is for the the postclassification of detection, for example in the Multibox 
context. This includes the analysis of a relative small patch
of the image containing a single object with some context.
The tasks is to decide whether the center part of the patch
corresponds to some object and determine the class of the
object if it does. The challenge is that objects tend to be

The common wisdom is that models employing higher
resolution receptive fields tend to result in significantly improved recognition performance. However it is important to
distinguish between the effect of the increased resolution of
the first layer receptive field and the effects of larger model
capacitance and computation. If we just change the resolution of the input without further adjustment to the model,
then we end up using computationally much cheaper models to solve more difficult tasks. Of course, it is natural,
that these solutions loose out already because of the reduced
computational effort. In order to make an accurate assessment, the model needs to analyze vague hints in order to
be able to “hallucinate” the fine details. This is computationally costly. The question remains therefore: how much
does higher input resolution helps if the computational effort is kept constant. One simple way to ensure constant
effort is to reduce the strides of the first two layer in the
case of lower resolution input, or by simply removing the
first pooling layer of the network

For this purpose we have performed the following three
experiments:
- [X] 299 × 299 receptive field with stride 2 and maximum
pooling after the first layer.
- [X] 151 × 151 receptive field with stride 1 and maximum
pooling after the first layer.
- [X] 79 × 79 receptive field with stride 1 and without pooling after the first layer.


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

