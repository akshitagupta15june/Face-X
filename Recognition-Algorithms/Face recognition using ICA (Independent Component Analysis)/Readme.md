<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-using-IOT/readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20Recognition%20Using%20Inception%20V3%20model/Images/Unt1.png" height="80%" width="70%" /></a><br /><br /></p>

# Face recognition using ICA (Independent Component Analysis)

## Introduction 
`Independent component analysis (ICA)`, a generalization of PCA, is one such method. We used a version of ICA derived from the principle of optimal information transfer through sigmoidal neurons. ICA was performed on face images in the `FERET database` under two different architectures, one which treated the images as random variables and the pixels as outcomes, and a second which treated the pixels as random variables and the images as outcomes. The first architecture found spatially local basis images for the faces. The second architecture produced a factorial face code. Both ICA representations were superior to representations based on PCA for recognizing faces across days and changes in expression. A classifier that combined the two ICA representations gave the best performance.
 
 
 ## Independent Component Analysis
 ICA is a data analysis tool derived from the `"source separation"` signal processing techniques.
The aim of source separation is to recover original signals Si, from known observations Xj,
where each observation is an (unknown) mixture of the original signals. Under the
assumption that the original signals Si are statistically independent, and under mild conditions
on the mixture, it is possible to recover the original signals from the observations. The
algorithmic techniques making this task possible are often called `ICA`, as they factorise the
observations as a combination of original sources. If the mixing is linear, ICA estimates the
inverse of the mixing matrix. 
              The number of observations N (1 ≤ j ≤ N) must be at least equal
to the number of original signals M (1 ≤ i ≤ M); often it is assumed that N = M. It is not
necessary to have signals Xj to consider using ICA: Xj may also be multi-dimensional data
(vectors). Assuming that each Xj is an unknown, different combination of original "source
vectors" Si, ICA will expand each signal Xj into a weighted sum of source vectors Si (ICA
estimates both the source vectors Si and the coefficients of the weighted sum). 

This view is not far from the PCA expansion: the eigenvectors of PCA are replaced by the independent
source vectors in ICA. For a review of ICA techniques and properties, see for example [5].
In our case, we assume that the faces in the learning set, viewed as high-dimensional vectors,
are linear combination of unknown independent source vectors. This may not be strictly true,
depending on the respective number of images in the database and size of an image in pixels,
but in any case ICA will find estimates of independent source vectors that are optimal to
reconstruct the original images (observations) in the least-square sense. The idea is then to
substitute PCA with ICA, and to use the coefficients of the ICA expansion (instead of those
from PCA) as feature vectors for the faces. It is expected that, ICA source vectors being
independent (instead of PCA eigenvectors being uncorrelated only), they will be closer to
natural features of images, and thus more able to represent differences between faces.

 The optimal weight w on x for maximizing information transfer is the one that best matches the probability
density of x to the slope of the nonlinearity. The optimal w produces the flattest possible output density, which in
other words, maximizes the entropy of the output.

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20recognition%20using%20ICA%20(Independent%20Component%20Analysis)/Images/ICA.PNG" height="400px"  />

Optimal information flow in sigmoidal neurons. The input x is passed through a nonlinear function, g(x).
The information in the output density  `fy(y)`  depends on matching the mean and variance of f, (x) to the slope and
threshold of `g(x)`. Right: `fy(y)` is plotted for different values of the weight, w. The optimal weight, w,,t transmits
the most information

The optimal weight is found by gradient ascent on the entropy of the output, y with respect to w. When there are
multiple inputs and outputs, maximizing the joint entropy of the output encourages the individual outputs to move
towards statistical independence. When the form of the nonlinear transfer function g is the same as the cumulative
density functions of the underlying independent components (up to a scaling and translation) it can be shown that
maximizing the mutual information between the `input X`  and the `output Y ` also minimizes the mutual information
between the  `Ui`

The update rule for the weight matrix, W, for multiple inputs and outputs is given by
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20recognition%20using%20ICA%20(Independent%20Component%20Analysis)/Images/ICA1.PNG" height="100px" align="right"/>

We employed the logistic transfer function, g(u) = (1 /1 + e^-N), giving y' = (1 - 2yi).

The algorithm includes a "sphering" step prior to learningg The row means are subtracted from the dataset,
X, and then X is passed through the zero-phase whitening filter, W,, which is twice the inverse square root of the
covariance matrix:
<br/><br />
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20recognition%20using%20ICA%20(Independent%20Component%20Analysis)/Images/ICA2.PNG" height="80px" align="center"/>

This removes both the first and the second order statistics of the data; both the mean and covariances are set to
zero and the variances are equalized. The full transform from the zero-mean input was calculated as the product of
the sphering matrix and the learned matrix, WI = W * Wz. The pre-whitening filter in the ICA algorithm has the
Mexican-hat shape of retinal ganglion cell receptive fields which remove much of the variability due to lighting.

## 2. INDEPENDENT COMPONENT REPRESENTATIONS OF FACE IMAGES 
### 1. Statistically independent basis images
To find a set of statistically independent basis images for the set of faces, we separated the independent components
of the face images according to the image synthesis model of Figure 2. The face images in X were assumed to be
a linear mixture of an unknown set of statistically independent source images S, where A is an unknown mixing
matrix. The sources were recovered by a matrix of learned filters, WI, which produced statistically independent
outputs, U. This synthesis model is related to that used to perform blind separation on an unknown mixture of
auditory signals1 and to separate the sources of EEG signals7 and fMRI image

 Image synthesis model. For finding a set of independent component images, the images in X are
considered to be a linear combination of statistically independent basis images, S, where A is an unknown mixing
matrix. The basis images were recovered by a matrix of learned filters, WI, that produced statistically independent
outputs, U.
The images comprised the rows of the input matrix, X. With the input images in the rows of X, the ICA
outputs in the rows of WIX = U were also images, and provided a set of independent basis images for the faces
. These basis images can be considered a set of statistically independent facial features, where the pixel
values in each feature image were statistically independent from the pixel values in the other feature images. The
ICA representation consisted of the coefficients for the linear combination of independent basis images in U that
comprised each face image .
The number of independent components found by the ICA algorithm corresponds with the dimensionality of the
input. In order to have control over the number of independent components extracted by the algorithm, instead
of performing ICA on the n original images, we performed ICA on a set of m linear combinations of those images,


The independent basis image representation consisted of the coefficients, b, for the linear combination of
independent basis images, u, that comprised each face image x.

  where m < n. Recall that the image synthesis model assumes that the images in X are a linear combination of a
set of unknown statistically independent sources. The image synthesis model is unaffected by replacing the original
images with some other linear combination of the images. 

### 2.A factorial code 

The previous analysis produced statistically independent basis images. The representational code consisted of the
set of coefficients for the linear combination of the independent basis images from which each face image could
be reconstructed. Although the basis images were spatially independent, the coefficients were not. By altering the
architecture of the independent component analysis, we defined a second representation in which the coefficients were
statistically independent, in other words, the new ICA outputs formed a factorial code for the face images. Instead
of separating the face images to find sets of independent images, as in Architecture 1, we separated the elements
of the face representation to find a set of independent variables for coding the faces. The alteration in architecture
corresponded to transposing the input matrix X such that the images were in columns and the pixels in row
Under this architecture, the filters (rows of WI) were images, as were the columns of A = W;'. The
columns of A formed a new set of basis images for the faces, and the coefficients for reconstructing each face were
contained in the columns of the ICA outputs, U

Two architectures for performing ICA on images. Left: Architecture for finding statistically independent
basis images. Performing source separation on the face images produced independent component images in the rows
of U. Right: Architecture for finding a factorial code. Performing source separation on the pixels produced a factorial
code in the columns of the output matrix, U
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20recognition%20using%20ICA%20(Independent%20Component%20Analysis)/Images/A-basic-ICA-model-for-blind-source-separation.png" align="center"/>



## 3. FACE RECOGNITION PERFORMANCE

Face recognition performance was evaluated for the two ICA representations using the FERET face database.12
The data set contained images of 425 individuals. There were up to four frontal views of each individual: a neutral
expression and a change of expression from one session, and a neutral expression and change of expression from a
second session that occurred up to two years after the first. Examples of the four views are shown in Figure 7.
The two algorithms were trained on a single frontal view of each individual, and tested for recognition under three
different conditions: same session, different expression; different session, same expression; and different session,
different expression

Coordinates for eye and mouth locations were provided with the FERET database. These coordinates were used
to center the face images, crop and scale them to 60 x 50 pixels based on the area of the triangle defined by the eyes
and mouth. The luminance was normalized. For the subsequent analyses, the rows of the images were concatenated
to produce 1 x 3000 dimensional vectors.
### 1. Independent basis architecture
The principal component axes of the Training Set were found by calculating the eigenvectors of the pixelwise covariance matrix over the set of face images. Independent component analysis was then performed on the first 200
of these eigenvectors, Pzoo. The 1 x 3000 eigenvectors in Pzoo comprised the rows of the 200 x 3000 input matrix
X. The input matrix X was sphered according to Equation 2, and the weights, W, were updated according to
Equation 1 for 1600 iterations. The learning rate was initialized at 0.001 and annealed down to 0.0001. Training

took 90 minutes on a Dec Alpha 2100a quad processor. Following training, a set of statistically independent source
images were contained in the rows of the output matrix U.
Figure 8 shows a subset of 25 source images. A set of principal component basis images (PCA axes), are shown
in Figure 9 for comparison. The ICA basis images were more spatially local than the principal component basis
images. Two factors contribute to the local property of the ICA basis images: The ICA algorithm produces sparse
 output^,^ and secondly, most of the statistical dependencies may be in spatially proximal image locations.
These source images in the rows of U were used as the basis of the ICA representation. The coefficients for the
zero-mean training images were contained in the rows of B = Rzoo * w;' according to Equation 3, and coefficients
for the test images were contained in the rows of Btest = RTest * wil where RT~~~ = Test * Pzoo.
Face recognition performance was evaluated for the coefficient vectors b by the nearest neighbor algorithm.
Coefficient vectors in the test set were assigned the class label of the coefficient vector in the training set with the
most similar angle, as evaluated by the cosine

Face recognition performance for the principal component representation was evaluated by an identical procedure,
using the principal component coefficients contained in the rows of R. Figure 10 gives face recognition performance
with both the ICA and the PCA based representations. Face recognition performance with the ICA representation
was superior to that with the PCA representation. Recognition performance is also shown for the PCA based representation using the first 20 principal component vectors, which was the representation used by Pentland, Moghaddam
and Starner.13 Best performance for PCA was obtained using 200 coefficients. Excluding the first 1, 2, or 3 principal
components did not improve PCA performance, nor did selecting intermediate ranges of components from 20 through

`D = B.Test * B.Train / || B.Test || * || B.Train||`

### 2. Factorial code architecture 

Face recognition performance was evaluated by the nearest neighbor procedure of Section 3.1. Figure 13 compares the face recognition performance using the ICA factorial code representation to the ICA representation using
independent basis images and to the PCA representation, each with 200 coefficients. The two ICA representations
gave similar recognition performance, and both outperformed the PCA representation. Class discriminability of the
new ICA coefficients was calculated according to Equation 5. Again, the ICA coefficients had consistently higher
class discriminabilility than the PCA coefficients. Face recognition performance was reevaluated using subsets of
components selected by class discriminability. Best performance was obtained with the 180 most discriminable components. The improvement in face recognition obtained by the discriminability analysis is illustrated by the gray
extensions

 Basis images for the ICA factorial representation, obtained by training on the first 200 principal
component coeficients of the face images, oriented in columns of the input. Bases were contained in the columns of
A= W,'. 
