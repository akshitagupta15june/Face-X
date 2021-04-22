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
algorithmic techniques making this task possible are often called `ICA `, as they factorise the
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

<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Face%20recognition%20using%20ICA%20(Independent%20Component%20Analysis)/Images/ICA.PNG" height="400px"  />
