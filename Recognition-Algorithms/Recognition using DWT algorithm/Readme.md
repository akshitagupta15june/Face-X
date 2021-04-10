# Recognition using DWT Alogrithm
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Recognition%20using%20DWT%20algorithm/Images/C.png" height="440px" align="left"/>

##  Introduction
Image compression is important for many applications that
involve huge data storage, transmission and retrieval such
as for multimedia, documents, videoconferencing, and medical
imaging. Uncompressed images require
considerable storage capacity and transmission bandwidth.
The objective of image compression technique is to reduce
redundancy of the image data in order to be able to store
or transmit data in an efficient form. This results in the
reduction of file size and allows more images to be stored
in a given amount of disk or memory space [1-3]. In a
lossless compression algorithm, compressed data can be
used to recreate an exact replica of the original; no
information is lost to the compression process. This type
of compression is also known as entropy coding. This
name comes from the fact that a compressed signal is
generally more random than the original; patterns are
removed when a signal is compressed. While lossless
compression is useful for exact reconstruction, it generally
does not provide sufficiently high compression ratios to be
truly useful in image compression.

In numerical analysis and functional analysis, a discrete wavelet transform (DWT) is any wavelet transform for which the wavelets are discretely sampled. As with other wavelet transforms, a key advantage it has over Fourier transforms is temporal resolution: it captures both frequency and location information (location in time).

## Image Compression and Reconstruction
 The image compression system is composed of
two distinct structural blocks: an encoder and a decoder.
Image f(x,y) is fed into the encoder, which creates a set of
symbols from the input data and uses them to represent
the image. Image f
ˆ (x,y) denotes an approximation of the
input image that results from compressing and
subsequently decompressing the input image. 

### 1.PROPOSED SYSTEM
<img src ="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Recognition%20using%20DWT%20algorithm/Images/1-Figure.png" align="right"/>
concerns face recognition using multi
resolution analysis, namely wavelet decomposition. The wavelet transform provides a powerful
mathematical tool for analysing non-stationary signals. The images used in this paper have been
taken from the ORL database.
  A. Wavelet Transform
Wavelet Transform is a popular tool in image processing and computer vision. Many applications, such as compression, detection, recognition, image retrieval et al. have been investigated. WT has the nice features of space frequency localization and multi-resolutions.
1-D continuous WT of function f(t) defined as

is wavelet basis function

is called mother wavelet which has at least one vanishing moment. The arguments and denote the scale and location parameters, respectively. The oscillation in the basis functions increases with a decrease in a. The transform can be discretized by restraining and to a discrete lattice. 2-D DWT is generally carried out using a separable approach, by first calculating the 1-D DWT on the rows, and then
the `1-D DWT` on the columns :
`DWTn[DWTm[x[m,n]]`. 
Two-dimensional WT decomposes an image into 4 “subbands” that are localized in frequency and orientation, by LL, HL, LH, HH.
Each of these sub bands can be thought of as a smaller version of the image representing different image properties. The band LL is a coarser approximation to the original image. The bands LH and HL record the changes of the image along horizontal and vertical directions, respectively. The HH band shows the high frequency component of the
image. Second level decomposition can then be conducted on the LL sub band. Fig.2 shows a two-level wavelet decomposition of two images of size 112X92 pixels.
They found that facial expressions and small occlusions affect the intensity manifold locally. Under frequency-based representation, only high-frequency spectrum is
affected, called high-frequency phenomenon.

### B. Haar Wavelet Transform (HWT)
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Recognition%20using%20DWT%20algorithm/Images/hqdefault%20.jpg" align="right"/>

HWT decomposition works on an averaging and differencing process as follows: 
 
 It can be seen that the number of decomposition steps is `22 = 4`.
Given an original image, the Harr wavelet transform method separates high frequency and low frequency bands of the image by high-pass and low-pass filters from the horizontal direction and so does the vertical direction of the image.
 - 1) Two-dimensional Haar wavelet transforms: 
 There are two ways we can use wavelets to transform the pixel values within an image. Each is a generalization to two dimensions of the one-dimensional wavelet transform .To obtain the standard decomposition of an image; we first apply the one-dimensional wavelet transform to each row of pixel values. This operation gives us an average value along with detail coefficients for each row. Next, we treat these transformed rows as if they were themselves an image and apply the one-dimensional transform to each column. The resulting values are all detail coefficients except for a single overall average coefficient. The algorithm below computes the standard decomposition. Figure 3 illustrates each step of its operation.
 
```
procedure StandardDecomposition (C: array [1. . h,
1. . w] of reals)
for row 1 to h do
Decomposition (C[row, 1. . w])
end for
for col 1 to w do
Decomposition (C[1. . h, col])
end for
end procedure
```
The second type of two-dimensional wavelet  transform, called the nonstandard decomposition, alternates between operations on rowsand columns. First, we perform one step of horizontal pairwise averaging and differencing on the pixel values in averaging and differencing to each column of the result. To complete the transformation, we repeat this
process recursively only on the quadrant containing averages in both directions.
```
the steps involved in the nonstandard decomposition.
procedure NonstandardDecomposition(C: array [1. .
h, 1. . h] of reals)
C C=h (normalize input coefficients)
while h > 1 do
for row 1 to h do
DecompositionStep (C[row, 1. . h])
end for
for col 1 to h do
DecompositionStep (C[1. . h, col])
end for
h h=2
end while
end procedure 

```
B. Biorthogonal 9/7
The family of biorthogonal filters considered here are of lengths 9 and 7 and represent a super-set of the 9/7 pair that is used for face recognition. A biorthogonal
wavelet is a wavelet where the associated wavelet transform is invertible but not necessarily orthogonal. Designing biorthogonal wavelets allows more degrees of freedom than orthogonal wavelets. One additional degree of freedom is the possibility to construct symmetric wavelet functions biorthogonal filters are the shortest odd length filter pair with the following properties:
 - 1. minimum number of vanishing moments
    (which is 2 for any linear phase odd length filter);
 - 2. two degrees of freedom;
    the structure of one stage of a twochannel biorthogonal filter bank. For the 9/7 DWT, filters H(z) and G(z) are symmetric FIR filters with nine and seven taps,         respectively. Traditionally, the filters are implemented using convolution. This implementation is non-polyphase, and suffers from inefficient hardware utility and low  throughput. 


































<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Recognition-Algorithms/Recognition%20using%20DWT%20algorithm/Images/Gorup1.png" height="470px" align="left"/>
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

**Open Source First**

 best practices for managing all aspects of distributed services. Our shared commitment to the open-source spirit push the Face-X community and its projects forward.</p>
