<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Video-BG-Substraction/Assets/Untitled%20(5).png"  width="90%" /></a><br /><br /></p>

# 1.Introduction
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Video-BG-Substraction/Assets/remove_image_background.jpg" height="50%" width="50%" align="right"/>
Background subtraction (BGS) has been an active research
area in the past decades. The main task is to differentiate the
foreground (i.e. moving objects) from the background (i.e. the
static parts of a given scene) . A large number of realworld applications, such as person re-identification , object
tracking , gesture recognition , vehicle tracking , video
recognition , action recognition , crowd analysis 
and even use cases of the medical domain , depend
on accurate and robust background subtraction as a first step
of their pipelines.
Sudden illumination changes signify a particularly difficult
challenge, since they cannot be captured by a background
model. Such changes in lighting conditions can be caused
either by weather conditions or electric lights and result in
color changes of a significant amount of pixels. Due to the
difference of visual appearance in consecutive frames, BGS
becomes inaccurate. The timing of these changes could be
short, such as switching a light on/off, or a piece of cloud
blocking the sun, making it tough for the system to adjust to
the new condition in a timely manner

State-of-the-art deep learning algorithms allow adapting to
sudden illumination changes if a huge amount of training data
is provided. However, obtaining labelled data is very costly
and there is only limited datasets available in the community

As a solution, data augmentation methods are proposed to
perform image-based operations on the data, such as mirroring or cropping, to synthesize a larger dataset. However,
simple image tricks cannot effectively generate images with
realistic illumination changes. Another solution is adding a
small amount of noise to create a new, synthetic image that
is similar to the original in context but different in color
distribution. However, since the added noise does not have any
semantic meaning, the synthetic images only slightly increase
the generalisation power of the model, as they do not offer
any additional knowledge of different lighting conditions of
the same scene

Background subtraction is a major preprocessing steps in many vision based applications. For example, consider the cases like visitor counter where a static camera takes the number of visitors entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In all these cases, first you need to extract the person or vehicles alone. Technically, you need to extract the moving foreground from static background.

## 2.OverCome this Challenges : 
To overcome this challenge, we propose a new data augmentation technique by synthesising the light-based effects of
different degrees of brightness. Such effects include shadows
and halos of different size, placed in random locations of the
input image. In addition, global illumination changes are also
included, in order to increase the generalisation abilities of
the model to scenes filmed at various times of the day and
night. Such augmented data allows us to provide extra semantic information to the BGS model in terms of illumination
for better generalisation performance. The results show that
the proposed technique is superior to regular augmentation
methods and can significantly boost the segmentation results
even in scenes that feature illumination conditions unseen to
the model. Our experiments indicate that the proposed method
improves the BGS results in our quantitative and qualitative
evaluations on the benchmark dataset

## 3.METHODOLOGY :
synthesise images of
different illumination with both local and global changes, and
then combine them as a unified augmentation method that
covers all scenarios simultaneously.

#### 1.Local Changes
To synthesize local changes of illumination, we generate
the synthetic images by locally altering the illumination of
the input image, therefore creating either a ”lamp-post” light
source or a shadow effect. First, we randomly select a pixel of
the image that serves as the centre of the illumination circle
to be drawn: p = I(w, h), w ∈ W, h ∈ H, I = W × H, where
W, H the width and height of the input image I respectively.
Once the coordinates of the centre pixel are determined, we
randomly select the diameter d of the illumination circle. Since
we want our model to be robust to both small and large
shadows and flashes of light, we choose the diameter to be
between one fifth and half of the smallest dimension of the
input image: d = k × min(W, H), k ∈ (
1
5
,
1
2
).
Since modifying all pixels within the circle uniformly
generate unrealistic results, we proposed a more sophisticated
approach to model the effect of the light. First, we calculate the
binary mask M1 of the pixels to be altered using the following
formula:

M1(x, y) = 1 ⇔ (x − w)
2 + (y − h)
2 ≤ d
2

This means that the pixels of our mask have the value of 1
if they reside within the drawn circle and zero everywhere
else. We then use the Euclidean Distance Transform (EDT) to
model the light attenuation. Given a binary mask B, EDT is
defined as:
EDTx(B) = minb(||x − b||L2
), ∀b ∈ B, 
where L2 is the Euclidean norm. Now, we can calculate the
mask for local changes M2 by applying the EDT on M1:
M2 = EDT(M1) 
Once the new mask has been created, we proceed to alter
the pixels of the original image that lie within the circle. The
new synthetic image Is is calculated as:
Is = I ± (M2 × z), z ∈ [120, 160], 
where I the original image, M2 the mask calculated with the
distance transform, z a random integer, and ± is either pixelwise addition or subtraction, chosen with probability p = 0.5.
When the addition operation is chosen, a lamp-post effect will
be created in a random part of the image. Conversely, the
subtraction operation creates shadows. The application of the
aforementioned local masks are depicted in . It can
be seen that the final light source effect looks realistic

#### 2.Global changes
global illumination changes can occur. For
example, a lightning during a storm may instantly increase
the brightness, and once the rain is over the global illumination will change again. In order to model such illumination
changes, we need to alter the pixels of the whole image, rather
than a small patch.
We synthesize global illumination changes as:
Is = I ± z, z ∈ [40, 80],

where I, z and ± are as previously defined. In this case the
illumination noise z needs to be slightly diminished, since the
whole image is affected.

#### 3.Combined changes
To capture both local and global illumination changes in
the scene, we combine equation 4 and equation 5 into the
following:
Is = z1 ± (I ± (M2 × z2), z1 ∈ [40, 80], z2 ∈ [120, 160] (6)
Sample images synthesised from our system can be found
in figure 2. Since both the positioning and the intensity
of the masks is random, this method can effectively cover
all kinds of illumination changes. Additionally, hundreds of
different synthetic images can be generated from a single
frame. Therefore, given a small video, we can generate enough
unique synthetic images to train a very deep network.

#### 4.Illumination-invariant Deep Network
the synthetic images to train multiple deep
learning networks for BGS and evaluate their performances.
Due to the use of images with synthetic illumination changes,
these networks become invariant to lighting conditions.
addition, the ReLu
non-linearity is applied after each convolutional layer. Finally,
once the spatial size has been restored, we add a final 3x3
convolutional layer, followed by a sigmoid layer to convert
the output of the model to a foreground probability map

## 4.Background Subtraction

#### 1.Preprocessing: 
we firstly use simple temporal and spacial smoothing to reduce camera noise. Smoothing
can also be used to remove transient environmental
noise. Then to ensure real-time capabilities, we have
to decide on the frame-size and frame-rate which are
the determining factors of the data processing rate.
Another key issue in preprocessing is the data format
used by the particular background subtraction algorithm. Most of the algorithms handle only luminance
intensity, which is one scalar value per each pixel.
However, color image, in either RGB or YUV color
space, is becoming more popular these days. In the
case of a mismatch, some time will be spent on converting the output data from the driver of the camera
to the required input data type for the algorithm

#### 2.Color Model:
The input to our algorithm is a time series of spatially registered and time-synchronized color images
obtained by a static camera in the YUV color space.
This allows us to separate the luminance and chroma
components which has been decoded in a 4:2:0 ratio
by our camera hardware. The observation at pixel i at
time t can then be written as:
IC = (Y, U, V )

#### 3.Texture Model
In our implementation, we used the texture information available in the image in our algorithm by including and considering the surrounding neighbors of the
pixel. This can be obtained several ways. In our implementation, we have decided to take the image gradient of the Y component in the x and y directions.
The gradient is then denoted as follows:


```
IG = GY = (Gx, Gy)
        =  (G^2 x + G^2y)^2
        
 ```
 
This is obtained using a Sobel operator. The Sobel
operator combines Gaussian smoothing and differentiation so the result is more robust to noise. We have
decided to use the following Sobel filter after experimenting with several other filters.
```
1 2 1
2 0 2
1 2 1

```


the behavior of the camera sensor to
the neighboring pixel and the effect of the surrounding
pixels for any given pixel. Notice that the edges of the
red, blue and green color zones are not uniform.

#### 4.Shadow Model : 
At this point, we still have a major inconvenience in
the model. Shadows are not translated as being part of
the background and we definitely do not want them to
be considered as an object of interest. To remedy this,
we have chosen to classify shadows as regions in the
image that differ in Y but U,V rest unchanged. The
shadow vector Is is then given as follows:

IS = (YS, U, V )

where YS = Y ± ∆Y . This is in fact not a disadvantage. Since the Y component is only sensible to
illumination changes, it is in fact redundant for foreground or background object discrimination.

#### 5.Learning Vector Model : 
We can now form the final vectors which we are going
to observe at pixel level. They are given as follows:
B1 = (GY , U, V )
B2 = (YS)
B3 = (U, V )

where B1 is the object background model which combines the color and texture information but ignores
luminance that is incorporated in B2 which is the
shadow background model and B3 is taken as a safety

## 5.Development of Background Subtraction Algorithm
During run-time, the reference image is subtracted from the current image to obtain a mask which will highlight all foreground objects. Once the mask is obtained, the background model can be updated. There a four major steps in the background subtraction algorithm. These are detailed in
the following subsections

#### 1.Background Learning :
for each incoming frame, at pixel level, we store the number of samples n, the sum of the observed vector a, b, c grouped as U and the sum of
the cross-product of the observed vector d, e and f grouped as V .

<img src=""/>

This stage will be defined as the learning phase
and is required for initialization. From our experiments, about 100 frames is necessary to sufficiently
learn the variations in the background. This corresponds to about 2 to 4 seconds of initialization using
our camera

#### 2.Parameter Estimation :
At the end of the learning phase, the required variables
for the Gaussian models need to be calculated. They
are given as follows:
µ1 =
a
n
, µ2 =
b
n
, µ3 =
c
n
CB1 =
1
n
(d) −
1
n2
(a × a
T
)
CB2 =
1
n
(e) −
1
n2
(b × b
T
)
CB3 =
1
n
(f) −
1
n2
(c × c
T
)

These variables are calculated in a very efficient
manner such that real-time compatibility is always
maintained. In our implementation this stage is referred to the Gaussian distribution parameter estimation stage.

## 6.Foreground Detection : 
Foreground detection compares the input video frame
with the background reference image and identifies
candidate foreground pixels from the input frame. The
most commonly used approach for foreground detection is to check whether the input pixel is significantly different from the corresponding background
estimate. In the MoG method, we can do this by expanding the characterizing equation for each Gaussian
distribution.
P(Bj ) = f( Bj , µj , CBj
)
= ( 2π
n
2 ×
q
|CBj
| )
−1
× exp { (Bj − µj )
T
× C
−1
Bj
× (Bj − µj ) }

where j = 1, 2, 3. To compute this probability, it is
not entirely necessary to evaluate the whole expression. The first term is a constant and the remaining term is popularly known as the Mahalanobis Distance (MD). Hence, the decision making process is
streamed down to the following three categories,

MDj = (Bj − µj )
T × C
−1
Bj
× (Bj − µj )

#### 1.Model Update
µt =
Ut
nt
Ct =
1
nt−1 + 1
×
nhVt−1 + (Bt × B
T
t
)
i
−
h
(Ut−1 + Bt) × (U
T
t−1 + BT
t
)
nt−1 + 1
io

In order to fully capture the changing dynamic environment, the background model has to be updated. In
the proposed algorithm, we need to update the mean
vector, the covariance matrix and its inverse. In order
to avoid recalculating all the coefficients altogether, a
recursive update is preferred. The will allow us to obtain the new values of the model parameters from the
old ones. From the general expression for the mean
vector and the correlated Gaussian distributions,

## 7.Data Validation:
We define data validation as the process of improving
the candidate foreground mask based on information
obtained from outside the background model. Inaccuracies in threshold levels, signal noise and uncertainty in the background model can sometimes lead
to pixels easily mistaken as true foreground objects
and typically results in small false-positive or falsenegative regions distributed randomly across the candidate mask. The most common approach is to combine morphological filtering and connected component grouping to eliminate these regions. Applying morphological filtering on foreground masks eliminates isolated foreground pixels and merges nearby disconnected foreground regions.
<img src=""/>

Opening and closing are two important operators
that are both derived from the fundamental operations
of erosion and dilation. These operators are normally
applied to binary images. The basic effect of an opening is somewhat like erosion in that it tends to remove
some of the foreground (bright) pixels from the edges
of regions of foreground pixels. However, it is less
destructive than erosion in general.

the result after applying the opening (erosion) filter to
 which eliminates the foreground pixels in the
scene that is misjudged by the background subtraction
algorithm or due to improper thresholding. We use a
5 × 5 mask to eliminate this foreground noise that is
not part of the scene or the object of interest. Another
added advantage of using this filter is to remove any
unconnected pixel that does not belong to the object.
The size of the mask is determined empirically from
our experiments.

applying an opening and closing filter to We apply a 5 × 5 closing mask after applying the
opening filter. The effect of applying both operators
is to preserve background regions that have a similar shape to the structuring element, or that can completely contain the structuring element, while eliminating all other regions of background pixels. The advantage of this is that the foreground pixels which are
misjudged as background pixels make the object to
look as if it is not connected. Opening and closing are
themselves often used in combination to achieve more
subtle results. It is very clear that applying morphological opening and closing filters has a positive effect
on the process of extracting object from the scene by
removing the noise from the subtracted foreground.


### Algorithm used: BackgroundSubtractorMOG2

One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

Here, you have an option of selecting whether shadow to be detected or not. If `detectShadows = True` (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.

### Input:
![resframe](https://user-images.githubusercontent.com/60208804/113537714-106d6e80-95f7-11eb-8590-7d7b12e7760b.jpg)

### Output:
![resmog](https://user-images.githubusercontent.com/60208804/113537728-195e4000-95f7-11eb-8f3d-edcaf79ddc36.jpg)
