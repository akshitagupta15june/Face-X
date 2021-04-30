<p style="text-align:center;" align="center"><a href="https://github.com/Vi1234sh12/Face-X/blob/master/Readme.md"><img align="center" style="margin-bottom:20px;" src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Video-BG-Substraction/Assets/Untitled%20(5).png"  width="90%" /></a><br /><br /></p>

# 1.Introduction
<img src="https://github.com/Vi1234sh12/Face-X/blob/master/Awesome-face-operations/Video-BG-Substraction/Assets/magic-clipper.png" align="right"/>
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

# 2.OverCome this Challenges : 
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

### METHODOLOGY :
synthesise images of
different illumination with both local and global changes, and
then combine them as a unified augmentation method that
covers all scenarios simultaneously.

### 1.Local Changes
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

### 2.Global changes
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

### 3.Combined changes
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

### 4. Illumination-invariant Deep Network
the synthetic images to train multiple deep
learning networks for BGS and evaluate their performances.
Due to the use of images with synthetic illumination changes,
these networks become invariant to lighting conditions.
addition, the ReLu
non-linearity is applied after each convolutional layer. Finally,
once the spatial size has been restored, we add a final 3x3
convolutional layer, followed by a sigmoid layer to convert
the output of the model to a foreground probability map

### Background Subtraction


### Algorithm used: BackgroundSubtractorMOG2

One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel. (Remember, in last case, we took a K gaussian distributions throughout the algorithm). It provides better adaptibility to varying scenes due illumination changes etc.

Here, you have an option of selecting whether shadow to be detected or not. If `detectShadows = True` (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.

### Input:
![resframe](https://user-images.githubusercontent.com/60208804/113537714-106d6e80-95f7-11eb-8590-7d7b12e7760b.jpg)

### Output:
![resmog](https://user-images.githubusercontent.com/60208804/113537728-195e4000-95f7-11eb-8f3d-edcaf79ddc36.jpg)
