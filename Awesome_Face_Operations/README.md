# Awesome-face-operations

![image](https://user-images.githubusercontent.com/78999467/112627758-1bd3d380-8e5a-11eb-9c41-39a98e11c1c1.png)

# Face Morphing
This is a tool that creates a morphing effect. It takes two facial images as input and returns morphing from the first image to the second.
### Example:
![image](https://user-images.githubusercontent.com/78999467/112639582-87bd3880-8e68-11eb-8506-5a3800aef529.png)

### Steps:
```diff
- Find point-to-point correspondences between the two images.
- Find the Delaunay Triangulation for the average of these points.
- Using these corresponding triangles in both initial and final images, perform Warping and Alpha Blending and obtain intermediate images. 
```

# Converting an image into a ghost image.

Used OpenCV and Numpy to convert an image into a ghost image.

### Steps:
```diff
- Imported the required libraries ( Numpy, Matplotlib, Cv2)
- Read the input image using cv2
```
### Methods applied Using Cv2
```diff
- Used Bilateral Filter
- Used Median Blur
- Used Adaptive Threshold
- Used Bitwise Xor
- Finally converted the image into a ghost image
```

### Original Image
![image](https://user-images.githubusercontent.com/78999467/112639805-c6eb8980-8e68-11eb-9312-312a5df65aa1.png)



### Ghost Image
![image](https://user-images.githubusercontent.com/78999467/112639825-cce16a80-8e68-11eb-9920-7d515ff158e4.png)



# Pencil Sketch In Python Using OpenCV
### OpenCV

OpenCV is an open-source computer vision and machine learning software library. It is a BSD-licence product thus free for both business and academic purposes. OpenCV is written natively in C/C++. It has C++, C, Python, and Java interfaces and supports Windows, Linux, Mac OS, iOS, and Android. OpenCV was designed for computational efficiency and targeted for real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

### Pencil Sketch in OpenCV

OpenCV 3 comes with a pencil sketch effect right out of the box. The cv2.pencilSketch function uses a domain filter introduced in the 2011 paper Domain transform for edge-aware image and video processing, by Eduardo Gastal and Manuel Oliveira. For customizations, other filters can also be developed.

###  Libraries Used

#### imread()
cv2.imread() method loads an image from the specified file. If the image cannot be read (because of missing file, improper permissions, unsupported or invalid format) then this method returns an empty matrix.
#### cvtColor()
cv2.cvtColor() method is used to convert an image from one color space to another. There are more than 150 color-space conversion methods available in OpenCV. 
#### bitwise_not()
To make brighter regions lighter and lighter regions darker so that we could find edges to create a pencil sketch.
#### GaussianBlur()
In the Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced. It also smoothens or blurs the image. You can perform this operation on an image using the Gaussianblur() method of the imgproc class.
#### dodgeV2()
It is used to divide the grey-scale value of the image by the inverse of the blurred image which highlights the sharpest edges.
### Results Obtained

![image](https://user-images.githubusercontent.com/78999467/112639271-2dbc7300-8e68-11eb-8c99-314d1bffa1b1.png)

![image](https://user-images.githubusercontent.com/78999467/112639296-344aea80-8e68-11eb-85a9-401529d63164.png)

![image](https://user-images.githubusercontent.com/78999467/112639322-3a40cb80-8e68-11eb-8a6e-266b923b038e.png)


<h1> Image Segmentation Using Color space and Opencv</h1>
<h2>Introduction</h2>
<p>
The process of partitioning a digital image into multiple segments is defined as image segmentation. Segmentation aims to divide an image into regions that can be more representative and easier to analyze.</p>

<h2>What are color spaces?</h2>
<p>Basically, Color spaces represent color through discrete structures (a fixed number of whole number integer values), which is acceptable since the human eye and perception are also limited. Color spaces are fully able to represent all the colors that humans are able to distinguish between.</p>

 
## Steps followed for implementation
```diff
- Converted the image into HSV
- Choosing swatches of the desired color, In this, shades of light and dark orange have been taken.
- Applying an orange shade mask to the image
- Adding the second swatches of color, Here shades of white were chosen i.e light and dark shades
- Apply the white mask onto the image
- Now combine the two masks, Adding the two masks together results in 1 value wherever there is an orange shade or white shade.
- Clean up the segmentation using a blur 
```

 
 ### Default  image in BGR color space
 
![image](https://user-images.githubusercontent.com/78999467/112638972-e59d5080-8e67-11eb-91a6-aff48f35c1c0.png)

 ### Image converted to RGB color space
 
![image](https://user-images.githubusercontent.com/78999467/112638902-d3bbad80-8e67-11eb-9885-e7e2e367bb8c.png)

 ### Image converted to GRAY color space
 
![image](https://user-images.githubusercontent.com/78999467/112638849-c4d4fb00-8e67-11eb-9d10-413da262d1d2.png)

### Image converted to HSV color space
 
![image](https://user-images.githubusercontent.com/78999467/112638768-b38bee80-8e67-11eb-9f94-037ed3acf9ea.png)


### Segmented images
![image](https://user-images.githubusercontent.com/78999467/112638705-a2db7880-8e67-11eb-89f3-87f16f1ed8d2.png)


# More Awesome Face Operations That Can Be Added Here 

![Face_Alignment](https://raw.githubusercontent.com/YadiraF/PRNet/master/Docs/images/alignment.jpg "Facial Alignment Analysis")

### Face Detection
### Face Alignment
### Face Recognition
### Face Identification
### Face Verification
### Face Representation
### Face Alignment
### Face(Facial) Attribute & Face(Facial) Analysis
### Face Reconstruction
### Face 3D
### Face Tracking
### Face Clustering
### Face Super-Resolution
### Face Deblurring
### Face Hallucination
### Face Generation
### Face Synthesis
### Face Completion
### Face Restoration
### Face De-Occlusion
### Face Transfer
### Face Editing
### Face Anti-Spoofing
### Face Retrieval
### Face Application

---
## Piplines

- [seetaface/SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine)
---
## DataSets

- Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner .[FaceForensics: A Large-scale Video Dataset for Forgery Detection in  Human Faces](https://arxiv.org/pdf/1803.09179) .[J] arXiv preprint arXiv:1803.09179.
- Ziyi Liu, Jie Yang, Mengchen Lin, Kenneth Kam Fai Lai, Svetlana Yanushkevich, Orly Yadid-Pecht .[WDR FACE: The First Database for Studying Face Detection in Wide Dynamic Range](https://arxiv.org/pdf/2101.03826) [J]. arXiv preprint arXiv:2101.03826.
- Jianglin Fu, Ivan V. Bajic, Rodney G. Vaughan .[Datasets for Face and Object Detection in Fisheye Images](https://arxiv.org/pdf/1906.11942) .[J] arXiv preprint arXiv:1906.11942.
- Huang G B, Mattar M, Berg T, et al. [Labeled faces in the wild: A database forstudying face recognition in unconstrained environments](http://vis-www.cs.umass.edu/lfw/lfw.pdf)[C]//Workshop on faces in'Real-Life'Images: detection, alignment, and recognition. 2008.
- Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, Jianfeng Gao .[MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition](https://arxiv.org/pdf/1607.08221) .[J] arXiv preprint arXiv:1607.08221.
- Ankan Bansal, Anirudh Nanduri, Carlos Castillo, Rajeev Ranjan, Rama Chellappa .[UMDFaces: An Annotated Face Dataset for Training Deep Networks](https://arxiv.org/pdf/1611.01484) .[J] arXiv preprint arXiv:1611.01484.
- Tianyue Zheng, Weihong Deng, Jiani Hu .[Cross-Age LFW: A Database for Studying Cross-Age Face Recognition in  Unconstrained Environments](https://arxiv.org/pdf/1708.08197) .[J] arXiv preprint arXiv:1708.08197.
- Cao Q, Shen L, Xie W, et al. [Vggface2: A dataset for recognising faces across pose and age](https://arxiv.org/abs/1710.08092)[C]//Automatic Face & Gesture Recognition (FG 2018), 2018 13th IEEE International Conference on. IEEE, 2018: 67-74.
- Mei Wang, Weihong Deng, Jiani Hu, Jianteng Peng, Xunqiang Tao, Yaohai Huang .[Racial Faces in-the-Wild: Reducing Racial Bias by Deep Unsupervised Domain Adaptation](https://arxiv.org/pdf/1812.00194) .[J] arXiv preprint arXiv:1812.00194.
- Michele Merler, Nalini Ratha, Rogerio S. Feris, John R. Smith .[Diversity in Faces](https://arxiv.org/pdf/1901.10436) .[J] arXiv preprint arXiv:1901.10436.
- Shan Jia, Chuanbo Hu, Guodong Guo, Zhengquan Xu .[A database for face presentation attack using wax figure faces](https://arxiv.org/pdf/1906.11900) .[J] arXiv preprint arXiv:1906.11900.
- Muhammad Haris Khan, John McDonagh, Salman Khan, Muhammad Shahabuddin, Aditya Arora, Fahad Shahbaz Khan, Ling Shao, Georgios Tzimiropoulos .[AnimalWeb: A Large-Scale Hierarchical Dataset of Annotated Animal Faces](https://arxiv.org/pdf/1909.04951) .[J] arXiv preprint arXiv:1909.04951.
- Zhongyuan Wang, Guangcheng Wang, Baojin Huang, Zhangyang Xiong, Qi Hong, Hao Wu, Peng Yi, Kui Jiang, Nanxi Wang, Yingjiao Pei, Heling Chen, Yu Miao, Zhibing Huang, Jinbi Liang .[Masked Face Recognition Dataset and Application](https://arxiv.org/pdf/2003.09093) .[J] arXiv preprint arXiv:2003.09093
- Viktor Varkarakis, Peter Corcoran .[Dataset Cleaning -- A Cross Validation Methodology for Large Facial Datasets using Face Recognition](https://arxiv.org/pdf/2003.10815) .[J] arXiv preprint arXiv:2003.10815.
- Raj Kuwar Gupta, Shresth Verma, KV Arya, Soumya Agarwal, Prince Gupta .IIITM Face: A Database for Facial Attribute Detection in Constrained and Simulated Unconstrained Environments .[J] arXiv preprint arXiv:1910.01219.
- Shifeng Zhang, Xiaobo Wang, Ajian Liu, Chenxu Zhao, Jun Wan, Sergio Escalera, Hailin Shi, Zezheng Wang, Stan Z. Li .CASIA-SURF: A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing .[J] arXiv preprint arXiv:1812.00408.
-  Liming Jiang, Wayne Wu, Ren Li, Chen Qian, Chen Change Loy .[DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection](https://arxiv.org/pdf/2001.03024) .[J] arXiv preprint arXiv:2001.03024
- Jian Han, Sezer Karaoglu, Hoang-An Le, Theo Gevers .[Improving Face Detection Performance with 3D-Rendered Synthetic Data](https://arxiv.org/pdf/1812.07363) .[J] arXiv preprint arXiv:1812.07363.
-  Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner .[FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971) .[J] arXiv preprint arXiv:1901.08971.<br>[data:[ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)]
- Ziyi Liu, Jie Yang, Mengchen Lin, Kenneth Kam Fai Lai, Svetlana Yanushkevich, Orly Yadid-Pecht .[WDR FACE: The First Database for Studying Face Detection in Wide Dynamic Range](https://arxiv.org/pdf/2101.03826) [J]. arXiv preprint arXiv:2101.03826.
- Kai Zhang, Vítor Albiero, Kevin W. Bowyer .[A Method for Curation of Web-Scraped Face Image Datasets](https://arxiv.org/pdf/2004.03074) [J]. arXiv preprint arXiv:2004.03074.
- 【Datasets】Philipp Terhörst, Daniel Fährmann, Jan Niklas Kolf, Naser Damer, Florian Kirchbuchner, Arjan Kuijper .[MAAD-Face: A Massively Annotated Attribute Dataset for Face Images](https://arxiv.org/pdf/2012.01030) [J]. arXiv preprint arXiv:2012.01030.
- Domenick Poster, Matthew Thielke, Robert Nguyen, Srinivasan Rajaraman, Xing Di, Cedric Nimpa Fondje, Vishal M. Patel, Nathaniel J. Short, Benjamin S. Riggan, Nasser M. Nasrabadi, Shuowen Hu .[A Large-Scale, Time-Synchronized Visible and Thermal Face Dataset](https://arxiv.org/pdf/2101.02637) [J]. arXiv preprint arXiv:2101.02637.
- Anselmo Ferreira, Ehsan Nowroozi, Mauro Barni .[VIPPrint: A Large Scale Dataset of Printed and Scanned Images for Synthetic Face Images Detection and Source Linking](https://arxiv.org/pdf/2102.06792) [J]. arXiv preprint arXiv:2102.06792.


> * **2D face recognition**   
> * **Video face recognition**   
> * **3D face recognition**   
> * **Anti-spoofing**   
> * **cross age and cross pose**   
> * **Face Detection**   
> * **Face Attributes**   
> * **Others**   

### 📌 2D Face Recognition

| Datasets                   | Description                                                  | Links                                                        | Publish Time |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **CASIA-WebFace**          | **10,575** subjects and **494,414** images                   | [Download](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) | 2014         |
| **MegaFace**🏅              | **1 million** faces, **690K** identities                     | [Download](http://megaface.cs.washington.edu/)               | 2016         |
| **MS-Celeb-1M**🏅           | about **10M** images for **100K** celebrities   Concrete measurement to evaluate the performance of recognizing one million celebrities | [Download](http://www.msceleb.org)                           | 2016         |
| **LFW**🏅                   | **13,000** images of faces collected from the web. Each face has been labeled with the name of the person pictured.  **1680** of the people pictured have two or more distinct photos in the data set. | [Download](http://vis-www.cs.umass.edu/lfw/)                 | 2007         |
| **VGG Face2**🏅             | The dataset contains **3.31 million** images of **9131** subjects (identities), with an average of 362.6 images for each subject. | [Download](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)  | 2017         |
| **UMDFaces Dataset-image** | **367,888 face annotations** for **8,277 subjects.**         | [Download](http://www.umdfaces.io)                           | 2016         |
| **Trillion Pairs**🏅        | Train: **MS-Celeb-1M-v1c** &  **Asian-Celeb** Test: **ELFW&DELFW** | [Download](http://trillionpairs.deepglint.com/overview)      | 2018         |
| **FaceScrub**              | It comprises a total of **106,863** face images of male and female **530** celebrities, with about **200 images per person**. | [Download](http://vintage.winklerbros.net/facescrub.html)    | 2014         |
| **Mut1ny**🏅                | head/face segmentation dataset contains over 17.3k labeled images | [Download](http://www.mut1ny.com/face-headsegmentation-dataset) | 2018         |
| **IMDB-Face**              | The dataset contains about 1.7 million faces, 59k identities, which is manually cleaned from 2.0 million raw images. | [Download](https://github.com/fwang91/IMDb-Face)             | 2018         |

### 📌 Video Face Recognition 

| Datasets                    | Description                                                  | Links                                                        | Publish Time |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **YouTube Face**🏅           | The data set contains **3,425** videos of **1,595** different people. | [Download](http://www.cs.tau.ac.il/%7Ewolf/ytfaces/)         | 2011         |
| **UMDFaces Dataset-video**🏅 | Over **3.7 million** annotated video frames from over **22,000** videos of **3100 subjects.** | [Download](http://www.umdfaces.io)                           | 2017         |
| **PaSC**                    | The challenge includes 9,376 still images and 2,802 videos of 293 people. | [Download](https://www.nist.gov/programs-projects/point-and-shoot-face-recognition-challenge-pasc) | 2013         |
| **YTC**                     | The data consists of two parts: video clips (1910 sequences of 47 subjects) and initialization data(initial frame face bounding boxes, manually marked). | [Download](http://seqamlab.com/youtube-celebrities-face-tracking-and-recognition-dataset/) | 2008         |
| **iQIYI-VID**🏅              | The iQIYI-VID dataset **contains 500,000 videos clips of 5,000 celebrities, adding up to 1000 hours**. This dataset supplies multi-modal cues, including face, cloth, voice, gait, and subtitles, for character identification. | [Download](http://challenge.ai.iqiyi.com/detail?raceId=5b1129e42a360316a898ff4f) | 2018         |

### 📌3D Face Recognition 

| Datasets       | Description                                                  | Links                                                        | Publish Time |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **Bosphorus**🏅 | 105 subjects and 4666 faces 2D & 3D face data                | [Download](http://bosphorus.ee.boun.edu.tr/default.aspx)     | 2008         |
| **BD-3DFE**    | Analyzing **Facial Expressions** in **3D** Space             | [Download](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) | 2006         |
| **ND-2006**    | 422 subjects and 9443 faces 3D Face Recognition              | [Download](https://sites.google.com/a/nd.edu/public-cvrl/data-sets) | 2006         |
| **FRGC V2.0**  | 466 subjects and 4007 of 3D Face, Visible Face Images        | [Download](https://sites.google.com/a/nd.edu/public-cvrl/data-sets) | 2005         |
| **B3D(AC)^2**  | **1000** high quality, dynamic **3D scans** of faces, recorded while pronouncing a set of English sentences. | [Download](http://www.vision.ee.ethz.ch/datasets/b3dac2.en.html) | 2010         |

### 📌 Anti-Spoofing  

| Datasets          | \# of subj. / \# of sess. | Links                                                        | Year | Spoof attacks attacks | Publish Time |
| ----------------- | :-----------------------: | ------------------------------------------------------------ | ---- | --------------------- | ------------ |
| **NUAA**          |           15/3            | [Download](http://parnec.nuaa.edu.cn/xtan/data/nuaaimposterdb.html) | 2010 | **Print**             | 2010         |
| **CASIA-MFSD**    |           50/3            | Download(link failed)                                        | 2012 | **Print, Replay**     | 2012         |
| **Replay-Attack** |           50/1            | [Download](https://www.idiap.ch/dataset/replayattack)        | 2012 | **Print, 2 Replay**   | 2012         |
| **MSU-MFSD**      |           35/1            | [Download](https://www.cse.msu.edu/rgroups/biometrics/Publications/Databases/MSUMobileFaceSpoofing/index.htm) | 2015 | **Print, 2 Replay**   | 2015         |
| **MSU-USSA**      |          1140/1           | [Download](http://biometrics.cse.msu.edu/Publications/Databases/MSU_USSA/) | 2016 | **2 Print, 6 Replay** | 2016         |
| **Oulu-NPU**      |           55/3            | [Download](https://sites.google.com/site/oulunpudatabase/)   | 2017 | **2 Print, 6 Replay** | 2017         |
| **Siw**           |           165/4           | [Download](http://cvlab.cse.msu.edu/spoof-in-the-wild-siw-face-anti-spoofing-database.html) | 2018 | **2 Print, 4 Replay** | 2018         |

### 📌 Cross-Age and Cross-Pose

| Datasets     | Description                                                  | Links                                                        | Publish Time |
| ------------ | :----------------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| **CACD2000** | The dataset contains more than 160,000 images of 2,000 celebrities with **age ranging from 16 to 62**. | [Download](http://bcsiriuschen.github.io/CARC/)              | 2014         |
| **FGNet**    | The dataset contains more than 1002 images of 82 people with **age ranging from 0 to 69**. | [Download](http://www-prima.inrialpes.fr/FGnet/html/benchmarks.html) | 2000         |
| **MPRPH**    | The MORPH database contains **55,000** images of more than **13,000** people within the age ranges of **16** to **77** | [Download](http://www.faceaginggroup.com/morph/)             | 2016         |
| **CPLFW**    | we construct a Cross-Pose LFW (CPLFW) which deliberately searches and selects **3,000 positive face pairs** with **pose difference** to add pose variation to intra-class variance. | [Download](http://www.whdeng.cn/cplfw/index.html)            | 2017         |
| **CALFW**    | Thereby we construct a Cross-Age LFW (CALFW) which deliberately searches and selects **3,000 positive face pairs** with **age gaps** to add aging process intra-class variance. | [Download](http://www.whdeng.cn/calfw/index.html)            | 2017         |

### 📌Face Detection

| Datasets       | Description                                                  | Links                                                       | Publish Time |
| -------------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------ |
| **FDDB**🏅      | **5171** faces in a set of **2845** images                   | [Download](http://vis-www.cs.umass.edu/fddb/index.html)     | 2010         |
| **Wider-face** | **32,203** images and label **393,703** faces with a high degree of variability in scale, pose and occlusion, organized based on **61** event classes | [Download](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) | 2015         |
| **AFW**        | AFW dataset is built using Flickr images. It has **205** images with **473** labeled faces. For each face, annotations include a rectangular **bounding box**, **6 landmarks** and the **pose angles**. | [Download](http://www.ics.uci.edu/~xzhu/face/)              | 2013         |
| **MALF**       | MALF is the first face detection dataset that supports fine-gained evaluation. MALF consists of **5,250** images and **11,931** faces. | [Download](http://www.cbsr.ia.ac.cn/faceevaluation/)        | 2015         |

### 📌 Face Attributes 

| Datasets                             | Description                                                  | Links                                                        | Key features                                 | Publish Time |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- | ------------ |
| **CelebA**                           | **10,177** number of **identities**,  **202,599** number of **face images**, and  **5 landmark locations**, **40 binary attributes** annotations per image. | [Download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | **attribute & landmark**                     | 2015         |
| **IMDB-WIKI**                        | 500k+ face images with **age** and **gender** labels         | [Download](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | **age & gender**                             | 2015         |
| **Adience**                          | Unfiltered faces for **gender** and **age** classification   | [Download](http://www.openu.ac.il/home/hassner/Adience/data.html) | **age & gender**                             | 2014         |
| **WFLW**🏅                            | WFLW contains **10000 faces** (7500 for training and 2500 for testing) with **98 fully manual annotated landmarks**. | [Download](https://wywu.github.io/projects/LAB/WFLW.html)    | **landmarks**                                | 2018         |
| **Caltech10k Web Faces**             | The dataset has 10,524 human faces of various resolutions and in **different settings** | [Download](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/#Description) | **landmarks**                                | 2005         |
| **EmotioNet**                        | The EmotioNet database includes**950,000 images** with **annotated AUs**.  A **subset** of the images in the EmotioNet database correspond to **basic and compound emotions.** | [Download](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/index.html#overview) | **AU and Emotion**                           | 2017         |
| **RAF( Real-world Affective Faces)** | **29672** number of **real-world images**,  including **7** classes of basic emotions and **12** classes of compound emotions,  **5 accurate landmark locations**,  **37 automatic landmark locations**, **race, age range** and  **gender** **attributes** annotations per image | [Download](  <http://www.whdeng.cn/RAF/model1.html>)         | **Emotions、landmark、race、age and gender** | 2017         |

### 📌 Others

| Datasets           | Description                                                  | Links                                                        | Publish Time |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| **IJB C/B/A**🏅     | IJB C/B/A is currently running **three challenges** related to  **face detection, verification, identification, and identity clustering.** | [Download](https://www.nist.gov/programs-projects/face-challenges) | 2015         |
| **MOBIO**          | **bi-modal** (**audio** and **video**) data taken from 152 people. | [Download](https://www.idiap.ch/dataset/mobio)               | 2012         |
| **BANCA**          | The BANCA database was captured in four European languages in **two modalities** (**face** and **voice**). | [Download](http://www.ee.surrey.ac.uk/CVSSP/banca/)          | 2014         |
| **3D Mask Attack** | **76500** frames of **17** persons using Kinect RGBD with eye positions (Sebastien Marcel). | [Download](https://www.idiap.ch/dataset/3dmad)               | 2013         |
| **WebCaricature**  | **6042** **caricatures** and **5974 photographs** from **252 persons** collected from the web | [Download](https://cs.nju.edu.cn/rl/WebCaricature.htm)       | 2018         |