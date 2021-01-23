# FACE-X


<div align="center">
<img src="https://github.com/akshitagupta15june/Face-X/blob/master/Cartoonify%20Image/facex.jpeg" width="350px" height="350px" align='center'>
</div>

### Demonstration of different algorithms and operations on faces 

#### [Recognition-Algorithms](https://github.com/akshitagupta15june/Face-X/tree/master/Recognition-Algorithms)

There are several approaches for recognizing a face. The algorithm can use statistics, try to find a pattern which represents a specific person or use a convolutional neural network. 
<div align="center">
<img src="https://media.giphy.com/media/AXorq76Tg3Vte/giphy.gif" width="20%"><br>
</div>

The algorithms used for the tests are Eigenfaces, Fisherfacesand local binary patterns histograms which all come from the library OpenCV. Eigenfaces and Fisher faces are used with a Euclidean distance to predict the person. The algorithm which is using a deep convolutional neural network is the project called OpenFace.

This can be used for automatic face detection attendance system in recent technology.



`
Recognition of faces by different algorithms and frameworks. Despite a variety of open-source face recognition frameworks available, there was 
no ready-made solution to implement. So In this project all kind of algorithms are implemented and even with various operations that can be implemented
in a frontal face. The available algorithms processed only high-resolution static shots and performed insufficiently.
`


### Requirements 👇
- Python3.6+
- virtualenv (`pip install virtualenv`)

### Installation 🖥
- `virtualenvv env`
- `source venv/bin/activate` (Linux)
- `venv\Scripts\activate` (Windows)
- `pip install -r requirements.txt`
- Create an .env file, copy the content from .env.sample and add your data path. Example: `DATA_PATH = "./foto_reco/"`

## Comparative Study of the Algorithms used here :

This project holds different type of deep learning models on different frameworks. every single model has it's uniqueness and contribute vastly to the deep learning domain .
If we try to compare them, we might find better understanding over those and this would be great for all of us :)

Model | Creator | Published | Misc 
--- | --- | --- | --- 
LBPH | C. Silva | March,2015 | Got the highest accuracy in all experiments, but this algorithm has the higher impact of the negative light exposure and high noise level more than the others that are statistical approach.
LBP_SVM | C. Silva | March,2015 | The accuracy is reported at 90.52% using SVM which has a gamma value of 0.0000015 and penalty parameter of the error term C = 2.5 while using the RBF kernel.
MobileNetV2 | Google AI | April,2018 | Faster for the same accuracy across the entire latency spectrum. In particular, the new models use 2x fewer operations, need 30% fewer parameters and are about 30-40% faster on a Google Pixel phone than MobileNetV1 models, all while achieving higher accuracy.
EffecientNet | Google AI | May, 2019 | On the ImageNet challenge, with a 66M parameter calculation load, EfficientNet reached 84.4% accuracy and took its place among the state-of-the-art.
EigenFaceRecogniser | M. Turk and A. Pentland | 1991 | The accuracy of Eigenface is satisfactory (over 90 %) with frontal faces. Eigenface uses PCA. A drawback is that it is very sensitive for lightening conditions and the position of the head. Fisherface is similar to Eigenface but with improvement in better classification of different classes image.
FisherFaceRecogniser | Aleix Martinez | 2011 | Fisherface is a technique similar to Eigenfaces but it is geared to improve clustering of classes.  While Eigenfaces relies on PCA, Fischer faces relies on LDA (aka Fischer’s LDA) for dimensionality reduction.
GhostNet | Huwayei Noah | Recent | GhostNet can achieve higher recognition performance (75% , top-1 accuracy) than MobileNetV3 with similar computational cost on the ImageNet ILSVRC-2012 classification dataset.
KNN | Evelyn Fix and Joseph Hodges | 1951 | K-Nearest Neighbor face recognition delivered best accuracy 91.5% on k=1. KNN showed the faster execution time compared with PCA and LDA. Time execution of KNN to recognize face was 0.152 seconds on high-processor. Face detection and recognition only need 2.66 second to recognize on low-power ARM11 based system
ResNet-50 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | Dec,2015 | ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
DenseNet121 | Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger | Jan,2018 | Got Best Paper Award with over 2000 citations. It is jointly invented by Cornwell University, Tsinghua University and Facebook AI Research (FAIR)
VGG-19 | Karen Simonyan, Andrew Zisserman | April,2015 | This model achieves 75.2% top-1 and 92.5% top-5 accuracy on the ImageNet Large Scale Visual Recognition Challenge 2012 dataset.
MTCNN | Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao | April,2016 | One of the hottest model used most widely recently for its high precision and outstanding real time performance among the state-of-art algorithms for face detection. Then, the first basic application of portrait classification is researched based on MTCNN and FaceNet. Its direction is one of the most classical and popular area in nowadays AI visual research, and is also the base of many other industrial branches.

![](https://miro.medium.com/max/1416/0*6wtXZPL89Apg2rlH) 

![](https://imgs.developpaper.com/imgs/1527989268-5de6f88a07966_articlex.png)


## Channels 📞
Join official discord channel for discussion by clicking [here](https://discord.gg/d5GfFfy8)


## Get Started with Open Source now 👨‍💻

[Start Open Source](https://anush-venkatakrishna.medium.com/part-1-winter-or-summer-take-your-baby-steps-into-opensource-now-7d661235d7ff) an article by [Anush Krishna](https://github.com/anushkrishnav)


## Open-source Programs ❄

<p align="center">
  <a>
   <img  width="140" height="140" src="https://njackwinterofcode.github.io/images/nwoc-logo.png">
   <img align="center" src="https://devscript.tech/woc/img/WOC-logo.png" width="140" height="140"/>
   <img  width="140" height="140" src="https://media-exp1.licdn.com/dms/image/C560BAQGh8hr-FgbrHw/company-logo_200_200/0/1602422883512?e=2159024400&v=beta&t=s8IX2pN1J2v5SRRbgzVNzxnQ2rWeeMq2Xb__BYW60qE">
</p>
 
</br>


We have participated in various open-source programs mentioned below:
```
1. NWOC(NJack Winter of Code)
```
```
2. DWOC(Devscript Winter of Code)
```
```
3. SWOC(Script Winter of Code)
```
```
4. DSC IIT KALYANI WINTER OF CODE
```
```
5. UACEIT WINTER OF CODE
```

## Contributors 🌟 

Thanks goes to these wonderful people ✨✨:
<table>
	<!--   ROW 1 -->
	<tr>
		<td align="center">
			<a href="https://github.com/akshitagupta15june">
				<img src="https://avatars0.githubusercontent.com/u/57909583?v=4" width="100px" alt="" />
				<br /> <sub><b>akshitagupta15june</b></sub>
			</a>
			<br /> <a href="https://github.com/akshitagupta15june"> 
                👑 Author
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/Aayush-hub">
				<img src="https://avatars1.githubusercontent.com/u/65889104?v=4" width="100px" alt="" />
				<br /> <sub><b>Aayush-hub</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=Aayush-hub">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/Halix267">
				<img src="https://avatars1.githubusercontent.com/u/63572018?v=4" width="100px" alt="" />
				<br /> <sub><b>Halix267</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=Halix267">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/smriti1313">
				<img src="https://avatars1.githubusercontent.com/u/52624997?v=4" width="100px" alt="" />
				<br /> <sub><b>smriti1313</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=smriti1313">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/SoyabulIslamLincoln">
				<img src="https://avatars1.githubusercontent.com/u/55865931?v=4" width="100px" alt="" />
				<br /> <sub><b>SoyabulIslamLincoln</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=SoyabulIslamLincoln">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/ashwani-rathee">
				<img src="https://avatars3.githubusercontent.com/u/54855463?v=4" width="100px" alt="" />
				<br /> <sub><b>ashwani-rathee</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=ashwani-rathee">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/KerinPithawala">
				<img src="https://avatars3.githubusercontent.com/u/46436993?v=4" width="100px" alt="" />
				<br /> <sub><b>KerinPithawala</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=KerinPithawala">
                💻
            </a>
		</td>
	</tr>
	<!--   ROW 2 -->
	<tr>
		<td align="center">
			<a href="https://github.com/koolgax99">
				<img src="https://avatars0.githubusercontent.com/u/55532999?v=4" width="100px" alt="" />
				<br /> <sub><b>koolgax99</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=koolgax99">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/Sloth-Panda">
				<img src="https://avatars2.githubusercontent.com/u/70213384?v=4" width="100px" alt="" />
				<br /> <sub><b>Sloth-Panda</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=Sloth-Panda">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/amandp13">
				<img src="https://avatars0.githubusercontent.com/u/55224891?v=4" width="100px" alt="" />
				<br /> <sub><b>amandp13</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=amandp13">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/Bhagyashri2000">
				<img src="https://avatars1.githubusercontent.com/u/43903254?v=4" width="100px" alt="" />
				<br /> <sub><b>Bhagyashri2000</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=Bhagyashri2000">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/musavveer">
				<img src="https://avatars2.githubusercontent.com/u/62888562?v=4" width="100px" alt="" />
				<br /> <sub><b>musavveer</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=musavveer">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/RaghavModi">
				<img src="https://avatars1.githubusercontent.com/u/52846588?v=4" width="100px" alt="" />
				<br /> <sub><b>RaghavModi</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=RaghavModi">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/Karnak123">
				<img src="https://avatars1.githubusercontent.com/u/39977582?v=4" width="100px" alt="" />
				<br /> <sub><b>Karnak123</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=Karnak123">
                💻
            </a>
		</td>
	</tr>
	<!--   ROW 3 -->
	<tr>
		<td align="center">
			<a href="https://github.com/himanshu007-creator">
				<img src="https://avatars2.githubusercontent.com/u/65963997?v=4" width="100px" alt="" />
				<br /> <sub><b>himanshu007-creator</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=himanshu007-creator">
                💻
            </a>
		</td>
		<td align="center">
			<a href="https://github.com/saiharsha-22">
				<img src="https://avatars1.githubusercontent.com/u/61947484?v=4" width="100px" alt="" />
				<br /> <sub><b>saiharsha-22</b></sub>
			</a>
			<br /> <a href="https://github.com/Jayshah6699/datascience-mashup/commits?author=saiharsha-22">
                💻
            </a>
		</td>
	</tr>
</table>
