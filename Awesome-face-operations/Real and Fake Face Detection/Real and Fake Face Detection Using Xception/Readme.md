<h1>Xception Algorithm</h1>
<br>

![Modified Deptthwise Separable Convolution in Xception](https://github.com/kanishkakataria/Images/assets/85161519/84790e95-81e7-495e-9ba8-f96d7391330c)
<br>
Xception is a deep convolutional neural network architecture that involves Depthwise Separable Convolutions. It was developed by Google researchers. Google presented an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads them to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions.<br>
The modified depthwise separable convolution is the pointwise convolution followed by a depthwise convolution. This modification is motivated by the inception module in Inception-v3 that 1×1 convolution is done first before any n×n spatial convolutions. Thus, it is a bit different from the original one. (n=3 here since 3×3 spatial convolutions are used in Inception-v3.)
<br>
<h2>Two minor differences:</h2>
<br>
1. The order of operations: As mentioned, the original depthwise separable convolutions as usually implemented (e.g. in TensorFlow) perform first channel-wise spatial convolution and then perform 1×1 convolution whereas the modified depthwise separable convolution perform 1×1 convolution first then channel-wise spatial convolution. This is claimed to be unimportant because when it is used in stacked setting, there are only small differences appeared at the beginning and at the end of all the chained inception modules.
<br>
2. The Presence/Absence of Non-Linearity: In the original Inception Module, there is non-linearity after first operation. In Xception, the modified depthwise separable convolution, there is NO intermediate ReLU non-linearity.


<h2>CONFUSION MATRIX</h2>

![confusion matrix_Xception](https://github.com/kanishkakataria/Images/assets/85161519/4c9b6600-9a20-4a09-8f50-356b524315e5)<br>
<h2>ACCURACY CURVE</h2>

![Acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/4b43ac5e-d89e-4941-acd8-e67c37175e09)<br>
<h2>LOSS CURVE</h2>

![loss_Curve](https://github.com/kanishkakataria/Images/assets/85161519/7a5d1c76-1118-4881-8b0b-aeec9bec6e33)
