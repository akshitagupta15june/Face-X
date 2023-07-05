<h1>MobilenetV2</h1>
<br>

![mobilenet conv blocks](https://github.com/kanishkakataria/Images/assets/85161519/142333e3-efcf-4386-8bfd-eede3aa842d2)<br>
Depthwise Separable Convolution is introduced which dramatically reduce the complexity cost and model size of the network, which is suitable to Mobile devices, or any devices with low computational power. In MobileNetV2, a better module is introduced with inverted residual structure. Non-linearities in narrow layers are removed this time. With MobileNetV2 as backbone for feature extraction, state-of-the-art performances are also achieved for object detection and semantic segmentation. 

<br>
<ol>
<li>In MobileNetV2, there are two types of blocks. One is residual block with stride of 1. Another one is block with stride of 2 for downsizing.</li>
<li>There are 3 layers for both types of blocks.</li>
<li>This time, the first layer is 1×1 convolution with ReLU6.</li>
<li>The second layer is the depthwise convolution.</li>
<li>The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.</li>
<li>The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.</li>
</ol>

<h2>Confusion Matrix</h2>

![confusion matrix](https://github.com/kanishkakataria/Images/assets/85161519/d55ce591-3733-4281-bfdc-b246e0fe8d0a)<br>
<h2>Accuracy Curve</h2>

![acc_curve](https://github.com/kanishkakataria/Images/assets/85161519/308d7b80-bd5e-4c02-afaf-4132516b6992)<br>
<h2>Loss Curve</h2>

![loss_curve](https://github.com/kanishkakataria/Images/assets/85161519/0b6ce9aa-8251-489a-a0bf-6f088fb566c3)
