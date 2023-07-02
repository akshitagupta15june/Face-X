<h1>MobilenetV2</h1>
<br>
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

<h1>Confusion Matrix</h1>
<h1>Accuracy Curve</h1>
<h1>Loss Curve</h1>
