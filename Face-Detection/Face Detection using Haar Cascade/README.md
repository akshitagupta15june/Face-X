<h1>Face Detection using Harr Cascade</h1>
<p>Harr Cascade is a powerful detection algorithm to detect faces,eyes,lips etc.Harr cascade was proposed by Viola and Jones in the research paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. Haar features are used to identify edges,lines in the faces(ie. change in intensities of pixels.)</p>
![image](https://user-images.githubusercontent.com/51399803/115982788-45982b80-a5bb-11eb-97d3-01a295988438.png)

<p>The haar feature continuously traverses from the top left of the image to the bottom right to search for the particular feature.
there’s a set of features which would capture certain facial structures like eyebrows or the bridge between both the eyes, or the lips etc. But originally the feature set was not limited to this. The feature set had an approx. of 180,000 of them.A boosting technique known as AdaBoost is used for feature selection as some features are irrelevant. Using AdaBoost the feature set is reduced to 6000 features.</p>


