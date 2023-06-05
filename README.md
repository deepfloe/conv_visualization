# Visualisations of Convolutional Neural Networks
Deep neural networks are often seen as black box models. However, image classifiers can be visualised in interesting ways, because the output of convolutional layers can itself be plotted as an image. We implement interfaces for various techniques to visualise CNNs. To keep computations a bit less costly we use the somehwat out-dated VGG16 model.
## Class activation maps
One common way are so called class activation maps.
The motivation is to visualize "which part of the image" contributed to the final prediction. Given an image, the idea is to compute the gradient of the probability share of the top predicted class with respect to the output of the last convolutional layer. In the case of VGG, this gradient has shape (7,7,512). To avoid visualising 512 channels, we compute a weighted average of this gradient with the output of the last convolutional layer, giving a single array of size (7,7). This array encodes which arrays contributed to the top predicted class. So we create a heat map of this image, resize it and superimpose it with the original image in gray. In this example the house has been most influential in the final decision so it is coloured in red. 

<p align="center">
  <img width="460*1.8" height="300*1.8" src="https://github.com/deepfloe/conv_visualization/assets/53785628/576a4c6d-3398-4030-a974-c9affa8cddd5">
</p>

