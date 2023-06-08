# Visualisations of Convolutional Neural Networks
Deep neural networks are often seen as black box models. However, there are many informative ways to visualise their inner workings because the output of convolutional layers can be plotted as an image. We implement interfaces for various techniques to visualise CNNs. To keep computations less costly, we use the somewhat outdated VGG16 model.
## Class activation maps
One common way are so called class activation maps.
The motivation is to visualize "which part of the image" contributed to the final prediction. Given an image, the idea is to compute the gradient of the probability share of the top predicted class with respect to the output of the last convolutional layer. In the case of VGG, this gradient has shape (7,7,512). To avoid visualising 512 channels, we compute a weighted average of this gradient with the output of the last convolutional layer, giving a single array of size (7,7). This array encodes which arrays contributed to the top predicted class. So we create a heat map of this image, resize it and superimpose it with the original image in gray. In this example the house has been most influential in the final decision so it is coloured in red. 

<p align="center">
  <img width="460" height="300" src="https://github.com/deepfloe/conv_visualization/assets/53785628/576a4c6d-3398-4030-a974-c9affa8cddd5">
</p>

## Intermediate activation
The first 18 layers of the VGG16 model are either pooling or convolutional layers. These layers preserve the matrix structure of the input image. In each stage, they consist of a number of channels, which all output 2D arrays that are visualised as grayscale images.
The layers are stacked in such a way that the number of channels increases while the array shape becomes smaller. We visualize the first 64 channels, the user can choose any of the 18 hidden layers via a slider.

<p align="center">
  <img width="580" height="260" src="https://github.com/deepfloe/conv_visualization/assets/53785628/f2236936-635b-46f7-8798-2cb34a0fa546">
</p>

## How to use
The interfaces are built using the python package [gradio](https://gradio.app/docs/). To launch them either execute ´ćlass_activation_map.py´ or `intermediate_activation.py`, you can then submit your input image.

## References
- Chollet, Deep learning with Python. Simon and Schuster, 2021
- Ramprasaath R. Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, [arxiv](https://arxiv.org/abs/1610.02391), 2017
- Brownlee, How to Visualize Filters and Feature Maps in Convolutional Neural Networks, [blog entry](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/), 2019, retrieved May 2023
 
