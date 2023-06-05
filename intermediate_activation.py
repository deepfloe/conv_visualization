import keras
from keras.applications import vgg16
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.image import resize

def get_conv_base():
    '''Return the convolutional base of the VGG model'''
    conv_base = keras.applications.vgg16.VGG16(
      weights = "imagenet",
      include_top = False,
    )
    return conv_base

def run_conv_base(input_image, depth):
    '''Run the convolutional base and preprocessing pipeline of vgg to a variable depth.'''
    model = get_conv_base() #this loads the full convolutional base
    # we instantiate a new keras model, that outputs the result of the depth layer
    vgg_truncated = keras.Model(inputs = model.inputs, outputs = model.layers[depth].output)
    # this turns the shape of the array into the form (1, width, heigth, 3), which is needed for the preprocessing pipelien
    input_image = np.array([input_image])
    pre_processed = keras.applications.vgg16.preprocess_input(input_image)
    return vgg_truncated(pre_processed)


def plot_channels(input_image, depth):
    '''Plot a grid of 8x8 images of features of the vgg model applied to the input image at variable depth.'''
    # To keep computation time low, we resize the image
    input_image_resized = resize(input_image, (512, 384), preserve_aspect_ratio=True)
    result = run_conv_base(input_image_resized, depth)
    # the shape of result is (1,width,height, number of channels). In the later layers, there are up to 512 channels but we prefer to only plot the first 64

    fig, ax = plt.subplots(nrows=8, ncols=8)
    for k in range(64):
        i = k % 8
        j = k // 8
        output_channel = result[0, :, :, k]
        ax[i, j].imshow(output_channel, cmap=plt.cm.gray)
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    # sets the space between subplots to 0
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def launch_gradio_plot():
    '''Produces a gradio visualization for the plot of vgg outputs as variable layer depth'''
    outputs = gr.Plot()
    demo = gr.Interface(fn=plot_channels, inputs=["image",gr.Slider(1, 18, step =1)], outputs=outputs)
    demo.launch()

if __name__ == '__main__':
    launch_gradio_plot()