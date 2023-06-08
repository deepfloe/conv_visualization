import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gradio as gr

def get_vgg_bottom():
    '''Get the convolutional base of the VGG16 model
    :returns: a Keras model with input shape (None, 224, 224, 3) and output shape (None, 7, 7, 512)
    '''
    model = VGG16(weights = "imagenet")
    # the name of the last conv layer in the VGG model. If another image classifier is used, these needs to be changed
    last_conv_layer_name = "block5_pool"
    last_conv_layer = model.get_layer(name = last_conv_layer_name)
    # the output of the models bottom part is the last con layer
    model_bottom = keras.Model(inputs = model.inputs, outputs = last_conv_layer.output)
    return model_bottom

def get_vgg_top():
    '''Get the layers of VGG16 model coming after the convolutional base.
    :returns: a keras Model with input shape (None, 7, 7, 512) and output shape (None, 1000)
    '''
    model = VGG16(weights = "imagenet")
    last_conv_layer_name = "block5_pool"
    last_conv_layer = model.get_layer(name = last_conv_layer_name)
    # the input of the top part is the output of the last conv layer
    input_top = keras.Input(last_conv_layer.output.shape[1:])
    #The names of the VGG layers that come after the convolutional base. If another image classifier is used, these need to be changed
    layer_names_top = ["flatten", "fc1", "fc2", "predictions"]
    x = input_top
    for layer_name in layer_names_top:
        x = model.get_layer(name = layer_name)(x)
    # when the for loop is run through,x is the ouput of the last vgg layer
    model_top = keras.Model(input_top, x)
    return model_top

def convert_image_to_tensor(img):
    ''' Converts an image into a tensor that can be input for the VGG model.
    :param img: a PIL image of size (224,224)
    :returns: a numpy array of shape (1,224,224,3)
    '''
    img_array = img_to_array(img)
    preprocessed = preprocess_input(img_array)
    return np.expand_dims(preprocessed, axis = 0)

def load_test_image(name = "elephant.jpg", url = "https://img-datasets.s3.amazonaws.com/elephant.jpg"):
    '''Load a test image of an elephant
    :returns a PIL image of size (224,224)
    '''
    img_path = tf.keras.utils.get_file(fname = name, origin=url)
    img = load_img(img_path, target_size = (224,224))
    return img

def vgg_prediction(img):
    '''Return the top 3 predictions of VGG'''
    model = VGG16(weights="imagenet")
    img_tensor = convert_image_to_tensor(img)
    vgg_pred = model.predict(img_tensor)
    preds = decode_predictions(vgg_pred, top=3)[0]
    formatted_preds = ''
    for pred in preds:
        formatted_preds += pred[1] + ': ' + str(round(pred[2], 4)) + '\n'
    return formatted_preds

def gradient_top(bottom_output, model_top):
    '''Take the gradient of the function whose input is the output of the last convolutional layer and whose output is the
     probability of the most likely class. The gradient is a tensor of shape ( ,512), where the first two numbers represent the shape of the last conv layer and 512 is the number of channels.
     :param bottom_output: a tf tensor of shape (7,7,512)
     :param model_top: a keras Model, with input shape (None, 7, 7, 512) and output shape (None, 1000), intended to be the output get_vgg_top()
     :returns: tf tensor of shape (7,7,512)
     '''
    with tf.GradientTape() as tape:
        tape.watch(bottom_output)
        pred = model_top(bottom_output)
        index_max = tf.argmax(pred[0])
        top_channel = pred[:, index_max]
    return tape.gradient(top_channel, bottom_output)

def apply_cmap(cmap, array):
    '''Return a 3D tensor representing an RGB image computed from an array and a colormap.
    :param cmap: a matplotlib.cm Colormap
    :param array: 2D numpy array of arbitrary size
    :returns: 3D numpy array of shape (shape(array),3) where the third component represents RGB values
    '''
    array = np.uint8(array)
    cmap_colors = cmap(np.arange(256))[:,:3]
    return array_to_img(cmap_colors[array])

def weigh_output_gradient(bottom_output, grads):
    '''The bottom output has 512 channels. The function computes a weighted average of these channels to return a 2D array.
    The weights are determined by how much each channel contributes to the vgg prediction, computed by the function gradient_top.
    :param bottom_output: a tf tensor of shape (7,7,512)
    :param grads: a tf tensor of shape, which measures the gradient of the most likely class with respect to the last conv layer. Intended to be the output of gradient_top
    :returns: tf array of size (7,7), which displays in red tones which areas of the last conv layer have contributed to the class prediction
    '''
    bottom_output_np = bottom_output.numpy()[0]
    grads_mean = tf.reduce_mean(grads, axis = (0,1,2)).numpy()
    for i in range(len(grads_mean)):
        bottom_output_np[:,:,i] *= grads_mean[i]
    heatmap = np.mean(bottom_output_np, axis = -1)
    #normalise the heatmap to have values between 0 and 255
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    heatmap *= 255
    red = cm.get_cmap("Reds")
    return apply_cmap(red, heatmap)


def superimpose_image(image1, image2, factor):
    '''Compute the superposition of two images. Factor is the weight of the first image, for example if factor=1, the function
    return image1. If factor = 0, the function returns image2.
    :param image1, image2: two PIL images of the same shape
    :param factor: float between 0 and 1
    :returns: PIL image with same shape as image1 and image2
    '''
    array1 = img_to_array(image1)
    array2 = img_to_array(image2)
    array_super = array1*factor+array2*(1-factor)
    return array_to_img(array_super)

def rgb2gray(rgb):
    '''Convert an rgb image to a gray image.
    '''
    #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    gray = cm.get_cmap("Greys")
    rgb = img_to_array(rgb)
    #print(rgb.shape)
    img = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    image_gray = apply_cmap(gray, img)
    return image_gray

def superimpose_image_with_cnn_gradient(image):
    '''Turn an image into gray scale and superimpose it with a heatmap that displays which area of the image has contributed to the vgg class prediction. The other output is the vgg prediction itself.
    :param image. PIL object of arbitrary shape
    :returns: string: the top three classes of the vgg_prediction , image: a gray-red PIL image
    '''
    model_bottom = get_vgg_bottom()
    model_top = get_vgg_top()
    image_tensor = convert_image_to_tensor(image)
    bottom_output = model_bottom(image_tensor)
    grads = gradient_top(bottom_output, model_top)
    heatmap = weigh_output_gradient(bottom_output, grads)
    heatmap_resize = heatmap.resize((224, 224))
    image_gray = rgb2gray(image)
    combined_image = superimpose_image(image_gray, heatmap_resize, 0.6)
    return vgg_prediction(image), combined_image

def launch_gradio_demo():
    demo = gr.Interface(fn = superimpose_image_with_cnn_gradient, inputs = gr.Image(shape=(224, 224)), outputs = ["text","image"])
    demo.launch(debug = True)

if __name__ == '__main__':
    launch_gradio_demo()
