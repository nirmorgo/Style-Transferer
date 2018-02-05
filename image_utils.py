from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
import urllib.request, urllib.error, urllib.parse, io
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

"""
Utility functions used for viewing and processing images.
"""
# The mean to subtract from the input to the VGG model. This is the mean that
# when the VGG was used to train. Minor changes to this will make a lot of
# difference to the performance of model.
VGG19_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def preprocess_image(img):
    """Preprocess an image for squeezenet.
    
    Subtracts the pixel mean and divides by the standard deviation.
    """
    img = np.reshape(img, ((1,) + img .shape))
    # Input to the VGG model expects the mean to be subtracted.
    img = img - VGG19_MEAN
    return img


def deprocess_image(img, rescale=False, net = 'squeezenet'):
    """Undo preprocessing on an image and convert back to uint8."""
    img = img + VGG19_MEAN
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
	
	Modification (Nir) - instead of saving the data to a temp file, we store it
	in a local variable using BytesIO and Image class from PIL. Not gross anymore. 
    """
    try:
        f = urllib.request.urlopen(url)
        file = io.BytesIO(f.read())
        img = np.array(Image.open(file))
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        img = imresize(img, scale_factor)
    return img


def generate_noise_image(img, noise_ratio=0.01):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(-20, 20,
            (1, img.shape[1], img.shape[2], img.shape[3])).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    output = noise_image * noise_ratio + img * (1 - noise_ratio)
    return output

def plot_input_images(content_img, style_img):
    # Plot inputs
    f, axarr = plt.subplots(1,2,figsize=(12,12))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(content_img)
    axarr[1].imshow(style_img)
    plt.show()