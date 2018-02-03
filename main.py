from scipy.misc import imread, imresize
import numpy as np
import matplotlib.pyplot as plt

# Helper functions to deal with image preprocessing
from image_utils import load_image, preprocess_image, deprocess_image
from style_transfer_funcs import *
import tensorflow as tf
from net.squeezenet import SqueezeNet

#%%
def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

tf.reset_default_graph() # remove all existing variables in the graph 
sess = get_session()
SAVE_PATH = 'net/squeezenet_tf/squeezenet.ckpt'
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)

#%%
# Composition VII + Tubingen
params1 = {
    'sess' : sess,
    'model' : model,    
    'content_image' : 'data/mit_building.jpg',
    'style_image' : 'data/gray_picasso.jpg',
    'image_size' : 340,
    'style_size' : 800,
    'content_layer' : 3,
    'content_weight' : 0.01, 
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (10, 2000, 300, 4),
    'tv_weight' : 0.03,
}

#style_transfer(**params1)

transferer = Style_Transferer_Squeezenet(**params1)
transferer.generate_image(iter_num=2000, draw_every=1000)
