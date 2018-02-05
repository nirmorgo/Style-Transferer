import matplotlib.pyplot as plt
import tensorflow as tf
from image_utils import deprocess_image, preprocess_image, load_image, generate_noise_image
from net.vgg19net import load_vgg_model

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def content_loss(content_current, content_target):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    H = tf.shape(content_current)[1]
    W = tf.shape(content_current)[2]
    C = tf.shape(content_current)[3]
    _, h, w, c = content_target.shape

    content_current = tf.reshape(content_current, (H*W,C))    
    content_target = tf.reshape(content_target, (H*W,C))

    loss = tf.reduce_sum(tf.abs(content_current - content_target))
    loss = (1/(4*h*w*c)) * loss
    return loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    H = tf.shape(features)[1]
    W = tf.shape(features)[2]
    C = tf.shape(features)[3]
    features = tf.reshape(features, (H*W,C))
    featuresT = tf.transpose(features)
    gram = tf.matmul(featuresT, features)
    if normalize:
        gram = tf.divide(gram, tf.cast(H*W*C, tf.float32))
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    losses = 0
    N = len(style_layers)
    for n in range(N):
        gram = gram_matrix(feats[style_layers[n]])
        gram_target = style_targets[n]
        loss = (gram - gram_target)**2
        losses += tf.reduce_sum(loss) * style_weights[n]
        
    return losses 

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    img = tf.reshape(img, (H,W,3))
    imgoh = tf.slice(img, [0,0,0],[H-1,W,3])
    imgh = tf.slice(img, [1,0,0],[H-1,W,3])
    imgow = tf.slice(img, [0,0,0],[H,W-1,3])
    imgw = tf.slice(img, [0,1,0],[H,W-1,3])
    loss = tf.reduce_sum((imgoh-imgh)**2) + tf.reduce_sum((imgow-imgw)**2)
    return tv_weight * loss


class Style_Transferer_VGG19():
    def __init__(self, content_image, content_layer, content_to_style_ratio, style_image, 
                   style_layers_list, tv_weight=0,initial_lr=10, decay_lr_at=500,
                   decayed_lr=1, init_random = False):
        """Initialize style transfer!
        
        Inputs:
        - content_image: np.array, content image in the shape of ?x?x?x3
        - style_image: np.array, style image in the shape of ?x?x?x3
        - content_layer: name of layer to use for content loss
        - content_weight: weighting on content loss
        - style_layers: list of tuples with names of layers to use for style loss and their weight ('name','weight')
        - tv_weight: weight of total variation regularization term
        - init_random: initialize the starting image to uniform random noise
        """
        content_image = preprocess_image(content_image)
        style_image = preprocess_image(style_image)
        # Extract features from the content image
        tf.reset_default_graph() # remove all existing variables in the graph 
        self.sess = get_session()
        VGG_MODEL = 'net/VGG19/imagenet-vgg-verydeep-19.mat'
        model = load_vgg_model(VGG_MODEL)
        
        content_weight=1000
        content_target = self.sess.run(model[content_layer],{model['input']:content_image})
        
        style_layers = [layer[0] for layer in style_layers_list]
        style_weights = [layer[1] for layer in style_layers_list]
        
        # Extract features from the style image   
        style_feat_vars = [model[layer] for layer in style_layers]
             
        # Compute list of TensorFlow Gram matrices
        style_target_vars = []
        for style_feat_var in style_feat_vars:
            style_target_vars.append(gram_matrix(style_feat_var))
        
        # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
        style_targets = self.sess.run(style_target_vars, {model['input']:style_image})
        
        # Initialize generated image to content image
        if init_random:
            self.img_var = tf.Variable(tf.random_uniform(content_image.shape, 0, 1), name="image", dtype=tf.float32)
        else:
            self.img_var = tf.Variable(generate_noise_image(content_image), name="image", dtype=tf.float32)
        
        # get current image features from VGG19
        img_feats = load_vgg_model(VGG_MODEL, self.img_var)
        
        # Compute all loss functions
        c_loss = content_loss(img_feats[content_layer], content_target)
        s_loss = style_loss(img_feats, style_layers, style_targets, style_weights)
        if tv_weight != 0:
            t_loss = tv_loss(self.img_var, tv_weight)   #tensor variation regularization
        else:
            t_loss = 0
        alpha = content_to_style_ratio
        beta = 1 - content_to_style_ratio
        self.loss = alpha*c_loss + beta*s_loss + t_loss
        self.runtime_loss = c_loss + s_loss + t_loss
        
        # Set up optimization hyperparameters
        self.initial_lr = initial_lr
        self.decayed_lr = decayed_lr
        self.global_t = 0
        self.min_loss = 999999999999999 # will be used to find the best image in case of divergence. kind of clumsy... but it works!
        
        # Create and initialize the Adam optimizer
        self.lr_var = tf.Variable(self.initial_lr, name="lr")
        # Create train_op that updates the generated image when run
        with tf.variable_scope("optimizer") as opt_scope:
            self.train_op = tf.train.AdamOptimizer(self.lr_var).minimize(self.loss, var_list=[self.img_var])
        # Initialize the generated image and optimization variables
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
        self.sess.run(tf.variables_initializer([self.lr_var, self.img_var] + opt_vars))
        # Create an op that will clamp the image values when run
                
    
    def generate_image(self, iter_num, draw_every=300, print_every=50, 
                       decay_lr_at=400, initial_lr=None, decayed_lr=None):
        '''
        the train operation of the style tranferer
        '''        
        # If user updated the kearning rate parameters, the graph would be updated
        if initial_lr != None and self.global_t < decay_lr_at:
            self.sess.run(tf.assign(self.lr_var, initial_lr))
        elif decayed_lr != None and self.global_t > decay_lr_at:
            self.sess.run(tf.assign(self.lr_var, decayed_lr))
        # run the train iterations
        for t in range(iter_num):
            # Take an optimization step to update img_var
            t += self.global_t
            self.sess.run(self.train_op)
            loss = self.sess.run(self.runtime_loss)
            if loss <= self.min_loss:
                self.min_loss = loss
                img = self.sess.run(self.img_var)
                self.best_image = deprocess_image(img[0])
            
            if t == decay_lr_at:
                self.sess.run(tf.assign(self.lr_var, self.decayed_lr))
#            if t % 750 ==0 and gradual_decay_lr:
#                self.decayed_lr * 0.9
#                self.sess.run(tf.assign(self.lr_var, self.decayed_lr))
            if t % print_every == 0:                
                print('Iteration {}, loss= {:.3f}'.format(t, loss))
            
            if t % draw_every == 0:
                plt.figure(figsize=(10,10))                
                plt.imshow(self.best_image)
                plt.axis('off')
                title = 'Iteration: '+str(t)
                plt.title(title)
                plt.show()
        self.global_t = t+1
        loss = self.sess.run(self.runtime_loss)
        print('Iteration {}, loss= {:.3f}'.format(t, loss))
        img = self.sess.run(self.img_var)        
        plt.figure(figsize=(10,10))
        title = 'Iteration: '+str(t+1)
        plt.title(title)
        plt.imshow(deprocess_image(img[0]))
        plt.axis('off')
        plt.show()
      