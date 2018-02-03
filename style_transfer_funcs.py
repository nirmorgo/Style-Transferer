import matplotlib.pyplot as plt
import tensorflow as tf
from image_utils import deprocess_image, preprocess_image, load_image

def content_loss(content_weight, content_current, content_target):
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
       
    content_current = tf.reshape(content_current, (H*W,C))    
    content_target = tf.reshape(content_target, (H*W,C))
    
    loss = (content_current - content_target)**2    
    loss = tf.reduce_sum(loss)
    
    return content_weight * loss


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
        loss = (gram - gram_target) ** 2
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


class Style_Transferer_Squeezenet():
    def __init__(self, sess, model, content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
        """Initialize style transfer!
        
        Inputs:
        - content_image: filename of content image
        - style_image: filename of style image
        - image_size: size of smallest image dimension (used for content loss and generated image)
        - style_size: size of smallest style image dimension
        - content_layer: layer to use for content loss
        - content_weight: weighting on content loss
        - style_layers: list of layers to use for style loss
        - style_weights: list of weights to use for each layer in style_layers
        - tv_weight: weight of total variation regularization term
        - init_random: initialize the starting image to uniform random noise
        """
        # Extract features from the content image
        self.sess = sess
        content_img = preprocess_image(load_image(content_image, size=image_size))
        feats = model.extract_features(model.image)
        content_target = sess.run(feats[content_layer],
                                  {model.image: content_img[None]})
        # Extract features from the style image
        style_img = preprocess_image(load_image(style_image, size=style_size))
        style_feat_vars = [feats[idx] for idx in style_layers]
        style_target_vars = []
        
        # Plot inputs
        f, axarr = plt.subplots(1,2)
        axarr[0].axis('off')
        axarr[1].axis('off')
        axarr[0].set_title('Content Source Img.')
        axarr[1].set_title('Style Source Img.')
        axarr[0].imshow(deprocess_image(content_img))
        axarr[1].imshow(deprocess_image(style_img))
        plt.show()
        
        # Compute list of TensorFlow Gram matrices
        for style_feat_var in style_feat_vars:
            style_target_vars.append(gram_matrix(style_feat_var))
        # Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
        style_targets = self.sess.run(style_target_vars, {model.image: style_img[None]})
    
        # Initialize generated image to content image        
        if init_random:
            self.img_var = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
        else:
            self.img_var = tf.Variable(content_img[None], name="image")
    
        # Extract features on generated image
        feats = model.extract_features(self.img_var)
        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(self.img_var, tv_weight)
        self.loss = c_loss + s_loss + t_loss
        
        # Set up optimization hyperparameters
        self.initial_lr = 3.0
        self.decayed_lr = 0.1
        self.decay_lr_at = 180 
        self.global_t = 0
        self.min_loss = 999999999999999 # will be used to find the best image in case of divergence
    
        # Create and initialize the Adam optimizer
        self.lr_var = tf.Variable(self.initial_lr, name="lr")
        # Create train_op that updates the generated image when run
        with tf.variable_scope("optimizer") as opt_scope:
            self.train_op = tf.train.AdamOptimizer(self.lr_var).minimize(self.loss, var_list=[self.img_var])
        # Initialize the generated image and optimization variables
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
        self.sess.run(tf.variables_initializer([self.lr_var, self.img_var] + opt_vars))
        # Create an op that will clamp the image values when run
        self.clamp_image_op = tf.assign(self.img_var, tf.clip_by_value(self.img_var, -1.5, 1.5))
    
    def generate_image(self, iter_num, draw_every=300, decay_lr=True):
        
        for t in range(iter_num):
            # Take an optimization step to update img_var
            t += self.global_t
            self.sess.run(self.train_op)
            loss = self.sess.run(self.loss)
            if loss <= self.min_loss:
                self.min_loss = loss
                img = self.sess.run(self.img_var)
                self.best_image = deprocess_image(img[0], rescale=True)
            
            if t < self.decay_lr_at:
                self.sess.run(self.clamp_image_op)
            if t == self.decay_lr_at:
                self.sess.run(tf.assign(self.lr_var, self.decayed_lr))
            if t % 1000 ==0 and decay_lr:
                self.decayed_lr * 0.9
                self.sess.run(tf.assign(self.lr_var, self.decayed_lr))
            if t % 100 == 0:                
                print('Iteration {}, loss= {:.3f}'.format(t, loss))
            
            if t % draw_every == 0:
                plt.figure()                
                plt.imshow(deprocess_image(self.best_image, rescale=True))
                plt.axis('off')
                title = 'Iteration: '+str(t)
                plt.title(title)
                plt.show()
        self.global_t = t+1
        loss = self.sess.run(self.loss)
        print('Iteration {}, loss= {:.3f}'.format(t, loss))
        img = self.sess.run(self.img_var)        
        plt.figure()
        title = 'Iteration: '+str(t)
        plt.title(title)
        plt.imshow(deprocess_image(img[0], rescale=True))
        plt.axis('off')
        plt.show()



        