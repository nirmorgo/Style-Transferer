# Style-Transferer
Deep learning style transfer! good times :) 
This is a tensorflow implementation of https://arxiv.org/abs/1508.06576 published on 2015.
had a great time when working on the examples in http://cs231n.github.io/ and thought that it would be nice to turn it into an app.
took the original code from assignment3, and modified it to an improved version that works with VGG19 pretrained layer.
some of the code was taken from: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style (the pre-trained weights loading part).

# Instructions
1. Load images into Style_Transferer_VGG19() when initializing the object
2. Use the method: generate_image(iter_num=500) #500 iterations are usually enough.

you can use the default parameters defined on main_vgg19.py they work pretty well for most cases

# Requirements
1. python3.5 
2. tensorflow=1.4 with GPU support (proper CUDA and cudnn)
3. numpy=1.14
4. scipy=1.0.0

A decent GPU. unless you like waiting for long periods of time.

# Examples
![alt text](https://i.imgur.com/b3w9XFj.jpg "fancy kitten")
![alt text](https://i.imgur.com/0P85lEa.jpg "pasta trump!")


