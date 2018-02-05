from style_transfer_funcs_vgg19 import Style_Transferer_VGG19
from image_utils import load_image, plot_input_images

#%%

content_image_path= 'data/trump.jpg'
style_image_path= 'data/pasta.jpg'
img_resize = 720
content_img = load_image(content_image_path, img_resize)
style_img = load_image(style_image_path)
plot_input_images(content_img, style_img)

transferer_params = {
    'content_image' : content_img,
    'style_image' : style_img,
    'content_layer' : 'conv4_2',
    'content_to_style_ratio' : 1e-3,
    'style_layers_list' : [
                    ('conv1_1', 0.2),
                    ('conv2_1', 0.2),
                    ('conv3_1', 0.2),
                    ('conv4_1', 0.2),
                    ('conv5_1', 1.0)],
    'tv_weight' : 1e-2,
    'initial_lr': 12.5,  #optimizer learning rate at begining
    'decayed_lr': 3
    }
transferer = Style_Transferer_VGG19(**transferer_params)

opt_params = {
        'iter_num':500, 
        'draw_every':200, 
        'print_every':50, 
        'decay_lr_at':500,
        'initial_lr':None, # can be used to update learning rate without initializing the entire graph
        'decayed_lr':None  # can be used to update learning rate without initializing the entire graph
        }
transferer.generate_image(**opt_params)

