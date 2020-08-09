import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add ,Lambda,Concatenate,GlobalAvgPool1D
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten, merge, ZeroPadding2D, AveragePooling2D,MaxPooling2D,GlobalAveragePooling1D
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D

""" Define layers block functions """
def relu6(x):
  return K.relu(x, max_value=6)

def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

# ** update custom Activate functions
get_custom_objects().update({'custom_activation': Activation(Hswish)})

def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
    #x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias,kernel_initializer='he_normal')(_inputs)
    #x = BatchNormalization()(x)
    if activation == 'RE':
        x = ReLU(name=name)(x)
    elif activation == 'HS':
        x = Activation(Hswish, name=name)(x)
    elif activation == 'RE6':
        x = Activation(relu6, name=name)(x)
    else:
        raise NotImplementedError
    return x

def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same',kernel_initializer='he_normal')(_inputs)
    #x = BatchNormalization()(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation(Hswish)(x)
    elif activation == 'RE6':
        x = Activation(relu6)(x)
    else:
        raise NotImplementedError
    return x

def __global_depthwise_block(_inputs):
    assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    kernel_size = _inputs._keras_shape[1]
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid',kernel_initializer='he_normal')(_inputs)
    return x

def __se_block(_inputs, ratio=4, pooling_type='avg'):
    filters = _inputs._keras_shape[-1]
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = GlobalAveragePooling2D()(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs)
    else:
        raise NotImplementedError
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([_inputs, se])

def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True, is_use_se=True, activation='RE', num_layers=0, *args):
    with tf.name_scope('bottleneck_block'):
        # ** to high dim 
        bottleneck_dim = expansion_dim

        # ** pointwise conv 
        x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias, activation=activation)

        # ** depthwise conv
        x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation, num_layers=num_layers)

        # ** pointwise conv
        x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',kernel_initializer='he_normal')(x)
        #x = BatchNormalization()(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(_inputs)[-1]
            if in_dim != out_dim:
                ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
                x = Add()([x, ins])
            else:
                x = Add()([x, _inputs])
    return x

def mobilenet_test(shape, num_classes,num_conv,kernel_size,stride, model_type='large', pooling_type='avg', include_top=True):
    # ** input layer
    #inputs = Input(batch_size=20,shape=shape, sparse=True)
    inputs = Input(shape=shape,) 

    # ** feature extraction layers
    net = __conv2d_block(inputs, num_conv, kernel=(kernel_size, kernel_size), strides=(stride, stride), padding='same', is_use_bias=False,activation='RE') 
    if model_type == 'small':
        config_list = small_config_list
    elif model_type == 're':
        config_list = re_config_list
    elif model_type == 'try':
        config_list = try_config_list
    elif model_type == 'test':
        config_list = test_config_list
    else:
        raise NotImplementedError
        
    for config in config_list:
        net = __bottleneck_block(net, *config)
    
    # ** final layers
    net = __conv2d_block(net, num_conv, kernel=(kernel_size, kernel_size), strides=(stride, stride), padding='same', is_use_bias=False,activation='HS', name='output_map')
    net = Reshape(target_shape=((1,-1)),name='reshape')(net)
    O_seq = GlobalAvgPool1D()(net)
    outputs = Dense(num_classes,activation='relu')(O_seq)

    model = Model(inputs=inputs, outputs=outputs)

    return model

""" define bottleneck structure """
# ** 
# **               
global small_config_list
global try_config_list
global re_config_list
global test_config_list
re_config_list = [[16,  (3, 3), (1, 1), 32,  False, True, False,  'RE', 1],
                     [32,  (3, 3), (1, 1), 48,  False, True, False, 'RE', 2],
                     [48,  (3, 3), (1, 1), 64,  False, True, False,  'HS', 3],
                     [64,  (3, 3), (1, 1), 80,  False, True, False, 'HS', 4],
                     [80,  (3, 3), (1, 1), 96,  False, True, False, 'HS', 5]]

small_config_list = [[80,  (3, 3), (1, 1), 96,  False, True, False,  'RE', 1],
                     [64,  (3, 3), (1, 1), 80,  False, True, False, 'RE', 2],
                     [48,  (3, 3), (1, 1), 64,  False, True, False,  'HS', 3],
                     [32,  (3, 3), (1, 1), 48,  False, True, False, 'HS', 4],
                     [16,  (3, 3), (1, 1), 32,  False, True, False, 'HS', 5]]

try_config_list =   [[40,  (3, 3), (1, 1), 56,  False, True, False,  'RE', 0],
                     [24,  (3, 3), (1, 1), 40,  False, True, False, 'RE', 2],
                     [8,  (3, 3), (1, 1), 24,  False, True, False,  'HS', 3],
                     [24,  (3, 3), (1, 1), 40,  False, True, False, 'HS', 4],
                     [40,  (3, 3), (1, 1), 56,  False, True, False, 'HS', 5]]

test_config_list =   [[80,  (3, 3), (1, 1), 96,  False, True, False,  'RE', 0],
                     [48,  (3, 3), (1, 1), 64,  False, True, False, 'RE', 1],
                     [16,  (3, 3), (1, 1), 32,  False, True, False,  'HS', 2],
                     [48,  (3, 3), (1, 1), 64,  False, True, False, 'HS', 3],
                     [80,  (3, 3), (1, 1), 96,  False, True, False, 'HS', 4]]