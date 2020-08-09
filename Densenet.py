import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add ,Lambda,Concatenate,GlobalAvgPool1D
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten, merge, ZeroPadding2D, AveragePooling2D,MaxPooling2D,GlobalAveragePooling1D
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D

def _DenseLayer(input, nb_filter, bn_size, dropout_rate):
    #x = BatchNormalization()(input)
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter*bn_size, (1, 1), kernel_initializer="he_uniform", padding="same" )(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same" )(x)

    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    return x

def _DenseBlock(x, num_layers, num_features, bn_size, growth_rate, dropout_rate):
    feature_list = [x]
    for i in range(num_layers):
        x = _DenseLayer(x, growth_rate, bn_size, dropout_rate)
        feature_list.append(x)
        x = Concatenate()(feature_list)
        num_features += growth_rate

    return x, num_features

def densenet_(nb_class, input_dim, growth_rate=12, nb_dense_block=4, layer=5, nb_filter=32, dropout_rate=0.2):

    model_input = Input(shape=input_dim)

    # Initial convolution
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False)(model_input)
    #x = BatchNormalization(name='batch1')(x)
    x = Activation('relu', name='relu1')(x)

    # Add dense blocks
    num_features = nb_filter
    num_layers = layer
    x, nb_filter = _DenseBlock(x, num_layers=num_layers, num_features=num_features, bn_size=nb_dense_block, growth_rate=growth_rate, dropout_rate=dropout_rate)

    # The last 
    x = BatchNormalization(name='batch_last')(x)
    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", name="last_conv2D", use_bias=False)(x)
    x = Reshape(target_shape=((-1,nb_classes)),name='reshape')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(nb_classes, activation='relu')(x)

    densenet = Model(inputs=model_input, outputs=x)

    return densenet