import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add ,Lambda,Concatenate,GlobalAvgPool1D
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten, merge, ZeroPadding2D, AveragePooling2D,MaxPooling2D,GlobalAveragePooling1D
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D

def _Convlstm(nb_class, input_dim,):

    model_input = Input(shape=input_dim)

    # Initial convolution
    x = ConvLSTM2D(filters=128, kernel_size=(3,3), activation='relu',return_sequences = False)(model_input)
    x = Flatten()(x)
    x = Dense(nb_class, activation='relu')(x)

    Convlstm = Model(inputs=model_input, outputs=x)

    return Convlstm

def CNNLSTM(num_classes, shape):
    x_in = Input(shape=shape) #(4,400,1)

    #CNN layer 
    inner = Conv2D(32,(3,3),strides=(1, 1),padding='same',name='conv1',kernel_initializer='he_normal')(x_in) #(4,200,64)
    inner = Activation('relu')(inner)
    inner = Conv2D(32,(3,3),strides=(1, 1),padding='same',name='conv3',kernel_initializer='he_normal')(inner) #(4,10,512)
    inner = Activation('relu')(inner)
    inner = Conv2D(96,(3,3),strides=(1, 1),padding='same',name='conv5',kernel_initializer='he_normal')(inner) #(4,10,512)
    inner = Activation('relu')(inner)
    inner = Conv2D(96,(3,3),strides=(1, 1),padding='same',name='conv6',kernel_initializer='he_normal')(inner) #(4,10,512)
    inner = Activation('relu')(inner)
    inner = Conv2D(96,(3,3),strides=(1, 1),padding='same',name='conv7',kernel_initializer='he_normal')(inner) #(4,10,512)
    inner = Activation('relu')(inner)
    #CNN to lstm
    inner = Reshape(target_shape=((-1,num_classes)),name='reshape')(inner)
    inner = LSTM(256,return_sequences=True,name='lstm')(inner)

    outputs = Dense(num_classes,activation='relu',name='Dense_output')(inner)

    model = Model(inputs=x_in, outputs=outputs)

    return model

def _cnn(num_classes,shape): 
    inputs = Input(shape=shape)
    #CNN layer 
    x = Conv2D(64,(3,3),strides=(3, 3),padding='same',name='conv1',kernel_initializer='he_normal')(inputs) # (10,10,64)
    x = Activation('relu', name='relu1')(x)
    x = Conv2D(128,(3,3),strides=(1, 3),padding='same',name='',kernel_initializer='he_normal')(x) # (10,10,64)
    x = Activation('relu', name='')(x)
    x = Conv2D(128,(3,3),strides=(1, 3),padding='same',name='conv2',kernel_initializer='he_normal')(x) # (10,10,64)
    x = Activation('relu', name='relu2')(x)
    x = Conv2D(256,(3,3),strides=(1, 3),padding='same',name='',kernel_initializer='he_normal')(x) # (10,10,64)
    x = Activation('relu', name='')(x)
    x = Conv2D(256,(3,3),strides=(1, 3),padding='same',name='conv3',kernel_initializer='he_normal')(x) # (10,10,64)
    x = Activation('relu', name='relu3')(x)

    x = Reshape(target_shape=((1,-1)),name='reshape')(x)
    x = GlobalAvgPool1D()(x)

    outputs = Dense(num_classes,activation='relu',name='Dense_output')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model