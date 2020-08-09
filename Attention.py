import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.models import Sequential,Model
from keras.layers import *
from keras.optimizers import SGD,Adam 

class OurLayer(Layer):
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs

class SelfAttention(OurLayer):

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        self.attention = Attention_1(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )
    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            o = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.reuse(self.attention, [x, x, x])
        return o
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)

def selfattention_timeseries(nb_class, input_dim,):
    model_input = Input(shape=input_dim)
    #model_input = SinCosPositionEmbedding(4)(model_input)
    O_seq = SelfAttention(16,32)(model_input)
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(1,activation='relu')(O_seq)
    
    model = Model(inputs=model_input, outputs=outputs)
    
    return model