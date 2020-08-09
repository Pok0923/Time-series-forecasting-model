from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,GRU

def build_Lstm(cell_num,input_shape,hidden_layer):
    model = Sequential()
    if hidden_layer==1:
        model.add(LSTM(units=cell_num,input_shape=input_shape,return_sequences=False))
    elif hidden_layer >1:
        for i in range(1,hidden_layer):
            model.add(LSTM(units=cell_num,input_shape=input_shape,return_sequences=True))
        model.add(LSTM(units=cell_num,input_shape=input_shape,return_sequences=False))
    model.add(Dense(1))

    return model

def build_GRU(cell_num,input_shape,hidden_layer):
    model = Sequential()
    if hidden_layer==1:
        model.add(GRU(units=cell_num,input_shape=input_shape,return_sequences=False))
    elif hidden_layer >1:
        for i in range(1,hidden_layer):
            model.add(GRU(units=cell_num,input_shape=input_shape,return_sequences=True))
        model.add(GRU(units=cell_num,input_shape=input_shape,return_sequences=False))
    model.add(Dense(1))

    return model 