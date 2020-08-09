import h5py
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error ,mean_absolute_error
from data_pre import split_,train_data,inverse_transform
from RNN import *
from Attention import selfattention_timeseries
from CNN import *
from Densenet import densenet_
from Mobilenet import mobilenet_test



filename = 'data.h5'
file = h5py.File(filename)
data_x,data_y = split_(file)
train_x,test_x,train_y,test_y = train_data(data_x,data_y)

model=mobilenet_test((n_seq,n_steps, feature),n_steps,'try')
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_x, train_y, epochs=100, batch_size=32, verbose=0)

testPrediction = model.predict(test_x)
testprediction_0 = testPrediction[:,546:547]
test_y_0 = test_y[:,546:547]

# Get something which has as many features as dataset
testPrediction_extended_0 = np.zeros((len(testprediction_0), 1))
# Put the predictions there
testPrediction_extended_0[:, ] = testprediction_0[:, ]
# Inverse transform it and select the 3rd column.
testprediction_0 = inverse_transform(testPrediction_extended_0,max_,min_)

# extend
test_y_extended = np.zeros((len(test_y_0), 1))
# swap
test_y_extended[:, ] = test_y_0
# inverse
test_y_0 = inverse_transform(test_y_extended,max_,min_)

#rmse
rmse_testScore_1 = math.sqrt(mean_squared_error(test_y_0, testprediction_0))

