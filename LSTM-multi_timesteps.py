#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:30:36 2017

@author: yhsui
"""

from math import sqrt
from numpy import concatenate
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert data to multiple time steps
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names

#define MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

i = 10

#select seed value
seed_list = [12043,241596,9131,45921,3013,45921,32107,11193,11107,34076,23491,1000,2391,2143]

#seed_list = [12043,241596]

for seed in seed_list:
    
    np.random.seed(seed)

# load dataset
    dataset = read_csv('data.csv', header=0, index_col=0)
    values = dataset.values
# integer encode direction
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])

    values = values.astype('float32')

# normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

# specify the number of lag hours
    n_hours = 6 # 3 and 12 respectively
    n_features = 7
    
# frame the dataset
    reframed = series_to_supervised(scaled, n_hours, 1)
    print(reframed.shape)

# split into train and test sets
    values = reframed.values
    n_train_hours = 5200  # about 60%
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
# split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse', optimizer='adam')
# fit network
    model.fit(train_X, train_y, epochs=50, batch_size = 5, validation_split = 0.2, validation_data=(test_X, test_y), verbose=2, shuffle=True)

# make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

    train_yhat = model.predict((train_X))
    train_X = train_X.reshape((train_X.shape[0],n_hours*n_features))

# invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    inv_train_yhat = concatenate((train_yhat,train_X[:,-6:]),axis = 1)
    inv_train_yhat = scaler.inverse_transform(inv_train_yhat)
    inv_train_yhat = inv_train_yhat[:,0]

# invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    train_y = train_y.reshape((len(train_y),1))
    inv_train_y = concatenate((train_y,train_X[:,-6:]),axis = 1)
    inv_train_y = scaler.inverse_transform(inv_train_y)
    inv_train_y = inv_train_y[:,0]

# get Feb and Sep data
    summer1 = 720-n_hours
    summer2 = 791-n_hours+1
    winter1 = 5064-n_hours
    winter2 = 5135-n_hours+1

    summerResult = inv_train_yhat[summer1:summer2]
    summerResult.tofile('resultSummer'+str(i)+'.csv',sep='\n',format='%10.5f')

    winterResult = inv_train_yhat[winter1:winter2]
    winterResult.tofile('resultWinter'+str(i)+'.csv',sep='\n',format='%10.5f')
    
# calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    train_rmse = sqrt(mean_squared_error(inv_train_y, inv_train_yhat))
    print('Train RMSE: %.3f' % train_rmse)
    print('Test RMSE: %.3f' % rmse)

    test_mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    train_mape = mean_absolute_percentage_error(inv_train_y, inv_train_yhat)
    print('Train MAPE: %.3f' % train_mape)
    print('Test MAPE: %.3f' % test_mape)
    
    i= i+1
