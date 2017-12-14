#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:24:39 2017

@author: yhsui
"""
import csv
import math
from numpy import concatenate
import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

i = 1

#select seed value
seed_list = [12043,241596,9131,45921,3013,45921,32107,11193,11107,34076,23491,1000,2391,2143]

#for seed in seed_list:

# load dataset
dataset = read_csv('data_shuffle.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
values = reframed.values
shuffleValues = values[:8615,:]
forTesting = values[8615:,:]
np.random.shuffle(shuffleValues)
newValues = concatenate((shuffleValues,forTesting),axis = 0)

# split into train and test sets
n_train_hours = 5200
train = newValues[:n_train_hours, :]
test = newValues[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1,activation='sigmoid'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(train_X, train_y, epochs=50, batch_size=5, validation_data=(test_X, test_y), verbose=1, shuffle=True)
model.fit(train_X, train_y, epochs=1, batch_size=5, validation_split=0.2, verbose=2, shuffle=True)

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

train_yhat = model.predict(train_X)
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

inv_train_yhat = concatenate((train_yhat, train_X[:, 1:]), axis=1)
inv_train_yhat = scaler.inverse_transform(inv_train_yhat)
inv_train_yhat = inv_train_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

train_y = train_y.reshape((len(train_y), 1))
inv_train_y = concatenate((train_y, train_X[:, 1:]), axis=1)
inv_train_y = scaler.inverse_transform(inv_train_y)
inv_train_y = inv_train_y[:,0]

sampleNum = len(inv_yhat) - 144
sampleResult = inv_yhat[sampleNum:]
sampleResult.tofile('result.csv',sep='\n',format='%10.5f')

# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
train_rmse = math.sqrt(mean_squared_error(inv_train_y, inv_train_yhat))
print('Train RMSE: %.3f' % train_rmse)
print('Test RMSE: %.3f' % rmse)

test_mape = mean_absolute_percentage_error(inv_y, inv_yhat)
train_mape = mean_absolute_percentage_error(inv_train_y, inv_train_yhat)
print('Train MAPE: %.3f' % train_mape)
print('Test MAPE: %.3f' % test_mape)

#i+=1