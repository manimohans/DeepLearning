#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:03:05 2018

@author: mani

recurrent neural networks
for predicting stock prices
uses google data for past 5 years.
"""

#RNN
#Data pre-processing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt #viz
import pandas as pd #import and manage dataset

#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #upper bound excluded (2)

#feature scaling
#standardization or normalization
#recommended: normalization. whenever building RNN and sigmoid is 
#activation function in RNN, use normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1)) #normalized value between 0/1
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 time steps and 1 output
#at each time T, RNN will look at 60 prices before time T, and based on that, 
#it will predict next result at time T+1
"""
the number of time steps is really crucial. Lower val might result in
under-fitting, and higher val might result in over-fitting. should find
the optimal time step value
"""
X_train = []
y_train = []
#x_train is input value of last 60 days, y_train is prediction for next day
for i in range(60, 1258): #1257+1 , start from 60th day
    X_train.append(training_set_scaled[i-60:i, 0]) #past 60 days
    y_train.append(training_set_scaled[i, 0]) #i'th day

X_train, y_train = np.array(X_train), np.array(y_train) 
#MUST BE CHANGED TO NUMPY ARRAY

#Reshaping
#addition of unit - number of predictors we can use to predict what we want
#'open google stock price' is an indicator. we can add more indicators
#by doing the following (if you want). we are not adding much now
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #a, newshape, order='C'
#shape[0] is number of rows, shape[1] is number columns - timesteps
#new shape from keras
"""
Input shape

3D tensor with shape (batch_size, timesteps, input_dim).

Output shape

if return_state: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each with shape (batch_size, units).
if return_sequences: 3D tensor with shape (batch_size, timesteps, units).
else, 2D tensor with shape (batch_size, units).
"""
#end of pre-processing - on to RNN

#Building the RNN
#importing keras libs and pkgs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing a rnn
regressor = Sequential() #(not called classifier coz its predicting cont values))

#adding first LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#LSTM takes 3 args
#1 - number of units -num of lstm cells u want in the layer
#2 - return sequences, we have to set it == true, since we use data from back
#3 - input_shape, shape of input of x_train - very important, 3-dimension,
#just last two-dim is enough to specify in 3rd argument.

#dropout
regressor.add(Dropout(0.2)) #20% of neuron during training will be dropped out

#adding a 2nd LSTM with some dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding a 3rd LSTM with some dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding a 4th LSTM with some dropout
regressor.add(LSTM(units = 50)) #return_sequences removed since last LSTM
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1)) #fully connected to last LSTM layer
#output value is just one value so units == 1

#compiling the RNN
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')
#for RNN and keras doc, rmsprop optimizer is recommended
#but for this example, tutor has tested and found adam to be more
#effective. So we will use that.
#its not a classification so loss is not binary cross entropy
#since we are predicting values, we use mean square error loss type

#fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
#epochs can be anything but tutor found 100 is a good choice.

#Making predictions and visualization
#get real stock price of 2017
#importing the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values #upper bound excluded (2)

#get predicted stock price of 2017
#be very careful about using test values with already finished
#feature scaled data. use with actual data frames instead of fit_transform
#dataset.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']),
                          axis=0)
#contains the 'open' column of both datasets
#vertical axis is 0 horizontal is 1
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
#the difference is exactly jan 3 2017 the first day we wanted

#reshaping to get right numpy shape
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs) #dont call fit_transform since it was already fitted
#get same scaling as how regressor was trained

#copying from above and modifying
X_test = []
for i in range(60, 60+20): #last 60, plus twenty days
    X_test.append(inputs[i-60:i, 0]) #past 60 days

X_test = np.array(X_test)
#changing to 3d structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
#going back from scaled to original dimension value:
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the results in a plot
#use matplotlib plt function for plotting the values real vs pred
plt.plot(real_stock_price, color='red', label='Real $GOOGL')
plt.plot(predicted_stock_price, color='green', label='Predicted $GOOGL')
plt.title('LSTM prediction for $GOOGL stock price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

"""
as seen in the practical lectures, the RNN we built was a regressor. Indeed, we were dealing with Regression because we were trying to predict a continuous outcome (the Google Stock Price). For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.

However for our specific Stock Price Prediction problem, evaluating the model with the RMSE does not make much sense, since we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. We want to check if our predictions follow the same directions as the real stock price and we don’t really care whether our predictions are close the real stock price. The predictions could indeed be close but often taking the opposite direction from the real stock price.

Nevertheless if you are interested in the code that computes the RMSE for our Stock Price Prediction problem, please find it just below:

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
Then consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.

here are different ways to improve the RNN model:

Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.

you can do some Parameter Tuning on the RNN model we implemented.

Remember, this time we are dealing with a Regression problem because we predict a continuous outcome (the Google Stock Price).

Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters.
"""