import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import math
import pickle
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn import metrics
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
plt.style.use('fivethirtyeight')

crypto_currency = 'BTC'
against_currency = 'USD'
start = dt.datetime (2018,1,1)
end =  dt.datetime.now()
#data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo',start , end)

data = yf.download(tickers='BTC-USD', period = '700d', interval = '1d')
data.dropna(inplace=True)

#training preparation

close_prices = data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

train_data = scaled_data[0: training_data_len, :]
prediction_days = 60
future_day = 1

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data) - future_day):
  x_train.append(scaled_data[x-prediction_days:x, 0])
  y_train.append(scaled_data[x + future_day, 0])


x_train, y_train = np.array(x_train), np.array(y_train)
#x_train, y_train = np.atleast_3d(x_train, y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#test set preparation
test_data = scaled_data[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape

#neural network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=15, batch_size=32)

#model evaluation
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


#plot
data = data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()