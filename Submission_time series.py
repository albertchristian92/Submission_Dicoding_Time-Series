#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Submission-Time series

#Nama: Albert Budi Christian
#Email: albert.christian92@gmail.com


# In[2]:


#import library yang dibutuhkan

import numpy as np 
import pandas as pd 
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ## DATASET 
# Download from kaggle 
# link : https://www.kaggle.com/mczielinski/bitcoin-historical-data?select=bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv

# In[3]:


#baca data
train = pd.read_csv("btc.csv")


# In[4]:


#cek data
train.tail(20)


# In[5]:


#cek apakah ada data yang kosong
train.isnull().sum()


# In[6]:


#hapus data yang kosong
train = train.dropna()
train.isnull().sum()


# In[7]:


#convert timestamp ke format waktu untuk melakukan grouping dataset
train['date'] = pd.to_datetime(train['Timestamp'],unit='s')
group = train.groupby('date')
Harga_btc = group['Weighted_Price'].mean()


# In[8]:


#Dikarenakan ini prediksi harga btc, train dan test dipilih 
#dengan memprediksi seberapa banyak data yang akan diprediksi dengan cara mengatur variable pred_range

pred_range = 120
data_train= Harga_btc[:len(Harga_btc)-pred_range].values.reshape(-1,1)
data_test= Harga_btc[len(Harga_btc)-pred_range:].values.reshape(-1,1)

#normalisasi data dilakukan untuk memudahkan dan mempercepat proses training dikarenakan harga btc yang rangenya tinggi
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(data_train)

#data test ini tidak digunakan untuk train, tapi untuk test hasil prediksi, untuk validasi kita split lagi data_train di atas 
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(data_test)


# In[9]:


#pre-process dataset untuk mencari label Y
def preprocess_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    npx = np.array(dataX)
    npy = np.array(dataY)
    rshapeX = np.reshape(npx, (npx.shape[0], 1, npx.shape[1]))
    return rshapeX, npy

train_x, train_y = preprocess_dataset(scaled_train)
test_x, test_y = preprocess_dataset(scaled_test)


# In[38]:


#define model
model = Sequential()
model.add(LSTM(10,activation="sigmoid",return_sequences = True,input_shape = (None, 1)))
model.add(Dense(units = 1))
model.summary()


# In[39]:


#compile model dan fit
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["mae"])
callback = EarlyStopping(monitor='loss', patience=5) #callback untuk stop berdasarkan loss
history = model.fit(train_x, train_y, batch_size = 2048, epochs = 50, verbose=1, shuffle=False, validation_split=0.2, callbacks=[callback])


# In[40]:


#plot history loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[41]:


# prediksi harga BTC
pred_harga_BTC = model.predict(test_x)
pred_harga_BTC = scaler_test.inverse_transform(pred_harga_BTC.reshape(-1, 1))

true = scaler_test.inverse_transform(test_y.reshape(-1, 1))


# In[42]:


rmse = sqrt(mean_squared_error(true, pred_harga_BTC))
print('Test RMSE: %.3f' % rmse)

mae = mean_absolute_error(true, pred_harga_BTC)
print('Test MAE: %.3f' % mae)


# In[43]:


pyplot.plot(pred_harga_BTC, label='predict')
pyplot.plot(true, label='true')
pyplot.legend()
pyplot.show()

