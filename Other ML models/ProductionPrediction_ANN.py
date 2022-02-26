# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import seaborn as sns
import os


raw_data = pd.read_csv('interview_dataset.csv')

data = raw_data.copy()
data.tail()


n = len(data['production'])
data['treatment company'] = pd.to_numeric(data['treatment company'].str.replace('treatment company ',''))
data['operator'] = pd.to_numeric(data['operator'] .str.replace('operator ',''))

raw_data['date'] = pd.to_datetime(raw_data['date on production'], format='%m/%d/%Y')
raw_data['age'] = ((pd.to_datetime("2020-11-01") - raw_data['date']).dt.days)/365.25

data['date on production'] = raw_data['age']


X = data.iloc[:,0:27]
y = data.iloc[:,-1]

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# count nan in each col
nancount = data.isna().sum()

# drop the two columns with large number of nan values
data = data.drop(columns=['water saturation', 'breakdown pressure'])

# data = data.drop(columns=['well spacing', 'isip'])

#data = data.drop(columns=['treatment company', 'azimuth', 'operator', 'well spacing', 'porpoise deviation', ...
#                          'porpoise count'])

data_selected = data
data_selected = data[['md (ft)', 'date on production', 'footage lateral length', 'total number of stages', 'proppant volume', 'production']].copy()

# drop nan
data_selected = data_selected.dropna()
data_selected = data_selected

# normalized data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

data_selected.iloc[:,0:len(data_selected.columns)-1] = normalize(data_selected.iloc[:,0:len(data_selected.columns)-1])

      
# define dependent and independent variables
# X = data_selected.iloc[:,0:25]
# y = data_selected.iloc[:,-1]

#%%
train_dataset = data_selected.sample(frac=0.8, random_state=0)
test_dataset = data_selected.drop(train_dataset.index)

#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('production')
test_labels = test_features.pop('production')


train_dataset.describe().transpose()[['mean', 'std']]

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

#%%
'''
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
  

horsepower = np.array(train_features['proppant volume'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

horsepower_model.predict(horsepower[:10])

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = horsepower_model.fit(
    train_features['proppant volume'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 2500])
  plt.xlabel('Epoch')
  plt.ylabel('Error [production]')
  plt.legend()
  plt.grid(True)
  
plot_loss(history)

#%%
test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['proppant volume'],
    test_labels, verbose=0)

x = horsepower[100:300]
y = horsepower_model.predict(x)
'''

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 2500])
  plt.xlabel('Epoch')
  plt.ylabel('Error [production]')
  plt.legend()
  plt.grid(True)
  
  
def plot_horsepower(x, y):
  plt.scatter(train_features['proppant volume'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('proppant volume')
  plt.ylabel('production')
  plt.legend()
  

#%%
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:26])

linear_model.layers[1].kernel

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


plot_loss(history)

test_results = {}
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

#%%
# Deep Neural Network (DNN) Model
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.01))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=300)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [production]']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [production]')
plt.ylabel('Predictions [production]')
lims = [0, 7500]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

 
 