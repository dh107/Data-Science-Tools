# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

for i in range (0, n):
    data['date on production'][i] = data['date on production'][i][-4:]

data['date on production'] = pd.to_numeric(data['date on production'])

#data.insert(1, "col_name", "") 

#df.rename(columns = d, inplace = False)

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
      
# define dependent and independent variables
X = data_selected.iloc[:,0:25]
y = data_selected.iloc[:,-1]


train_dataset = data_selected.sample(frac=0.8, random_state=0)
test_dataset = data_selected.drop(train_dataset.index)

#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

X_train = train_dataset.copy()
X_test = test_dataset.copy()

y_train = X_train.pop('production')
y_test = X_test.pop('production')

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

#%%

feature_columns = []
for col_name in data_selected.columns: 
    feature_columns.append(tf.feature_column.numeric_column(col_name, dtype=tf.float32))


tf.random.set_seed(123)


# create the input functions. These will specify how data will be read into our model for both training and inference
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = dataset.repeat(n_epochs)
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(X_train, y_train)
test_input_fn = make_input_fn(X_test, y_test, shuffle=False, n_epochs=1)


n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not
# based on the number of steps.
est.train(train_input_fn, max_steps=100)

# Eval.
result = est.evaluate(test_input_fn)
clear_output()
print(pd.Series(result))





