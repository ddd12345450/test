# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:39:06 2019

@author: TONG
"""
#This notebook introduced a few techniques to handle a regression problem.
#
#Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
#Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
#When input data features have values with different ranges, each feature should be scaled independently.
#If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
#Early stopping is a useful technique to prevent overfitting.


from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

# =============================================================================
# Getting file and importing using pandas
# =============================================================================
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy() #copy() method copies the data using shallow copy ie. the original data will be unaffected by any changes to the oopied data, and vice versa
print(dataset.tail()) #shows the last 5 data in the dataset

#Check for unknown values
dataset.isna().sum()
#Drop rows with unknown values
dataset = dataset.dropna()

#Convert "Origin" column to numeric
origin = dataset.pop('Origin') #remove the origin column and change to numbers
dataset['USA'] = (origin == 1)*1.0 #adds an zero USA column where, at indices where the in the popped column origin==1, changes the value in that cell to 1
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()


# =============================================================================
# TRAINING 
# =============================================================================
train_dataset = dataset.sample(frac=0.8,random_state=0) #samples data,0.8 for training, 0.2 for testing, random_state is the seed for number generator for repeatability
test_dataset = dataset.drop(train_dataset.index) #removes columns or rows( in this case remove rows with the specified indices)

#pairplot
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde") # looks at data distribution

#descriptive statistics describe()
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#separate labels(the label that will be used for prediction) from the dataset
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
#MPG is a measure of fuel efficiency

#NORMALIZATION
#As the descriptive statistics show that there is a distinctive variation in the feature value ranges,
#the data should be normalized (optional)
#here we normalize using (x-mean)/std
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#BUILDING MODEL
def build_model():
  model = keras.Sequential([
          #3 layers
          #1. two dense layers - first layer taking in inputs of nine datapoints
          #2. single node output returning a continuous value
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

#TRAINING 
# ***Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()]) #callback method(what to print)

#visualize training progress
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

plot_history(history)


model = build_model()

# from the graphs we see that error stops decreasing after 1000+ runs
# using callback that tests training condition for every epoch
# if no improvement is shown after a few epochs, then stop training
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# =============================================================================
# USING THE TEST SET
# =============================================================================
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100]) #if true=predicted, points should follow a straight line from -100,-100 to 100,100

# histogram of difference between predicted and true labels
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")



