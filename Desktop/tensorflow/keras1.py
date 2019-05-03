# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:15:59 2019

@author: TONG
"""

import tensorflow as tf
from tensorflow.keras import layers

# prints keras version
print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

#Compiling the model to be used for training
#3 important arguments
#1. optimizer - the optimisation procedure, can use tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, or tf.train.GradientDescentOptimizer.
#2. loss - the function to minimise, can use square error (mse), categorical_crossentropy, and binary_crossentropy
#3. metrics - training monitor
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])

#EXAMPLES 
## Configure a model for mean-squared error regression.
#model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#              loss='mse',       # mean squared error
#              metrics=['mae'])  # mean absolute error
#
## Configure a model for categorical classification.
#model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#              loss=tf.keras.losses.categorical_crossentropy,
#              metrics=[tf.keras.metrics.categorical_accuracy])

# =============================================================================
# 
# =============================================================================
## Create a sigmoid layer:
#layers.Dense(64, activation='sigmoid')
## Or:
#layers.Dense(64, activation=tf.sigmoid)
#
## A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
#layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#
## A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
#layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
#
## A linear layer with a kernel initialized to a random orthogonal matrix:
#layers.Dense(64, kernel_initializer='orthogonal')
#
## A linear layer with a bias vector initialized to 2.0s:
#layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
#

# =============================================================================
# DATA IMPORT
# =============================================================================
import numpy as np

#creates data containing random numbers from 0 to 1 uniformly distributed in 1000 x 32 array
data = np.random.random((1000,32)) 
labels = np.random.random((1000,10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

#fitting the model (note: epoch = enitre dataset, batch = within one epoch, data is sliced into smaller divisions)
#1. epochs - one epoch is one iteration over entire input data
#2. batch_size - size of each batch during training
#3. validation_data - performance monitor using validation data and labels
model.fit(data,labels,epochs=10,batch_size=32,validation_data=(val_data, val_labels))

# =============================================================================
# INPUT tf.data datasets using datasets API to scale to large datasets
# =============================================================================
# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
#steps_per_epoch = number of training steps run per epoch before moving on to next epoch


# =============================================================================
# EVALUATION AND PREDICTION
# =============================================================================
#evaluate loss and metrics (evaluating the model)
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)

#predict output of last layer in inference for the data provided
#(testing the model)
result = model.predict(data, batch_size=32)
print(result.shape)
#
#predictions = model.predict(x_test)
#print('First prediction:', predictions[0])
# 
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])