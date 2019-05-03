# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:42:37 2019

@author: TONG
"""
# =============================================================================
# ADVANCED KERAS using FUNCTIONAL API (being able to return input and output)
# =============================================================================
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

#Instantiate model given inputs and outputs
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, batch_size=32, epochs=5)