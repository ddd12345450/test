# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:16:15 2019

@author: TONG
"""

#sharing models-model building code and parameters

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model


# Create a basic model instance
model = create_model()
model.summary()


# =============================================================================
# Save Checkpoints during Training
# =============================================================================
# save checkpoints DURING and at the END of training
#Saves weights
# Restoring model: Create a new model with same architecture as original model and share weights
# uses tf.keras.callbacks.ModelCheckpoint (callback method)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()
model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

#New model
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#Loading the trained weights into new model
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# =============================================================================
# Saving checkpoints with options
# =============================================================================
# Unique name ({epoch:04d})
# Number of checkpoints (periods)
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt" #04d changes a number to 4 digit places ie. 1 will become 0001
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

#Check the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

#Reloading weights into new model
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# =============================================================================
# Manually saving weights
# =============================================================================
# Model.save_weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# =============================================================================
# Saving entire model (weights, model architecture and optimizer configuration, but needs to re-compile)
# =============================================================================
# HDF5 format
model = create_model()

# You need to use a keras.optimizer to restore the optimizer state from an HDF5 file.
model.compile(optimizer='adam', 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# Checking
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# =============================================================================
# Saved model
# =============================================================================
model = create_model()

model.fit(train_images, train_labels, epochs=5)

#saving model
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

#reloading model
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model

# The optimizer was not restored, re-attach a new one.
new_model.compile(optimizer='adam', 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))