# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:04:20 2019

@author: TONG
"""

#Build, train and predict
# Eager execution makes TensorFlow evaluate operations immediately,
# returning concrete values instead of 
# creating a computational graph that is executed later. 

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
#os.path.basename returns the name of the last item of the path
print("Local copy of the dataset file: {}".format(train_dataset_fp))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

# =============================================================================
# SHUFFLE DATA AND PARSE DATA - important function to convert data table into desired shape for training model
# =============================================================================
# tf.contrib.data.make_csv_dataset = returns a tf.data.Dataset...
# ... {'feature_name': value}, a dictionary
# batch size

batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size, 
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset)) # a batch of 32 features
# features
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels.numpy(),
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length");

# STACKING ALL FEATURES IN ONE ARRAY OF batch size*number of features
# tf.stack = creates a feature matrix
# arguments = feature dictionary and axis(the stacking dimension, 0=stacking row by row, 1=stacking column by column)
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

#print(features[:5]) #inspect 


# =============================================================================
# BUILDING THE MODEL
# =============================================================================
model = tf.keras.Sequential([
        #have to have input_shape in first layer-takes  in number of features
        #have to have activation in each layer, without activation the layers behave as one layer, ReLU is common
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
#initially the model will return logit for each class
    #softmax function converts logit for each class into probability for each class
    #tf.nn.softmax
    
################## PREDICTION BEFORE TRAINING ##################################
##initially the model will return logit for each class
#predictions = model(features)
##softmax function converts logit for each class into probability for each class
#tf.nn.softmax(predictions[:5])
#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#print("    Labels: {}".format(labels))


################################################################################


# =============================================================================
# TRAINING
# =============================================================================
# =============================================================================
# DEFINE LOSS
# =============================================================================
## Sparse softmax cross entropy
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#
#l = loss(model, features, labels)
#print("Loss test: {}".format(l))

## Gradient for gradient descent method
  #expect loss to go down after applying changes to variables(weights)
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)
# Gradient of loss with respect to each variable = can check which variables have more contribution

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #applies compputed gradients to the weights to minimise loss function

global_step = tf.Variable(0) #step counter

#loss_value, grads = grad(model, features, labels)
#print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
#                                          loss_value.numpy()))
## GradientDescentOptimizer.apply_gradients
#optimizer.apply_gradients(zip(grads, model.variables), global_step)
#print("Step: {},         Loss: {}".format(global_step.numpy(),
#                                          loss(model, features, labels).numpy()))
# =============================================================================
# LOOPING
# =============================================================================
#WITHIN EACH EPOCH,
#1. TAKE ONE EXAMPLE
#2. MAKE PREDICTION
#3. CALCULATE LOSS AND GRADIENT
#4. UPDATE VARIABLES (optimizer here)
#5. EXTRA COUNTERS/STATS FOR VISUALIZATION/TRACKING
#6. REPEAT OVER REST OF THE EXAMPLES
#7. REPEAT OVER TOTAL NUMBER OF EPOCHS

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #applies compputed gradients to the weights to minimise loss function

global_step = tf.Variable(0) #step counter

num_epochs = 201

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.metrics.Mean()
  epoch_accuracy = tf.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))                                      
# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results);

# =============================================================================
# PREDICTION
# =============================================================================
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)
#shuffling and parsing
test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size, 
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

tf.stack([y,prediction],axis=1)


# =============================================================================
#     #UNLABELED DATA PREDICTION
# =============================================================================
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))