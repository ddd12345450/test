# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:04:31 2019

@author: TONG
"""

import tensorflow as tf

tf.enable_eager_execution()

# Python is a stateful programming language
# Machine learning models need to have changing states - changing weights
# Tensorflow has built-in stateful operations
# To represent weights, use Tensorflow variables
# Tensorflow variables - an object
#                       -stores a value
#                       -can be used in Tensorflow computations (tf.assign_sub, tf.scatter_update)
#                       -computations are automatically recorded (for gradient computation)
#                       -embeddings - tensorflow does sparse updates by default-more computation and memory efficient
# =============================================================================
# VARIABLE
# =============================================================================
v = tf.Variable(1.0)
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0



# =============================================================================
# EXAMPLE- Using Tensor, GradientTape, Variable
# =============================================================================
#Steps
#1. Define model
#2. Define loss function
#3. Obtain Training data
#4. Training - Run through data, and use "optimizer" to adjust variables to fit data

# Simple linear model f(x) = x*W+b with W and b as variables
# Data is synthesised such that a well trained model would have W = 3.0 and b = 2.0

# =============================================================================
# DEFINE MODEL - using class
# =============================================================================
class Model(object):
  def __init__(self):
    # Initialize variable to (5.0, 0.0)
    # In practice, these should be initialized to RANDOM VALUES.
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)
    
  def __call__(self, x):   #creates a method when model is called with an argument
    return self.W * x + self.b
  
model = Model()

assert model(3.0).numpy() == 15.0

# =============================================================================
# DEFINE LOSS FUNCTION - minimise least squares
# =============================================================================
def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y)) #L2 norm/loss, sum of squared differences, least squares
#reduce_mean calculates mean of tensor
# =============================================================================
# OBTAIN TRAINING DATA
# =============================================================================
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES]) #This is the training data
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

################################################################
##Plotting model
#import matplotlib.pyplot as plt
#
#plt.scatter(inputs, outputs, c='b')  #DESIRED
#plt.scatter(inputs, model(inputs), c='r')  #PREDICTED
#plt.show()
#
#print('Current loss: '),
#print(loss(model(inputs), outputs).numpy())  #PREDICTED-DESIRED
#################################################################

# =============================================================================
# DEFINE TRAINING LOOP - update variables so loss goes down using gradient descent
# =============================================================================
# update variables to reduce gradient so as to minimise loss -minimise loss via gradient descent
# many gradient descent schemes in tf.train.Optimizer
# Here we implement the basic math
# in the function. we have 4 arguments
#1. the model,
#2. the training data (input),
#3. the outputs of the training data, and
#4. the learning rate
#ONE ITERATION:
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)  #set up for gradient calculation
  dW, db = t.gradient(current_loss, [model.W, model.b]) #calculate gradient of loss (ys) with respect to W and b (xs) of current model
  model.W.assign_sub(learning_rate * dW) #assign the result after subtraction in bracket
  model.b.assign_sub(learning_rate * db)
  
## REAL TRAINING - setting up the loop
model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)   # 10 iterations
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy()) # add current W and b to array
  current_loss = loss(model(inputs), outputs) #calculate loss

  train(model, inputs, outputs, learning_rate=0.1) #train with function
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--', # multiply list [] with a digit is equal to expanding the array to the size of the digit
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()
  