# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:09:48 2019

@author: TONG
"""

# Automatic differentiation - optimise machine learning models

import tensorflow as tf
tf.enable_eager_execution()


#GradientTape - compute gradient of a computation with respect to input variables
#ie. z with respect to x
#1. records all operations onto a "tape"
#2. uses tape and gradients of each operation to compute the gradient of the computation
# by using reverse mode differentiation
x = tf.ones((2, 2))
  
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
    
#or gradient with respect to intermediate values
    x = tf.ones((2, 2))
  
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

# GradientTape.gradient() releases resources. TO compute multiple gradients, use persistent=True
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape

# Python control flow are naturally handled, eg. if and while
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x) 

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

# Higher order computations
# ie. gradient computations within gradient computations
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
  dy_dx = t2.gradient(y, x) #first gradient (from inside to outside)
d2y_dx2 = t.gradient(dy_dx, x) #second gradient(outer gradient using first gradient)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
