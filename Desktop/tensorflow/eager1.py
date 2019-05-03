# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:26:25 2019

@author: TONG
"""

import tensorflow as tf

# tensor is a multi-dimensional array, has datatype(dtype) and shape
# operations with tensorflow changes object into tensor type
# Tensors are immutable and can be backed by GPU,TPU accelerator memory
# Easy conversion: NumPy operations convert Tensors into ndarrays, and vice versa
# can also use tensor.numpy() for direct conversion
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

# matmul: matrix multiplication
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


# =============================================================================
# GPU
# =============================================================================
# Tensorflow automatically decides whether to use CPU or GPU memory
# to perform operations on Tensors
# copies Tensors to the device if needed
# Checking whether GPU is used
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# Tensor.device
# provides fully qualified of string name of the device hosting the ccontents of the tensor
# + all details

#Explicit placement of Tensors on specific devices (not automatic)
# tf.device
def time_matmul(x):
  %timeit tf.matmul(x, x)

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000]) # random_uniform returns randomly distributed numbers with [min val, max val] as argument
  assert x.device.endswith("CPU:0") #assert : checks whether condition is True and stops if False
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
    
    
# =============================================================================
#   Dataset  
# =============================================================================
# tf.data.Dataset
# With this API, pipelines can be built to feed data into the model

# Creating source Dataset
    # Dataset.from_tensors
    # Dataset.from_tensor_slices
    # TextLineDataset
    # TFRecordDataset
    
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp() #creates secure temporary file

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
    

# Transforming dataset
    # map
    # batch
    # shuffle
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

# Iterating datasets
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)