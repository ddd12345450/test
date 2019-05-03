# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:03:27 2019

@author: TONG
"""
#In tensorflow keras.layers package, layers are objects
#can use pre-existing layers or custom-made layers to achieve higher level of abstraction
#can avoid having to changing individual variables directly
# pre-existing layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers
# Dense,  Conv2D, LSTM, BatchNormalization, Dropout, ...
import tensorflow as tf

tf.enable_eager_execution()

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to 
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# To use a layer, simply call it.
layer(tf.zeros([10, 5]))

# Layers have many useful methods. For example, you can inspect all variables
# in a layer by calling layer.variables. In this case a fully-connected layer
# will have variables for weights and biases.
layer.variables
# The variables are also accessible through nice accessors
layer.kernel, layer.bias


# =============================================================================
# CUSTOM-MADE LAYER
# =============================================================================
#Best way: extend tf.keras.layers.Layer class + 
# 1. implement __init__ = input-dependent initialization
# 2. implement build = shapes of input tensors + create variables
# 3. implement call = forward computation(output of layer)
### creating variables in "build" allows late-variable creation based on shape of inputs
### ie. no need for explicit specification of shapes in build(already built in class method)

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__() #inheritance from tf.keras.layers.Layer class
    self.num_outputs = num_outputs
    
    #call build to state input shape
  def build(self, input_shape):
    self.kernel = self.add_variable("kernel", 
                                    shape=[int(input_shape[-1]), 
                                           self.num_outputs])  # weight matrix = input shape*output shape
    #call call to fill in inputs
  def call(self, input):
    return tf.matmul(input, self.kernel) # input * weight matrix = output
  
layer = MyDenseLayer(10)
#print(layer(tf.zeros([10, 5])))
#print(layer.variables)

# =============================================================================
# COMPOSING LAYERS
# =============================================================================
# Composition of pre-existing layers
# inheritance from tf.keras.Model
# example of ResNet
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters): #filters are the sliding windows in convoluted neural networks(also known as multi-layer perceptrons)
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1)) 
    # filter (Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)) 
    # and kernel size/dimensions
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)  #rectified linear unit activation= function, always used with convoluted neural layer

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

    
block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.variables])


#using tf.keras.Sequential - the simple way
 my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(2, 1, 
                                                      padding='same'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(3, (1, 1)),
                               tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))