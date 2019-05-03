# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()
print("Number of Images in Training set: ", len(train_labels))
print("Matrix dimension of each Training Image: ", train_images[0].shape)
#train_images.shape
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("The labels consist of: ")
for i in range(len(class_names)):
    print(class_names[i])

##Plotting the images
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
    
    
# =============================================================================
# Scaling values from 0-255 to 0-1
# =============================================================================
train_images = train_images / 255.0

test_images = test_images / 255.0

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#    
# =============================================================================
# Setting up the neural network
# =============================================================================
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #first layer (flattening) - for reformating data into 28*28 = 784 nodes
    keras.layers.Dense(128, activation=tf.nn.relu), ##Dense(fully connected) networks
    keras.layers.Dense(10, activation=tf.nn.softmax) #softmax later that returns the probability that the image is one of the 10 layers(10 nodes)
])
    
#Compiling model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 

#Training model
model.fit(train_images, train_labels, epochs=5)


#TESTING model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
predictions[0]


# =============================================================================
# Plotting
# =============================================================================
#Functions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
#Plotting 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  
  
# =============================================================================
#   PREDICTING A SINGLE IMAGE
# =============================================================================
# Grab an image from the test dataset
img = test_images[0]

# Add the image to a batch where it's the only member. (making a list)
img = (np.expand_dims(img,0)) #expanding dimensions

predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])