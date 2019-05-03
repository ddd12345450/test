# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:42:34 2019

@author: TONG
"""

# =============================================================================
# BINARY CLASSIFICATION OF IMDB DATASET (POSITIVE OR NEGATIVE)
# =============================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("""Number of Training Data: %d Number of Test Data: %d""" %(len(train_data),len(test_data)))

if np.shape(train_data[0])[0] == np.shape(train_data[1])[0]: #or len(train_data[0])
    print("The data has the same shape, which is %d" %(np.shape(train_data[0])))
else:
    print("The data has different shapes")
    
#We need to convert all data into the same shape
    #first we convert the integers to text(string)
    
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved(move the indices back)
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,"?") for i in text])

#TWO ways to make all data has the same shape(size)
    #1. Pad all the data to the same length as the largest data
    #2. Make a distribution table(frequency of occurence table), with number of indices*number of occurrence size
    
# =============================================================================
#     PADDING
# =============================================================================
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# =============================================================================
# BUILDING AND TRAINING MODEL 
# =============================================================================
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
#Embedding layer - learns the embedding vector for each word
model.add(keras.layers.Embedding(vocab_size, 16))
#Global Average Pooling - averages and converts the vectors into a fixed-length output vector
model.add(keras.layers.GlobalAveragePooling1D())
#Dense layer
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#Output with sigmoid activation between 0 and 1
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary() #shows the model summary

#binary_crossentropy is better for dealing with probabilities
#It measures the distance between probability distributions - ie. between the true distribution and predicted distribution

#COMPILING and TRAINING with VALIDATION
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1) #verbose>0 slows down printing out of information

results = model.evaluate(test_data,test_labels)

print(results)

# =============================================================================
# Create a graph of accuracy and loss over time
# =============================================================================
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# =============================================================================
# 
# =============================================================================
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#
#Notice the training loss decreases with 
#each epoch and the training accuracy increases 
#with each epoch. This is expected when using a 
#gradient descent optimization—it should minimize 
#the desired quantity on every iteration.
#
#This isn't the case for the validation loss and 
#accuracy—they seem to peak after about twenty 
#epochs. This is an example of overfitting: the 
#model performs better on the training data than 
#it does on data it has never seen before. After 
#this point, the model over-optimizes and learns 
#representations specific to the training data that 
#do not generalize to test data.
#
#For this particular case, we could prevent 
#overfitting by simply stopping the training after 
#twenty or so epochs. Later, you'll see how to do 
#this automatically with a callback.