from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from pylab import *
import numpy as np
import tensorflow
from keras.utils.data_utils import get_file
import os
np.random.seed(123)  # for reproducibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Loading the dataset
def load_data(path='mnist.npz'):
        path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz', file_hash='8a61469f7ea1b51cbae51d4f78837e45') #Downloads a file from a URL if it not already in the cache 
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        return (x_train, y_train), (x_test, y_test)

(X_train, Y_train), (X_test, Y_test) = load_data()

# Preprocess input data - Adding the depth parameter to the training and testing samples X_*.reshape(no_of_samples, depth=1(grayscale image), width, height) + convert to float32 and convert the pixel values to the range 0 - 1
def preprocess_image(X_train, X_test):
	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	return X_train, X_test

# create a binary class matrix where - 1 is filled in one of the 10 columns if the image belongs to that class
def preprocess_labels(Y_train, Y_test):
	Y_train = np_utils.to_categorical(Y_train, 10)
	Y_test = np_utils.to_categorical(Y_test, 10)
	return Y_train, Y_test


(X_train, Y_train), (X_test, Y_test) = load_data()
X_train, X_test = preprocess_image(X_train, X_test)
Y_train, Y_test = preprocess_labels(Y_train, Y_test)

# Define the model 

def define_model():
	model = Sequential()
	model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(1,28,28), data_format="channels_first"))
	model.add(Convolution2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	return model
	

# Compile model

model = define_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model on training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
