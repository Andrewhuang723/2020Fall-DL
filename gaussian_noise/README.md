# Gaussian noise on image wafer

## Introduction

Addiing Gaussian noise on the latent vector of the autoencoder of wafer images.

Requirements: numpy

### 1. Conv.py:

Consists of some classes which are necessary to build an overall neural network for prediction.

### (1) Convolution(kernel_size, channels, out_channels, strides, padding)

#### _CLASS_

* filter: The function which generates filters for convolution

* __call__(inputs): The forward pass including convolution & padding process

### (2) Pooling(size): Pooling layer

#### _CLASS_

* __call__(inputs, pooling="max"): Pooling methods include "max" (default) for maxpooling & "average" for average-pooling


### (3) net(n_linear, activation_list, batch_size)

#### _CLASS_

* Conv_2d(inputs): Construct several convolution and pooling layers with activation function operations.

* fit(inputs, outputs, x_val, y_val, epochs): Training the model by input training data and validation data through iterations

* eval(X_test, y_test, model): Evaluate the model with testing data

### 2. Network.py:

#### Backbone of neural network
