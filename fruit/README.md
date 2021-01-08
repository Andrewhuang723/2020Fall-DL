## Fruit_Prediction

### Introduction
_Construct a 2D convolution and pooling model for predicting fruit images, which should be classified as "Carambula", "Lychee", "Pear"._

There are 4 python files: Conv.py, Network.py, fruit_model.py, fruit_predict.py

### Requirements: numpy

## 1. Conv.py:

Consists of some classes which are necessary to build an overall neural network for prediction.

#### Convolution(kernel_size, channels, out_channels, strides, padding)

_CLASS_

function:

(a) filter: The function which generates filters for convolution

(b) __call__(inputs): The forward pass including convolution & padding process

_input_shape: (batch_size, channels, w_in, h_in)_

_output_shape: (batch_size, out_channels, w_out, h_out)_

#### Pooling(size)

_CLASS_

function:

(a) __call__(inputs, pooling="max"): Pooling methods include "aax" (default) for maxpooling & "average" for average-pooling

_input_shape: (batch_size, channels, w_in, h_in)_

_output_shape: (batch_size, out_channels, w_out, h_out)_

#### net(n_linear, activation_list, batch_size)

_CLASS_

function:

(a) Conv_2d(inputs): Construct several convolution and pooling layers with activation function operations.

(b) fit(inputs, outputs, x_val, y_val, epochs): Training the model by input training data and validation data through iterations

(c) eval(X_test, y_test, model): Evaluate the model with testing data

### 2. Network.py:

#### Model

_CLASS_

function:
    
(a) init_parameters(input_shape, n_hidden, activation_list): initialized the parameters for each hidden layer.

(b) output(inputs, weights, bias, activation_function=None): calculated the output and activation output for a hidden layer.

(c) forward_pass(n_hidden, activation_list, parameters): calculated the outputs and activation outputs from all the hidden layers

(d) backward(dcda_2, a_1, z, weights_1, activation_function): calculated the gradients of loss (dL/dW, dL/db, dL/da) for a hidden layer.

(e) backpropagation(pred, expected, parameters, n_hidden, activation_list): calculated the gradients (dL/dW, dL/db) of loss for each hidden layer.

(f) update(alpha, gradients, parameters, n_hidden): updated gradients according to gradient descent.

### 3. fruit_model.py
Read and augment images from training data

(1) shuffle(X, Y): Shuffle the datasets by random index

(2) split(X, Y, rate): Splitting data into training data and validation data

### 4. fruit_predict.py
Read and predict images from testing data by constructed model, and visualized the results.
