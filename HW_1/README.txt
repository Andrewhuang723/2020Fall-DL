Neural network scratch
==============================
There are 2 scripts in this file.

1. Network.py: Contains several functions and classes, including ReLU, softmax, crossentropy and other utility functions. Additionally, the feed forward and back propagation algorithm are implemented in the script.
    (1) class nn:
        --init_parameters: For initializing parameters.
        "n_hidden" is the structure of the network, "activation list" is the activation function of each hidden layer.
        --output: It is quite similar to a single layer of feed forward network, which outputs the value (or matrix) after function operation.
        output function will output "z", which is "weights" dot "inputs" plus "bias", "a" is the activation function output of "z". This function could be regarded as a single hidden layer.
    (2) class Model: super from 'nn'
        --forward_pass: By passing through the parameters, outputs of every hidden layer are solved
        concatenate all the outputs of each hidden layer, which return from the function "outputs". More details about the feed forward pass network, please look at page 5 of the slides "scratch.pptx"
        --backward: a simple algorithm of back propagation
        dcda_2 is the derivative of the loss to the output of the current hidden layer, z is the current pre-activation value, activation_function is the activation function of current layer"a_1" is the output of the previous hidden layer
        --back_propagation:
        "n_hidden", "activation_list" is the concatenate of hidden layers, activation functions. "pred" is the output of the forward_pass, "expected" is the ground-truth
        "parameters" is the concatenate of all the weights and bias in different activation function of hidden layer. Return the gradients of loss.
        --update: W := W - (learning rate) * gradients
        alpha = learining rate
    (3) mini_batch:
        Slicing the data into n_batches, each batch contains "batch_size" X and Y.
    (4) get_acc:
        According to the probability distribution based on softmax, if the index in prediction has the maximum probability, and matches the maximum value in ground truth data, then it is a correct prediction.
        the function returns the percentage in which the correct prediction labels over total predictions.
        y is ground truth and y_hat is prediction.
    (5) mini_batch_training:
        Slicing data into batches for training by "mini_batch". Get the initial parameters from data, "X_train", "y_train" and the model structure, "n_hidden", "activation_list".
        For each epoch in "epochs", each batch of data will passing through the feed forward network, class "Model().forward_pass()", returns predictions. Passing backward by class "Model().backpropagation()",
        and updated by gradient descent from class "Model().update()", the updated parameters is solved, iteratively, the terminal parameters are saved in the dictionary "model_para".
        Additionally, the loss and accuracy of each epoch are saved in the dictionary, "history". Same process with validation data, "X_val", "y_val".
    (6) predict:
        Predict the testing data "X_test", "y_test" with the optimized parameter "model_para" by "mini_batch_training", outputs the prediction, loss, and accuracy: "Y_pred", "loss", "accuracy".

2. Model.py: contains reading data and data preprocessing, and last, the model training as well as prediction is carried out.
    (1) num_images:
        Returns the number of images, which are saved in zip file
    (2) read_image:
        Read images from the file, the images are 784 pixels, by slicing the data in 784 per index and reshape to (28, 28), returns the array of the images.
    (3) read_label:
        Read labels from the file, returns the array with shape (n, 1)
    (4) one_hot_encode:
        Presenting label data as one hot encoded label.
    (5) shuffle:
        Make the data more flexible by shuffling the indices of data.
    (6) split:
        Splitting training data into training data and validation data.
