import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from numba import jit, vectorize

@jit
def crossentropy(pred, test):
    for i in pred.reshape(-1):
        if i == 0:
            i += 0.00001
        elif i == 1:
            i -= 0.00001
    n = test.shape[0] #(n,1,10)
    loss = -test * np.log(pred)
    loss = 1.0 /n * np.sum(loss)
    return loss
@jit
def ReLU(x):
    y = np.array(x).reshape(-1)
    for i in range(len(y)):
        if y[i] <= 0:
            y[i] = 0.00001

    y = y.reshape(x.shape)
    return y
@jit
def softmax(x):
    x_0 = np.exp(x - np.max(x))
    SUM = np.sum(x_0)
    prob = x_0 / SUM
    return prob
AA=np.arange(10,20)

@jit
def ReLU_grad(z):
    z = z.reshape(-1)
    z = np.diag(z)
    J = np.zeros([len(z), len(z)])
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i, j] > 0:
                J[i, j] += 1
            else:
                J[i, j] += 0.01
    return J
@jit
def softmax_grad(z):
    z = z.reshape(-1)
    J = np.zeros([len(z), len(z)])
    for j in range(len(z)):
        for i in range(len(z)):
            if j == i:
                J[i, j] += softmax(z)[i] * (1 - softmax(z)[i])
            else:
                J[i, j] += - softmax(z)[i] * softmax(z)[j]
    return J

def mini_batch(X, Y, batch_size, shuffle=True):
    size = np.arange(X.shape[0])
    if shuffle:
        rand = np.random.shuffle(size)
        X = X[rand].reshape(X.shape)
        Y = Y[rand].reshape(Y.shape)
    start = 0
    n_batches = len(size) // batch_size
    X_batch = []
    Y_batch = []
    for i in range(n_batches):
        X_i = X[start:start + batch_size]
        Y_i = Y[start:start + batch_size]
        start += batch_size
        X_batch.append(X_i)
        Y_batch.append(Y_i)
    res = len(size) % batch_size
    if res != 0:
        res_x = X[len(size) - res:]
        res_y = Y[len(size) - res:]
        X_batch.append(res_x)
        Y_batch.append(res_y)
        n_batches += 1
    return X_batch, Y_batch, n_batches

class nn:
    def __init__(self, inputs=None, ):
        self.inputs = inputs

    def init_parameters(self, input_shape, n_hidden, activation_list):
        '''"n_hidden" is the structure of the network,
        "activation list" is the activation function of each hidden layer.'''
        theta = {}
        first_inputs = input_shape
        for i in range(len(n_hidden)):
            np.random.seed(0)
            weights = np.random.normal(0.0, 0.1, size=(first_inputs, n_hidden[i]))
            first_inputs = n_hidden[i]
            theta["W" + str(i)] = weights
            np.random.seed(1)
            bias = np.random.normal(0.0, 0.01, size=(1, n_hidden[i]))
            theta["b" + str(i)] = bias
            theta["activation" + str(i)] = activation_list[i]
        return theta

    def output(self, inputs, weights, bias, activation_function=None):
        '''output function will output "z", which is "weights" dot "inputs" plus "bias",
        "a" is the activation function output of "z". This function could be regarded as a single hidden layer.'''
        z = np.matmul(inputs, weights) + bias

        if activation_function is None:
            a = z
        else:
            a = activation_function(z)
        return a, z



class Model(nn):
    def forward_pass(self, n_hidden, activation_list, parameters):
        'concatenate all the outputs of each hidden layer in n_hidden and activation_list, which return from the function "outputs"'
        output_1 = self.inputs
        function_output = {}
        for i in range(len(n_hidden)):
            layer = nn(inputs=output_1) # (784,)
            output_1, output_2 = layer.output(inputs=output_1, weights=parameters["W" + str(i)], bias=parameters["b" + str(i)], activation_function=activation_list[i])
            output_1 = output_1.reshape(1,-1)
            Z_2 = output_2.reshape(1,-1)
            function_output["A" + str(i)] = output_1
            function_output["Z" + str(i)] = Z_2
        return function_output

    def backward(self, dcda_2, a_1, z, weights_1, activation_function):
        'dcda_2 is the derivative of the loss to the output of the current hidden layer, z is the current pre-activation value, activation_function is the activation function of current layer"a_1" is the output of the previous hidden layer'
        d_activate = ReLU_grad
        if activation_function is ReLU:
            d_activate = ReLU_grad
        elif activation_function is softmax:
            d_activate = softmax_grad
        #print(z.shape, dcda_2.shape, a_1.T.shape)
        dcda_2 = dcda_2.reshape(1,-1)
        #print(z.shape)
        #print(dcda_2.shape)
        dcdz = np.matmul(dcda_2, d_activate(z))
        #print(d_activate(z).shape)
        #print(dcdz.shape)
        #print(a_1.shape)
        #print("----")
        dcdw = np.matmul(a_1.T, dcdz)
        dcdb = dcdz
        #print(dcdz.shape, weights_1.shape)
        dcda = np.matmul(dcdz, weights_1.T)
        return dcdw, dcdb, dcda

    def back_propagation(self, pred, expected, parameters, n_hidden, activation_list):
        '''n_hidden, activation_list is the concatenate of hidden layers, activation functions.
        pred is the output of the forward_pass, expected is the ground-truth
        parameters is the concatenate of all the weights and bias in different activation function of hidden layer.'''
        gradient = {}
        pred = pred.reshape(expected.shape)
        for i in pred:
            if i == 0:
                pred += 0.00001
        dcda_2 = - expected / pred #(1,10)
        #print(dcda_2.shape)
        function_output = self.forward_pass(n_hidden=n_hidden, activation_list=activation_list, parameters=parameters)
        for n, m in enumerate(reversed(range(len(n_hidden) - 1))): # m is the previous layer and n is the current layer
            n = m + 1
            A = function_output["A" + str(m)]
            Z = function_output["Z" + str(n)]
            W = parameters["W" + str(n)]
            b = parameters["W" + str(n)]
            dcdw, dcdb, dcda_2 = self.backward(dcda_2, A, Z, W, activation_list[n])
            gradient["dcdw" + str(n)] = dcdw
            gradient["dcdb" + str(n)] = dcdb
        Z_0, A_0 = self.output(inputs=self.inputs, weights=parameters["W0"], bias=parameters["b0"], activation_function=None)
        #print(dcda_2.shape, Z_0.shape)
        dcdw0, dcdb0, dcda0 = self.backward(dcda_2, self.inputs.reshape(1,-1), Z_0, parameters["W0"], activation_function=activation_list[0])
        gradient["dcdw0"] = dcdw0
        gradient["dcdb0"] = dcdb0
        return gradient

    def update(self, alpha, gradients, parameters, n_hidden):
        new_theta = {}
        for i in range(len(n_hidden)):
            dLdw = gradients["dcdw" + str(i)]
            dLdb = gradients["dcdb" + str(i)]
            weights = parameters["W" + str(i)]
            bias = parameters["b" + str(i)]
            weights = weights - alpha * dLdw
            bias = bias - alpha * dLdb
            new_theta["W" + str(i)] = weights
            new_theta["b" + str(i)] = bias
        return new_theta




def get_acc(y, y_hat):
    'y is ground truth and y_hat is prediction'
    correct = 0
    for j in range(len(y)):
        i = np.where(y_hat[j] == np.amax(y_hat[j]))
        if y[j][i] == np.max(y[j]):
            correct += 1
    return correct / len(y)

def mini_batch_training(X_train, y_train, n_hidden, activation_list, batch_size, epochs, X_val, y_val):
    X_batch, y_batch, n_batches = mini_batch(X_train, y_train, batch_size=batch_size)
    X_val_batch, y_val_batch, val_n_batches = mini_batch(X_val, y_val, batch_size=batch_size)
    initial_parameters = nn().init_parameters(input_shape=X_train.shape[-1], n_hidden=n_hidden, activation_list=activation_list)
    All_para = {}
    for j in range(n_batches):
        All_para["theta" + str(j)] = initial_parameters # the dictionary combined all batches of parameters, which will be updated from initial value

    Accuracy = []
    Loss = []
    Val_Accuracy = []
    Val_Loss = []
    for t in range(epochs):
        ini_para = All_para
        acc = 0
        loss = 0
        for j in range(n_batches):
            batch_theta = ini_para["theta" + str(j)]

            ww = []
            bb = []
            Y_pred = []
            train_loss = 0
            if len(X_batch[j]) != batch_size:
                batch_size = len(X_batch[j])
            for i in range(batch_size):
                function_output = Model(inputs=X_batch[j][i]).forward_pass(n_hidden=n_hidden, activation_list=activation_list,
                                                                   parameters=batch_theta)
                y_pred = function_output["A3"]
                Y_pred.append(y_pred)

                grads = Model(inputs=X_batch[j][i]).back_propagation(pred=y_pred,
                                                            expected=y_batch[j][i], parameters=batch_theta,
                                                            n_hidden=n_hidden, activation_list=activation_list)

                train_loss += crossentropy(pred=y_pred, test=y_batch[j][i])

                theta = Model().update(alpha=0.01, gradients=grads, parameters=batch_theta, n_hidden=n_hidden)

                w = []; b = []
                for h in range(len(n_hidden)):
                    w.append(theta["W" + str(h)])
                    b.append(theta["b" + str(h)])
                ww.append(w)
                bb.append(b)

            for h in range(len(n_hidden)):
                sum_w = 0; sum_b = 0
                for i in range(batch_size):
                    sum_w += ww[i][h]
                    sum_b += bb[i][h]

                All_para["theta" + str(j)]["W" + str(h)] = sum_w / batch_size
                All_para["theta" + str(j)]["b" + str(h)] = sum_b / batch_size

            train_loss = train_loss / batch_size
            Y_pred = np.array(Y_pred).reshape(batch_size, -1)
            train_acc = get_acc(y_batch[j], Y_pred)

            print(str(j) + "/" + str(n_batches) + " loss: {0:.8f}".format(train_loss), " Accuracy: {0:.4f}".format(train_acc))
            acc += train_acc
            loss += train_loss
        acc = acc / n_batches
        loss = loss / n_batches
        Accuracy.append(acc)
        Loss.append(loss)
        print("--------- epochs: " + str(t + 1), "\n--------- accurcay: ", acc, "\n--------- loss: ", loss)

        model_para = {}
        for h in range(len(n_hidden)):
            weights = 0
            bias = 0
            for j in range(n_batches):
                weights += All_para["theta" + str(j)]["W" + str(h)]
                bias += All_para["theta" + str(j)]["b" + str(h)]
            model_para["W" + str(h)] = weights / n_batches
            model_para["b" + str(h)] = bias / n_batches

        val_loss = 0
        val_acc = 0
        for b in range(val_n_batches):
            batch_loss = 0
            Y_val_pred = []
            if len(X_val_batch[b]) != batch_size:
                batch_size = len(X_val_batch[b])
            for i in range(batch_size):
                function_output = Model(inputs=X_val_batch[b][i]).forward_pass(n_hidden=n_hidden, activation_list=activation_list,
                                                                        parameters=model_para)
                y_val_pred = function_output["A3"]
                Y_val_pred.append(y_val_pred)
                loss = crossentropy(y_val_pred, y_val_batch[b][i])
                batch_loss += loss
            Y_val_pred = np.array(Y_val_pred).reshape(batch_size, -1)
            val_acc += get_acc(y_val_batch[b], Y_val_pred)
            val_loss += batch_loss
        val_loss = val_loss / X_val.shape[0]
        val_acc = val_acc / val_n_batches
        Val_Loss.append(val_loss)
        Val_Accuracy.append(val_acc)
        print("--------- validation loss: ", val_loss, "\n-------- validation accuracy: ", val_acc)

    history = {}
    history["loss"] = Loss
    history["accuracy"] = Accuracy
    history["val_loss"] = Val_Loss
    history["val_accuracy"] = Val_Accuracy
    return model_para, history

def predict(X_test, y_test, model_para, n_hidden, activation_list):
    Y_pred = []
    Loss = []
    for i in range(X_test.shape[0]):
        function_output = Model(inputs=X_test[i]).forward_pass(n_hidden=n_hidden, activation_list=activation_list,
                                                                    parameters=model_para)
        y_pred = function_output["A3"]
        loss = crossentropy(y_pred, y_test[i])
        Y_pred.append(y_pred)
        Loss.append(loss)
    Y_pred = np.array(Y_pred).reshape(X_test.shape[0], -1)
    loss = sum(Loss) / X_test.shape[0]
    acc = get_acc(Y_pred, y_test)
    return Y_pred, loss, acc


