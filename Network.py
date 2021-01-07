import numpy as np
from numba import jit

@jit
def crossentropy(pred, test):
    for i in pred.reshape(-1):
        if i == 0.:
            i += 1e-6
        elif i == 1.:
            i -= 1e-6
    n = test.shape[0] #(n,1,10)
    loss = -test * np.log(pred)
    loss = 1.0 / n * np.sum(loss)
    return loss

@jit
def ReLU(x):
    y = np.array(x, dtype="float64").reshape(-1)
    for i in range(len(y)):
        if y[i] <= 0.:
            y[i] = 1e-6

    y = y.reshape(x.shape)
    return y

@jit
def softmax(x):
    x_0 = np.exp(x - np.max(x))
    SUM = np.sum(x_0)
    prob = x_0 / SUM
    return prob

@jit
def ReLU_grad(z):
    z = z.reshape(-1)
    z = np.diag(z)
    J = np.zeros([len(z), len(z)], dtype="int64")
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i, j] > 0.:
                J[i, j] += 1.
            else:
                J[i, j] += 1e-6
    return J

@jit
def softmax_grad(z):
    z = z.reshape(-1)
    J = np.zeros([len(z), len(z)], dtype="float32")
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
        np.random.seed(12)
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

class Model():
    def __init__(self, n_hidden, activation_list, inputs=None):
        self.inputs = inputs
        self.n_hidden = n_hidden
        self.activation_list = activation_list

    def init_parameters(self, input_shape):
        '''"n_hidden" is the structure of the network,
        "activation list" is the activation function of each hidden layer.'''
        theta = {}
        first_inputs = input_shape
        for i in range(len(self.n_hidden)):
            np.random.seed(0)
            weights = np.random.normal(0.0, 0.1, size=(first_inputs, self.n_hidden[i]))
            first_inputs = self.n_hidden[i]
            theta["W" + str(i)] = weights
            np.random.seed(1)
            bias = np.random.normal(0.0, 0.01, size=(1, self.n_hidden[i]))
            theta["b" + str(i)] = bias
            theta["activation" + str(i)] = self.activation_list[i]
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

    def forward_pass(self, parameters):
        'concatenate all the outputs of each hidden layer in n_hidden and activation_list, which return from the function "outputs"'
        output_1 = self.inputs
        function_output = {}
        for i in range(len(self.n_hidden)):
            #layer = nn(inputs=output_1) # (784,)
            output_1, output_2 = self.output(inputs=output_1, weights=parameters["W" + str(i)], bias=parameters["b" + str(i)], activation_function=self.activation_list[i])
            output_1 = output_1.reshape(1, -1)
            Z_2 = output_2.reshape(1, -1)
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
        dcda_2 = dcda_2.reshape(1, -1)
        dcdz = np.matmul(dcda_2, d_activate(z))
        dcdw = np.matmul(a_1.T, dcdz)
        dcdb = dcdz
        dcda = np.matmul(dcdz, weights_1.T)
        return dcdw, dcdb, dcda

    def back_propagation(self, pred, expected, parameters):
        '''n_hidden, activation_list is the concatenate of hidden layers, activation functions.
        pred is the output of the forward_pass, expected is the ground-truth
        parameters is the concatenate of all the weights and bias in different activation function of hidden layer.'''
        gradient = {}
        pred = pred.reshape(expected.shape)
        for i in pred:
            if i == 0:
                pred += 1e-6
        dcda_2 = - expected / pred #(1,10)

        function_output = self.forward_pass(parameters=parameters)
        for n, m in enumerate(reversed(range(len(self.n_hidden) - 1))): # m is the previous layer and n is the current layer
            n = m + 1
            A = function_output["A" + str(m)]
            Z = function_output["Z" + str(n)]
            W = parameters["W" + str(n)]
            b = parameters["W" + str(n)]
            dcdw, dcdb, dcda_2 = self.backward(dcda_2, A, Z, W, self.activation_list[n])
            gradient["dcdw" + str(n)] = dcdw
            gradient["dcdb" + str(n)] = dcdb
        Z_0, A_0 = self.output(inputs=self.inputs, weights=parameters["W0"], bias=parameters["b0"], activation_function=None)
        dcdw0, dcdb0, dcda0 = self.backward(dcda_2, self.inputs.reshape(1, -1), Z_0, parameters["W0"], activation_function=self.activation_list[0])
        gradient["dcdw0"] = dcdw0
        gradient["dcdb0"] = dcdb0
        return gradient

    def update(self, alpha, gradients, parameters):
        new_theta = {}
        for i in range(len(self.n_hidden)):
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
        if y[j][i] == np.max(y[j]) and y[j][i] >= 0.5:
            correct += 1
    return correct / len(y)

def train(inputs, n_hidden, activation_list, theta, lr,  mode="Train", y_test=None):
    net = Model(inputs=inputs, n_hidden=n_hidden, activation_list=activation_list)
    function_output = net.forward_pass(parameters=theta)
    y_pred = function_output["A" + str(len(n_hidden) - 1)]
    if mode is "Train":
        grads = net.back_propagation(pred=y_pred, expected=y_test, parameters=theta)
        theta = net.update(alpha=lr, gradients=grads, parameters=theta)
    elif mode is "Validation":
        theta = theta
    return y_pred, theta

def batch_train(batch_theta, bX, bY, batch_size, n_hidden, activation_list, lr):
    ww = []
    bb = []
    Y_pred = []
    train_loss = 0
    if len(bX) != batch_size:
        batch_size = len(bX)
    for i in range(batch_size):
        y_pred, theta = train(bX[i], n_hidden, activation_list, batch_theta, mode="Train", y_test=bY[i], lr=lr)
        train_loss += crossentropy(pred=y_pred, test=bY[i])

        Y_pred.append(y_pred)
        w = []; b = []
        for h in range(len(n_hidden)):
            w.append(theta["W" + str(h)])
            b.append(theta["b" + str(h)])
        ww.append(w)
        bb.append(b)

    Y_pred = np.array(Y_pred).reshape(batch_size, -1)
    train_loss /= batch_size
    train_acc = get_acc(bY, Y_pred)

    return train_loss, train_acc, ww, bb


def mini_batch_training(X_train, y_train, n_hidden, activation_list, batch_size, epochs, lr, X_val, y_val,):

    X_batch, y_batch, n_batches = mini_batch(X_train, y_train, batch_size=batch_size)
    X_val_batch, y_val_batch, val_n_batches = mini_batch(X_val, y_val, batch_size=batch_size)
    m = Model(n_hidden=n_hidden, activation_list=activation_list)
    initial_parameters = m.init_parameters(input_shape=X_train.shape[-1])
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
            train_loss, train_acc, ww, bb = batch_train(batch_theta=batch_theta, bX=X_batch[j], bY=y_batch[j], lr=lr,
                                                        batch_size=batch_size, n_hidden=n_hidden, activation_list=activation_list)

            for h in range(len(n_hidden)):
                sum_w = 0;
                sum_b = 0
                if len(ww) != batch_size:
                    batch_size = len(ww)
                for i in range(batch_size):
                    sum_w += ww[i][h]
                    sum_b += bb[i][h]
                All_para["theta" + str(j)]["W" + str(h)] = sum_w / batch_size
                All_para["theta" + str(j)]["b" + str(h)] = sum_b / batch_size
            acc += train_acc
            loss += train_loss
        acc /= n_batches
        epoch_loss = loss / n_batches

        Accuracy.append(acc)
        Loss.append(epoch_loss)

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
                y_val_pred, _ = train(inputs=X_val_batch[b][i], n_hidden=n_hidden, activation_list=activation_list, lr=lr,
                                      theta=model_para, mode="Validation")
                Y_val_pred.append(y_val_pred)
                batch_loss += crossentropy(y_val_pred, y_val_batch[b][i])

            batch_loss /= batch_size
            Y_val_pred = np.array(Y_val_pred).reshape(batch_size, -1)
            val_acc += get_acc(y_val_batch[b], Y_val_pred)
            val_loss += batch_loss

        val_loss /= val_n_batches
        val_acc /= val_n_batches
        Val_Loss.append(val_loss)
        Val_Accuracy.append(val_acc)
        print("Epochs {} | loss {:.6f} | accuracy {:.6f} | val_loss {:.6f}".format(t + 1, epoch_loss, acc, val_loss))

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
        test_model = Model(inputs=X_test[i], n_hidden=n_hidden, activation_list=activation_list)
        function_output = test_model.forward_pass(parameters=model_para)
        y_pred = function_output["A" + str(len(n_hidden) - 1)]
        loss = crossentropy(y_pred, y_test[i])
        Y_pred.append(y_pred)
        Loss.append(loss)
    Y_pred = np.array(Y_pred).reshape(X_test.shape[0], -1)
    loss = sum(Loss) / X_test.shape[0]
    acc = get_acc(Y_pred, y_test)
    return Y_pred, loss, acc


