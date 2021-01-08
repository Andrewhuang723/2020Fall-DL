import numpy as np
from Network import Model, ReLU, mini_batch_training, predict

'''2. 2D-Convolution'''
class Convolution():
    def __init__(self, kernel_size, channels, out_channels, strides, padding):
        self.kernel_size = kernel_size
        self.channels = channels
        self.out_channels = out_channels
        self.strides = strides
        self.padding = padding

    def filter(self):
        output = []
        for i in range(self.out_channels):
            np.random.seed(5)
            kernel = np.random.randint(0, 2, size=(self.kernel_size, self.kernel_size))
            output.append(kernel)
        return output

    def __call__(self, inputs):
        h_out = int((inputs.shape[-2] - (self.kernel_size - 1) - 1) / self.strides + 1)
        w_out = int((inputs.shape[-1] - (self.kernel_size - 1) - 1) / self.strides + 1)
        k = []
        #print(inputs.shape)
        for batch in range(inputs.shape[0]):
            for f in self.filter():
                for h in range(0, h_out, self.strides):
                    for w in range(0, w_out, self.strides):
                        k.append(np.sum(inputs[batch][:, h: h + self.kernel_size, w: w + self.kernel_size] * f))
        k_1 = np.array(k).reshape(inputs.shape[0], self.out_channels, h_out, w_out)

        copy = np.zeros([inputs.shape[0], self.out_channels, inputs.shape[-2], inputs.shape[-1]])
        for batch in range(inputs.shape[0]):
            for c in range(self.out_channels):
                if self.padding == "same":
                    add = int((inputs.shape[-1] - h_out) / 2)
                    zero_pad_1 = np.zeros([add, h_out])
                    pad = np.vstack((zero_pad_1, k_1[batch][c], zero_pad_1))
                    copy[batch][c] = np.hstack((np.zeros([pad.shape[0], add]), pad, np.zeros([pad.shape[0], add])))

                elif self.padding == "None":
                    copy[batch][c] = k_1[batch][c]

                else:
                    zero_pad_1 = np.zeros([self.padding, h_out])
                    pad = np.vstack((zero_pad_1, k_1[batch][c], zero_pad_1))
                    copy[batch][c] = np.hstack((np.zeros([pad.shape[0], self.padding]), pad, np.zeros([pad.shape[0], self.padding])))
        return copy


class Pooling:
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, pooling="max"):
        h_out = int(inputs.shape[-1] / self.size)
        w_out = int(inputs.shape[-2] / self.size)
        out = np.zeros([inputs.shape[0], inputs.shape[1], h_out, w_out])
        for batch in range(inputs.shape[0]):
            for i in range(inputs.shape[1]):
                for h in range(0, inputs.shape[-1], self.size):
                    for w in range(0, inputs.shape[-2], self.size):
                        if pooling == "max":
                            out[batch][i][int(h / 2), int(w / 2)] += np.max(inputs[batch][i][h: h + self.size, w: w + self.size])
                        elif pooling == "average":
                            out[batch][i][int(h / 2), int(w / 2)] += np.mean(inputs[batch][i][h: h + self.size, w: w + self.size])
        return out

class Linear(Model):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def init_parameters(self):
        theta = {}
        np.random.seed(0)
        weights = np.random.normal(0.0, 0.1, size=(self.input_size, self.output_size))
        np.random.seed(1)
        bias = np.random.normal(0.0, 0.01, size=(1, self.output_size))
        theta["W"] = weights
        theta["b"] = bias
        return theta

    def __call__(self, inputs):
        z = np.matmul(inputs, self.init_parameters()["W"]) + self.init_parameters()["b"]
        return z

def Flatten(inputs):
    out = inputs.reshape(inputs.shape[0], -1)
    return out



class net(Model):
    def __init__(self, n_linear, activation_list, batch_size):
        super(net, self).__init__(n_hidden=n_linear, activation_list=activation_list)
        self.conv_1 = Convolution(kernel_size=3, channels=2, out_channels=3, strides=1, padding="same")
        self.pool_1 = Pooling(size=2)
        self.conv_2 = Convolution(kernel_size=3, channels=3, out_channels=5, strides=1, padding="same")
        self.n_linear = n_linear
        self.activation_list = activation_list
        self.batch_size = batch_size

    def Conv_2d(self, inputs):
        out = ReLU(self.conv_1(inputs))
        out = self.pool_1(out)
        out = ReLU(self.conv_2(out))
        out = self.pool_1(out)
        out = Flatten(out)
        return out

    def fit(self, inputs, outputs, x_val, y_val, epochs):
        inputs = self.Conv_2d(inputs)
        x_val = self.Conv_2d(x_val)
        model, history = mini_batch_training(X_train=inputs / np.max(inputs), y_train=outputs, n_hidden=[inputs.shape[-1]] + self.n_linear,
                                             activation_list=[ReLU] + self.activation_list, lr=0.01,
                                             batch_size=self.batch_size, X_val=x_val / np.max(inputs), y_val=y_val, epochs=epochs)
        return model, history

    def eval(self, X_test, y_test, model):
        inputs = self.Conv_2d(X_test)
        y_pred, loss, acc = predict(inputs / np.max(inputs), y_test, model, n_hidden=[inputs.shape[-1]] + self.n_linear,
                                         activation_list=[ReLU] + self.activation_list,)
        return y_pred, loss, acc

