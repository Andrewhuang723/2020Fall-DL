import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from numba import jit, vectorize


def nums_of_images(path):
    data = open(path, "rb")
    data = data.read()
    magic, images, rows, columns = struct.unpack_from('>iiii', data, 0)
    return images
train_nums = nums_of_images("train-images.idx3-ubyte")
test_nums = nums_of_images("t10k-images.idx3-ubyte")
print("training data: ", train_nums, "\ntesting data: ", test_nums)

def read_image(path, n_samples):
    data = open(path, "rb")
    data = data.read()
    index = 0
    index += struct.calcsize(">iiii")
    Info = []
    for i in range(n_samples):
        image = struct.unpack_from(">784B", data, index)
        image = np.reshape(image, (28, 28))
        Info.append(image)
        index += struct.calcsize(">784B")
    return np.array(Info)

def read_label(path, n_samples):
    data = open(path, "rb")
    data = data.read()
    index = 0
    index += struct.calcsize(">II")
    labels = np.array(struct.unpack_from(str(">") + str(n_samples) + str("B"), data, index))
    labels = labels[:, np.newaxis]
    return labels

X_train = read_image("train-images.idx3-ubyte", train_nums)
X_test = read_image("t10k-images.idx3-ubyte", test_nums)
y_train = read_label("train-labels.idx1-ubyte", train_nums)
y_test = read_label("t10k-labels.idx1-ubyte", test_nums)
print("X_train: ", X_train.shape, "\nX_test: ", X_test.shape,
      "\ny_train: ", y_train.shape, "\ny_test: ", y_test.shape)
#plt.imshow(X_train[10])
#plt.show()



### Normalization
max_train = np.max(X_train)
max_test = np.max(X_test)
max = max(max_train, max_test)
X_train = X_train / max + 0.01
X_test = X_test / max

## t-SNE

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
t_sne = TSNE(n_components=2, perplexity=30)
'''
t_sne_test = t_sne.fit_transform(X_test.reshape(-1,784))
print(t_sne_test.shape)
data = np.concatenate((t_sne_test, y_test), axis=1)
print(data.shape)
data = pd.DataFrame(data, columns=["feature_1", "feature_2", "label"])
sns.FacetGrid(data, hue="label", size=6).map(plt.scatter, "feature_1", "feature_2", s=5).add_legend()
plt.show()
'''

def one_hot_encoder(data):
    Y = []
    for label in data:
        label = int(label)
        encode = np.zeros(10)
        for i in range(len(encode)):
            if i == label:
                encode[i] += 1
        Y.append(encode)
    return np.array(Y)

'''Give bias to y to prevent Nan from crossentropy loss'''
y_train = one_hot_encoder(y_train) + 0.01
y_test = one_hot_encoder(y_test) + 0.01
print(y_train.shape, y_test.shape)

def reversed_one_hot(y):
    k = []
    for i in y:
        j = np.where(i == np.amax(i))
        k.append(j)
    k = np.array(k).reshape(-1, 1)
    return k




def shuffle(X, Y):
    np.random.seed(1)
    randomlist = np.arange(X.shape[0])
    np.random.shuffle(randomlist)
    return X[randomlist], Y[randomlist]

''' 1. Splitting into Training & Validation data'''
def split(X, Y, rate):
    X_train = X[int(X.shape[0]*rate):]
    y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    y_val = Y[:int(Y.shape[0]*rate)]
    return X_train,y_train,X_val,y_val

X_train, y_train = shuffle(X_train, y_train)
X_train, y_train, X_val, y_val = split(X_train, y_train, rate=0.3)
X_train = X_train.reshape(-1,784) #(42000, 784)
X_val = X_val.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
print("X_train",X_train.shape, "X_val: ", X_val.shape, "\ny_val: ", y_val.shape)

### tsne
t_sne = TSNE(n_components=2, perplexity=30)
t_sne_y_val = reversed_one_hot(y_val)
print(t_sne_y_val.shape)
t_sne_val = t_sne.fit_transform(X_val)
val_data = np.concatenate((t_sne_val, t_sne_y_val), axis=1)
print(val_data.shape)
val_data = pd.DataFrame(val_data, columns=["feature_1", "feature_2", "label"])
sns.FacetGrid(val_data, hue="label", size=6).map(plt.scatter, "feature_1", "feature_2", s=5).add_legend()
plt.show()




'''Network construct'''
from Network import ReLU, softmax, mini_batch, predict
from Network import mini_batch_training

X_batch, y_batch, n_batches = mini_batch(X_train, y_train, batch_size=32)

'''Model structure'''
structure = [100, 50, 20, 10]
act_list = [ReLU, ReLU, ReLU, softmax]

'''Training Model'''
model, history = mini_batch_training(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                     n_hidden=structure, activation_list=act_list, batch_size=128, epochs=100)

'''Prediction & Evaluation'''
y_pred, loss, acc = predict(X_test=X_test, y_test=y_test, model_para=model, n_hidden=structure, activation_list=act_list)
print("test: ", "\nloss: ", loss, " acc: ", acc)

import pandas as pd
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_excel("prediction.xlsx")

'''Plot'''
loss = history["loss"]
val_loss = history["val_loss"]
accuracy = history["accuracy"]
val_acc = history["val_accuracy"]

plt.figure()
t = range(len(loss))
plt.plot(t, loss, c="k",)
plt.plot(t, val_loss, c="b",)
plt.legend(["Training", "Validation"], fontsize=8)
plt.title("Loss v.s epochs")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(t, accuracy, c="k")
plt.plot(t, val_acc, c="b")
plt.legend(["Training", "Validation"], fontsize=8)
plt.title("Accuracy v.s epochs")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show()

