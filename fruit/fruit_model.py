import numpy as np
import PIL.Image as image
import os
import pickle
from Network import ReLU, softmax
import warnings
warnings.filterwarnings('ignore')

data_train = os.listdir("./Data_train")
fn_1 = "./Data_train/" + data_train[0]
fn_2 = "./Data_train/" + data_train[1]
fn_3 = "./Data_train/" + data_train[2]

A = os.listdir(fn_1)
B = os.listdir(fn_2)
C = os.listdir(fn_3)

carambula_train = [np.array(image.open(fn_1 + "/" + A[i])) for i in range(len(A))]
lychee_train = [np.array(image.open(fn_2 + "/" + B[i])) for i in range(len(B))]
pear_train = [np.array(image.open(fn_3 + "/" + C[i])) for i in range(len(C))]

y = ["Carambula"] * len(A) + ["Lychee"] * len(B) + ["Pear"] * len(C)
label = np.diag(np.ones(3))
y_train = [label[0] for i in y if i == "Carambula"] + [label[1] for i in y if i == "Lychee"] + [label[2] for i in y if i == "Pear"]
y_train = np.array(y_train)
X_train = np.array(carambula_train + lychee_train + pear_train)

X_augment_1 = np.array([np.rot90(X_train[i]) for i in range(len(X_train))])
np.random.seed(100)
X_noise = X_train + np.random.normal(1, 0.01, X_train.shape)

X_train = np.concatenate([X_train,X_augment_1, X_noise], axis=0).transpose(0, 3, 1, 2)
X_train /= np.max(X_train)
y_train = np.concatenate([y_train, y_train, y_train], axis=0)
print(y_train.shape, X_train.shape)

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
    return X_train, y_train, X_val, y_val

from HW_3.Conv import net

X_train, y_train = shuffle(X_train, y_train)
X_train, y_train, X_val, y_val = split(X_train, y_train, rate=0.3)
print("train: ", len(X_train), "validation: ", len(X_val), y_train[0])

'''4. FC layers were included in structure'''
structure = [100, 50, 3]

'''3. ReLU layer'''
'''5. Softmax output'''
activation_list = [ReLU, ReLU, softmax]

Net = net(n_linear=structure, activation_list=activation_list, batch_size=32)

'''6. Cross-entropy loss is the default loss function'''
'''7. Backpropagation included in Network.py'''
model, history = Net.fit(inputs=X_train, outputs=y_train, x_val=X_val, y_val=y_val, epochs=100)

with open("./model.pickle", "wb") as f:
    pickle.dump((model, history), f)