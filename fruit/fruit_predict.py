import numpy as np
import os
import PIL.Image as image
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

data_test = os.listdir("./Data_test")
fn_1 = "./Data_test/" + data_test[0]
fn_2 = "./Data_test/" + data_test[1]
fn_3 = "./Data_test/" + data_test[2]

A = os.listdir(fn_1)
B = os.listdir(fn_2)
C = os.listdir(fn_3)

carambula_test = [np.array(image.open(fn_1 + "/" + A[i])) for i in range(len(A))]
lychee_test = [np.array(image.open(fn_2 + "/" + B[i])) for i in range(len(B))]
pear_test = [np.array(image.open(fn_3 + "/" + C[i])) for i in range(len(C))]

y = ["Carambula"] * len(A) + ["Lychee"] * len(B) + ["Pear"] * len(C)
label = np.diag(np.ones(3))
y_test = [label[0] for i in y if i == "Carambula"] + [label[1] for i in y if i == "Lychee"] + [label[2] for i in y if i == "Pear"]
y_test = np.array(y_test)
X_test = np.array(carambula_test + lychee_test + pear_test) / 255
X_test = X_test.transpose(0, 3, 1, 2)
print(y_test.shape, X_test.shape)

from HW_3.Conv import net
from Network import ReLU, softmax

with open("./model.pickle", "rb") as f:
    model, history = pickle.load(f)

'''Plot'''
loss = history["loss"]
loss.remove(max(loss))
val_loss = history["val_loss"]
val_loss.remove(max(val_loss))
accuracy = history["accuracy"]
accuracy.remove(min(accuracy))
val_acc = history["val_accuracy"]
val_acc.remove(min(val_acc))

'''8. Validation loss & accuracy'''
print(min(val_loss), max(val_acc))

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

structure = [100, 50, 3]
activation_list = [ReLU, ReLU, softmax]
Net = net(n_linear=structure, activation_list=activation_list, batch_size=32)
y_pred, loss, acc = Net.eval(X_test, y_test, model)

'''9. Testing loss & accuracy'''
print("\nloss: ", loss, " acc: ", acc)
print(y_pred.shape, y_test.shape)

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred_df = pd.DataFrame(np.concatenate([y_pred, np.array(y).reshape(-1, 1)], axis=1), columns=["Carambula", "Lychee", "Pear", "Truth value"])
y_pred_df.to_csv("./y_pred.csv")

y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)
cf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
label = ["Carambula", "Lychee", "Pear"]
sns.heatmap(cf_matrix / np.sum(cf_matrix), xticklabels=label, yticklabels=label, annot=True, fmt='.2%', cmap='Blues')
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()