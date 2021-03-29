import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Conv2d, ConvTranspose2d, MaxPool2d, Upsample
from sklearn.model_selection import train_test_split, cross_validate
from torch.utils.data import DataLoader, Dataset
from skimage.util import random_noise


device = torch.device("cuda")

data = np.load("data.npy")
label = np.load("label.npy")

data = torch.tensor(data, dtype=torch.float)
label_images = torch.tensor(np.load("label_images.npy"), dtype=torch.float)

data = data.view(-1, 3, 26, 26)
label = torch.tensor(label, dtype=torch.long)

class Data_image(Dataset):
    def __init__(self, images, label_images):
        self.images = images
        self.label_images = label_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.label_images[item]

All_data_loader = DataLoader(dataset=Data_image(data, label_images), batch_size=64, shuffle=False)

from torch import optim
from torch.nn import CrossEntropyLoss
from torchsummary import summary
import time
from Auto_encoder import Auto_encoder

net = Auto_encoder().cuda()
print(summary(net, input_size=(3, 26, 26)))
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = CrossEntropyLoss().cuda()

n_iter = 2000

def train(model, epochs, loss_func, data_loader):
    train_epoch_losses = []
    dur = []
    for epoch in range(epochs):
        train_epoch_loss = 0
        if epoch >= 1:
            t0 = time.time()
        for j, (images, label) in zip(range(len(data_loader)), data_loader):
            images = images.cuda()
            label = label.to(torch.long).cuda()
            prediction, hidden = model(images)
            loss = 0
            prediction = prediction.reshape(-1, images.shape[2] * images.shape[3], images.shape[1])
            label = label.reshape(-1, label.shape[1] * label.shape[2])
            for i in range(len(label)):
                loss += loss_func(prediction[i,:,:], label[i,:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += (loss / len(images))
            if j == 0:
                latent = hidden
            else:
                latent = torch.cat([latent, hidden]).cuda()
        train_epoch_loss /= len(data_loader)
        if epoch >= 1:
            dur.append(time.time() - t0)
        print('Epoch {} | loss {:.4f} | Time(s) {:.4f} '.format(epoch, train_epoch_loss, np.mean(dur)))
        train_epoch_losses.append(train_epoch_loss)
    return model, latent, train_epoch_losses

model, latent, loss = train(net, epochs=n_iter, loss_func=criterion, data_loader=All_data_loader, )

dict = net.state_dict()
dict["loss"] = loss
torch.save(dict, "./model_output/auto_encoder_para.pkl")
torch.save(latent, "./model_output/latent_vector.pkl")




