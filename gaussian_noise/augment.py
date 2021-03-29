import numpy as np
import torch
from skimage.util import random_noise
import matplotlib.pyplot as plt
from Auto_encoder import Auto_encoder
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dict = torch.load("./model_output/auto_encoder_para.pkl", map_location=lambda storage, loc: storage)

'''plot loss '''
loss = dict["loss"]
plt.plot(range(len(loss)), loss)
plt.title("act_func: Leaky_ReLU, Conv: 4, Encode:[-1, 160, 6, 6] ")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

encode = torch.load("./model_output/latent_vector.pkl")

def gaussian_noise(data, nums_img):
    augm_img = []
    for img in data:
        noise = []
        for i in range(nums_img):
            gauss_img = random_noise(img.detach().cpu().numpy(), mode="gaussian", mean=0.0, var=0.0001)
            noise.append(torch.from_numpy(gauss_img).cuda())
        noise_0 = torch.stack(noise).cuda()
        augm_img.append(noise_0)
    augm_img = torch.stack(augm_img)
    augm_img = augm_img.reshape(-1, data.shape[1], data.shape[2], data.shape[3])
    return augm_img

nums_image = 5
noise = gaussian_noise(encode, nums_img=nums_image)
noise = torch.tensor(noise, dtype=torch.float).cpu()
print("noise: ", noise.shape, type(noise))

model = Auto_encoder()
del dict["loss"]
model.load_state_dict(dict)

encode = encode.cpu()
out = model.decode(noise)
print("generate: ", out.shape)
np.save("./data/gen_images.npy", out.detach().cpu().numpy())

from keras.preprocessing.image import array_to_img
from PIL import Image
sample = model.decode(encode)
Image._show(array_to_img(sample[0].detach().cpu().numpy()))

def visualize(data, nums_img):
    v = []
    for i in range(0, len(data) - 1, nums_img):
        if i + nums_img <= len(data):
            sample = torch.cat(list(data[i:i + nums_img]), dim=1)
            sample = sample.detach().cpu().numpy()
            v.append(sample)
    return v

vis = visualize(out, nums_img=nums_image)
print(vis[0].shape)

for i in range(len(vis)):
    noise_img = array_to_img(vis[i])
    noise_img.save("./noise_images/" + str(i) + ".png")

