from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, Upsample, MaxPool2d

class Auto_encoder(nn.Module):
    def __init__(self):
        super(Auto_encoder, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.5),
            MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            Conv2d(10, 20, 3, 1, 1),
            nn.LeakyReLU(0.5),
            MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            Conv2d(20, 40, 3, 1, 1),
            Conv2d(40, 160, 3, 1, 1)
            #nn.LeakyReLU(0.5),
        )
        self.deconv1 = nn.Sequential(
            Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.LeakyReLU(0.5),
            ConvTranspose2d(160, 40, 3, 1, 1),
            ConvTranspose2d(40, 20, 3, 1, padding=2, dilation=2, output_padding=1),
            nn.LeakyReLU(0.5),
            Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvTranspose2d(20, 10, 3, 1, 1),
            #nn.LeakyReLU(0.5),
            ConvTranspose2d(10, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        hx = self.conv2(x)
        hx = self.conv3(hx)
        output = self.deconv1(hx)
        return output, hx

    def decode(self, hidden):
        output = self.deconv1(hidden)
        output = output.reshape(-1, 26, 26, 3)
        output = nn.Softmax(dim=3)(output)
        return output