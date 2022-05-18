import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

"""
Model wrappers
"""


class Unet(smp.Unet):
    def __init__(self):
        super(Unet, self).__init__()
        self.model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class FPN(smp.fpn.FPN):
    def __init__(self):
        super(FPN, self).__init__()
        self.model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class DeepLab(smp.DeepLabV3Plus):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.model = smp.DeepLabV3Plus(in_channels=3, classes=1,
                                       activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class TriUnet(nn.Module):
    def __init__(self):
        super(TriUnet, self).__init__()
        self.Unet1 = Unet()
        self.Unet2 = Unet()
        self.Unet3 = smp.Unet(in_channels=2, classes=1, activation="sigmoid")

    def forward(self, x):
        mask1, mask2 = self.Unet1(x), self.Unet2(x)
        mask3 = self.Unet3(torch.cat((mask1, mask2), 1))
        return mask3

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask
