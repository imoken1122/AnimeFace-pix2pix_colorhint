
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(dim),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(dim),
            nn.Conv2d(dim, dim, kernel_size=kernel_size))
        

    def forward(self, x):

        y = self.conv_block(x)
        out = x + y
        return out

class SLEBlock(nn.Module):
    def __init__(self,in_ch1,in_ch2,):
        super(SLEBlock, self).__init__()
        self.layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((4,4)),
                nn.Conv2d(in_ch2,in_ch2,kernel_size=4,stride=1,padding=0),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_ch2,in_ch1,kernel_size=1,stride=1,padding=0),
                nn.Sigmoid()

        )
    def forward(self,x1_skip,x2):
        x2 = self.layer(x2)

        output = x1_skip * x2
        return output



# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out